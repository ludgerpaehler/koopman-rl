# GPU-enabling KoopmanRL: environments & algorithms

**Date:** 2026-05-22
**Status:** Approved (design); pending implementation plan
**Author:** Ludger Paehler (with Claude)

## Goal

Make the KoopmanRL package run its reinforcement-learning algorithms **and** its 4 control
environments on an NVIDIA GPU (target hardware: RTX 5090 / Blackwell `sm_120`, validated with
`torch==2.9.1+cu128`). The Koopman-assisted algorithms (SKVI, SAKC) currently contain parts that
cannot run on GPU at all (numpy round-trips inside the Koopman tensor, default-device tensor
allocations, hardcoded `device="cpu"`). This effort fixes that and validates the GPU paths against
the existing CPU behavior.

## Baseline (already verified on CPU, 2026-05-22)

- `tests/test_environments.py`: 8/8 passed.
- `tests/test_rl.py`: 20/20 passed (LQR, SAC-Q, SAC-V, SKVI, SAKC × {LinearSystem, FluidFlow,
  Lorenz, DoubleWell}).

This is the reference behavior the GPU port is validated against.

## Decisions (locked)

| Topic | Decision |
|---|---|
| Env GPU model | **Hybrid**: keep the gym single-env numpy API; add batched, device-native env dynamics for data-gen + training. |
| ODE integrator (FluidFlow/Lorenz) | **Substepped fixed-step RK4** in torch, batched. Replaces `scipy.integrate.solve_ivp` (RK45) on the batched GPU path only. |
| Precision | **FP64 default, `--fp32` opt-in.** |
| Validation | **Parity tests added to the suite** (CPU vs GPU within tolerance, plus GPU smoke runs). |
| Algorithm scope | SKVI, SAKC, both Optuna wrappers, **both** CleanRL SAC variants. **LQR stays CPU** (classical Riccati via `control`/scipy). |
| Code structure | **Single source of truth**: `koopman_tensor/torch_tensor.py` becomes the one device-aware `KoopmanTensor`; SKVI & SAKC import it instead of their inline copies. |
| Default device | **`--cuda` stays opt-in** (default CPU, preserves exact reproducibility of existing runs). Tests exercise `--cuda`. |

## Non-goals

- Porting **LQR** to GPU (classical scipy/`control` Riccati solve; no GPU benefit).
- Changing the **default device** (GPU is opt-in via `--cuda`).
- Changing SAC **algorithm logic** — only restoring device plumbing.
- Touching `interpretability_discrete_value_iteration.py` beyond optional alignment (it is already
  device-aware and not in the test suite).

---

## Architecture

### 1. Device & dtype foundation

Add helpers to `koopmanrl/utils.py`:

```python
def resolve_device(cuda: bool) -> torch.device:
    return torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

def resolve_dtype(fp32: bool) -> torch.dtype:
    return torch.float32 if fp32 else torch.float64
```

- Replace the hardcoded `device = torch.device("cpu")` in `soft_koopman_value_iteration.py`,
  `soft_actor_koopman_critic.py`, `sac_continuous_action.py`,
  `value_based_sac_continuous_action.py`, and `opt_wrappers.py` with `resolve_device(args.cuda)`.
  (The `--cuda` flag is currently parsed but ignored in all of these.)
- Add a `--fp32` flag (default off → FP64) to the Koopman algorithms and the Optuna wrappers.
  Default dtype handling stays FP64 unless `--fp32` is set.

### 2. Canonical Koopman tensor (`koopman_tensor/torch_tensor.py`)

Make this the single device-aware implementation; have SKVI & SAKC import `KoopmanTensor`,
`Regressor`, `generate_koopman_tensor` from here (delete the inline duplicates). Fixes the three
CPU detours:

1. **`kron_matrix` Python loop** (`for i in range(N): torch.kron(...)`) →
   `torch.einsum("in,jn->ijn", Psi_U, Phi_X).reshape(psi_dim * phi_dim, N)`.
   This matches `torch.kron(u_col, x_col)` ordering: element `[i*phi_dim + j, n] = Psi_U[i,n]*Phi_X[j,n]`.
2. **`self.M.numpy()` + per-row Fortran-order reshape into `self.K`** →
   `K = M.reshape(phi_dim, psi_dim, phi_dim).transpose(-1, -2).contiguous()`.
   (F-order reshape of a length-`phi_dim*psi_dim` row into `(phi_dim, psi_dim)` equals the C-order
   reshape into `(psi_dim, phi_dim)` transposed.) Stays on-device, no numpy.
3. Constants and inputs carry `device`/`dtype` (inferred from input `X` when it is a tensor).

The regressors (`ols/rrr/sindy/ridge`) are already `torch.linalg` and GPU-capable.

**Risk:** `torch.linalg.lstsq` on CUDA uses the `gels` driver (assumes full rank, ignores `rcond`).
Mitigation: the existing rank checks; validate numerically; fall back to `torch.linalg.pinv`-based
solve (or a one-time CPU solve) for the tensor build if instability appears. The tensor build is a
one-time, small operation (phi_dim ≈ 10, N ≈ 30k).

### 3. Device-aware observables (`koopman_tensor/observables/torch_observables.py`)

`monomials.__call__/diff/ddiff` must allocate output tensors on `x.device`/`x.dtype` and move the
powers matrix `c` to `x.device` (today they create CPU default-device tensors, e.g.
`torch.ones([n, m])`). `koopman_observables.py` re-exports/aligns with this canonical module.

### 4. Batched, device-native env dynamics

Each env (`LinearSystem`, `FluidFlow`, `Lorenz`, `DoubleWell`) gains vectorized, device-aware batch
ops on `(batch, state_dim)` torch tensors. **The existing numpy `step`/`f`/`reset`/`continuous_f`
stay untouched** — they remain the gym-compatible CPU path and the parity reference.

- `f_batch(states, actions) -> next_states` (batch, state_dim):
  - LinearSystem: `states @ A.T + actions @ B.T`.
  - FluidFlow / Lorenz: substepped torch RK4 of `continuous_f`, batched (`n_substeps` configurable,
    default chosen to match RK45 accuracy at dt=0.01).
  - DoubleWell: torch Euler–Maruyama (drift + state-dependent diffusion), torch RNG.
- `reset_batch(n, generator) -> states`: uniform sampling within state bounds, on device.
- Env constants (`A, B, Q, R, reference_point, continuous_A/B`) lazily converted to torch tensors on
  the requested device/dtype and cached.
- `vectorized_cost_fn` made device-correct: convert `Q/R/reference_point` to torch on the input
  states' device. (Today it mixes numpy `Q/R` with torch tensors — works on CPU by coincidence,
  breaks on GPU.)

### 5. Batched GPU rollout for Koopman data generation

`generate_koopman_tensor` gains a batched code path: when on CUDA, collect `X/Y/U` by running all
`num_paths` rollouts **in parallel** via `reset_batch` + `f_batch` for `num_steps_per_path` steps,
with random actions sampled on-device within the action bounds. This replaces ~`num_paths *
num_steps_per_path` (≈30k) sequential `solve_ivp`/`step` calls with `num_steps_per_path` batched GPU
steps — the single biggest speedup. The current sequential gym loop is retained as the CPU
fallback / parity reference.

### 6. Algorithm wiring

- **SKVI** (`soft_koopman_value_iteration.py`):
  - Import canonical `KoopmanTensor`; move tensor (`K, B, Phi_X, X`), `all_actions`, value-function
    weights, and costs to `device`.
  - `DiscreteKoopmanValueIterationPolicy` takes a `device`; its internal `torch.zeros([...])`
    allocations (in `pis`, `discrete_bellman_error`, `train`) get `device=`.
  - **Vectorize the per-action Python loop** (`for action_index in range(K_us.shape[0])`) into
    `torch.einsum("aij,jb->aib", K_us, phi_x_batch)` followed by a batched value evaluation —
    removes the 101-iteration loop (big GPU win) in `pis`, `discrete_bellman_error`, and `train`.
- **SAKC** (`soft_actor_koopman_critic.py`): device resolution; SB3 `ReplayBuffer(device=cuda)`;
  `koopman_tensor.phi_f` and `SoftKoopmanVNetwork` on device; obs/action tensors on device.
- **SAC variants** (`sac_continuous_action.py`, `value_based_sac_continuous_action.py`): restore
  CleanRL-style device resolution from the hardcoded CPU. No algorithm-logic changes.
- **Optuna wrappers** (`opt_wrappers.py`): same device wiring; add `cuda`/`fp32` parameters.

---

## Validation & testing

### New: `tests/test_gpu_parity.py` (skipped when CUDA unavailable)

- **Env dynamics parity (per env):**
  - CPU-RK4 vs GPU-RK4 of `f_batch` on identical inputs → **tight** parity (FP64, atol≈1e-9).
  - GPU-RK4 vs CPU numpy `f` (RK45 reference) → closeness within an **integrator tolerance**
    (looser, reflects RK4-vs-RK45 difference), single-step / short-horizon only (Lorenz is chaotic;
    no long-trajectory parity).
  - DoubleWell: parity on the **drift** term with injected identical noise.
- **Cost parity:** `vectorized_cost_fn` CPU vs GPU within tol.
- **Koopman tensor parity:** build from identical `X/Y/U` on CPU vs GPU → `K`, `B`, `M`, `phi_f`,
  `f` match within tol; `monomials` outputs match.

### Extend `tests/test_rl.py`

- Add `--cuda` smoke runs (assert returncode 0) for SKVI, SAKC, and the SAC variants on at least one
  env. Existing CPU tests remain unchanged.

### Acceptance criteria

1. All existing CPU tests still pass (28/28).
2. New GPU parity tests pass on the RTX 5090.
3. SKVI, SAKC, SAC-Q, SAC-V run end-to-end with `--cuda` on all 4 envs (returncode 0), with tensors
   confirmed resident on the GPU.
4. The Koopman tensor build and SKVI action evaluation contain no numpy round-trips or Python
   per-element loops on the hot path.

## Risks & mitigations

- **`lstsq` on CUDA** (full-rank `gels`, no `rcond`): rely on rank checks; fall back to `pinv`/CPU
  solve for the one-time build if needed.
- **FP64 on consumer Blackwell** (~1:64 throughput): `--fp32` opt-in for speed.
- **Lorenz chaos**: parity only at single-step/short horizon, not full trajectories.
- **DoubleWell stochastic RNG**: numpy↔torch draws can't match bit-for-bit; parity on drift with
  injected noise.
- **Triplicated KoopmanTensor**: consolidating to one module changes imports in SKVI/SAKC and
  `opt_wrappers`; verify `generate_tensor.py` (already importing the canonical module) still works.

## Affected files

- `koopmanrl/utils.py` (device/dtype helpers)
- `koopmanrl/koopman_tensor/torch_tensor.py` (canonical, device-aware; add `generate_koopman_tensor`)
- `koopmanrl/koopman_tensor/observables/torch_observables.py` (device-aware)
- `koopmanrl/koopman_observables.py` (align/re-export)
- `koopmanrl/environments/{linear_system,fluid_flow,lorenz,double_well}.py` (batch ops, device-aware cost)
- `koopmanrl/soft_koopman_value_iteration.py` (import canonical tensor; device; vectorize action loop)
- `koopmanrl/soft_actor_koopman_critic.py` (import canonical tensor; device)
- `koopmanrl/sac_continuous_action.py`, `koopmanrl/value_based_sac_continuous_action.py` (device)
- `koopmanrl/opt_wrappers.py` (device/dtype params)
- `tests/test_gpu_parity.py` (new), `tests/test_rl.py` (GPU smoke runs)
- Docs: update `koopmanrl/environments/AGENTS.md` ("run on CPU" → CPU + batched GPU path).
