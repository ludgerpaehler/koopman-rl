# GPU-enabling KoopmanRL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run KoopmanRL's environments and RL algorithms (SKVI, SAKC, both Optuna wrappers, both CleanRL SAC variants) on an NVIDIA GPU, with batched device-native env dynamics and a single device-aware Koopman tensor, validated against the CPU baseline.

**Architecture:** Keep the gym single-env numpy API untouched as the CPU/parity reference. Add batched, device-native `f_batch`/`reset_batch` to each env (substepped torch RK4 for FluidFlow/Lorenz, torch Euler–Maruyama for DoubleWell). Consolidate the triplicated `KoopmanTensor` into one device-aware module (`koopman_tensor/torch_tensor.py`) with the numpy round-trips and Python loops removed. Wire `--cuda`/`--fp32` through the algorithms (currently parsed but ignored). LQR stays on CPU.

**Tech Stack:** Python 3.10, uv, PyTorch 2.9.1+cu128 (Blackwell sm_120), Gymnasium 1.0, pytest. Run everything via `uv run`.

**Spec:** `docs/superpowers/specs/2026-05-22-gpu-koopmanrl-design.md`

**Conventions for every task:** run tests with `uv run pytest ...`; commit on branch `gpu-acceleration`; end commit messages with the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer. CPU baseline to preserve: `uv run pytest tests/test_environments.py tests/test_rl.py` = 28 passed.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `koopmanrl/utils.py` | Add `resolve_device`/`resolve_dtype` | Modify |
| `koopmanrl/koopman_tensor/observables/torch_observables.py` | Device-aware monomials | Modify |
| `koopmanrl/koopman_observables.py` | Re-export canonical observables | Modify |
| `koopmanrl/koopman_tensor/torch_tensor.py` | Canonical device-aware `KoopmanTensor` + `generate_koopman_tensor` (incl. batched rollout) | Modify |
| `koopmanrl/environments/linear_system.py` | `f_batch`/`reset_batch`, device-aware cost | Modify |
| `koopmanrl/environments/fluid_flow.py` | Substepped RK4 `f_batch`/`reset_batch`, device cost | Modify |
| `koopmanrl/environments/lorenz.py` | Substepped RK4 `f_batch`/`reset_batch`, device cost | Modify |
| `koopmanrl/environments/double_well.py` | Euler–Maruyama `f_batch`/`reset_batch`, device cost | Modify |
| `koopmanrl/soft_koopman_value_iteration.py` | Import canonical tensor; device; vectorize action loop | Modify |
| `koopmanrl/soft_actor_koopman_critic.py` | Import canonical tensor; device | Modify |
| `koopmanrl/sac_continuous_action.py` | Device resolution | Modify |
| `koopmanrl/value_based_sac_continuous_action.py` | Device resolution | Modify |
| `koopmanrl/opt_wrappers.py` | Device/dtype params | Modify |
| `tests/test_gpu_parity.py` | CPU/GPU parity + integrator accuracy tests | Create |
| `tests/test_rl.py` | `--cuda` smoke runs | Modify |
| `koopmanrl/environments/AGENTS.md` | Note CPU + batched-GPU path | Modify |

---

## Task 1: Device & dtype helpers

**Files:**
- Modify: `koopmanrl/utils.py`
- Test: `tests/test_device_utils.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_device_utils.py
import torch
from koopmanrl.utils import resolve_device, resolve_dtype


def test_resolve_device_cpu_when_cuda_false():
    assert resolve_device(False) == torch.device("cpu")


def test_resolve_device_cuda_when_requested_and_available():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device(True).type == expected


def test_resolve_dtype():
    assert resolve_dtype(False) == torch.float64
    assert resolve_dtype(True) == torch.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_device_utils.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_device'`.

- [ ] **Step 3: Implement**

Append to `koopmanrl/utils.py` (add `import torch` at top of file):

```python
import torch


def resolve_device(cuda: bool) -> torch.device:
    """Return the CUDA device when requested and available, else CPU."""
    return torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")


def resolve_dtype(fp32: bool) -> torch.dtype:
    """FP64 by default; FP32 when opted in (faster on consumer Blackwell)."""
    return torch.float32 if fp32 else torch.float64
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_device_utils.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add koopmanrl/utils.py tests/test_device_utils.py
git commit -m "feat: add resolve_device/resolve_dtype helpers"
```

---

## Task 2: Device-aware observables (monomials)

The hot path is `monomials.__call__`. It currently allocates `torch.ones([n, m])` on the default (CPU) device and uses a CPU powers matrix `c`, so it breaks when `x` is on CUDA. `diff`/`ddiff` are only used for the unused generator path and additionally call `.copy()` (a numpy-ism) on a torch tensor — fix them too for device-correctness and to remove the latent bug.

**Files:**
- Modify: `koopmanrl/koopman_tensor/observables/torch_observables.py`
- Test: `tests/test_gpu_parity.py` (create with the monomials test; later tasks extend it)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_parity.py
import pytest
import torch

from koopmanrl.koopman_tensor.observables.torch_observables import monomials

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
ATOL = 1e-9


@CUDA
def test_monomials_cpu_gpu_parity():
    phi = monomials(3)
    x_cpu = torch.randn(3, 64, dtype=torch.float64)
    x_gpu = x_cpu.cuda()
    y_cpu = phi(x_cpu)
    y_gpu = phi(x_gpu)
    assert y_gpu.is_cuda
    assert torch.allclose(y_cpu, y_gpu.cpu(), atol=ATOL)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gpu_parity.py -v`
Expected: FAIL (RuntimeError about tensors on different devices, or a CPU-resident result).

- [ ] **Step 3: Implement**

In `koopmanrl/koopman_tensor/observables/torch_observables.py`, edit `monomials.__call__` so allocations follow `x`:

```python
    def __call__(self, x):
        [d, m] = x.shape
        c = allMonomialPowers(d, self.p).to(device=x.device, dtype=x.dtype)
        n = c.shape[1]
        y = torch.ones([n, m], device=x.device, dtype=x.dtype)
        for i in range(n):
            for j in range(d):
                y[i] *= torch.pow(x[j], c[j, i])
        return y
```

Edit `diff` and `ddiff` the same way — allocate `y = torch.zeros([...], device=x.device, dtype=x.dtype)`, set `c = allMonomialPowers(d, self.p).to(device=x.device, dtype=x.dtype)`, and replace `e = c[:, i].copy()` with `e = c[:, i].clone()` (both occurrences). Leave `allMonomialPowers` as-is (small CPU build, `.to(...)` moves it).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gpu_parity.py -v`
Expected: PASS.

- [ ] **Step 5: Run the CPU env tests to confirm no regression**

Run: `uv run pytest tests/test_environments.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add koopmanrl/koopman_tensor/observables/torch_observables.py tests/test_gpu_parity.py
git commit -m "feat: make monomials observables device-aware"
```

---

## Task 3: Canonical device-aware KoopmanTensor

Make `koopman_tensor/torch_tensor.py` the single device-aware implementation: remove the `self.M.numpy()` + per-row Fortran-order reshape, vectorize the `kron` loop, and make `ridgeRegression` device-aware.

**Files:**
- Modify: `koopmanrl/koopman_tensor/torch_tensor.py`
- Test: `tests/test_gpu_parity.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_parity.py`:

```python
from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor


def _make_tensor(device):
    torch.manual_seed(0)
    N, state_dim, action_dim = 500, 3, 1
    X = torch.randn(state_dim, N, dtype=torch.float64, device=device)
    Y = X + 0.01 * torch.randn(state_dim, N, dtype=torch.float64, device=device)
    U = torch.randn(action_dim, N, dtype=torch.float64, device=device)
    return KoopmanTensor(X, Y, U, phi=monomials(2), psi=monomials(2), regressor=Regressor.OLS)


@CUDA
def test_koopman_tensor_cpu_gpu_parity():
    kt_cpu = _make_tensor(torch.device("cpu"))
    kt_gpu = _make_tensor(torch.device("cuda"))
    assert kt_gpu.K.is_cuda and kt_gpu.B.is_cuda
    assert torch.allclose(kt_cpu.K, kt_gpu.K.cpu(), atol=1e-6)
    assert torch.allclose(kt_cpu.B, kt_gpu.B.cpu(), atol=1e-6)
    # phi_f / f parity on a small sample
    x = torch.randn(3, 16, dtype=torch.float64)
    u = torch.randn(1, 16, dtype=torch.float64)
    f_cpu = kt_cpu.f(x, u)
    f_gpu = kt_gpu.f(x.cuda(), u.cuda())
    assert torch.allclose(f_cpu, f_gpu.cpu(), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gpu_parity.py::test_koopman_tensor_cpu_gpu_parity -v`
Expected: FAIL (the current code calls `self.M.numpy()` on a CUDA tensor → `TypeError: can't convert cuda ... to numpy`).

- [ ] **Step 3: Implement — vectorize kron**

In `KoopmanTensor.__init__`, replace the kron loop:

```python
        # Build matrix of kronecker products between u_i and x_i for all 0 <= i <= N
        # torch.kron(Psi_U[:,i], Phi_X[:,i])[a*phi_dim + b] = Psi_U[a,i] * Phi_X[b,i]
        self.kron_matrix = torch.einsum("an,bn->abn", self.Psi_U, self.Phi_X).reshape(
            self.psi_dim * self.phi_dim, self.N
        )
```

- [ ] **Step 4: Implement — pure-torch K reshape**

Replace the numpy reshape block:

```python
        # reshape M into tensor K without leaving the device.
        # Original used a per-row Fortran-order reshape into (phi_dim, psi_dim);
        # that equals the C-order reshape into (psi_dim, phi_dim) transposed.
        self.M = self.M.contiguous()
        self.K = self.M.reshape(self.phi_dim, self.psi_dim, self.phi_dim).transpose(-1, -2).contiguous()
```

(Delete the `self.K = np.empty(...)`, `self.M = self.M.numpy()`, the `for i in range(self.phi_dim)` loop, and the two trailing `torch.tensor(...)` casts.)

- [ ] **Step 5: Implement — device-aware ridge**

Make `ridgeRegression` device-aware:

```python
def ridgeRegression(X, y, lamb=0.05):
    eye = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.inv(X.T @ X + lamb * eye) @ X.T @ y
```

- [ ] **Step 6: Run parity test**

Run: `uv run pytest tests/test_gpu_parity.py::test_koopman_tensor_cpu_gpu_parity -v`
Expected: PASS. If `torch.linalg.lstsq` raises on CUDA (full-rank `gels` driver), fall back: in `ols`, when `X.is_cuda`, use `torch.linalg.lstsq(X, Y, driver="gels").solution`; if still unstable, `return torch.linalg.pinv(X) @ Y`. Re-run until PASS.

- [ ] **Step 7: Run generate_tensor + env tests (no regression)**

Run: `uv run python -m koopmanrl.koopman_tensor.generate_tensor --env_id LinearSystem-v0 --num_paths 5 --num_steps_per_path 20`
Expected: prints estimation-error stats, exit 0.
Run: `uv run pytest tests/test_environments.py -v`
Expected: 8 passed.

- [ ] **Step 8: Commit**

```bash
git add koopmanrl/koopman_tensor/torch_tensor.py tests/test_gpu_parity.py
git commit -m "feat: device-aware KoopmanTensor (vectorized kron, torch-only K reshape)"
```

---

## Task 4: LinearSystem batched dynamics + device-aware cost

**Files:**
- Modify: `koopmanrl/environments/linear_system.py`
- Test: `tests/test_gpu_parity.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_parity.py`:

```python
import gymnasium as gym
import numpy as np
import koopmanrl.environments  # noqa: F401


def test_linear_system_f_batch_matches_numpy_cpu():
    env = gym.make("LinearSystem-v0").unwrapped
    rng = np.random.default_rng(0)
    states = rng.uniform(-5, 5, size=(8, env.state_dim))
    actions = rng.uniform(-5, 5, size=(8, env.action_dim))
    ref = np.stack([env.f(states[i], actions[i]) for i in range(8)])
    out = env.f_batch(
        torch.tensor(states, dtype=torch.float64),
        torch.tensor(actions, dtype=torch.float64),
    )
    assert torch.allclose(out, torch.tensor(ref), atol=1e-9)


@CUDA
def test_linear_system_f_batch_cpu_gpu_parity():
    env = gym.make("LinearSystem-v0").unwrapped
    s = torch.randn(8, env.state_dim, dtype=torch.float64)
    a = torch.randn(8, env.action_dim, dtype=torch.float64)
    out_cpu = env.f_batch(s, a)
    out_gpu = env.f_batch(s.cuda(), a.cuda())
    assert out_gpu.is_cuda
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-9)


@CUDA
def test_linear_system_cost_cpu_gpu_parity():
    env = gym.make("LinearSystem-v0").unwrapped
    s = torch.randn(8, env.state_dim, dtype=torch.float64)
    a = torch.randn(3, env.action_dim, dtype=torch.float64)
    c_cpu = env.vectorized_cost_fn(s, a)
    c_gpu = env.vectorized_cost_fn(s.cuda(), a.cuda())
    assert torch.allclose(c_cpu, c_gpu.cpu(), atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gpu_parity.py -k linear_system -v`
Expected: FAIL (`AttributeError: 'LinearSystem' object has no attribute 'f_batch'`).

- [ ] **Step 3: Implement `f_batch`/`reset_batch` and device-aware cost**

In `koopmanrl/environments/linear_system.py`, replace `vectorized_cost_fn` body and add the batch methods:

```python
    def vectorized_cost_fn(self, states, actions):
        Q = torch.as_tensor(self.Q, dtype=states.dtype, device=states.device)
        R = torch.as_tensor(self.R, dtype=states.dtype, device=states.device)
        ref = torch.as_tensor(self.reference_point, dtype=states.dtype, device=states.device)
        _states = (states - ref).T
        state_cost = torch.einsum("bi,ij,bj->b", _states.T, Q, _states.T).unsqueeze(-1)
        mat = state_cost + torch.pow(actions.T, 2) * R
        return mat.T

    def f_batch(self, states, actions, generator=None):
        """Batched dynamics on (batch, state_dim) torch tensors. generator unused (deterministic)."""
        A = torch.as_tensor(self.A, dtype=states.dtype, device=states.device)
        B = torch.as_tensor(self.B, dtype=states.dtype, device=states.device)
        return states @ A.T + actions @ B.T

    def reset_batch(self, n, device, dtype=torch.float64, generator=None):
        low = torch.as_tensor(self.state_minimums, device=device, dtype=dtype)
        high = torch.as_tensor(self.state_maximums, device=device, dtype=dtype)
        return low + (high - low) * torch.rand(n, self.state_dim, device=device, dtype=dtype, generator=generator)
```

Note: `state_cost` uses `einsum("bi,ij,bj->b", ...)` (equals `diag(_states.T @ Q @ _states)`) to avoid an O(batch²) intermediate.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gpu_parity.py -k linear_system -v`
Expected: PASS (3, skips for CUDA-only when no GPU).

- [ ] **Step 5: Commit**

```bash
git add koopmanrl/environments/linear_system.py tests/test_gpu_parity.py
git commit -m "feat: LinearSystem batched dynamics + device-aware cost"
```

---

## Task 5: FluidFlow substepped-RK4 batched dynamics + device cost

**Files:**
- Modify: `koopmanrl/environments/fluid_flow.py`
- Test: `tests/test_gpu_parity.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_parity.py`:

```python
def test_fluid_flow_rk4_matches_scipy_accuracy():
    env = gym.make("FluidFlow-v0").unwrapped
    rng = np.random.default_rng(1)
    states = rng.uniform(-1, 1, size=(8, env.state_dim))
    actions = rng.uniform(-2, 2, size=(8, env.action_dim))
    ref = np.stack([env.f(states[i], actions[i]) for i in range(8)])  # scipy RK45
    out = env.f_batch(torch.tensor(states), torch.tensor(actions))
    # Looser integrator-accuracy tolerance (RK4 substeps vs adaptive RK45)
    assert torch.allclose(out, torch.tensor(ref), atol=1e-4, rtol=1e-4)


@CUDA
def test_fluid_flow_f_batch_cpu_gpu_parity():
    env = gym.make("FluidFlow-v0").unwrapped
    s = torch.rand(8, env.state_dim, dtype=torch.float64)
    a = torch.randn(8, env.action_dim, dtype=torch.float64)
    out_cpu = env.f_batch(s, a)
    out_gpu = env.f_batch(s.cuda(), a.cuda())
    assert out_gpu.is_cuda
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-9)  # same RK4 both sides → tight
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gpu_parity.py -k fluid_flow -v`
Expected: FAIL (`AttributeError: ... 'f_batch'`).

- [ ] **Step 3: Implement**

In `koopmanrl/environments/fluid_flow.py`, add `self.n_substeps = 4` in `__init__`, replace `vectorized_cost_fn` body with the device-aware version (identical to Task 3's pattern), and add:

```python
    def _deriv_batch(self, states, actions):
        x, y, z = states[:, 0], states[:, 1], states[:, 2]
        u = actions[:, 0]
        x_dot = self.mu * x - self.omega * y + self.A * x * z
        y_dot = self.omega * x + self.mu * y + self.A * y * z + u
        z_dot = -self.lamb * (z - x.pow(2) - y.pow(2))
        return torch.stack([x_dot, y_dot, z_dot], dim=1)

    def f_batch(self, states, actions, generator=None):
        """Batched dynamics via substepped RK4 over a single dt. generator unused (deterministic)."""
        h = self.dt / self.n_substeps
        x = states
        for _ in range(self.n_substeps):
            k1 = self._deriv_batch(x, actions)
            k2 = self._deriv_batch(x + 0.5 * h * k1, actions)
            k3 = self._deriv_batch(x + 0.5 * h * k2, actions)
            k4 = self._deriv_batch(x + h * k3, actions)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def reset_batch(self, n, device, dtype=torch.float64, generator=None):
        low = torch.as_tensor(self.state_minimums, device=device, dtype=dtype)
        high = torch.as_tensor(self.state_maximums, device=device, dtype=dtype)
        return low + (high - low) * torch.rand(n, self.state_dim, device=device, dtype=dtype, generator=generator)
```

Use the device-aware `vectorized_cost_fn` from Task 4 (copy that exact method body).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_gpu_parity.py -k fluid_flow -v`
Expected: PASS. If the accuracy test fails, raise `self.n_substeps` (e.g. 8) and re-run.

- [ ] **Step 5: Commit**

```bash
git add koopmanrl/environments/fluid_flow.py tests/test_gpu_parity.py
git commit -m "feat: FluidFlow batched RK4 dynamics + device-aware cost"
```

---

## Task 6: Lorenz substepped-RK4 batched dynamics + device cost

**Files:**
- Modify: `koopmanrl/environments/lorenz.py`
- Test: `tests/test_gpu_parity.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_parity.py`:

```python
def test_lorenz_rk4_matches_scipy_accuracy():
    env = gym.make("Lorenz-v0").unwrapped
    rng = np.random.default_rng(2)
    states = rng.uniform(-10, 10, size=(8, env.state_dim))
    actions = rng.uniform(-20, 20, size=(8, env.action_dim))
    ref = np.stack([env.f(states[i], actions[i]) for i in range(8)])
    out = env.f_batch(torch.tensor(states), torch.tensor(actions))
    assert torch.allclose(out, torch.tensor(ref), atol=1e-3, rtol=1e-3)


@CUDA
def test_lorenz_f_batch_cpu_gpu_parity():
    env = gym.make("Lorenz-v0").unwrapped
    s = torch.randn(8, env.state_dim, dtype=torch.float64) * 5
    a = torch.randn(8, env.action_dim, dtype=torch.float64) * 5
    out_cpu = env.f_batch(s, a)
    out_gpu = env.f_batch(s.cuda(), a.cuda())
    assert out_gpu.is_cuda
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gpu_parity.py -k lorenz -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement**

In `koopmanrl/environments/lorenz.py`, add `self.n_substeps = 4` in `__init__`, replace `vectorized_cost_fn` body with the device-aware version, and add (note the control `u` enters `x_dot`, matching `continuous_f`):

```python
    def _deriv_batch(self, states, actions):
        x, y, z = states[:, 0], states[:, 1], states[:, 2]
        u = actions[:, 0]
        x_dot = self.sigma * (y - x) + u
        y_dot = (self.rho - z) * x - y
        z_dot = x * y - self.beta * z
        return torch.stack([x_dot, y_dot, z_dot], dim=1)

    def f_batch(self, states, actions, generator=None):
        h = self.dt / self.n_substeps
        x = states
        for _ in range(self.n_substeps):
            k1 = self._deriv_batch(x, actions)
            k2 = self._deriv_batch(x + 0.5 * h * k1, actions)
            k3 = self._deriv_batch(x + 0.5 * h * k2, actions)
            k4 = self._deriv_batch(x + h * k3, actions)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def reset_batch(self, n, device, dtype=torch.float64, generator=None):
        low = torch.as_tensor(self.state_minimums, device=device, dtype=dtype)
        high = torch.as_tensor(self.state_maximums, device=device, dtype=dtype)
        return low + (high - low) * torch.rand(n, self.state_dim, device=device, dtype=dtype, generator=generator)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_gpu_parity.py -k lorenz -v`
Expected: PASS. If accuracy fails, raise `self.n_substeps` and re-run.

- [ ] **Step 5: Commit**

```bash
git add koopmanrl/environments/lorenz.py tests/test_gpu_parity.py
git commit -m "feat: Lorenz batched RK4 dynamics + device-aware cost"
```

---

## Task 7: DoubleWell batched Euler–Maruyama + device cost

DoubleWell is stochastic. The batched `f_batch` takes an optional `noise` tensor of shape `(batch, 2, 1)` so the deterministic **drift** can be parity-tested with injected noise; when `noise is None` it samples on-device via `generator`.

**Files:**
- Modify: `koopmanrl/environments/double_well.py`
- Test: `tests/test_gpu_parity.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_parity.py`:

```python
def test_double_well_drift_matches_numpy():
    env = gym.make("DoubleWell-v0").unwrapped
    env.reset(seed=0)
    rng = np.random.default_rng(3)
    states = rng.uniform(-2, 2, size=(8, env.state_dim))
    actions = rng.uniform(-5, 5, size=(8, env.action_dim))
    # numpy drift-only reference (diffusion term zeroed)
    ref = np.stack([
        states[i] + env.continuous_f(actions[i])(0, states[i]) * env.dt
        for i in range(8)
    ])
    zero_noise = torch.zeros(8, env.state_dim, 1, dtype=torch.float64)
    out = env.f_batch(torch.tensor(states), torch.tensor(actions), noise=zero_noise)
    assert torch.allclose(out, torch.tensor(ref), atol=1e-9)


@CUDA
def test_double_well_drift_cpu_gpu_parity():
    env = gym.make("DoubleWell-v0").unwrapped
    s = torch.randn(8, env.state_dim, dtype=torch.float64)
    a = torch.randn(8, env.action_dim, dtype=torch.float64)
    zn = torch.zeros(8, env.state_dim, 1, dtype=torch.float64)
    out_cpu = env.f_batch(s, a, noise=zn)
    out_gpu = env.f_batch(s.cuda(), a.cuda(), noise=zn.cuda())
    assert out_gpu.is_cuda
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gpu_parity.py -k double_well -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement**

In `koopmanrl/environments/double_well.py`, replace `vectorized_cost_fn` body with the device-aware version, and add. The numpy drift is `b_x = [4x - 4x³, -2y] + u` per `continuous_f`; the diffusion uses `sigma_x = [[0.7, x],[0,0.5]]`:

```python
    def _drift_batch(self, states, actions):
        x, y = states[:, 0], states[:, 1]
        u = actions[:, 0]
        x_dot = 4 * x - 4 * x.pow(3) + u
        y_dot = -2 * y + u
        return torch.stack([x_dot, y_dot], dim=1)

    def f_batch(self, states, actions, generator=None, noise=None):
        """Batched Euler-Maruyama. noise: (batch, 2, 1) standard-normal draws; sampled if None."""
        dt = self.dt
        drift = self._drift_batch(states, actions) * dt
        if noise is None:
            noise = torch.randn(states.shape[0], 2, 1, device=states.device, dtype=states.dtype, generator=generator)
        x = states[:, 0]
        # sigma_x = [[0.7, x], [0, 0.5]] per-sample
        sigma = torch.zeros(states.shape[0], 2, 2, device=states.device, dtype=states.dtype)
        sigma[:, 0, 0] = 0.7
        sigma[:, 0, 1] = x
        sigma[:, 1, 1] = 0.5
        diffusion = torch.bmm(sigma, noise).squeeze(-1) * (dt ** 0.5)
        return states + drift + diffusion

    def reset_batch(self, n, device, dtype=torch.float64, generator=None):
        low = torch.as_tensor(self.state_minimums, device=device, dtype=dtype)
        high = torch.as_tensor(self.state_maximums, device=device, dtype=dtype)
        return low + (high - low) * torch.rand(n, self.state_dim, device=device, dtype=dtype, generator=generator)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_gpu_parity.py -k double_well -v`
Expected: PASS.

- [ ] **Step 5: Run full parity + env suite**

Run: `uv run pytest tests/test_gpu_parity.py tests/test_environments.py -v`
Expected: all pass (CUDA-only tests run on the GPU box; skipped otherwise).

- [ ] **Step 6: Commit**

```bash
git add koopmanrl/environments/double_well.py tests/test_gpu_parity.py
git commit -m "feat: DoubleWell batched Euler-Maruyama + device-aware cost"
```

---

## Task 8: Batched GPU rollout in `generate_koopman_tensor`

Add a batched, device-native rollout to the canonical `generate_koopman_tensor` (in `torch_tensor.py`). When `device` is CUDA it runs all paths in parallel via `reset_batch`/`f_batch`; otherwise it keeps the sequential gym loop (parity reference). Returns a `KoopmanTensor` built on `device`.

**Files:**
- Modify: `koopmanrl/koopman_tensor/torch_tensor.py`
- Test: `tests/test_gpu_parity.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_parity.py`:

```python
from koopmanrl.koopman_tensor.torch_tensor import generate_koopman_tensor


@CUDA
def test_generate_koopman_tensor_batched_gpu():
    kt = generate_koopman_tensor(
        env_id="Lorenz-v0", seed=0, num_paths=16, num_steps_per_path=20,
        state_order=2, action_order=2, regressor="ols",
        device=torch.device("cuda"),
    )
    assert kt.K.is_cuda
    assert kt.X.shape[1] == 16 * 20
    # tensor is usable on-device
    x = torch.randn(3, 5, dtype=torch.float64, device="cuda")
    u = torch.randn(1, 5, dtype=torch.float64, device="cuda")
    assert kt.f(x, u).is_cuda
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gpu_parity.py::test_generate_koopman_tensor_batched_gpu -v`
Expected: FAIL (`generate_koopman_tensor` has no `device` parameter / not importable from here yet).

- [ ] **Step 3: Implement**

Add `generate_koopman_tensor` to `koopman_tensor/torch_tensor.py` (import `gymnasium as gym`, `numpy as np`, and `from koopmanrl.koopman_tensor.observables.torch_observables import monomials`; ensure `import koopmanrl.environments` registers the envs):

```python
def generate_koopman_tensor(
    env_id, seed, num_paths, num_steps_per_path, state_order, action_order, regressor,
    device=None, dtype=torch.float64,
):
    import koopmanrl.environments  # noqa: F401 (register custom envs)

    device = device or torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    base = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if device.type == "cuda" and hasattr(base, "f_batch"):
        gen = torch.Generator(device=device).manual_seed(seed)
        low = torch.as_tensor(env.action_space.low, device=device, dtype=dtype)
        high = torch.as_tensor(env.action_space.high, device=device, dtype=dtype)
        states = base.reset_batch(num_paths, device=device, dtype=dtype, generator=gen)
        Xs, Ys, Us = [], [], []
        for _ in range(num_steps_per_path):
            actions = low + (high - low) * torch.rand(
                num_paths, action_dim, device=device, dtype=dtype, generator=gen
            )
            next_states = base.f_batch(states, actions, generator=gen)
            Xs.append(states); Us.append(actions); Ys.append(next_states)
            states = next_states
        X = torch.stack(Xs, dim=1).reshape(-1, state_dim).T
        Y = torch.stack(Ys, dim=1).reshape(-1, state_dim).T
        U = torch.stack(Us, dim=1).reshape(-1, action_dim).T
    else:
        X = torch.zeros((num_paths, num_steps_per_path, state_dim), dtype=dtype)
        Y = torch.zeros_like(X)
        U = torch.zeros((num_paths, num_steps_per_path, action_dim), dtype=dtype)
        for p in range(num_paths):
            state, _ = env.reset()
            for s in range(num_steps_per_path):
                X[p, s] = torch.as_tensor(state, dtype=dtype)
                action = env.action_space.sample()
                U[p, s] = torch.as_tensor(action, dtype=dtype)
                state, _, _, _, _ = env.step(action)
                Y[p, s] = torch.as_tensor(state, dtype=dtype)
        n = num_paths * num_steps_per_path
        X = X.reshape(n, state_dim).T.to(device)
        Y = Y.reshape(n, state_dim).T.to(device)
        U = U.reshape(n, action_dim).T.to(device)

    kwargs = dict(phi=monomials(state_order), psi=monomials(action_order), regressor=Regressor(regressor))
    try:
        return KoopmanTensor(X, Y, U, dt=base.dt, **kwargs)
    except Exception:
        return KoopmanTensor(X, Y, U, **kwargs)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_gpu_parity.py::test_generate_koopman_tensor_batched_gpu -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add koopmanrl/koopman_tensor/torch_tensor.py tests/test_gpu_parity.py
git commit -m "feat: batched GPU rollout in generate_koopman_tensor"
```

---

## Task 9: SKVI — canonical tensor, device, vectorized action loop

Replace SKVI's inline `KoopmanTensor`/`generate_koopman_tensor`/`Regressor`/regressors with imports from the canonical module, resolve the device from `--cuda`, add `--fp32`, move all tensors to device, and vectorize the per-action Python loops.

**Files:**
- Modify: `koopmanrl/soft_koopman_value_iteration.py`
- Test: existing `tests/test_rl.py` (CPU) + new GPU smoke in Task 13

- [ ] **Step 1: Replace inline tensor code with imports**

Delete the inline `checkMatrixRank`, `checkConditionNumber`, `ols/OLS/SINDy/rrr/RRR/ridgeRegression`, `Regressor`, `KoopmanTensor`, and `generate_koopman_tensor` definitions. Add near the top:

```python
from koopmanrl.koopman_tensor.torch_tensor import (
    KoopmanTensor,
    Regressor,
    generate_koopman_tensor,
)
```

- [ ] **Step 2: Add `--fp32`, resolve device, set dtype**

In `ArgumentParser` add: `fp32: bool = False  # use float32 instead of float64`.
In `main()` replace `device = torch.device("cpu")` with:

```python
    from koopmanrl.utils import resolve_device, resolve_dtype  # or add to existing import
    device = resolve_device(args.cuda)
    torch.set_default_dtype(resolve_dtype(args.fp32))
```

Pass `device`/`dtype` into `generate_koopman_tensor(...)`. Move `all_actions` to device: append `.to(device)` to the `all_actions = torch.from_numpy(...).T` expression. Pass `device=device` to `DiscreteKoopmanValueIterationPolicy(...)`.

- [ ] **Step 3: Device-thread the policy**

Add a `device=None` parameter to `DiscreteKoopmanValueIterationPolicy.__init__`; store `self.device = device or torch.device("cpu")`; allocate `self.value_function_weights = torch.zeros((phi_dim, 1), device=self.device)` (and the `requires_grad` branch likewise). In `pis`, `discrete_bellman_error`, and `train`: (a) build batch index tensors on device — `x_batch_indices = torch.as_tensor(np.random.choice(...)).to(self.device)`; (b) replace `costs = torch.Tensor(self.cost(...))` with `costs = self.cost(...)` (it already returns a device tensor); (c) replace each per-action loop.

- [ ] **Step 4: Vectorize the per-action loops**

In all three methods replace the pattern

```python
        K_us = self.dynamics_model.K_(self.all_actions)
        phi_x_prime_batch = torch.zeros([num_actions, phi_dim, B])
        V_x_prime_batch = torch.zeros([num_actions, B])
        for action_index in range(K_us.shape[0]):
            phi_x_prime_hat_batch = K_us[action_index] @ phi_x_batch
            phi_x_prime_batch[action_index] = phi_x_prime_hat_batch
            V_x_prime_batch[action_index] = self.V_phi_x(phi_x_prime_batch[action_index])
```

with

```python
        K_us = self.dynamics_model.K_(self.all_actions)  # (num_actions, phi_dim, phi_dim)
        phi_x_prime_batch = torch.einsum("aij,jb->aib", K_us, phi_x_batch)  # (num_actions, phi_dim, B)
        w = self.value_function_weights.squeeze(-1)  # (phi_dim,)
        V_x_prime_batch = torch.einsum("p,apb->ab", w, phi_x_prime_batch)  # (num_actions, B)
```

(`phi_xs` in `pis` plays the role of `phi_x_batch`.) Leave the softmax / OLS update logic unchanged.

- [ ] **Step 5: Run CPU regression for SKVI**

Run: `uv run pytest "tests/test_rl.py::test_skvi" -v`
Expected: 4 passed (CPU path unchanged behavior; numerics within solver tolerance).

- [ ] **Step 6: GPU smoke run**

Run: `uv run python -m koopmanrl.soft_koopman_value_iteration --env_id Lorenz-v0 --total_timesteps 200 --cuda --num_paths 20 --num_steps_per_path 50 --num_training_epochs 5`
Expected: exit 0, runs without device-mismatch errors.

- [ ] **Step 7: Commit**

```bash
git add koopmanrl/soft_koopman_value_iteration.py
git commit -m "feat: SKVI on GPU (canonical tensor, device, vectorized action loop)"
```

---

## Task 10: SAKC — canonical tensor + device

**Files:**
- Modify: `koopmanrl/soft_actor_koopman_critic.py`
- Test: existing `tests/test_rl.py` (CPU) + GPU smoke in Task 13

- [ ] **Step 1: Replace inline tensor code with imports**

Delete the inline `checkMatrixRank/checkConditionNumber/ols/OLS/SINDy/rrr/RRR/ridgeRegression/Regressor/KoopmanTensor/generate_koopman_tensor` definitions. Add:

```python
from koopmanrl.koopman_tensor.torch_tensor import (
    KoopmanTensor,
    Regressor,
    generate_koopman_tensor,
)
```

(Keep the `ReplayBuffer` import — `opt_wrappers` imports `ReplayBuffer` from this module.)

- [ ] **Step 2: Add `--fp32`, resolve device, set dtype**

Add `fp32: bool = False` to `ArgumentParser`. In `main()` replace `device = torch.device("cpu")` with:

```python
    from koopmanrl.utils import resolve_device, resolve_dtype
    device = resolve_device(args.cuda)
    torch.set_default_dtype(resolve_dtype(args.fp32))
```

Pass `device=device, dtype=resolve_dtype(args.fp32)` to `generate_koopman_tensor(...)`. The networks already use `.to(device)`; the SB3 `ReplayBuffer(..., device, ...)` already takes `device`. `koopman_tensor.phi_f(...)` now runs on-device because the tensor is on `device`.

- [ ] **Step 3: Run CPU regression**

Run: `uv run pytest "tests/test_rl.py::test_sakc" -v`
Expected: 4 passed.

- [ ] **Step 4: GPU smoke run**

Run: `uv run python -m koopmanrl.soft_actor_koopman_critic --env_id FluidFlow-v0 --total_timesteps 6000 --learning_starts 1000 --cuda --num_paths 20 --num_steps_per_path 50`
Expected: exit 0, training updates run on GPU without device errors.

- [ ] **Step 5: Commit**

```bash
git add koopmanrl/soft_actor_koopman_critic.py
git commit -m "feat: SAKC on GPU (canonical tensor + device)"
```

---

## Task 11: SAC variants — device resolution

Restore device plumbing in the two CleanRL-derived files. No algorithm-logic changes.

**Files:**
- Modify: `koopmanrl/sac_continuous_action.py`, `koopmanrl/value_based_sac_continuous_action.py`
- Test: existing `tests/test_rl.py` (CPU) + GPU smoke in Task 13

- [ ] **Step 1: Edit both files**

In each, replace `device = torch.device("cpu")` with:

```python
    from koopmanrl.utils import resolve_device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
```

(The `--cuda` flag already exists in both `ArgumentParser`s; all networks/buffers already use `.to(device)` / `device=`.)

- [ ] **Step 2: Run CPU regression**

Run: `uv run pytest "tests/test_rl.py::test_sac_q" "tests/test_rl.py::test_sac_v" -v`
Expected: 8 passed.

- [ ] **Step 3: GPU smoke run**

Run: `uv run python -m koopmanrl.sac_continuous_action --env_id LinearSystem-v0 --total_timesteps 6000 --learning_starts 1000 --cuda`
Run: `uv run python -m koopmanrl.value_based_sac_continuous_action --env_id LinearSystem-v0 --total_timesteps 6000 --learning_starts 1000 --cuda`
Expected: both exit 0.

- [ ] **Step 4: Commit**

```bash
git add koopmanrl/sac_continuous_action.py koopmanrl/value_based_sac_continuous_action.py
git commit -m "feat: restore CUDA device resolution in SAC variants"
```

---

## Task 12: Optuna wrappers — device/dtype params

**Files:**
- Modify: `koopmanrl/opt_wrappers.py`
- Test: smoke import + a short run

- [ ] **Step 1: Update imports**

The wrappers import `generate_koopman_tensor` and `DiscreteKoopmanValueIterationPolicy` from `soft_koopman_value_iteration`, and `Actor/ReplayBuffer/SoftKoopmanVNetwork/SoftQNetwork` from `soft_actor_koopman_critic`. These still exist after Tasks 9–10 (the algos re-export `generate_koopman_tensor` via their import). Confirm: add `from koopmanrl.koopman_tensor.torch_tensor import generate_koopman_tensor` directly to `opt_wrappers.py` to avoid relying on re-export.

- [ ] **Step 2: Add params and resolve device in both wrappers**

Add `cuda: bool = False` and `fp32: bool = False` parameters to `skvi_tuning_wrapper` and `sakc_tuning_wrapper`. Replace `device = torch.device("cpu")` with:

```python
    from koopmanrl.utils import resolve_device, resolve_dtype
    device = resolve_device(cuda)
    torch.set_default_dtype(resolve_dtype(fp32))
```

Pass `device=device, dtype=resolve_dtype(fp32)` into `generate_koopman_tensor(...)`, pass `device=device` into `DiscreteKoopmanValueIterationPolicy(...)` (skvi wrapper), and append `.to(device)` to the `all_actions` tensor (skvi wrapper).

- [ ] **Step 3: Smoke test**

```bash
uv run python -c "
import torch
from koopmanrl.opt_wrappers import skvi_tuning_wrapper, sakc_tuning_wrapper
r = skvi_tuning_wrapper(env_id='LinearSystem-v0', total_timesteps=100, number_of_paths=10, number_of_steps_per_path=20, number_of_training_epochs=3, cuda=torch.cuda.is_available())
print('skvi keys:', list(r.keys())[:2])
r = sakc_tuning_wrapper(env_id='LinearSystem-v0', total_timesteps=2000, learning_starts=500, number_of_paths=10, number_of_steps_per_path=20, cuda=torch.cuda.is_available())
print('sakc keys:', list(r.keys())[:2])
"
```
Expected: prints both key lists, exit 0.

- [ ] **Step 4: Commit**

```bash
git add koopmanrl/opt_wrappers.py
git commit -m "feat: device/dtype support in Optuna wrappers"
```

---

## Task 13: GPU smoke tests in the suite + full validation + docs

**Files:**
- Modify: `tests/test_rl.py`, `koopmanrl/environments/AGENTS.md`

- [ ] **Step 1: Add CUDA smoke tests to `tests/test_rl.py`**

Append:

```python
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda_only
@pytest.mark.parametrize("env_id", ENVS)
def test_skvi_cuda(env_id):
    result = run_module(
        "koopmanrl.soft_koopman_value_iteration",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr


@cuda_only
@pytest.mark.parametrize("env_id", ENVS)
def test_sakc_cuda(env_id):
    result = run_module(
        "koopmanrl.soft_actor_koopman_critic",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda"],
    )
    assert result.returncode == 0, result.stderr


@cuda_only
@pytest.mark.parametrize("env_id", ENVS)
def test_sac_q_cuda(env_id):
    result = run_module(
        "koopmanrl.sac_continuous_action",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda"],
    )
    assert result.returncode == 0, result.stderr


@cuda_only
@pytest.mark.parametrize("env_id", ENVS)
def test_sac_v_cuda(env_id):
    result = run_module(
        "koopmanrl.value_based_sac_continuous_action",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda"],
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest tests/test_environments.py tests/test_gpu_parity.py tests/test_device_utils.py tests/test_rl.py -v`
Expected: all CPU tests pass (28 original + new), all CUDA tests pass on the GPU box. Investigate and fix any failure before proceeding (use systematic-debugging).

- [ ] **Step 3: Confirm tensors are actually on the GPU**

Run: `uv run python -c "
import torch
from koopmanrl.koopman_tensor.torch_tensor import generate_koopman_tensor
kt = generate_koopman_tensor('Lorenz-v0', 0, 32, 50, 2, 2, 'ols', device=torch.device('cuda'))
print('K on cuda:', kt.K.is_cuda, '| B on cuda:', kt.B.is_cuda, '| X on cuda:', kt.X.is_cuda)
"`
Expected: `K on cuda: True | B on cuda: True | X on cuda: True`.

- [ ] **Step 4: Update env AGENTS doc**

In `koopmanrl/environments/AGENTS.md`, change the Design Guide bullet "Environments are allowed to run with FP64, and are run on CPU" to note: the gym single-env path is FP64/CPU (parity reference), and each env additionally exposes batched, device-aware `f_batch`/`reset_batch` (substepped RK4 for FluidFlow/Lorenz, Euler–Maruyama for DoubleWell) used for GPU data-gen and training.

- [ ] **Step 5: Run lint**

Run: `uv run pre-commit run --all-files`
Expected: passes (fix any ruff/isort findings it reports).

- [ ] **Step 6: Commit**

```bash
git add tests/test_rl.py koopmanrl/environments/AGENTS.md
git commit -m "test: GPU smoke tests; docs: note batched GPU env path"
```

---

## Self-Review

**Spec coverage:** §1 device foundation → Task 1, 9–12. §2 canonical tensor → Tasks 3, 8, 9, 10. §3 observables → Task 2. §4 batched envs + device cost → Tasks 4–7. §5 batched data-gen → Task 8. §6 algorithm wiring → Tasks 9–12. §Validation parity + smoke → Tasks 2–8 (parity), 13 (smoke + full run). §Non-goals (LQR untouched) respected — no LQR task. ✓

**Placeholder scan:** No TBD/TODO; every code step shows real code; commands have expected output. ✓

**Type/name consistency:** `f_batch(states, actions, generator=None)` everywhere (DoubleWell adds `noise=None`); `reset_batch(n, device, dtype, generator)` everywhere; `resolve_device`/`resolve_dtype` names consistent; `generate_koopman_tensor(..., device=, dtype=)` consistent across Tasks 8–12; `vectorized_cost_fn` einsum form identical across envs. ✓

**Known risk handled in-plan:** Task 3 Step 6 carries the `lstsq`-on-CUDA fallback.
