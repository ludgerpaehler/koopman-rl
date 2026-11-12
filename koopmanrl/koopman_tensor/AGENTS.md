# Koopman Tensor Guide

- This directory owns the fitted, action-conditioned Koopman dynamics model. SKVI and SAKC fit it once from random-action transitions before policy/value training; they do not update it from the online replay buffer.
- Paths below are relative to this directory unless prefixed with `../` or `../../`.

## Start Here

- [torch_tensor.py](torch_tensor.py): active training backend; `generate_koopman_tensor` collects transitions, and `KoopmanTensor.__init__` lifts data, fits regressions, and constructs the tensor. `K_`, `phi_f`, and `f` are the prediction interface.
- [numpy_tensor.py](numpy_tensor.py): separate NumPy implementation of the same construction and prediction layout, with different solver behavior (see below).
- [observables/torch_observables.py](observables/torch_observables.py) and [observables/numpy_observables.py](observables/numpy_observables.py): lifting dictionaries; see [observables/AGENTS.md](observables/AGENTS.md). The training factory uses the Torch monomials here, not the older duplicate [../koopman_observables.py](../koopman_observables.py), even though the algorithm modules still import that duplicate.
- [generate_tensor.py](generate_tensor.py): standalone CPU generation/diagnostics CLI with its own rollout loop; it does not call `generate_koopman_tensor`.
- [utils.py](utils.py): pickle persistence. Paths are relative to the process working directory: `./koopman_tensor/saved_models/{env_id}/{saved_model_name}.pickle`, not the package source directory.

## Construction and Array Contracts

- Keep samples in columns: `X` and `Y` have shape `(state_dim, N)`, `U` has shape `(action_dim, N)`, and column `i` is the aligned transition `(x_i, u_i, x_next_i)`. Rollout buffers start as `(paths, steps, dimension)` and are flattened/transposed into this layout.
- `phi` lifts states and `psi` lifts actions. `Phi_X = phi(X)` and `Phi_Y = phi(Y)` have shape `(phi_dim, N)`; `Psi_U = psi(U)` has shape `(psi_dim, N)`. Monomials include the constant and all terms of total degree up to the requested order; for input dimension `d` and order `p`, there are `binomial(d + p, p)` terms. Preserve the `allMonomialPowers` ordering when changing dictionaries or using saved weights.
- Construct `kron_matrix[:, i] = kron(Psi_U[:, i], Phi_X[:, i])`, shape `(psi_dim * phi_dim, N)`. The state-feature index varies fastest. Torch uses an equivalent `einsum`/reshape rather than a per-sample loop.
- Fit `kron_matrix.T @ M.T ≈ Phi_Y.T` and the separate decoder `Phi_X.T @ B ≈ X.T`. Thus `M` has shape `(phi_dim, psi_dim * phi_dim)` and `B` has shape `(phi_dim, state_dim)`. The decoder is fitted from current states, not next states.
- Reshape `M` into `K` with shape `(phi_dim, phi_dim, psi_dim)`. NumPy reshapes each output row with Fortran order; Torch uses `M.reshape(phi_dim, psi_dim, phi_dim).transpose(-1, -2)`. Preserve this equivalence and the Kronecker ordering together.
- Default construction is a discrete one-step lifted predictor, `phi(x_next) ≈ K_(u) @ phi(x)`. For stochastic transitions this is a regression approximation to the conditional expectation of the lifted next state, not a sampled next state or an exact closure.

## Prediction and Batching

- Supply 2D column matrices even for a single state/action. `K_(u)` contracts the action dictionary with `K`; one action column returns `(phi_dim, phi_dim)`, while multiple action columns return `(num_actions, phi_dim, phi_dim)`.
- `phi_f(x, u)` returns predicted lifted states. A single action applies to every state column; multiple actions pair with corresponding state columns (or broadcast over a single state). It does not form an arbitrary all-actions-by-all-states grid.
- `f(x, u) = B.T @ phi_f(x, u)` decodes to state space. RL value targets use `phi_f` or `K_` directly; decoding and then lifting again is not equivalent for nonlinear observables.
- For the all-actions-by-all-states grid used in SKVI, contract `K_(all_actions)` with `phi(X)` using `einsum("aij,jb->aib", ...)`. This expects multiple action columns because `K_` removes the action axis for a singleton.

## Solvers, Precision, and Generator Mode

- Torch supports `ols`, `rrr`, `sindy`, and `ridge`; NumPy supports the first three through `KoopmanTensor` (its ridge helper is not dispatched by the constructor). The selected regressor fits both `M` and `B`.
- Torch OLS uses `torch.linalg.lstsq`; NumPy OLS uses normal equations with a pseudoinverse by default and exposes `p_inv`. Do not assume bitwise backend parity. RRR defaults to rank 8; SINDy uses threshold 0.05 and 10 threshold/refit passes; Torch ridge uses penalty 0.05.
- Rank and condition-number checks print diagnostics only; they do not reject deficient or ill-conditioned data. CUDA `lstsq` uses the full-rank `gels` driver, so these diagnostics are not a guarantee that a CUDA fit is valid.
- The Torch factory defaults to CPU and FP64. Algorithm entry points and tuning wrappers resolve `--cuda` / `--fp32` through [../utils.py](../utils.py) and pass device/dtype into generation. Keep inputs, lifted features, fitted tensors, and value weights on compatible devices/dtypes; `KoopmanTensor` is not an `nn.Module` with automatic `.to(...)` migration. Several Torch modules set the global default dtype to FP64 at import time.
- `is_generator=True` changes the regression target to a derivative approximation using `phi.diff`, `phi.ddiff`, and `dt`. The current second-order term uses the dataset-wide product `(Y-X) @ (Y-X).T`, not per-transition outer products. Inspect this formula before interpreting it as a stochastic generator estimator.
- Neither the factory nor the standalone CLI enables generator mode. Passing `dt` alone still fits a discrete operator. `K_`/`phi_f`/`f` perform the same contractions in generator mode; they do not exponentiate or integrate the generator into a next-state operator.

## Rollout Collection

- `generate_koopman_tensor` seeds NumPy, Torch, the environment, and the action space. CPU collection uses sequential Gymnasium `reset`/`step` calls and random sampled actions, then transfers the column matrices to the requested device.
- On CUDA, environments exposing `f_batch` take the batched path: `reset_batch` initializes all paths, actions are uniform within bounds, and `f_batch` advances all paths together for each step. See [../environments/AGENTS.md](../environments/AGENTS.md) for integrators and stochastic dynamics. These environment APIs use row batches `(paths, dimension)`, unlike the tensor's column layout.
- Both paths collect fixed-length trajectories; the sequential loop ignores termination/truncation flags. Check this assumption before extending collection to other environments.
- CPU and CUDA use different RNG streams, and the continuous environments use different numerical integration paths. For solver parity, fit identical transition data on both devices rather than comparing independently collected rollouts.
- The standalone CLI reports one-step errors on the same data it fitted, not a held-out validation set. It has separate seeding and a broad exception/retry around construction with `env.dt`; use the factory as the source of truth for training behavior.

## RL Consumers

- [../soft_koopman_value_iteration.py](../soft_koopman_value_iteration.py): `DiscreteKoopmanValueIterationPolicy` represents a cost value as `w.T @ phi(x)`. It evaluates every discretized action with `K_`, forms soft-policy logits `-(cost + discount * V_next) / alpha`, and fits value weights against the entropy-regularized Bellman expectation. The main script uses OLS value updates and discount `gamma ** dt` (`dt=None` becomes 1). Its `pis` input is a row batch despite the docstring; `V_x` takes column states.
- [../soft_actor_koopman_critic.py](../soft_actor_koopman_critic.py): `SoftKoopmanVNetwork` is linear in lifted states, with no extra bias because the monomial dictionary contains a constant. Q targets are `reward + (1-done) * gamma * V_target.linear(phi_f(x,u).T)`, using paired replay states/actions rather than sampled next observations. The actor and twin Q networks remain neural networks; their external batches use rows.
- [../opt_wrappers.py](../opt_wrappers.py): tuning orchestration reuses the factory and the policy/network classes above, but also duplicates the SAKC update loop. Check both the script and wrapper when changing algorithm behavior. [../sakc_optuna_opt.py](../sakc_optuna_opt.py) and [../skvi_optuna_opt.py](../skvi_optuna_opt.py) drive these wrappers with Ray/Optuna.
- [../interpretability_discrete_value_iteration.py](../interpretability_discrete_value_iteration.py): separate value-iteration/interpretability implementation loading `path_based_tensor` through `load_tensor`; it is not the policy class used by current training wrappers. [../../koopmanrl_utils/tsne_koopman_tensor.py](../../koopmanrl_utils/tsne_koopman_tensor.py) also loads persisted tensors.

## Commands and Verification

- Standalone generation: `uv run python -m koopmanrl.koopman_tensor.generate_tensor --env_id LinearSystem-v0`. Also supports `FluidFlow-v0`, `Lorenz-v0`, and `DoubleWell-v0`; use `--help` for rollout sizes, dictionary orders, regressors, animation, and `--save_model`.
- Device and dynamics checks: `uv run pytest tests/test_device_utils.py tests/test_gpu_parity.py -v`. CUDA-only tests skip without a GPU; this is not evidence of CUDA parity. The parity suite covers monomials, OLS tensor predictions, batched dynamics, and CUDA generation, not every regressor or generator mode.
- CPU algorithm smoke tests: `uv run pytest tests/test_rl.py -k 'test_skvi[LinearSystem-v0] or test_sakc[LinearSystem-v0]' -v`. Broader environment, CUDA, and FP32 cases live in the same file; tuning smoke tests live in `tests/test_hparam_opt.py` and require substantially more time/resources.
- When changing construction, check lifted predictions and decoded predictions on aligned transitions, plus singleton/shared-action and paired-action batch shapes. Use held-out transitions when assessing model accuracy; construction diagnostics and successful RL execution alone do not establish predictive accuracy.
