# Pluggable regression backends for the Koopman tensor

**Date:** 2026-05-25
**Status:** Approved (design); pending implementation plan
**Author:** Ludger Paehler (with Claude)

## Goal

At the core of the Koopman-assisted RL scheme is the regression that produces the Koopman tensor.
Today every supported regressor in `koopmanrl/koopman_tensor/torch_tensor.py` is either OLS or a
rudimentary variant of it. This effort:

1. Reimplements the **SINDy** and **Ridge** regressors *anew*, to the algorithmic best practices used
   by packages such as PySINDy / scikit-learn.
2. Adds a new **PySR** (symbolic-regression) backend that can be triggered like any other regressor.
3. Adds **extensive isolation tests** for each of the four regressors (OLS baseline, SINDy, Ridge,
   PySR) validating each scheme in isolation against known ground truth.
4. Runs the **wider test suite against all backends** and keeps continuous integration green.

The work is delivered as **three independent pull requests off `master`** — one per backend
(SINDy, Ridge, PySR) — each carrying its own regressor rewrite/addition and its tests.

## Baseline (current state, verified 2026-05-25)

- Canonical tensor: `koopmanrl/koopman_tensor/torch_tensor.py` (device/dtype-aware, used by SKVI &
  SAKC). It already exposes `ols`, `SINDy`, `rrr`, `ridgeRegression`, and a `Regressor` enum
  (`OLS`, `SINDy`, `RRR`, `RIDGE`). These same functions also exist on `master` (the GPU branch only
  added device-awareness, orthogonal to regression quality).
- `koopmanrl/koopman_tensor/numpy_tensor.py` is a **legacy** parallel implementation (no `RIDGE`
  member, used by nothing in the RL path).
- `pysindy`, `pysr`, and `scikit-learn` are **not installed**.
- CI = a single pre-commit / ruff `Lint` workflow (`.github/workflows/lint.yml`, line-length 120,
  rule sets `E`,`F`). No pytest job. Tests run locally:
  `tests/test_rl.py` (subprocess RL smoke), `tests/test_gpu_parity.py` (exercises the tensor
  directly, OLS only), `tests/test_environments.py`, `tests/test_device_utils.py`.
- There are **no dedicated regression unit tests** yet.

## Decisions (locked)

| Topic | Decision |
|---|---|
| SINDy / Ridge library strategy | **Reimplement in-house in pure torch**, following PySINDy / sklearn best practices. No heavy new runtime deps; stays device/dtype-aware. |
| PR structure | **3 PRs off `master`**, one per backend: SINDy, Ridge, PySR. Each carries its own tests. |
| Tensor scope | **`torch_tensor.py` only.** `numpy_tensor.py` (legacy) and the `RRR` regressor are left untouched. |
| PySR wiring | **Optional dependency group + graceful skip.** Lazy import; clear error if missing; PySR-Julia tests skip when unavailable (mirrors the existing CUDA-skip pattern). |
| PySR → tensor bridge | **Linear-coefficient extraction.** PySR runs per output column, biased to affine-in-features models; linear coefficients are read from the best expression via `sympy` to assemble a linear `M`. `K`/`K_`/`phi_f`/`f` keep working unchanged. |
| Common contract | Every regressor is `regress(X, Y, **kwargs) -> coef_matrix` of shape `(n_features, n_targets)`, pure torch, device/dtype-preserving. |
| Hyperparameters | Threaded through a new `regressor_kwargs: dict | None = None` argument on `KoopmanTensor.__init__` and `generate_koopman_tensor`. Defaults keep the current `0.05`. |
| CI scope | "CI passing" = existing pre-commit `Lint` workflow green. Full `pytest` run locally per backend and reported. No new pytest CI job (shared infra change none of these PRs should own; possible follow-up). |

## Non-goals

- Touching `numpy_tensor.py` or the `RRR` regressor.
- Adding a pytest GitHub Actions job.
- Adding `pysindy` / `scikit-learn` as runtime dependencies (best practices are reimplemented, not
  delegated).
- Changing the default regressor (`OLS` stays the default everywhere).
- Making the RL training entrypoints (SKVI/SAKC) accept a `--regressor` flag (out of scope; the
  cross-backend validation goes through the `KoopmanTensor` / `generate_koopman_tensor` API).

## Architecture

### Common regressor contract

All regressors share one signature and one return shape so the dispatch and the downstream tensor
machinery are agnostic to which is used:

```
regress(X, Y, **kwargs) -> Tensor of shape (n_features, n_targets)
```

- `X`: `(n_samples, n_features)`, `Y`: `(n_samples, n_targets)`.
- Pure torch; the returned coefficient matrix lives on `X.device` with `X.dtype`.
- Used identically for both regressions performed in `KoopmanTensor.__init__`:
  the operator regression `kron_matrix.T → regression_Y.T` (producing `M`) and the state-reconstruction
  regression `Phi_X.T → X.T` (producing `B`).

### 1. SINDy — best-practice STLSQ

Replaces the current fixed-`λ`, 10-iteration loop (no normalization, no inner regularization, no
convergence check).

`sindy(X, Y, threshold=0.05, alpha=0.0, max_iter=20, tol=1e-8) -> coef`

- **Column normalization** of the feature matrix before thresholding, so the threshold is
  scale-invariant; coefficients rescaled back to original units on return. (The headline practice the
  old implementation lacked.)
- **Inner solve = ridge-regularized least squares** with strength `alpha` (PySINDy STLSQ behaviour);
  `alpha=0` ⇒ ordinary least squares inner solve.
- **Sequential thresholding** per target column: zero out coefficients with normalized magnitude
  `< threshold`, refit on the surviving support.
- **Convergence**: iterate until the support (zero/non-zero pattern) is unchanged between iterations
  or `max_iter` reached — not a hardcoded 10.
- **Degenerate guard**: a column whose entire support is thresholded away yields an all-zero column
  rather than erroring on an empty design matrix.
- **Reduction property**: `threshold=0, alpha=0` reproduces OLS to numerical tolerance.

### 2. Ridge — best-practice closed form

Replaces `torch.linalg.inv(XᵀX + λI) @ Xᵀ y` (explicit inverse, ill-conditioned-unstable).

`ridge(X, Y, alpha=0.05) -> coef`

- Solve the normal equations via `torch.linalg.solve(XᵀX + αI, Xᵀ Y)` (no explicit inverse;
  Cholesky/`solve` is stable on near-collinear features).
- Multi-output; device/dtype preserved.
- **Reduction property**: `alpha=0` reproduces OLS.
- **Shrinkage property**: increasing `alpha` monotonically reduces `‖coef‖`.

### 3. PySR — evolutionary sparse-linear regression (optional backend)

`pysr_regression(X, Y, **pysr_kwargs) -> coef`

- **Lazy import**: `pysr` imported only when this regressor is selected. If absent, raise a clear,
  actionable error naming the optional dependency group.
- Added as an **optional dependency group** in `pyproject.toml` (`[dependency-groups] pysr`), never in
  core `dependencies`.
- For each target column: fit a `PySRRegressor` on the lifted features, with the search **biased toward
  affine-in-features models** (operator set and low complexity budget); sensible fast defaults
  (small `niterations`, fixed seed) overridable via `pysr_kwargs`.
- **Linear-coefficient extraction**: parse the best discovered expression with `sympy` and read the
  coefficient of each feature symbol (`Poly` / `.coeff`), dropping nonlinear/constant remainder, to
  fill that column of `M`. This guarantees a linear `M` and keeps the Koopman operator interface intact.
- The extraction helper is a **pure function** (`sympy expression + feature symbols -> coefficient
  vector`) and is **unit-testable in CI without Julia**.

### 4. Shared plumbing

- Add `PYSR = "pysr"` to the `Regressor` enum (`SINDy`, `RIDGE` already present).
- Add `regressor_kwargs: dict | None = None` to `KoopmanTensor.__init__` and to
  `generate_koopman_tensor(...)`; the dispatch forwards it (`**regressor_kwargs`) to the selected
  regressor for both the `M` and `B` solves. `rank` is retained for `RRR`.
- Changes are additive; existing call sites (which pass no `regressor_kwargs`) are unaffected.

## Testing strategy

Each PR adds `tests/test_regression_<name>.py` with isolation tests against synthetic data with known
ground truth, plus one end-to-end `KoopmanTensor`-with-this-backend test. The existing suite
(`test_rl.py`, `test_gpu_parity.py`, `test_environments.py`, `test_device_utils.py`) must remain green.

**SINDy isolation tests**
1. Exact sparse recovery: `Y = Θ Ξ` with known sparse `Ξ`, no noise ⇒ support and coefficients
   recovered within tolerance.
2. Spurious-feature elimination: irrelevant high-variance columns are thresholded to exactly 0.
3. Scale-invariance: scaling one feature column by `1e3` recovers the same support (the test the old
   non-normalized implementation fails).
4. Reduction: `threshold=0, alpha=0` matches `torch.linalg.lstsq` within tolerance.
5. Noisy recovery: small additive noise ⇒ correct support, coefficients close.
6. Convergence: terminates by support-stabilization before `max_iter` on an easy problem; deterministic.
7. Contract: returns `(n_features, n_targets)` on `X`'s device/dtype.

**Ridge isolation tests**
1. Closed-form correctness: matches `solve(XᵀX + αI, Xᵀ Y)` within tolerance.
2. Reduction: `alpha=0` matches OLS / `lstsq`.
3. Shrinkage monotonicity: `‖coef‖` decreases as `alpha` increases.
4. Ill-conditioned stability: near-collinear `X` ⇒ finite, low-error solution with no NaN/Inf
   (where the naive `inv` path degrades).
5. Multi-output contract and device/dtype preservation.

**PySR isolation tests**
1. sympy coefficient-extraction correctness on a known expression — **runs in CI without Julia**.
2. Graceful, clearly-worded error when `pysr` is unavailable (import monkeypatched).
3. Linear recovery on a tiny fixed-seed problem with small `niterations` — **skipped** when
   `pysr`/Julia is unavailable.
4. Contract: returns `(n_features, n_targets)`.

**Cross-backend integration ("wider tests against all backends")**
- In each PR, an end-to-end test builds a `KoopmanTensor` (or calls `generate_koopman_tensor`) with the
  PR's backend and asserts construction succeeds, `K` has the expected shape, and `f(x, u)` predicts
  sanely (compared against the OLS baseline / ground truth). The PySR end-to-end test skips when the
  backend is unavailable.
- Collectively the three PRs cover OLS (baseline), SINDy, Ridge, and PySR.

## Branch / PR / execution strategy

- Three branches off `master`: `regression/sindy`, `regression/ridge`, `regression/pysr` → three PRs.
- Developed **sequentially** to keep the shared `torch_tensor.py` coherent, but each PR is **independently
  mergeable**. The only overlap between PRs is the `Regressor` enum, the dispatch block, and the
  `regressor_kwargs` signature line — small, expected, resolved at merge time.
- Each PR: implement the backend → add its tests → run `uv run pytest` (full suite) → run
  `uv run pre-commit run --all-files` → open PR with results.

## Verification & CI

- **Lint CI** (`uv run pre-commit run --all-files`) green: ruff `E`/`F`, line-length 120.
- **Local full suite** (`uv run pytest`) green per backend; PySR-Julia tests skipped where the optional
  dependency is absent, exactly like the existing CUDA-skip tests.
- Evidence (command output) captured before any "passing" claim and included in each PR description.
