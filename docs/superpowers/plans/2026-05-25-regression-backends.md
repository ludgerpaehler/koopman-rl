# Pluggable Koopman-tensor regression backends — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement SINDy and Ridge regression to best-practice standards and add a new optional PySR backend in the canonical Koopman tensor, each delivered as its own PR off `master` with extensive isolation tests.

**Architecture:** All regressors share one contract — `regress(X, Y, **kwargs) -> coef (n_features, n_targets)`, pure torch, device/dtype-preserving — so `KoopmanTensor.K`/`K_`/`phi_f`/`f` keep working unchanged. Hyperparameters are threaded via a new `regressor_kwargs` argument. PySR runs symbolic regression per output column and extracts linear coefficients via sympy, keeping `M` linear. Three independent branches off `master`: `regression/sindy`, `regression/ridge`, `regression/pysr`.

**Tech Stack:** Python 3.10, PyTorch (float64 default), uv, pytest, pre-commit (ruff/vulture/isort). Optional: pysr (Julia backend) + sympy for the PySR PR.

---

## Shared conventions (read once before starting any Part)

- **Baseline:** all three branches are cut from `master` **after** the GPU work was merged (PR #26, merge commit `f28d66f`). So `koopmanrl/koopman_tensor/torch_tensor.py` is the **device-aware** version: pure-torch `K` reshape (no numpy round-trip, lines ~208-209), device-aware `ridgeRegression` (`torch.eye(..., device=X.device, dtype=X.dtype)`, line ~74), and `generate_koopman_tensor` lives in this file (line ~279). The test suite includes `tests/test_gpu_parity.py` and `tests/test_device_utils.py` (CUDA-marked tests **skip** without a GPU; their CPU paths run).
- **File being modified in every Part:** `koopmanrl/koopman_tensor/torch_tensor.py` (the canonical, algorithm-used tensor). `numpy_tensor.py` and the `RRR` regressor are **not** touched.
- **Contract:** every regressor takes `X (n_samples, n_features)`, `Y (n_samples, n_targets)`, returns `coef (n_features, n_targets)` on `X.device`/`X.dtype`. `Y` may arrive 1-D; unsqueeze to a column internally. Keep regressors pure-torch and device/dtype-preserving (master's `K` reshape consumes the returned tensor directly — no numpy).
- **Dispatch site:** `KoopmanTensor.__init__` solves twice — operator `M` from `kron_matrix.T → regression_Y.T`, and `B` from `Phi_X.T → X.T`. Both use the selected regressor.
- **Running tests:** `uv run pytest <path> -v`. The repo has no `python` on PATH — always go through `uv run`.
- **Network:** SSH to GitHub is unavailable in this environment; the gh CLI is configured as the git HTTPS credential helper, so `git push` / `gh pr create` work over HTTPS. `origin` may need its URL set to HTTPS (`git remote set-url origin https://github.com/ludgerpaehler/koopman-rl.git`) for `git push -u origin` to succeed.
- **Lint == CI:** `uv run pre-commit run --all-files` (ruff, ruff-format, vulture on `koopmanrl/`, isort, basic hooks). This is the only GitHub CI workflow.
- **Baseline caveat:** `tests/test_hparam_opt.py` (8 tests) fails on `master` already (hardcoded absolute `storage_dir`). If those 8 fail, confirm they fail identically on clean `master` — they are **not** regressions from this work. `tests/test_rl.py`, `tests/test_environments.py`, and the CPU paths of `tests/test_gpu_parity.py` / `tests/test_device_utils.py` are the green baseline.
- **Each Part is one PR.** All three branch off `master`; they overlap only on the `Regressor` enum, the dispatch `elif` block, and the `regressor_kwargs` signature/normalization lines. That overlap is small and resolved at merge time. Develop in order A → B → C.

---

# Part A — SINDy backend (branch `regression/sindy`)

This branch already exists, is rebased onto the post-GPU-merge `master` (`f28d66f`), and already holds **both** doc commits (design spec + this plan). Verify with `git branch --show-current` → `regression/sindy` and `git log --oneline -3` showing the two `docs:` commits on top of `f28d66f`.

### Task A0: Confirm planning docs are committed

**Files:**
- Already committed: `docs/superpowers/specs/2026-05-25-regression-backends-design.md`
- Already committed: `docs/superpowers/plans/2026-05-25-regression-backends.md`

- [ ] **Step 1: Verify branch and docs**

Run:
```bash
git branch --show-current   # expect: regression/sindy
git log --oneline -3        # expect: two docs: commits on top of f28d66f (Merge PR #26)
```
No commit needed — both docs are already present on this branch. Proceed to Task A1.

---

### Task A1: Write failing isolation tests for `sindy`

**Files:**
- Test: `tests/test_regression_sindy.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_regression_sindy.py`:

```python
import torch

from koopmanrl.koopman_tensor.torch_tensor import sindy

torch.manual_seed(0)


def test_sindy_exact_sparse_recovery():
    """Clean data from a known sparse linear map is recovered exactly."""
    g = torch.Generator().manual_seed(0)
    n_samples, n_features, n_targets = 200, 6, 2
    X = torch.randn(n_samples, n_features, generator=g)
    Xi_true = torch.zeros(n_features, n_targets)
    Xi_true[0, 0] = 2.0
    Xi_true[3, 0] = -1.5
    Xi_true[1, 1] = 0.7
    Xi_true[4, 1] = 3.0
    Y = X @ Xi_true
    Xi = sindy(X, Y, threshold=0.1, alpha=0.0)
    assert torch.allclose(Xi, Xi_true, atol=1e-6)


def test_sindy_eliminates_spurious_features():
    """Features absent from the true model are driven to exactly zero."""
    g = torch.Generator().manual_seed(3)
    X = torch.randn(300, 6, generator=g)
    Xi_true = torch.zeros(6, 1)
    Xi_true[1] = 2.0
    Xi_true[4] = -1.0
    Y = X @ Xi_true
    Xi = sindy(X, Y, threshold=0.1)
    for spurious in (0, 2, 3, 5):
        assert Xi[spurious, 0] == 0.0


def test_sindy_threshold_is_scale_invariant():
    """Rescaling a feature column must not change the recovered support
    (the property the old, non-normalized implementation failed)."""
    g = torch.Generator().manual_seed(2)
    X = torch.randn(200, 4, generator=g)
    Xi_true = torch.tensor([[1.5], [0.0], [2.0], [0.0]])
    Y = X @ Xi_true
    support_a = sindy(X, Y, threshold=0.1).abs() > 0
    X2 = X.clone()
    X2[:, 0] *= 1000.0
    support_b = sindy(X2, Y, threshold=0.1).abs() > 0
    assert torch.equal(support_a, support_b)


def test_sindy_threshold_zero_reduces_to_ols():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(100, 5, generator=g)
    Y = torch.randn(100, 3, generator=g)
    Xi = sindy(X, Y, threshold=0.0, alpha=0.0)
    ols_sol = torch.linalg.lstsq(X, Y, rcond=None).solution
    assert torch.allclose(Xi, ols_sol, atol=1e-8)


def test_sindy_recovers_support_under_noise():
    g = torch.Generator().manual_seed(4)
    X = torch.randn(500, 5, generator=g)
    Xi_true = torch.zeros(5, 1)
    Xi_true[0] = 3.0
    Xi_true[2] = -2.0
    Y = X @ Xi_true + 0.01 * torch.randn(500, 1, generator=g)
    Xi = sindy(X, Y, threshold=0.5)
    assert (Xi[[1, 3, 4], 0] == 0).all()
    assert torch.allclose(Xi[[0, 2], 0], Xi_true[[0, 2], 0], atol=0.05)


def test_sindy_ridge_inner_solve_keeps_support():
    g = torch.Generator().manual_seed(6)
    X = torch.randn(200, 5, generator=g)
    Xi_true = torch.zeros(5, 1)
    Xi_true[0] = 1.0
    Xi_true[3] = 2.0
    Y = X @ Xi_true
    Xi = sindy(X, Y, threshold=0.1, alpha=1e-3)
    assert torch.equal(Xi.abs() > 0, Xi_true.abs() > 0)


def test_sindy_is_deterministic():
    g = torch.Generator().manual_seed(5)
    X = torch.randn(200, 5, generator=g)
    Xi_true = torch.zeros(5, 2)
    Xi_true[0, 0] = 1.0
    Xi_true[3, 1] = 2.0
    Y = X @ Xi_true
    a = sindy(X, Y, threshold=0.1, max_iter=20)
    b = sindy(X, Y, threshold=0.1, max_iter=20)
    assert torch.equal(a, b)


def test_sindy_contract_shape_and_dtype():
    X = torch.randn(50, 4)
    Y = torch.randn(50, 3)
    Xi = sindy(X, Y)
    assert Xi.shape == (4, 3)
    assert Xi.dtype == X.dtype


def test_sindy_handles_1d_target():
    X = torch.randn(40, 3)
    y = torch.randn(40)
    Xi = sindy(X, y, threshold=0.0)
    assert Xi.shape == (3, 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_regression_sindy.py -v`
Expected: FAIL — `ImportError: cannot import name 'sindy'` (the lowercase `sindy` does not exist yet).

---

### Task A2: Implement best-practice `sindy` (STLSQ)

**Files:**
- Modify: `koopmanrl/koopman_tensor/torch_tensor.py` (replace the existing `SINDy` function, lines ~31-46)

- [ ] **Step 1: Replace the `SINDy` function with the new implementation**

In `koopmanrl/koopman_tensor/torch_tensor.py`, replace the whole existing `SINDy` definition (the `def SINDy(Theta, dXdt, lamb=0.05): ...` block — initial `lstsq`, a `for _ in range(10)` thresholding loop, `return L`) with:

```python
def sindy(X, Y, threshold=0.05, alpha=0.0, max_iter=20):
    """
    Sequentially thresholded least squares (STLSQ), following PySINDy practice.

    Solves ``Y ~= X @ Xi`` for a sparse coefficient matrix ``Xi`` by repeatedly
    least-squares fitting and zeroing coefficients whose column-normalized
    magnitude falls below ``threshold``. Column normalization makes ``threshold``
    scale-invariant; the inner solve is optionally ridge-regularized by ``alpha``.

    Parameters
    ----------
    X : (n_samples, n_features) tensor
        Feature library.
    Y : (n_samples, n_targets) tensor
        Regression targets (a 1-D target is treated as a single column).
    threshold : float
        Sparsity threshold applied to normalized coefficients (``0`` -> OLS).
    alpha : float
        Ridge regularization strength used inside each least-squares solve.
    max_iter : int
        Maximum number of thresholding iterations.

    Returns
    -------
    Xi : (n_features, n_targets) tensor
        Sparse coefficient matrix on ``X``'s device/dtype.
    """
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    n_features = X.shape[1]
    n_targets = Y.shape[1]

    # Column-normalize the library so the threshold does not depend on feature scale.
    col_norms = torch.linalg.norm(X, dim=0)
    col_norms = torch.where(col_norms > 0, col_norms, torch.ones_like(col_norms))
    Xn = X / col_norms

    def _inner(A, b):
        # Solve min ||A w - b||^2 + alpha ||w||^2 for a single target column.
        if A.shape[1] == 0:
            return torch.zeros(0, device=A.device, dtype=A.dtype)
        if alpha == 0.0:
            return torch.linalg.lstsq(A, b.unsqueeze(1), rcond=None).solution[:, 0]
        eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(A.T @ A + alpha * eye, A.T @ b)

    # Initial least-squares fit on the normalized library.
    Xi = torch.zeros(n_features, n_targets, device=X.device, dtype=X.dtype)
    for j in range(n_targets):
        Xi[:, j] = _inner(Xn, Y[:, j])

    # Sequentially threshold and refit until the support stops changing.
    prev_support = None
    for _ in range(max_iter):
        smallinds = torch.abs(Xi) < threshold
        Xi[smallinds] = 0.0
        support = ~smallinds
        for j in range(n_targets):
            biginds = support[:, j]
            if biginds.any():
                Xi[biginds, j] = _inner(Xn[:, biginds], Y[:, j])
        if prev_support is not None and torch.equal(support, prev_support):
            break
        prev_support = support

    # Undo column normalization: coefficients were fit against X / col_norms.
    Xi = Xi / col_norms.unsqueeze(1)
    return Xi


# Backwards-compatible alias (mirrors the ols/OLS, rrr/RRR naming convention).
def SINDy(X, Y, **kwargs):
    return sindy(X, Y, **kwargs)
```

- [ ] **Step 2: Run the isolation tests**

Run: `uv run pytest tests/test_regression_sindy.py -v`
Expected: PASS (all 9 tests).

- [ ] **Step 3: Commit**

```bash
git add koopmanrl/koopman_tensor/torch_tensor.py tests/test_regression_sindy.py
git commit -m "feat: best-practice STLSQ sindy regressor with isolation tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A3: Wire `sindy` into `KoopmanTensor` with `regressor_kwargs`

**Files:**
- Modify: `koopmanrl/koopman_tensor/torch_tensor.py` (`KoopmanTensor.__init__` signature + dispatch)

- [ ] **Step 1: Add `regressor_kwargs` to the signature**

Change the `__init__` signature line:

```python
    def __init__(self, X, Y, U, phi, psi, regressor=Regressor.OLS, rank=8, is_generator=False, dt=0.01):
```
to:
```python
    def __init__(
        self, X, Y, U, phi, psi, regressor=Regressor.OLS, rank=8, is_generator=False, dt=0.01, regressor_kwargs=None
    ):
```

- [ ] **Step 2: Normalize `regressor_kwargs` at the top of `__init__`**

Immediately after the docstring (before `# Save datasets`), add:

```python
        regressor_kwargs = regressor_kwargs or {}
```

- [ ] **Step 3: Update the SINDy dispatch branch**

Replace:
```python
        elif regressor == Regressor.SINDy:
            self.M = SINDy(self.kron_matrix.T, self.regression_Y.T).T
            self.B = SINDy(self.Phi_X.T, self.X.T)
```
with:
```python
        elif regressor == Regressor.SINDy:
            self.M = sindy(self.kron_matrix.T, self.regression_Y.T, **regressor_kwargs).T
            self.B = sindy(self.Phi_X.T, self.X.T, **regressor_kwargs)
```

- [ ] **Step 4: Append the end-to-end integration tests**

Append to `tests/test_regression_sindy.py`:

```python
def _build_dataset():
    g = torch.Generator().manual_seed(0)
    n, state_dim, action_dim = 300, 2, 1
    X = torch.randn(state_dim, n, generator=g)
    A = torch.tensor([[0.9, 0.1], [-0.2, 0.8]])
    Y = A @ X
    U = torch.randn(action_dim, n, generator=g)
    return X, Y, U, state_dim


def test_koopman_tensor_sindy_end_to_end():
    from koopmanrl.koopman_tensor.observables.torch_observables import monomials
    from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor

    X, Y, U, state_dim = _build_dataset()
    kt = KoopmanTensor(X, Y, U, phi=monomials(2), psi=monomials(2), regressor=Regressor.SINDy)
    assert kt.K.shape == (kt.phi_dim, kt.phi_dim, kt.psi_dim)
    x = torch.randn(state_dim, 8)
    u = torch.randn(1, 8)
    pred = kt.f(x, u)
    assert pred.shape == (state_dim, 8)
    assert torch.isfinite(pred).all()


def test_koopman_tensor_accepts_regressor_kwargs():
    from koopmanrl.koopman_tensor.observables.torch_observables import monomials
    from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor

    X, Y, U, _ = _build_dataset()
    kt = KoopmanTensor(
        X, Y, U, phi=monomials(2), psi=monomials(2),
        regressor=Regressor.SINDy, regressor_kwargs={"threshold": 0.2, "alpha": 1e-3},
    )
    assert kt.K is not None
```

- [ ] **Step 5: Run the SINDy test file**

Run: `uv run pytest tests/test_regression_sindy.py -v`
Expected: PASS (11 tests total).

- [ ] **Step 6: Commit**

```bash
git add koopmanrl/koopman_tensor/torch_tensor.py tests/test_regression_sindy.py
git commit -m "feat: thread regressor_kwargs through KoopmanTensor; SINDy integration tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A4: Verify (lint + wider suite) and open PR

- [ ] **Step 1: Run lint (the CI check)**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass (ruff may auto-fix formatting; if it modifies files, re-stage and re-run until clean).

- [ ] **Step 2: Run the wider suite against the SINDy backend**

Run: `uv run pytest tests/test_regression_sindy.py tests/test_environments.py tests/test_gpu_parity.py tests/test_device_utils.py tests/test_rl.py -v`
Expected: SINDy + environment tests PASS; `test_rl.py` PASS (20); `test_gpu_parity.py`/`test_device_utils.py` CPU tests PASS, CUDA-marked tests SKIP. If anything fails, debug before proceeding. (`test_hparam_opt.py` is excluded — pre-existing failures unrelated to this work.)

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin regression/sindy
gh pr create --base master --title "feat: best-practice SINDy (STLSQ) regression backend" --body "$(cat <<'EOF'
## Summary
- Reimplements the SINDy regressor as best-practice sequential thresholded least squares (STLSQ): column-normalized (scale-invariant) thresholding, optional ridge inner solve (`alpha`), support-stabilization convergence — replacing the previous fixed-λ, 10-iteration loop.
- Threads a `regressor_kwargs` argument through `KoopmanTensor` so regressor hyperparameters are configurable.
- Adds `tests/test_regression_sindy.py`: exact sparse recovery, spurious-feature elimination, scale-invariance, threshold→OLS reduction, noisy recovery, ridge inner solve, determinism, contract, plus end-to-end `KoopmanTensor` tests.

## Verification
- `uv run pytest tests/test_regression_sindy.py` — green (paste output).
- `uv run pytest tests/test_rl.py tests/test_environments.py` — green (paste output).
- `uv run pre-commit run --all-files` — green.

Design spec: `docs/superpowers/specs/2026-05-25-regression-backends-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# Part B — Ridge backend (branch `regression/ridge`)

### Task B1: Create the branch off `master`

- [ ] **Step 1: Branch off clean master**

Run:
```bash
git stash -u 2>/dev/null; true   # park any stray working-tree noise
git checkout -b regression/ridge master
git branch --show-current        # expect: regression/ridge
```
Expected: on `regression/ridge`, working tree matches `master` (the SINDy work is NOT present here — independent PR).

---

### Task B2: Write failing isolation tests for `ridge`

**Files:**
- Test: `tests/test_regression_ridge.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_regression_ridge.py`:

```python
import torch

from koopmanrl.koopman_tensor.torch_tensor import ridge

torch.manual_seed(0)


def test_ridge_matches_closed_form():
    """Matches the explicit-inverse closed form on a well-conditioned problem."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(80, 5, generator=g)
    Y = torch.randn(80, 3, generator=g)
    alpha = 0.5
    eye = torch.eye(5)
    ref = torch.linalg.inv(X.T @ X + alpha * eye) @ X.T @ Y
    assert torch.allclose(ridge(X, Y, alpha=alpha), ref, atol=1e-8)


def test_ridge_alpha_zero_reduces_to_ols():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(80, 5, generator=g)
    Y = torch.randn(80, 3, generator=g)
    ols_sol = torch.linalg.lstsq(X, Y, rcond=None).solution
    assert torch.allclose(ridge(X, Y, alpha=0.0), ols_sol, atol=1e-8)


def test_ridge_shrinks_coefficients_with_alpha():
    g = torch.Generator().manual_seed(2)
    X = torch.randn(80, 5, generator=g)
    Y = torch.randn(80, 3, generator=g)
    small = torch.linalg.norm(ridge(X, Y, alpha=0.01))
    large = torch.linalg.norm(ridge(X, Y, alpha=100.0))
    assert large < small


def test_ridge_is_stable_on_ill_conditioned_features():
    """Near-collinear columns: ridge stays finite where naive inverse degrades."""
    g = torch.Generator().manual_seed(3)
    base = torch.randn(100, 1, generator=g)
    X = torch.cat([base, base + 1e-9 * torch.randn(100, 1, generator=g), torch.randn(100, 2, generator=g)], dim=1)
    Y = torch.randn(100, 2, generator=g)
    Xi = ridge(X, Y, alpha=1.0)
    assert Xi.shape == (4, 2)
    assert torch.isfinite(Xi).all()


def test_ridge_contract_shape_and_dtype():
    X = torch.randn(30, 4)
    Y = torch.randn(30, 2)
    Xi = ridge(X, Y)
    assert Xi.shape == (4, 2)
    assert Xi.dtype == X.dtype


def test_ridge_handles_1d_target():
    X = torch.randn(40, 3)
    y = torch.randn(40)
    Xi = ridge(X, y, alpha=0.1)
    assert Xi.shape == (3, 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_regression_ridge.py -v`
Expected: FAIL — `ImportError: cannot import name 'ridge'`.

---

### Task B3: Implement best-practice `ridge`

**Files:**
- Modify: `koopmanrl/koopman_tensor/torch_tensor.py` (replace `ridgeRegression`, lines ~74-76)

The current implementation is already device-aware but forms an explicit inverse. The improvement is replacing `torch.linalg.inv` with `torch.linalg.solve` for numerical stability on ill-conditioned features.

- [ ] **Step 1: Replace `ridgeRegression` with a stable `solve`-based implementation**

Replace:
```python
def ridgeRegression(X, y, lamb=0.05):
    eye = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.inv(X.T @ X + lamb * eye) @ X.T @ y
```
with:
```python
def ridge(X, Y, alpha=0.05):
    """
    Ridge regression via the normal equations, solved without forming an
    explicit inverse (numerically stable on ill-conditioned feature matrices).

    Solves ``(X^T X + alpha I) Xi = X^T Y`` for ``Xi``.

    Parameters
    ----------
    X : (n_samples, n_features) tensor
    Y : (n_samples, n_targets) tensor
        Regression targets (a 1-D target is treated as a single column).
    alpha : float
        L2 regularization strength (``alpha=0`` -> OLS).

    Returns
    -------
    Xi : (n_features, n_targets) tensor
        Coefficient matrix on ``X``'s device/dtype.
    """
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    if alpha == 0.0:
        return torch.linalg.lstsq(X, Y, rcond=None).solution
    eye = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(X.T @ X + alpha * eye, X.T @ Y)


# Backwards-compatible alias.
def ridgeRegression(X, Y, alpha=0.05):
    return ridge(X, Y, alpha=alpha)
```

- [ ] **Step 2: Run the isolation tests**

Run: `uv run pytest tests/test_regression_ridge.py -v`
Expected: PASS (6 tests).

- [ ] **Step 3: Commit**

```bash
git add koopmanrl/koopman_tensor/torch_tensor.py tests/test_regression_ridge.py
git commit -m "feat: stable solve-based ridge regressor with isolation tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task B4: Wire `ridge` into `KoopmanTensor` with `regressor_kwargs`

**Files:**
- Modify: `koopmanrl/koopman_tensor/torch_tensor.py` (`KoopmanTensor.__init__`)

- [ ] **Step 1: Add `regressor_kwargs` to the signature**

Change:
```python
    def __init__(self, X, Y, U, phi, psi, regressor=Regressor.OLS, rank=8, is_generator=False, dt=0.01):
```
to:
```python
    def __init__(
        self, X, Y, U, phi, psi, regressor=Regressor.OLS, rank=8, is_generator=False, dt=0.01, regressor_kwargs=None
    ):
```

- [ ] **Step 2: Normalize `regressor_kwargs`**

After the docstring (before `# Save datasets`), add:
```python
        regressor_kwargs = regressor_kwargs or {}
```

- [ ] **Step 3: Update the RIDGE dispatch branch**

Replace:
```python
        elif regressor == Regressor.RIDGE:
            self.M = ridgeRegression(self.kron_matrix.T, self.regression_Y.T).T
            self.B = ridgeRegression(self.Phi_X.T, self.X.T)
```
with:
```python
        elif regressor == Regressor.RIDGE:
            self.M = ridge(self.kron_matrix.T, self.regression_Y.T, **regressor_kwargs).T
            self.B = ridge(self.Phi_X.T, self.X.T, **regressor_kwargs)
```

- [ ] **Step 4: Append end-to-end integration tests**

Append to `tests/test_regression_ridge.py`:

```python
def _build_dataset():
    g = torch.Generator().manual_seed(0)
    n, state_dim, action_dim = 300, 2, 1
    X = torch.randn(state_dim, n, generator=g)
    A = torch.tensor([[0.9, 0.1], [-0.2, 0.8]])
    Y = A @ X
    U = torch.randn(action_dim, n, generator=g)
    return X, Y, U, state_dim


def test_koopman_tensor_ridge_end_to_end():
    from koopmanrl.koopman_tensor.observables.torch_observables import monomials
    from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor

    X, Y, U, state_dim = _build_dataset()
    kt = KoopmanTensor(X, Y, U, phi=monomials(2), psi=monomials(2), regressor=Regressor.RIDGE)
    assert kt.K.shape == (kt.phi_dim, kt.phi_dim, kt.psi_dim)
    x = torch.randn(state_dim, 8)
    u = torch.randn(1, 8)
    pred = kt.f(x, u)
    assert pred.shape == (state_dim, 8)
    assert torch.isfinite(pred).all()


def test_koopman_tensor_ridge_accepts_alpha_kwarg():
    from koopmanrl.koopman_tensor.observables.torch_observables import monomials
    from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor

    X, Y, U, _ = _build_dataset()
    kt = KoopmanTensor(
        X, Y, U, phi=monomials(2), psi=monomials(2),
        regressor=Regressor.RIDGE, regressor_kwargs={"alpha": 1.0},
    )
    assert kt.K is not None
```

- [ ] **Step 5: Run the ridge test file**

Run: `uv run pytest tests/test_regression_ridge.py -v`
Expected: PASS (8 tests total).

- [ ] **Step 6: Commit**

```bash
git add koopmanrl/koopman_tensor/torch_tensor.py tests/test_regression_ridge.py
git commit -m "feat: thread regressor_kwargs through KoopmanTensor; ridge integration tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task B5: Verify (lint + wider suite) and open PR

- [ ] **Step 1: Lint**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass (re-stage/re-run if ruff auto-fixes).

- [ ] **Step 2: Wider suite against the ridge backend**

Run: `uv run pytest tests/test_regression_ridge.py tests/test_environments.py tests/test_gpu_parity.py tests/test_device_utils.py tests/test_rl.py -v`
Expected: ridge + environment tests PASS; `test_rl.py` PASS (20); `test_gpu_parity.py`/`test_device_utils.py` CPU tests PASS, CUDA-marked tests SKIP.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin regression/ridge
gh pr create --base master --title "feat: stable solve-based Ridge regression backend" --body "$(cat <<'EOF'
## Summary
- Reimplements Ridge regression via the normal equations using `torch.linalg.solve` (no explicit inverse) — numerically stable on ill-conditioned feature matrices; `alpha=0` reduces to OLS; device/dtype preserved.
- Threads a `regressor_kwargs` argument through `KoopmanTensor` so `alpha` is configurable.
- Adds `tests/test_regression_ridge.py`: closed-form correctness, alpha→OLS reduction, shrinkage monotonicity, ill-conditioned stability, contract/dtype, 1-D target, plus end-to-end `KoopmanTensor` tests.

## Verification
- `uv run pytest tests/test_regression_ridge.py` — green (paste output).
- `uv run pytest tests/test_rl.py tests/test_environments.py` — green (paste output).
- `uv run pre-commit run --all-files` — green.

Design spec: `docs/superpowers/specs/2026-05-25-regression-backends-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# Part C — PySR backend (branch `regression/pysr`)

### Task C1: Create the branch off `master`

- [ ] **Step 1: Branch off clean master**

Run:
```bash
git checkout -b regression/pysr master
git branch --show-current     # expect: regression/pysr
```

---

### Task C2: Add the optional `pysr` dependency group

**Files:**
- Modify: `pyproject.toml` (`[dependency-groups]`)

- [ ] **Step 1: Add the `pysr` group**

In `pyproject.toml`, in the `[dependency-groups]` table (which currently has `dev = [...]`), add a new group:

```toml
pysr = [
    "pysr>=1.0.0",
    "sympy>=1.12",
]
```

- [ ] **Step 2: Resolve the optional group (best effort)**

Run: `uv sync --group pysr`
Expected: resolves and installs `pysr` + `sympy`. NOTE: `pysr` pulls a Julia runtime on first use; if installation or Julia provisioning fails in this environment, continue — the sympy-only and missing-dependency tests still run, and the live PySR test is written to skip when the backend is unavailable. Record what happened.

- [ ] **Step 3: Commit the dependency change**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add optional pysr dependency group (pysr + sympy)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```
(If `uv.lock` was not modified because the group could not be resolved, commit only `pyproject.toml`.)

---

### Task C3: Write failing isolation tests for PySR

**Files:**
- Test: `tests/test_regression_pysr.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_regression_pysr.py`:

```python
import builtins

import pytest
import torch

torch.manual_seed(0)


def test_extract_linear_coeffs_reads_linear_terms():
    sympy = pytest.importorskip("sympy")
    from koopmanrl.koopman_tensor.torch_tensor import _extract_linear_coeffs

    x0, x1, x2 = sympy.symbols("x0 x1 x2")
    expr = 3.0 * x0 - 2.0 * x2 + 5.0  # x1 absent; constant 5 dropped
    coeffs = _extract_linear_coeffs(expr, [x0, x1, x2])
    assert coeffs == pytest.approx([3.0, 0.0, -2.0])


def test_extract_linear_coeffs_drops_nonlinear_terms():
    sympy = pytest.importorskip("sympy")
    from koopmanrl.koopman_tensor.torch_tensor import _extract_linear_coeffs

    x0, x1 = sympy.symbols("x0 x1")
    expr = 4.0 * x0 + x0 * x1 + x1**2  # cross & quadratic terms dropped
    coeffs = _extract_linear_coeffs(expr, [x0, x1])
    assert coeffs == pytest.approx([4.0, 0.0])


def test_pysr_regression_raises_clear_error_when_missing(monkeypatch):
    from koopmanrl.koopman_tensor import torch_tensor

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pysr" or name.startswith("pysr."):
            raise ImportError("simulated missing pysr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    X = torch.randn(20, 3)
    Y = torch.randn(20, 2)
    with pytest.raises(ImportError, match="pysr"):
        torch_tensor.pysr_regression(X, Y)


def test_pysr_regression_recovers_linear_map():
    pytest.importorskip("pysr")
    from koopmanrl.koopman_tensor.torch_tensor import pysr_regression

    g = torch.Generator().manual_seed(0)
    X = torch.randn(60, 2, generator=g)
    Xi_true = torch.tensor([[2.0], [-1.0]])
    Y = X @ Xi_true
    try:
        Xi = pysr_regression(X, Y, niterations=10)
    except Exception as e:  # Julia backend not provisioned, etc.
        pytest.skip(f"PySR/Julia unavailable: {e}")
    assert Xi.shape == (2, 1)
    assert torch.allclose(Xi, Xi_true, atol=0.25)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_regression_pysr.py -v`
Expected: FAIL — `ImportError: cannot import name '_extract_linear_coeffs'` / `pysr_regression` (functions not defined yet). The two extraction tests fail to import; the missing-dependency test fails; the recovery test errors on import.

---

### Task C4: Implement `_extract_linear_coeffs` and `pysr_regression`

**Files:**
- Modify: `koopmanrl/koopman_tensor/torch_tensor.py` (add functions after `ridgeRegression`)

- [ ] **Step 1: Add the two functions**

In `koopmanrl/koopman_tensor/torch_tensor.py`, after the `ridgeRegression` function and before the `""" Regressor enum """` comment, add:

```python
def _extract_linear_coeffs(expression, feature_symbols):
    """
    Read the linear coefficient of each feature symbol from a sympy expression.

    Constant and nonlinear (cross / higher-order) terms are dropped, so a feature
    that appears only nonlinearly — or not at all — contributes 0.

    Parameters
    ----------
    expression : sympy.Expr
        The discovered symbolic expression.
    feature_symbols : list of sympy.Symbol
        Feature symbols, in the order of the columns of the feature matrix.

    Returns
    -------
    list of float
        Linear coefficients aligned with ``feature_symbols``.
    """
    import sympy

    expr = sympy.expand(expression)
    coeffs = []
    for sym in feature_symbols:
        # coeff(sym, 1) is the multiplier of sym^1; it may still depend on other
        # features (e.g. for a cross term x_i*x_j), so keep only its feature-free part.
        raw = expr.coeff(sym, 1)
        const_part, _ = raw.as_independent(*feature_symbols, as_Add=True)
        coeffs.append(float(const_part))
    return coeffs


def pysr_regression(X, Y, **pysr_kwargs):
    """
    Symbolic-regression backend: fit each target column with PySR and assemble a
    linear coefficient matrix by extracting per-feature linear coefficients from
    the discovered expressions (keeping ``M`` linear so the Koopman tensor stays
    intact).

    Requires the optional ``pysr`` dependency (Julia backend). Raises a clear
    ImportError if it is unavailable.

    Parameters
    ----------
    X : (n_samples, n_features) tensor
    Y : (n_samples, n_targets) tensor
        Regression targets (a 1-D target is treated as a single column).
    **pysr_kwargs
        Overrides forwarded to ``PySRRegressor`` (e.g. ``niterations``).

    Returns
    -------
    Xi : (n_features, n_targets) tensor
        Linear coefficient matrix on ``X``'s dtype (CPU; PySR runs on numpy).
    """
    try:
        from pysr import PySRRegressor
    except ImportError as e:
        raise ImportError(
            "The 'pysr' regression backend requires the optional 'pysr' dependency "
            "(and a Julia runtime). Install it with `uv sync --group pysr`."
        ) from e
    import sympy

    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    n_features = X.shape[1]
    n_targets = Y.shape[1]

    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    feature_names = [f"x{i}" for i in range(n_features)]
    feature_symbols = [sympy.Symbol(name) for name in feature_names]

    # Defaults bias the search toward small, affine-in-features models and make
    # runs deterministic; callers may override any of these via pysr_kwargs.
    defaults = dict(
        niterations=20,
        binary_operators=["+", "*"],
        unary_operators=[],
        maxsize=2 * n_features + 5,
        progress=False,
        verbosity=0,
        random_state=0,
        deterministic=True,
        parallelism="serial",
    )
    defaults.update(pysr_kwargs)

    Xi = torch.zeros(n_features, n_targets, dtype=X.dtype)
    for j in range(n_targets):
        model = PySRRegressor(**defaults)
        model.fit(X_np, Y_np[:, j], variable_names=feature_names)
        coeffs = _extract_linear_coeffs(model.sympy(), feature_symbols)
        Xi[:, j] = torch.tensor(coeffs, dtype=X.dtype)
    return Xi
```

> **PySR version note:** the determinism kwargs (`deterministic`, `parallelism`) and `model.sympy()` reflect PySR ≥ 1.0. If `uv sync --group pysr` installed a different major version and `PySRRegressor(**defaults)` rejects a kwarg, adjust `defaults` to that version's API (older PySR used `procs=0, multithreading=False`). The sympy-extraction tests do not depend on PySR and validate the core logic regardless.

- [ ] **Step 2: Run the PySR test file**

Run: `uv run pytest tests/test_regression_pysr.py -v`
Expected: the two `_extract_linear_coeffs` tests PASS; `test_pysr_regression_raises_clear_error_when_missing` PASS; `test_pysr_regression_recovers_linear_map` PASSES if pysr+Julia are available, otherwise SKIPS. No failures.

- [ ] **Step 3: Commit**

```bash
git add koopmanrl/koopman_tensor/torch_tensor.py tests/test_regression_pysr.py
git commit -m "feat: PySR symbolic-regression backend with linear-coefficient extraction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C5: Register the PySR regressor in `KoopmanTensor`

**Files:**
- Modify: `koopmanrl/koopman_tensor/torch_tensor.py` (`Regressor` enum + dispatch + `__init__` signature)

- [ ] **Step 1: Add `PYSR` to the enum**

Change:
```python
class Regressor(str, Enum):
    OLS = "ols"
    SINDy = "sindy"
    RRR = "rrr"
    RIDGE = "ridge"
```
to:
```python
class Regressor(str, Enum):
    OLS = "ols"
    SINDy = "sindy"
    RRR = "rrr"
    RIDGE = "ridge"
    PYSR = "pysr"
```

- [ ] **Step 2: Add `regressor_kwargs` to the signature**

Change:
```python
    def __init__(self, X, Y, U, phi, psi, regressor=Regressor.OLS, rank=8, is_generator=False, dt=0.01):
```
to:
```python
    def __init__(
        self, X, Y, U, phi, psi, regressor=Regressor.OLS, rank=8, is_generator=False, dt=0.01, regressor_kwargs=None
    ):
```

- [ ] **Step 3: Normalize `regressor_kwargs`**

After the docstring (before `# Save datasets`), add:
```python
        regressor_kwargs = regressor_kwargs or {}
```

- [ ] **Step 4: Add the PYSR dispatch branch**

Find the `elif regressor == Regressor.RIDGE:` block and insert a new branch immediately after it (before the closing `else:` that raises):
```python
        elif regressor == Regressor.PYSR:
            self.M = pysr_regression(self.kron_matrix.T, self.regression_Y.T, **regressor_kwargs).T
            self.B = pysr_regression(self.Phi_X.T, self.X.T, **regressor_kwargs)
```

- [ ] **Step 5: Append the integration test (skips without the backend)**

Append to `tests/test_regression_pysr.py`:

```python
def test_koopman_tensor_pysr_end_to_end():
    pytest.importorskip("pysr")
    from koopmanrl.koopman_tensor.observables.torch_observables import monomials
    from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor

    g = torch.Generator().manual_seed(0)
    state_dim, action_dim, n = 2, 1, 200
    X = torch.randn(state_dim, n, generator=g)
    A = torch.tensor([[0.9, 0.1], [-0.2, 0.8]])
    Y = A @ X
    U = torch.randn(action_dim, n, generator=g)
    try:
        kt = KoopmanTensor(
            X, Y, U, phi=monomials(2), psi=monomials(2),
            regressor=Regressor.PYSR, regressor_kwargs={"niterations": 5},
        )
    except Exception as e:
        pytest.skip(f"PySR/Julia unavailable: {e}")
    assert kt.K.shape == (kt.phi_dim, kt.phi_dim, kt.psi_dim)
    pred = kt.f(torch.randn(state_dim, 4), torch.randn(1, 4))
    assert pred.shape == (state_dim, 4)
    assert torch.isfinite(pred).all()
```

- [ ] **Step 6: Run the PySR test file**

Run: `uv run pytest tests/test_regression_pysr.py -v`
Expected: extraction + missing-dependency tests PASS; live/integration tests PASS or SKIP. No failures.

- [ ] **Step 7: Commit**

```bash
git add koopmanrl/koopman_tensor/torch_tensor.py tests/test_regression_pysr.py
git commit -m "feat: register PYSR regressor in KoopmanTensor dispatch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C6: Verify (lint + wider suite) and open PR

- [ ] **Step 1: Lint**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass. (Vulture scans `koopmanrl/` at min-confidence 80; `_extract_linear_coeffs`/`pysr_regression` are referenced by the dispatch and won't be flagged.)

- [ ] **Step 2: Wider suite**

Run: `uv run pytest tests/test_regression_pysr.py tests/test_environments.py tests/test_gpu_parity.py tests/test_device_utils.py tests/test_rl.py -v`
Expected: PySR tests PASS/SKIP; environment + `test_rl.py` PASS (20); `test_gpu_parity.py`/`test_device_utils.py` CPU tests PASS, CUDA-marked tests SKIP.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin regression/pysr
gh pr create --base master --title "feat: optional PySR symbolic-regression backend" --body "$(cat <<'EOF'
## Summary
- Adds a `pysr` regressor: fits each target column with PySR symbolic regression and extracts per-feature linear coefficients via sympy, assembling a linear `M` so the Koopman operator interface is unchanged.
- `pysr` is an **optional** dependency group (`uv sync --group pysr`); the backend lazily imports it and raises a clear error if missing. PySR/Julia tests skip when unavailable (mirrors the CUDA-skip pattern).
- Adds `tests/test_regression_pysr.py`: sympy coefficient-extraction (runs without Julia), missing-dependency error, live linear recovery (skips without Julia), end-to-end `KoopmanTensor` (skips without Julia).

## Verification
- `uv run pytest tests/test_regression_pysr.py` — extraction + missing-dep tests green; live tests green or skipped (paste output, note skips).
- `uv run pytest tests/test_rl.py tests/test_environments.py` — green (paste output).
- `uv run pre-commit run --all-files` — green.

Design spec: `docs/superpowers/specs/2026-05-25-regression-backends-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Final self-review checklist (run after all three Parts)

- [ ] All three PRs open against `master` (`gh pr list`).
- [ ] `sindy`, `ridge`, `pysr_regression` all honor the `(n_features, n_targets)` contract and device/dtype.
- [ ] `regressor_kwargs` plumbing present and identical in each PR's `__init__` (expected trivial merge overlap).
- [ ] No regressions in `tests/test_rl.py` / `tests/test_environments.py`; `test_hparam_opt.py` failures (if any) confirmed pre-existing.
- [ ] `uv run pre-commit run --all-files` green on each branch.
