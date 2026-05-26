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
        X,
        Y,
        U,
        phi=monomials(2),
        psi=monomials(2),
        regressor=Regressor.SINDy,
        regressor_kwargs={"threshold": 0.2, "alpha": 1e-3},
    )
    assert kt.K is not None
