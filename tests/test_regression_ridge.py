import torch

from koopmanrl.koopman_tensor.torch_tensor import ridge


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
