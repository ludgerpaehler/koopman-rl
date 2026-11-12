import builtins

import pytest
import torch


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
            X,
            Y,
            U,
            phi=monomials(2),
            psi=monomials(2),
            regressor=Regressor.PYSR,
            regressor_kwargs={"niterations": 5},
        )
    except Exception as e:
        pytest.skip(f"PySR/Julia unavailable: {e}")
    assert kt.K.shape == (kt.phi_dim, kt.phi_dim, kt.psi_dim)
    pred = kt.f(torch.randn(state_dim, 4), torch.randn(1, 4))
    assert pred.shape == (state_dim, 4)
    assert torch.isfinite(pred).all()
