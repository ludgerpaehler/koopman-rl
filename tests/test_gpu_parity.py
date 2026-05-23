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


from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor


def _make_tensor(device):
    # Generate the data on CPU first, then move to the target device, so the CPU and
    # GPU tensors are trained on *identical* data. torch.randn uses different RNG
    # streams on CPU vs CUDA even with the same seed, so generating per-device would
    # train the two tensors on different data and defeat the parity comparison.
    torch.manual_seed(0)
    N, state_dim, action_dim = 500, 3, 1
    X = torch.randn(state_dim, N, dtype=torch.float64).to(device)
    Y = (X.cpu() + 0.01 * torch.randn(state_dim, N, dtype=torch.float64)).to(device)
    U = torch.randn(action_dim, N, dtype=torch.float64).to(device)
    return KoopmanTensor(X, Y, U, phi=monomials(2), psi=monomials(2), regressor=Regressor.OLS)


@CUDA
def test_koopman_tensor_cpu_gpu_parity():
    kt_cpu = _make_tensor(torch.device("cpu"))
    kt_gpu = _make_tensor(torch.device("cuda"))
    assert kt_gpu.K.is_cuda and kt_gpu.B.is_cuda
    assert torch.allclose(kt_cpu.K, kt_gpu.K.cpu(), atol=1e-6)
    assert torch.allclose(kt_cpu.B, kt_gpu.B.cpu(), atol=1e-6)
    x = torch.randn(3, 16, dtype=torch.float64)
    u = torch.randn(1, 16, dtype=torch.float64)
    f_cpu = kt_cpu.f(x, u)
    f_gpu = kt_gpu.f(x.cuda(), u.cuda())
    assert torch.allclose(f_cpu, f_gpu.cpu(), atol=1e-6)
