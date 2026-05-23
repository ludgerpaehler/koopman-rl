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
