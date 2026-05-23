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
