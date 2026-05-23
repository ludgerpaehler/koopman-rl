import pytest
import torch
import gymnasium as gym
import numpy as np
import koopmanrl.environments  # noqa: F401

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
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-9)  # same RK4 both sides -> tight


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


def test_double_well_diffusion_matches_numpy():
    """Validate the full f_batch (drift + diffusion) against the numpy formula with injected noise."""
    env = gym.make("DoubleWell-v0").unwrapped
    rng = np.random.default_rng(7)
    states = rng.uniform(-2, 2, size=(8, env.state_dim))
    actions = rng.uniform(-5, 5, size=(8, env.action_dim))
    noise = rng.standard_normal(size=(8, env.state_dim, 1))
    # numpy reference: state + drift*dt + (sigma_x @ noise * sqrt(dt))[:,0]
    ref = []
    for i in range(8):
        drift = np.array(env.continuous_f(actions[i])(0, states[i])) * env.dt
        sigma_x = np.array([[0.7, states[i, 0]], [0.0, 0.5]])
        diffusion = (sigma_x @ noise[i] * np.sqrt(env.dt))[:, 0]
        ref.append(states[i] + drift + diffusion)
    ref = np.stack(ref)
    out = env.f_batch(
        torch.tensor(states),
        torch.tensor(actions),
        noise=torch.tensor(noise),
    )
    assert torch.allclose(out, torch.tensor(ref), atol=1e-9)


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
    x = torch.randn(3, 5, dtype=torch.float64, device="cuda")
    u = torch.randn(1, 5, dtype=torch.float64, device="cuda")
    assert kt.f(x, u).is_cuda
