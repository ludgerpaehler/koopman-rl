import pytest
import torch

from tests.utils import run_module

TOTAL_TIMESTEPS = 1000
ENVS = ["LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0"]

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("env_id", ENVS)
def test_lqr(env_id):
    result = run_module(
        "koopmanrl.linear_quadratic_regulator",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_sac_q(env_id):
    result = run_module(
        "koopmanrl.sac_continuous_action",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_sac_v(env_id):
    result = run_module(
        "koopmanrl.value_based_sac_continuous_action",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_skvi(env_id):
    result = run_module(
        "koopmanrl.soft_koopman_value_iteration",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_sakc(env_id):
    result = run_module(
        "koopmanrl.soft_actor_koopman_critic",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


@cuda_only
@pytest.mark.parametrize("env_id", ENVS)
def test_skvi_cuda(env_id):
    result = run_module(
        "koopmanrl.soft_koopman_value_iteration",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr


@cuda_only
@pytest.mark.parametrize("env_id", ENVS)
def test_sakc_cuda(env_id):
    result = run_module(
        "koopmanrl.soft_actor_koopman_critic",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda", "--learning_starts=500"],
    )
    assert result.returncode == 0, result.stderr


@cuda_only
@pytest.mark.parametrize("env_id", ENVS)
def test_sac_q_cuda(env_id):
    result = run_module(
        "koopmanrl.sac_continuous_action",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda"],
    )
    assert result.returncode == 0, result.stderr


@cuda_only
@pytest.mark.parametrize("env_id", ENVS)
def test_sac_v_cuda(env_id):
    result = run_module(
        "koopmanrl.value_based_sac_continuous_action",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda"],
    )
    assert result.returncode == 0, result.stderr


@cuda_only
def test_skvi_fp32_cuda():
    result = run_module(
        "koopmanrl.soft_koopman_value_iteration",
        [f"--env_id={ENVS[0]}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda", "--fp32"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr


@cuda_only
def test_sakc_fp32_cuda():
    result = run_module(
        "koopmanrl.soft_actor_koopman_critic",
        [f"--env_id={ENVS[0]}", f"--total_timesteps={TOTAL_TIMESTEPS}", "--cuda", "--fp32", "--learning_starts=500"],
    )
    assert result.returncode == 0, result.stderr
