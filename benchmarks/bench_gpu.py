"""CPU-only vs GPU-hybrid timing benchmark for KoopmanRL.

Each comparison point is the mean (+/- stdev) of N_RUNS timed runs, after one
untimed warmup run. GPU timing uses torch.cuda.synchronize() on both sides.
Precision is FP64 (the shipped default).
"""

import contextlib
import io
import statistics
import time

import torch

import koopmanrl.environments  # noqa: F401 (register envs)
import gymnasium as gym
from koopmanrl.koopman_tensor.torch_tensor import generate_koopman_tensor
from koopmanrl.opt_wrappers import sakc_tuning_wrapper, skvi_tuning_wrapper

N_RUNS = 5
ENVS = ["LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0"]
DTYPE = torch.float64

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA required for this benchmark"
print(f"Device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}\n", flush=True)


def timed(fn, is_cuda):
    """Return (mean_seconds, stdev_seconds) over N_RUNS, after one warmup."""
    with contextlib.redirect_stdout(io.StringIO()):
        fn()  # warmup (kernel/cublas/cusolver init, caches)
    if is_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(N_RUNS):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


def print_table(title, rows):
    """rows: list of (label, (cpu_mean,cpu_std), (gpu_mean,gpu_std))."""
    print(f"\n### {title}  (mean of {N_RUNS} runs, FP64)\n", flush=True)
    print(f"| {'Case':<16} | {'CPU (s)':>16} | {'GPU (s)':>16} | {'Speedup':>9} |")
    print(f"|{'-'*18}|{'-'*18}|{'-'*18}|{'-'*11}|")
    for label, cpu, gpu in rows:
        cpu_s = f"{cpu[0]:.4f} ± {cpu[1]:.4f}"
        gpu_s = f"{gpu[0]:.4f} ± {gpu[1]:.4f}"
        spd = cpu[0] / gpu[0] if gpu[0] > 0 else float("nan")
        spd_s = f"{spd:.2f}x"
        print(f"| {label:<16} | {cpu_s:>16} | {gpu_s:>16} | {spd_s:>9} |", flush=True)


# ---------------------------------------------------------------------------
# Group 1: batched environment dynamics (f_batch)
# ---------------------------------------------------------------------------
def bench_env_dynamics():
    BATCH, STEPS = 8192, 200
    rows = []
    for env_id in ENVS:
        base = gym.make(env_id).unwrapped
        res = {}
        for dev in (CPU, CUDA):
            g = torch.Generator(device=dev).manual_seed(0)
            s0 = base.reset_batch(BATCH, device=dev, dtype=DTYPE, generator=g)
            low = torch.as_tensor(base.action_space.low, device=dev, dtype=DTYPE)
            high = torch.as_tensor(base.action_space.high, device=dev, dtype=DTYPE)
            acts = low + (high - low) * torch.rand(
                STEPS, BATCH, base.action_dim, device=dev, dtype=DTYPE, generator=g
            )

            def run(s0=s0, acts=acts, base=base, g=g):
                s = s0.clone()
                for t in range(STEPS):
                    s = base.f_batch(s, acts[t], generator=g)
                return s

            res[dev.type] = timed(run, dev.type == "cuda")
        rows.append((env_id.replace("-v0", ""), res["cpu"], res["cuda"]))
        print(f"  [g1] {env_id} done", flush=True)
    print_table(f"Group 1 - Batched env dynamics f_batch (batch={BATCH}, {STEPS} steps)", rows)


# ---------------------------------------------------------------------------
# Group 2: Koopman data-gen + tensor build (generate_koopman_tensor)
#   CPU path = sequential gym/scipy stepping; GPU path = batched RK4 rollout.
# ---------------------------------------------------------------------------
def bench_data_gen():
    NP, NS = 100, 100
    rows = []
    for env_id in ENVS:
        res = {}
        for dev in (CPU, CUDA):
            def run(env_id=env_id, dev=dev):
                return generate_koopman_tensor(env_id, 1, NP, NS, 2, 2, "ols", device=dev, dtype=DTYPE)

            res[dev.type] = timed(run, dev.type == "cuda")
        rows.append((env_id.replace("-v0", ""), res["cpu"], res["cuda"]))
        print(f"  [g2] {env_id} done", flush=True)
    print_table(f"Group 2 - Koopman data-gen + tensor build (num_paths={NP}, steps/path={NS})", rows)


# ---------------------------------------------------------------------------
# Group 3: end-to-end Koopman algorithms via opt_wrappers (data-gen+train+rollout)
# ---------------------------------------------------------------------------
def bench_skvi():
    rows = []
    for env_id in ENVS:
        res = {}
        for dev in (CPU, CUDA):
            cuda = dev.type == "cuda"
            def run(env_id=env_id, cuda=cuda):
                return skvi_tuning_wrapper(
                    env_id=env_id, seed=1, number_of_paths=50, number_of_steps_per_path=100,
                    number_of_training_epochs=5, batch_size=512, total_timesteps=200,
                    cuda=cuda,
                )
            res[dev.type] = timed(run, cuda)
        rows.append((env_id.replace("-v0", ""), res["cpu"], res["cuda"]))
        print(f"  [g3-skvi] {env_id} done", flush=True)
    print_table("Group 3a - SKVI end-to-end (50x100 data, 5 epochs, 200 steps)", rows)


def bench_sakc():
    rows = []
    for env_id in ENVS:
        res = {}
        for dev in (CPU, CUDA):
            cuda = dev.type == "cuda"
            def run(env_id=env_id, cuda=cuda):
                return sakc_tuning_wrapper(
                    env_id=env_id, seed=1, number_of_paths=50, number_of_steps_per_path=100,
                    total_timesteps=1500, learning_starts=500, buffer_size=50000,
                    cuda=cuda,
                )
            res[dev.type] = timed(run, cuda)
        rows.append((env_id.replace("-v0", ""), res["cpu"], res["cuda"]))
        print(f"  [g3-sakc] {env_id} done", flush=True)
    print_table("Group 3b - SAKC end-to-end (50x100 data, 1500 steps, learn@500)", rows)


if __name__ == "__main__":
    t_start = time.perf_counter()
    bench_env_dynamics()
    bench_data_gen()
    bench_skvi()
    bench_sakc()
    print(f"\nTotal benchmark wall time: {time.perf_counter() - t_start:.1f}s", flush=True)
