"""Scaled CPU vs GPU benchmark for KoopmanRL — sweep parallel dimensions to find
where GPU dominance kicks in. Real code paths. FP64 (shipped default).
Each point = mean +/- stdev of N_RUNS timed runs after one untimed warmup.
"""

import contextlib
import io
import statistics
import time

import torch

import koopmanrl.environments  # noqa: F401
import gymnasium as gym
from koopmanrl.koopman_tensor.torch_tensor import generate_koopman_tensor
from koopmanrl.soft_actor_koopman_critic import SoftQNetwork

N_RUNS = 5
DTYPE = torch.float64
CPU = torch.device("cpu")
CUDA = torch.device("cuda")
delta = torch.finfo(DTYPE).eps
assert torch.cuda.is_available()
print(f"Device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}\n", flush=True)


def timed(fn, is_cuda, n=N_RUNS):
    with contextlib.redirect_stdout(io.StringIO()):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        if is_cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def table(title, rows, runs=N_RUNS):
    print(f"\n### {title}  (mean of {runs} runs, FP64)\n", flush=True)
    print(f"| {'Scale':<18} | {'CPU (s)':>16} | {'GPU (s)':>16} | {'Speedup':>9} |")
    print(f"|{'-'*20}|{'-'*18}|{'-'*18}|{'-'*11}|")
    for label, cpu, gpu in rows:
        spd = cpu[0] / gpu[0] if gpu[0] > 0 else float("nan")
        print(f"| {label:<18} | {cpu[0]:>8.4f} ± {cpu[1]:<5.4f} | {gpu[0]:>8.4f} ± {gpu[1]:<5.4f} | {spd:>8.2f}x |", flush=True)


# 1) Env dynamics f_batch: batch-size sweep on Lorenz (RK4) + DoubleWell (EM)
def bench_dyn_sweep(env_id, batches, steps=50):
    rows = []
    base = gym.make(env_id).unwrapped
    for B in batches:
        res = {}
        for dev in (CPU, CUDA):
            g = torch.Generator(device=dev).manual_seed(0)
            s0 = base.reset_batch(B, device=dev, dtype=DTYPE, generator=g)
            low = torch.as_tensor(base.action_space.low, device=dev, dtype=DTYPE)
            high = torch.as_tensor(base.action_space.high, device=dev, dtype=DTYPE)
            acts = low + (high - low) * torch.rand(steps, B, base.action_dim, device=dev, dtype=DTYPE, generator=g)

            def run(s0=s0, acts=acts, base=base, g=g):
                s = s0.clone()
                for t in range(steps):
                    s = base.f_batch(s, acts[t], generator=g)
                return s

            res[dev.type] = timed(run, dev.type == "cuda")
        rows.append((f"batch={B}", res["cpu"], res["cuda"]))
        print(f"  [dyn {env_id}] batch={B} done", flush=True)
    table(f"Env dynamics f_batch sweep - {env_id.replace('-v0','')} ({steps} steps)", rows)


# 2) Koopman data-gen: num_paths sweep (GPU batched vs CPU sequential scipy)
def bench_datagen_sweep(env_id, paths_list, steps=20):
    rows = []
    for NP in paths_list:
        res = {}
        for dev in (CPU, CUDA):
            def run(NP=NP, dev=dev):
                return generate_koopman_tensor(env_id, 1, NP, steps, 2, 2, "ols", device=dev, dtype=DTYPE)

            res[dev.type] = timed(run, dev.type == "cuda")
        rows.append((f"paths={NP}", res["cpu"], res["cuda"]))
        print(f"  [datagen {env_id}] paths={NP} done", flush=True)
    table(f"Koopman data-gen sweep - {env_id.replace('-v0','')} ({steps} steps/path)", rows)


# 3) SKVI value-iteration core compute: batch sweep (the vectorized action einsums)
def bench_skvi_compute(batches, num_actions=201):
    rows = []
    kts = {dev.type: generate_koopman_tensor("Lorenz-v0", 1, 50, 50, 2, 2, "ols", device=dev, dtype=DTYPE)
           for dev in (CPU, CUDA)}
    for B in batches:
        res = {}
        for dev in (CPU, CUDA):
            kt = kts[dev.type]
            g = torch.Generator(device=dev).manual_seed(0)
            all_actions = torch.linspace(-75, 75, num_actions, device=dev, dtype=DTYPE).reshape(1, -1)
            states = torch.randn(3, B, generator=g, device=dev, dtype=DTYPE)
            w = torch.randn(kt.phi_dim, device=dev, dtype=DTYPE)
            costs = torch.randn(num_actions, B, device=dev, dtype=DTYPE)

            def run(kt=kt, all_actions=all_actions, states=states, w=w, costs=costs):
                phi_x = kt.phi(states)                                  # (phi_dim, B)
                K_us = kt.K_(all_actions)                               # (A, phi, phi)
                phi_xp = torch.einsum("aij,jb->aib", K_us, phi_x)       # (A, phi, B)
                V = torch.einsum("p,apb->ab", w, phi_xp)               # (A, B)
                inner = -(costs + 0.99 * V) / 1.0
                m = inner.amax(0)
                pi = torch.exp(inner - m) + delta
                return (pi / pi.sum(0))

            res[dev.type] = timed(run, dev.type == "cuda")
        rows.append((f"batch={B}", res["cpu"], res["cuda"]))
        print(f"  [skvi-compute] batch={B} done", flush=True)
    table(f"SKVI value-iteration core compute - batch sweep ({num_actions} actions)", rows)


# 4) SAKC critic update: batch sweep through the real SoftQNetwork (width 256)
def bench_sakc_update(batches):
    rows = []
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("Lorenz-v0")])
    for B in batches:
        res = {}
        for dev in (CPU, CUDA):
            qf = SoftQNetwork(envs).to(dev)
            opt = torch.optim.Adam(qf.parameters(), lr=1e-3)
            g = torch.Generator(device=dev).manual_seed(0)
            obs = torch.randn(B, 3, generator=g, device=dev, dtype=DTYPE)
            act = torch.randn(B, 1, generator=g, device=dev, dtype=DTYPE)
            tgt = torch.randn(B, generator=g, device=dev, dtype=DTYPE)

            def run(qf=qf, opt=opt, obs=obs, act=act, tgt=tgt):
                q = qf(obs, act).view(-1)
                loss = torch.nn.functional.mse_loss(q, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                return loss

            res[dev.type] = timed(run, dev.type == "cuda")
        rows.append((f"batch={B}", res["cpu"], res["cuda"]))
        print(f"  [sakc-update] batch={B} done", flush=True)
    table("SAKC critic fwd+bwd+step - batch sweep (real 256-wide SoftQNetwork)", rows)


if __name__ == "__main__":
    t0 = time.perf_counter()
    bench_dyn_sweep("Lorenz-v0", [8192, 65536, 262144, 1048576])
    bench_dyn_sweep("DoubleWell-v0", [8192, 65536, 262144, 1048576])
    bench_datagen_sweep("Lorenz-v0", [100, 400, 1600])
    bench_skvi_compute([1024, 16384, 131072])
    bench_sakc_update([256, 4096, 65536, 262144])
    print(f"\nTotal benchmark wall time: {time.perf_counter() - t0:.1f}s", flush=True)
