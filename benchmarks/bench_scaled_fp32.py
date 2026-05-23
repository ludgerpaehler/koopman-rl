"""Scaled CPU(FP64) vs GPU(FP64) vs GPU(FP32) benchmark for KoopmanRL.
Real code paths. Each point = mean +/- stdev of N_RUNS timed runs after one warmup.
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
CPU = torch.device("cpu")
CUDA = torch.device("cuda")
F64, F32 = torch.float64, torch.float32
# (label, device, dtype)
MODES = [("cpu", CPU, F64), ("g64", CUDA, F64), ("g32", CUDA, F32)]
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


def table(title, rows):
    # rows: list of (label, {mode: (mean,std)})
    print(f"\n### {title}  (mean of {N_RUNS} runs)\n", flush=True)
    print(f"| {'Scale':<16} | {'CPU FP64 (s)':>14} | {'GPU FP64 (s)':>14} | {'GPU FP32 (s)':>14} | {'FP64x':>7} | {'FP32x':>7} |")
    print(f"|{'-'*18}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*9}|{'-'*9}|")
    for label, r in rows:
        cpu, g64, g32 = r["cpu"], r["g64"], r["g32"]
        s64 = cpu[0] / g64[0] if g64[0] > 0 else float("nan")
        s32 = cpu[0] / g32[0] if g32[0] > 0 else float("nan")
        print(f"| {label:<16} | {cpu[0]:>9.4f}±{cpu[1]:<4.4f} | {g64[0]:>9.4f}±{g64[1]:<4.4f} | {g32[0]:>9.4f}±{g32[1]:<4.4f} | {s64:>6.2f}x | {s32:>6.2f}x |", flush=True)


def bench_dyn_sweep(env_id, batches, steps=50):
    rows = []
    base = gym.make(env_id).unwrapped
    for B in batches:
        r = {}
        for name, dev, dt in MODES:
            g = torch.Generator(device=dev).manual_seed(0)
            s0 = base.reset_batch(B, device=dev, dtype=dt, generator=g)
            low = torch.as_tensor(base.action_space.low, device=dev, dtype=dt)
            high = torch.as_tensor(base.action_space.high, device=dev, dtype=dt)
            acts = low + (high - low) * torch.rand(steps, B, base.action_dim, device=dev, dtype=dt, generator=g)

            def run(s0=s0, acts=acts, base=base, g=g):
                s = s0.clone()
                for t in range(steps):
                    s = base.f_batch(s, acts[t], generator=g)
                return s

            r[name] = timed(run, dev.type == "cuda")
        rows.append((f"batch={B}", r))
        print(f"  [dyn {env_id}] batch={B} done", flush=True)
    table(f"Env dynamics f_batch sweep - {env_id.replace('-v0','')} ({steps} steps)", rows)


def bench_datagen_sweep(env_id, paths_list, steps=20):
    rows = []
    for NP in paths_list:
        r = {}
        for name, dev, dt in MODES:
            def run(NP=NP, dev=dev, dt=dt):
                return generate_koopman_tensor(env_id, 1, NP, steps, 2, 2, "ols", device=dev, dtype=dt)

            r[name] = timed(run, dev.type == "cuda")
        rows.append((f"paths={NP}", r))
        print(f"  [datagen {env_id}] paths={NP} done", flush=True)
    table(f"Koopman data-gen sweep - {env_id.replace('-v0','')} ({steps} steps/path)", rows)


def bench_skvi_compute(batches, num_actions=201):
    rows = []
    kts = {(dev.type, dt): generate_koopman_tensor("Lorenz-v0", 1, 50, 50, 2, 2, "ols", device=dev, dtype=dt)
           for _, dev, dt in MODES}
    for B in batches:
        r = {}
        for name, dev, dt in MODES:
            kt = kts[(dev.type, dt)]
            g = torch.Generator(device=dev).manual_seed(0)
            all_actions = torch.linspace(-75, 75, num_actions, device=dev, dtype=dt).reshape(1, -1)
            states = torch.randn(3, B, generator=g, device=dev, dtype=dt)
            w = torch.randn(kt.phi_dim, device=dev, dtype=dt)
            costs = torch.randn(num_actions, B, device=dev, dtype=dt)
            eps = torch.finfo(dt).eps

            def run(kt=kt, all_actions=all_actions, states=states, w=w, costs=costs, eps=eps):
                phi_x = kt.phi(states)
                K_us = kt.K_(all_actions)
                phi_xp = torch.einsum("aij,jb->aib", K_us, phi_x)
                V = torch.einsum("p,apb->ab", w, phi_xp)
                inner = -(costs + 0.99 * V)
                m = inner.amax(0)
                pi = torch.exp(inner - m) + eps
                return pi / pi.sum(0)

            r[name] = timed(run, dev.type == "cuda")
        rows.append((f"batch={B}", r))
        print(f"  [skvi-compute] batch={B} done", flush=True)
    table(f"SKVI value-iteration core compute - batch sweep ({num_actions} actions)", rows)


def bench_sakc_update(batches):
    rows = []
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("Lorenz-v0")])
    for B in batches:
        r = {}
        for name, dev, dt in MODES:
            qf = SoftQNetwork(envs).to(dev).to(dt)
            opt = torch.optim.Adam(qf.parameters(), lr=1e-3)
            g = torch.Generator(device=dev).manual_seed(0)
            obs = torch.randn(B, 3, generator=g, device=dev, dtype=dt)
            act = torch.randn(B, 1, generator=g, device=dev, dtype=dt)
            tgt = torch.randn(B, generator=g, device=dev, dtype=dt)

            def run(qf=qf, opt=opt, obs=obs, act=act, tgt=tgt):
                q = qf(obs, act).view(-1)
                loss = torch.nn.functional.mse_loss(q, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                return loss

            r[name] = timed(run, dev.type == "cuda")
        rows.append((f"batch={B}", r))
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
