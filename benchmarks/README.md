# KoopmanRL GPU Benchmarks

Timing benchmarks comparing **CPU-only (FP64)** against the **GPU hybrid** paths
(batched env dynamics, Koopman data-generation, the SKVI value-iteration compute,
and the SAKC critic update). Each comparison point is the mean ± stdev of 5 timed
runs after one untimed warmup; GPU timings use `torch.cuda.synchronize()` on both
sides of the measured region.

## Requirements

- An NVIDIA GPU with a CUDA-enabled PyTorch (the scripts `assert torch.cuda.is_available()`).
  Validated on an **RTX 5090 (Blackwell, sm_120)** with `torch 2.9.1+cu128`.
- The project environment: `uv sync`.

## Running

Run from the repository root so `koopmanrl` is importable:

```bash
uv run python benchmarks/bench_gpu.py          # baseline-size, CPU vs GPU (FP64), 4 layers
uv run python benchmarks/bench_scaled.py       # batch/path sweeps, CPU vs GPU (FP64)
uv run python benchmarks/bench_scaled_fp32.py  # batch/path sweeps, CPU(FP64) vs GPU FP64 vs GPU FP32
```

Each script prints one Markdown table per benchmark layer. `bench_scaled_fp32.py` is
the most complete (adds the FP32 GPU column). Tweak `N_RUNS` and the sweep sizes at
the bottom of each script.

## What each layer measures

| Layer | What | CPU path | GPU path |
|---|---|---|---|
| Env dynamics | `env.f_batch` stepping a batch | torch on CPU | torch on CUDA (substepped RK4 / Euler–Maruyama) |
| Data-gen | `generate_koopman_tensor` | sequential gym + `scipy.solve_ivp` | batched RK4 rollout |
| SKVI core | vectorized value-iteration einsums | torch on CPU | torch on CUDA |
| SAKC critic | real 256-wide `SoftQNetwork` fwd+bwd+step | torch on CPU | torch on CUDA |

## Representative results (RTX 5090, torch 2.9.1+cu128, mean of 5 runs)

CPU column is FP64 (the "CPU-only" reference). Speedups are vs that baseline.

### Env dynamics `f_batch` — Lorenz (RK4), 50 steps

| Batch | CPU FP64 (s) | GPU FP64 (s) | GPU FP32 (s) | FP64× | FP32× |
|---|---|---|---|---|---|
| 8,192 | 0.0551 | 0.0312 | 0.0316 | 1.77× | 1.75× |
| 65,536 | 0.1783 | 0.0314 | 0.0323 | 5.69× | 5.53× |
| 262,144 | 0.8620 | 0.0316 | 0.0317 | 27.2× | 27.2× |
| 1,048,576 | 7.8104 | 0.1161 | 0.0583 | 67.3× | 134× |

### Env dynamics `f_batch` — DoubleWell (Euler–Maruyama), 50 steps

| Batch | CPU FP64 (s) | GPU FP64 (s) | GPU FP32 (s) | FP64× | FP32× |
|---|---|---|---|---|---|
| 8,192 | 0.0141 | 0.0030 | 0.0030 | 4.70× | 4.70× |
| 65,536 | 0.0918 | 0.0049 | 0.0032 | 18.8× | 29.1× |
| 262,144 | 0.3831 | 0.0161 | 0.0078 | 23.8× | 49.1× |
| 1,048,576 | 1.7108 | 0.0596 | 0.0288 | 28.7× | 59.4× |

### Koopman data-gen — Lorenz, num_paths sweep, 20 steps/path

| Paths | CPU FP64 (s) | GPU FP64 (s) | GPU FP32 (s) | FP64× | FP32× |
|---|---|---|---|---|---|
| 100 | 0.1400 | 0.0242 | 0.0162 | 5.77× | 8.66× |
| 400 | 0.5451 | 0.0323 | 0.0169 | 16.9× | 32.2× |
| 1,600 | 2.1812 | 0.0640 | 0.0194 | 34.1× | 112× |

### SKVI value-iteration core compute — batch sweep (201 actions)

| Batch | CPU FP64 (s) | GPU FP64 (s) | GPU FP32 (s) | FP64× | FP32× |
|---|---|---|---|---|---|
| 1,024 | 0.0021 | 0.0005 | 0.0004 | 3.84× | 4.76× |
| 16,384 | 0.0887 | 0.0018 | 0.0008 | 48.9× | 117× |
| 131,072 | 0.7216 | 0.0128 | 0.0042 | 56.2× | 170× |

### SAKC critic fwd+bwd+step — batch sweep (real 256-wide `SoftQNetwork`)

| Batch | CPU FP64 (s) | GPU FP64 (s) | GPU FP32 (s) | FP64× | FP32× |
|---|---|---|---|---|---|
| 256 | 0.0009 | 0.0010 | 0.0004 | 0.89× | 2.07× |
| 4,096 | 0.0068 | 0.0031 | 0.0005 | 2.22× | 14.8× |
| 65,536 | 0.2385 | 0.0383 | 0.0013 | 6.23× | 185× |
| 262,144 | 0.9676 | 0.1516 | 0.0047 | 6.38× | 204× |

## Takeaways

- **GPU dominance scales with the parallel dimension** (batch size, number of paths).
  At small/default sizes the GPU can even lose (SAKC critic at batch 256, FP64: 0.89×)
  because launch overhead beats tiny kernels; it then climbs steeply.
- **FP64 on consumer Blackwell (~1:64 throughput) is the main tax** for matmul-bound
  work. `--fp32` lifts it dramatically: SKVI core 56× → 170×, SAKC critic 6× → ~200×,
  data-gen 34× → 112×.
- **FP32 trades precision for speed.** Recommended for neural-network training
  (SAKC/SAC) and large-batch throughput; use with care for chaotic Lorenz trajectories
  and the Koopman regression (condition numbers ~7e3 here), where FP64 guards against
  error growth. This is why `--fp32` is opt-in, not the default.
