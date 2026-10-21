# moe-offload-lab

Why does a 284-billion-parameter model run at half a token per second on a laptop, and how fast could it go?

DeepSeek V4 Flash is a mixture-of-experts (MoE) model: 167 GB of weights, of which each token uses a small slice. [colibri](https://github.com/JustVugg/colibri) runs it on machines with far less memory than that by keeping the dense weights in RAM and streaming the experts each token needs from the SSD. On this laptop it decodes at about 0.5 tokens/s. This repo measures where that time goes, with a model of the workload, a storage benchmark and a kernel benchmark written for the purpose, and profiled runs of the engine.

**Machine:** AMD Ryzen AI Max+ 392 (12 Zen 5 cores, 24 threads, AVX-512), 64 GB LPDDR5X-8533, Samsung MZVL81T0HFLB 1 TB NVMe SSD, Windows 11. colibri v2.0.0 on the CPU path (its DeepSeek V4 GPU path needs CUDA).

## 1. The workload (`analysis/bytes_per_token.py`)

From the model's `config.json`: 43 layers, 256 routed experts per layer, 6 chosen per token. One expert is three 4096 x 2048 projections, 25.2 M weights, 13.4 MB in FP4 with its block scales.

| | |
|---|---|
| On disk | 166.9 GB = 147.2 GB routed experts + 19.7 GB everything else |
| Routed expert bytes per token | 3.45 GB (43 x 6 x 13.4 MB) |
| From disk per token, 8% of experts cached (colibri's plan) | 3.17 GB |

The 19.7 GB non-expert figure matches the "dense" size colibri's own planner reports, which is a check on the split.

## 2. The drive (`iobench/`)

`iobench` reads expert-sized blocks (12 MiB) at random offsets across the 48 weight files, from N threads, with the OS file cache bypassed (`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux). C++20, `std::jthread`, per-thread latency vectors so the hot loop shares nothing. Clean under ThreadSanitizer and AddressSanitizer (`iobench/sanitize.sh`).

| Threads | GB/s | p50 ms | p99 ms |
|---|---|---|---|
| 1 | 3.52 | 3.5 | 5.4 |
| 2 | 4.79 | 5.2 | 8.1 |
| 4 | 4.85 | 10.3 | 16.4 |
| 8 | 4.35 | 22.5 | 43.3 |

About 4.8 GB/s at two to four reads in flight. With 3.17 GB per token from disk, **the drive alone allows about 1.5 tokens/s**.

## 3. The engine, profiled (`profile/`)

`profile_decode.ps1` runs colibri on a fixed prompt (24 tokens, greedy) and samples the engine process once a second: CPU time across all logical cores and bytes read. colibri learns which experts are popular across runs (`.coli_usage`), which would make each run faster than the last, so `sweep.ps1` restores the same snapshot before every run; all four runs below hit the RAM cache 37% of the time.

| Configuration | Decode tok/s | CPU, share of machine | Disk read |
|---|---|---|---|
| 12 threads (colibri default: physical cores) | 0.49 | 17% | 0.72 GB/s |
| 24 threads | 0.52 | 19% | 0.74 GB/s |
| 6 threads | 0.30 | 10% | 0.50 GB/s |
| load/compute overlap off (`PIPE=0`) | 0.43 | 15% | 0.66 GB/s |

colibri's own byte counter for a run (74.4 GB) agrees with the sampler. Two things stand out:

- **The drive is not the limit.** It delivers 0.7 of the 4.8 GB/s it can.
- **Speed follows the core count**, so compute is in the critical path, yet the cores are busy only about a third of the time. Each layer's router must finish before the engine knows which experts to load, so a token is a chain of 43 dependent load-then-compute steps.

## 4. What the CPU can do with FP4 (`kernels/`)

`fp4_gemv` multiplies a vector by FP4 (E2M1, UE8M0 scale per 32 weights) expert matrices, streaming 48 experts (0.64 GB) so the weights come from DRAM, not cache. The AVX-512 kernel decodes 16 weights per permute instruction (`vpermps` as a 16-entry lookup table) and is checked against the scalar version on every run. Clean under ASan and TSan. Three repeats per point:

| Kernel, threads | ms per expert (mean, range) | GFLOP/s |
|---|---|---|
| scalar, 1 | 15.9 | 3.2 |
| AVX-512, 1 | 1.41 (1.37 to 1.44) | 36 |
| AVX-512, 4 | 0.55 (0.53 to 0.56) | 92 |
| AVX-512, 12 | 0.32 (0.26 to 0.42) | 158 |

A token's 258 expert evaluations take about **0.08 s** with this kernel on 12 cores.

## 5. Where the time goes

| Per token | Seconds |
|---|---|
| Observed (colibri, 12 threads) | ~2.0 |
| Disk, at the measured 4.8 GB/s and 37% cache hits | ~0.45 |
| Routed-expert compute, AVX-512 kernel above | ~0.08 |

If loads and compute overlapped, the drive would set the pace at roughly **1.5 to 2 tokens/s, three to four times today's speed**. The gap is in how the engine computes and sequences the work, not in the hardware.

## 6. Why, from colibri's source (tag v2.0.0)

- **The Windows build cannot use AVX-512.** The release workflow builds the Windows DeepSeek engine without an `ARCH` (`.github/workflows/release.yml:55`), so `c/Makefile.deepseek-v4:51` applies `-march=x86-64-v3` (AVX2), and the kernel variant is chosen at compile time with no runtime dispatch. Disassembling the shipped `deepseek_v4.exe` confirms it: 0 instructions use AVX-512 (`zmm`) registers, 6,224 use AVX2 (`ymm`). The source already has an AVX-512 FP4 kernel using the same permute-as-lookup idea as `kernels/fp4_gemv.cpp` (`c/deepseek_v4.c:19780`); this build never runs it.
- **Experts run one after another**, each a parallel-for over 16-row tiles with a fork-join per projection (`c/deepseek_v4.c:5552`, `5733`, `5792`): about 12 thread barriers per layer and 500 per token, with serial work (activation rounding, SwiGLU) between them.
- **Loads overlap only within a layer.** Up to 9 loader threads start reading a layer's experts once its router has chosen them, and the shared expert computes meanwhile; nothing is fetched for the next layer before its router runs.

## Next experiments

1. Build the DeepSeek engine from source with `-march=znver5` and repeat the sweep: how much of the gap is the instruction set alone?
2. Batch a layer's six experts into one parallel region (one barrier instead of twelve) and measure.
3. Predict the next layer's experts from the current hidden state and prefetch them, measuring the hit rate of the guess.

## Build

```
clang++ -std=c++20 -O2 -static iobench/iobench.cpp -o iobench.exe          # or: cmake -S iobench -B build && cmake --build build
clang++ -std=c++20 -O3 -march=znver5 -static kernels/fp4_gemv.cpp -o fp4_gemv.exe
```
Windows builds used [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) 20261006 (Clang 23); the sanitizer runs used GCC 15 in WSL Ubuntu.
