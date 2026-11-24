# Monte Carlo Option Pricer — Benchmark Report (Ryzen 5 5600X)

All figures below come directly from runs on this machine (`bench_pricer` and,
for the Phase 1 C++17 baseline before retirement, `bench_baseline`). Nothing is
estimated or extrapolated.

> **Hardware note**: Earlier revisions of this report (through commit `755f55e`)
> measured the C++17 `mcpricer` baseline on a 4-core / 8-thread Intel Core
> i5-1135G7 laptop. Because cross-machine speedup comparisons are invalid, the
> legacy C++17 `mcpricer` code was re-baselined in Phase 1 on this exact
> **AMD Ryzen 5 5600X** before retirement so every Before vs. After comparison
> below compares same-machine measurements under the same compiler (`g++ 16.2.0`).

---

## Environment

| Parameter | Value |
|---|---|
| CPU | AMD Ryzen 5 5600X 6-Core Processor (Zen 3, homogeneous `x86-64-v3`: AVX2 + FMA) |
| Physical cores | 6 (via `GetLogicalProcessorInformationEx` / `RelationProcessorCore`) |
| Logical processors | 12 (SMT enabled) |
| TSC frequency | `3.6931 GHz` (`0.27078 ns/cycle`, calibrated against `steady_clock`) |
| Compiler | `g++ 16.2.0` (MSYS2 MinGW64, Rev3) |
| Build system | CMake 4.4.3, Ninja generator, Release (`-O3 -march=x86-64-v3 -ffp-contract=off -fno-math-errno -fno-trapping-math -flto`) |
| Contract (default) | European Call, `S0 = K = 100`, `r = 0.05`, `q = 0.0`, `sigma = 0.20`, `T = 1.0` |
| Heston contract | Broadie–Kaya / Andersen Case I (`S0 = K = 100`, `r = 0.0319`, `v0 = 0.010201`, `kappa = 6.21`, `theta = 0.019`, `xi = 0.61`, `rho = -0.7`, `T = 1.0`, Feller-violating) |

## Parallel & Timing Methodology

- **Persistent `ThreadPool`**: spawns `pool_size - 1` `std::jthread` workers
  once outside the timed window; calling thread participates as worker 0.
- **Physical-core pinning**: owned workers `1..5` are pinned to distinct
  physical cores (`SetThreadAffinityMask` on Windows, `pthread_setaffinity_np`
  on Linux); worker 0 is temporarily bound to core 0 during multi-threaded
  `parallel_for` and unconditionally restored via RAII on return.
- **Chunk-keyed determinism & zero false sharing**: paths are partitioned into
  fixed `4096`-path (`2048`-pair) chunks seeded from `(seed, chunk_id)` via
  `SplitMix64` into 4-wide `Xoshiro256++ x4`. Each chunk accumulates locally in
  AVX2 registers with a first-vector-mean pilot shift, writes once to a
  64-byte-aligned slot, and reduces in chunk order via Chan's parallel merge.
- **Thermal-drift control**: all 1T vs. 6T, thread-scaling, and plain vs.
  antithetic comparisons use **round-robin** schedules across configurations
  with discarded warm-up rounds, reporting `p50`, `p95`, `p99` (and `p99.9`
  under `rdtscp` for sub-millisecond Black-Scholes kernels).

---

## 1. Before (`mcpricer` C++17) vs. After (`pricer` C++23 / AVX2)

### 1a. Black-Scholes Option Chain (`N = 16,384` calls)

| Kernel | Legacy C++17 (`Black_Scholes`) | Modern AVX2 SoA (`OptionChainSoA`) | Speedup |
|---|---:|---:|---:|
| Price only | `124.13 ns/opt` (`2.034 ms`) | `10.49 ns/opt` (`171.9 us`, `95.3 M/s`) | **11.8x** |
| Price + 5 Greeks | `548.87 ns/opt` (`8.993 ms`) | `12.39 ns/opt` (`203.0 us`, `80.7 M/s`) | **44.3x** |
| Implied Volatility (Halley) | *(not implemented)* | `77.28 ns/opt` (`1,266 us`, `12.9 M/s`) | — |

Full `OptionChainSoA` latency breakdown (`rdtscp`, 2000 samples for `N <= 1024`,
500 samples for `N = 16384`):

| Kernel | N | `ns/opt` | `M opt/s` | `p50 (us)` | `p95 (us)` | `p99 (us)` | `p99.9 (us)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| BS price (AVX2 SoA) | 64 | 10.64 | 93.94 | 0.68 | 0.75 | 0.80 | 1.05 |
| BS price+5G (AVX2 SoA) | 64 | 12.37 | 80.86 | 0.79 | 0.79 | 0.79 | 0.87 |
| BS IV Halley (AVX2 SoA) | 64 | 67.31 | 14.86 | 4.31 | 4.32 | 5.03 | 17.34 |
| BS price (AVX2 SoA) | 1,024 | 10.43 | 95.88 | 10.68 | 11.78 | 13.50 | 33.56 |
| BS price+5G (AVX2 SoA) | 1,024 | 12.13 | 82.43 | 12.42 | 13.76 | 20.77 | 43.21 |
| BS IV Halley (AVX2 SoA) | 1,024 | 71.70 | 13.95 | 73.42 | 91.68 | 101.88 | 119.71 |
| BS price (AVX2 SoA) | 16,384 | 10.49 | 95.29 | 171.94 | 217.39 | 240.01 | 249.61 |
| BS price+5G (AVX2 SoA) | 16,384 | 12.39 | 80.69 | 203.04 | 260.72 | 286.41 | 321.43 |
| BS IV Halley (AVX2 SoA) | 16,384 | 77.28 | 12.94 | 1,266.09 | 1,455.99 | 1,518.48 | 1,659.19 |

### 1b. European GBM Monte Carlo — Like-for-Like (252 steps vs. Legacy 252 steps)

Here the new engine runs with `force_multi_step = true` (`252` steps/path) to
isolate pure implementation speedups (AVX2 SIMD math, `Xoshiro256++ x4`, log-space
stepping, persistent thread pool) from the 1-step algorithmic optimization:

| Mode | Paths | Threads | Legacy `ParallelEngine` | New `pricer` (252-step) | Like-for-Like Speedup |
|---|---:|---:|---:|---:|---:|
| Plain | 100,000 | 1T | `3001.6 ms` (`30,016 ns/p`) | `45.98 ms` (`459.8 ns/p`) | **65.3x** |
| Plain | 100,000 | 6T | `632.7 ms` (`6,327 ns/p`) | `7.97 ms` (`79.7 ns/p`) | **79.4x** |
| Plain | 250,000 | 1T | `7693.5 ms` (`30,774 ns/p`) | `115.28 ms` (`461.1 ns/p`) | **66.7x** |
| Plain | 250,000 | 6T | `1495.6 ms` (`5,982 ns/p`) | `20.94 ms` (`83.7 ns/p`) | **71.4x** |
| Antithetic | 100,000 | 1T | `1858.3 ms` (`18,583 ns/p`) | `24.56 ms` (`245.6 ns/p`) | **75.7x** |
| Antithetic | 100,000 | 6T | `404.9 ms` (`4,049 ns/p`) | `4.32 ms` (`43.2 ns/p`) | **93.8x** |
| Antithetic | 250,000 | 1T | `4613.4 ms` (`18,454 ns/p`) | `61.83 ms` (`247.3 ns/p`) | **74.6x** |
| Antithetic | 250,000 | 6T | `878.1 ms` (`3,512 ns/p`) | `10.66 ms` (`42.6 ns/p`) | **82.4x** |
| 5-Greeks + Theta | 100,000 | 1T | `7227.9 ms` (`72,279 ns/p`) | `53.64 ms` (`536.4 ns/p`) | **134.8x** |
| 5-Greeks + Theta | 100,000 | 6T | `1402.6 ms` (`14,026 ns/p`) | `9.46 ms` (`94.6 ns/p`) | **148.3x** |
| 5-Greeks + Theta | 250,000 | 1T | `16957.7 ms` (`67,831 ns/p`) | `133.79 ms` (`535.2 ns/p`) | **126.7x** |
| 5-Greeks + Theta | 250,000 | 6T | `2937.6 ms` (`11,750 ns/p`) | `23.54 ms` (`94.2 ns/p`) | **124.8x** |

*(Legacy single-threaded `MonteCarloEngine` without `ParallelEngine` ran 100k × 252
in `1786.7 ms`, so the new 1T 252-step engine is **38.9x** faster than legacy
serial and **65.3x** faster than legacy 1T `ParallelEngine`.)*

### 1c. European GBM Monte Carlo — Algorithmic (Exact 1-Step)

For path-independent European options under GBM, Branch A samples terminal spot
$S_T$ directly in a single step:

| Workload | Paths | Threads | `p50 (ms)` | `p95 (ms)` | `p99 (ms)` | `ns/path` | `M paths/s` | Speedup vs. Legacy 252-step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GBM 1-step Plain | 100,000 | 1T | 0.410 | 0.483 | 0.535 | 4.10 | 244.0 | ~7,320x |
| GBM 1-step Plain | 100,000 | 6T | 0.093 | 1.688 | 2.154 | 0.93 | 1,073.0 | ~6,800x |
| GBM 1-step Antithetic | 100,000 | 6T | 0.052 | 0.099 | 0.165 | 0.52 | 1,938.0 | ~7,790x |
| GBM 1-step Plain | 1,000,000 | 1T | 4.006 | 4.170 | 4.214 | 4.01 | 249.7 | — |
| GBM 1-step Plain | 1,000,000 | 6T | 0.761 | 0.889 | 0.891 | 0.76 | 1,314.2 | — |
| GBM 1-step Antithetic | 1,000,000 | 6T | 0.525 | 2.590 | 2.762 | 0.53 | 1,903.3 | — |
| GBM 1-step Plain | 10,000,000 | 1T | 40.486 | 41.108 | 41.211 | 4.05 | 247.0 | — |
| GBM 1-step Plain | 10,000,000 | 6T | 6.751 | 7.309 | 7.337 | 0.68 | **1,481.4** | — |
| GBM 1-step Antithetic | 10,000,000 | 6T | 4.515 | 4.920 | 4.978 | 0.45 | **2,214.6** | — |
| GBM 1-step 5-Greeks | 100,000 | 6T | 0.197 | 0.234 | 0.270 | 1.98 | 506.3 | ~7,120x |
| GBM 1-step 5-Greeks | 250,000 | 6T | 0.498 | 0.544 | 0.551 | 1.99 | 501.6 | ~5,900x |
| GBM 1-step 5-Greeks | 1,000,000 | 6T | 1.590 | 2.692 | 2.844 | 1.59 | 628.8 | — |

*(Note: on sub-millisecond 6T runs like 100k 1-step, occasional OS timer-tick
preemptions explain why `p99` reaches ~1–2 ms while `p50` is `0.05–0.09 ms`.)*

---

## 2. Multi-Core Scaling & Exact Bitwise Reproducibility

Measured at `N = 1,000,000` paths × `252` steps (round-robin across `1, 2, 4, 6`
threads on a single pinned `ThreadPool`, 1 warm-up + 5 measured rounds):

### GBM 252-step Plain (`N = 1,000,000`)

| Threads | `p50 (ms)` | Speedup `T1/Tp` | Efficiency | Spread (min–max / p50) | Price | Bitwise equal vs. 1T |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 468.130 | 1.00x | 1.000 | 2.40% | `10.467102597230` | YES |
| 2 | 230.210 | 2.03x | 1.017 | 10.64% | `10.467102597230` | YES |
| 4 | 113.231 | 4.13x | 1.034 | 6.75% | `10.467102597230` | YES |
| 6 | 76.103 | **6.15x** | 1.025 | 6.09% | `10.467102597230` | YES |

### Heston Full Truncation Euler 252-step Plain (`N = 1,000,000`, Broadie–Kaya Case I)

| Threads | `p50 (ms)` | Speedup `T1/Tp` | Efficiency | Spread (min–max / p50) | Price | Bitwise equal vs. 1T |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 977.737 | 1.00x | 1.000 | 2.09% | `6.810687825788` | YES |
| 2 | 470.784 | 2.08x | 1.038 | 2.23% | `6.810687825788` | YES |
| 4 | 238.101 | 4.11x | 1.027 | 2.92% | `6.810687825788` | YES |
| 6 | 160.162 | **6.10x** | 1.017 | 1.88% | `6.810687825788` | YES |

**Notes on the scaling numbers**:
- Both workloads clear the `>= 4.8x` 6-core scaling target by a wide margin.
- The ~1–3% efficiency reading above `1.00` is not genuine superlinear cache
  scaling; it is measurement/scheduling noise (2–6% spread) combined with the
  fact that `1T` executes inline without `CallerAffinityGuard` pinning the
  calling thread while `2T/4T/6T` pin workers to distinct physical cores. In
  Section 1b's separate 1T/6T run (`N = 1,000,000` GBM 252-step), 1T ran in
  `460.53 ms` and 6T in `79.85 ms` (`5.77x`, `96.1%` efficiency).
- Both `price` and `std_error` are **bitwise identical** across `1, 2, 4, 6`
  threads (0 ULP difference), eliminating the 4–32 ULP floating-point
  reduction-order drift noted in the earlier C++17 report.

---

## 3. Path-Dependent Exotics & Heston Throughput

### Exotics under GBM (`100,000` paths × `252` steps, 1T vs. 6T)

| Exotic | Thr | Price | SE | `p50 (ms)` | `p99 (ms)` | `ns/path` | `M paths/s` | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Geometric Asian Call | 1T | 5.58553 | 0.02451 | 47.271 | 48.326 | 472.7 | 2.115 | 1.00x |
| Geometric Asian Call | 6T | 5.58553 | 0.02451 | 7.982 | 9.457 | 79.8 | 12.528 | 5.92x |
| Arithmetic Asian Call | 1T | 5.80247 | 0.02539 | 79.238 | 81.141 | 792.4 | 1.262 | 1.00x |
| Arithmetic Asian Call | 6T | 5.80247 | 0.02539 | 13.805 | 13.991 | 138.0 | 7.244 | 5.74x |
| Barrier UpOut Call (`B=130`) | 1T | 3.55610 | 0.02009 | 48.223 | 49.824 | 482.2 | 2.074 | 1.00x |
| Barrier UpOut Call (`B=130`) | 6T | 3.55610 | 0.02009 | 8.240 | 9.738 | 82.4 | 12.136 | 5.85x |
| Lookback Fixed Call | 1T | 18.38137 | 0.04880 | 48.226 | 50.365 | 482.3 | 2.074 | 1.00x |
| Lookback Fixed Call | 6T | 18.38137 | 0.04880 | 8.112 | 9.166 | 81.1 | 12.327 | 5.94x |
| Lookback Floating Call | 1T | 16.63751 | 0.04622 | 49.041 | 49.853 | 490.4 | 2.039 | 1.00x |
| Lookback Floating Call | 6T | 16.63751 | 0.04622 | 8.014 | 8.284 | 80.1 | 12.479 | 6.12x |

Note how Geometric Asian, Barrier, and both Lookbacks run at essentially
identical speed (~8.0 ms on 6T, `12.1–12.5 M paths/s`) because all three track
state 100% in log-space and call `exp_avx2` only once at expiry; Arithmetic
Asian calls `exp_avx2` at every monitoring step (`needs_spot_per_step = true`)
and takes `13.8 ms` (`7.24 M paths/s`).

### Heston Full Truncation Euler (`252` steps, Broadie–Kaya Case I, exact COS price `6.8061`)

| Mode | Paths | Thr | Price | SE | `p50 (ms)` | `p99 (ms)` | `ns/path` | `M paths/s` | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heston Plain | 100,000 | 1T | 6.79334 | 0.02341 | 97.585 | 99.749 | 975.9 | 1.025 | 1.00x |
| Heston Plain | 100,000 | 6T | 6.79334 | 0.02341 | 16.738 | 17.590 | 167.4 | 5.974 | 5.83x |
| Heston Antithetic | 100,000 | 1T | 6.76877 | 0.01738 | 53.407 | 55.178 | 534.1 | 1.872 | 1.00x |
| Heston Antithetic | 100,000 | 6T | 6.76877 | 0.01738 | 9.215 | 9.561 | 92.2 | 10.852 | 5.80x |
| Heston Plain | 1,000,000 | 1T | 6.81069 | 0.00743 | 993.889 | 998.676 | 993.9 | 1.006 | 1.00x |
| Heston Plain | 1,000,000 | 6T | 6.81069 | 0.00743 | 161.683 | 166.696 | 161.7 | 6.185 | 6.15x |
| Heston Antithetic | 1,000,000 | 1T | 6.80668 | 0.00551 | 556.180 | 558.745 | 556.2 | 1.798 | 1.00x |
| Heston Antithetic | 1,000,000 | 6T | 6.80668 | 0.00551 | 88.676 | 92.593 | 88.7 | **11.277** | 6.27x |

---

## 4. Equal-Work Antithetic Strike Sweep

`250,000` antithetic pairs (`= 500,000` simulated paths) vs. `500,000` plain
paths, 6 threads, round-robin (2 warm-up + 7 measured rounds):

### Exact 1-Step European GBM

| Strike | SE plain | SE anti | `var_ratio` | Implied `rho` | `time_ratio` (`t_anti/t_plain`) | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.028563 | 0.005807 | 24.19 | -0.959 | 0.64 | **37.63x** |
| 80 | 0.027129 | 0.009167 | 8.76 | -0.886 | 0.63 | **13.98x** |
| 100 | 0.020853 | 0.014685 | 2.02 | -0.504 | 0.60 | **3.35x** |
| 120 | 0.012305 | 0.011349 | 1.18 | -0.149 | 0.63 | **1.85x** |
| 140 | 0.006036 | 0.005873 | 1.06 | -0.053 | 0.68 | **1.56x** |

### 252-Step European GBM

| Strike | SE plain | SE anti | `var_ratio` | Implied `rho` | `time_ratio` (`t_anti/t_plain`) | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.028573 | 0.005844 | 23.91 | -0.958 | 0.54 | **44.04x** |
| 80 | 0.027133 | 0.009219 | 8.66 | -0.885 | 0.54 | **16.02x** |
| 100 | 0.020865 | 0.014736 | 2.00 | -0.501 | 0.53 | **3.77x** |
| 120 | 0.012326 | 0.011397 | 1.17 | -0.145 | 0.54 | **2.18x** |
| 140 | 0.006062 | 0.005928 | 1.05 | -0.044 | 0.53 | **1.96x** |

---

## 5. RNG Stream Independence & `1/sqrt(N)` Convergence

24 independent replications per `N` (distinct seeds, 1-step exact European GBM,
closed-form reference **10.4505835722**):

| N | Mean MC price | Bias vs. BS | Observed SE (24 reps) | Engine SE (single run) |
|---:|---:|---:|---:|---:|
| 1,000 | 10.5331208 | +0.0825372 | 0.4213513 | 0.4519622 |
| 10,000 | 10.3948431 | -0.0557405 | 0.1376817 | 0.1489339 |
| 100,000 | 10.4433309 | -0.0072526 | 0.0320505 | 0.0466049 |
| 1,000,000 | 10.4546710 | +0.0040874 | 0.0135344 | 0.0147535 |

- Least-squares fit of `log(SE)` on `log(N)`: **slope = -0.5113** (theoretical
  `-0.5000`), **`R^2 = 0.991398`**.
- Verdict: empirical cross-replication standard error tracks the engine's
  internal error bar and decays at `N^-0.5`, with no evidence of correlated
  streams across seeds or chunks.
