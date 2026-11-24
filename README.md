# Monte Carlo Option Pricer

[![CI](https://github.com/mmuurrrraayy/Monte-Carlo-Option-Pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/mmuurrrraayy/Monte-Carlo-Option-Pricer/actions/workflows/ci.yml)

A header-only C++23 / AVX2+FMA (`x86-64-v3`) Monte Carlo and Black-Scholes
option-pricing library with static policy-based polymorphism, path-dependent
exotics, Heston stochastic volatility, forward-mode automatic differentiation,
a batched SoA Black-Scholes / Implied Volatility engine, and a persistent
physical-core-pinned thread pool with exact bitwise thread-count invariance.

Every number in this README is measured on this repository by `bench_pricer` on
an **AMD Ryzen 5 5600X** (6 physical cores / 12 threads, `g++ 16.2.0`, CMake
Release). Earlier revisions of this README reported measurements from a 4-core
Intel Core i5-1135G7 laptop; the C++17 `mcpricer` baseline (`755f55e`) was
re-measured on this same Ryzen 5 5600X before retirement so every Before vs.
After comparison below is same-machine. See [`bench/REPORT.md`](bench/REPORT.md)
for the full methodology and raw tables.

## Quick start

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/price --s0 100 --k 100 --r 0.05 --sigma 0.2 --T 1 --paths 1000000 --greeks
```

```text
call  gbm  european  S0 100  K 100  r 0.05  q 0  sigma 0.2  T 1
paths 1000000  steps/path 1 (exact)  threads 6  seed 42

price      10.45137900
std_error  0.01472664
95% CI     [10.42251532, 10.48024268]

black_scholes 10.45058357   diff +0.00079543   (+0.05 se)

greek           value     std_error  black_scholes    err_se
delta      0.63708241    0.00057634     0.63683065      0.44
gamma      0.01876157    0.00003789     0.01876202     -0.01
vega      37.52314142    0.07577584    37.52403469     -0.01
theta     -6.41515852    0.00887151    -6.41402755     -0.13
rho       53.25686228    0.04721926    53.23248155      0.52
```

Other CLI modes:

```bash
# Heston stochastic volatility (Broadie-Kaya Case I) with antithetic variates & sensitivities
./build/price --model heston --s0 100 --k 100 --r 0.0319 --v0 0.010201 --kappa 6.21 \
              --heston-theta 0.019 --xi 0.61 --rho -0.7 --T 1 --paths 100000 --steps 252 \
              --antithetic --greeks

# Path-dependent exotics (Geometric/Arithmetic Asian, Barrier, Fixed/Floating Lookback)
./build/price --payoff barrier --barrier 130 --paths 100000 --steps 252 --greeks
./build/price --payoff asian-arith --paths 100000 --steps 252 --antithetic
./build/price --payoff lookback --lookback-style floating --paths 100000 --steps 252

# Batched AVX2 SoA Black-Scholes chain + Implied Volatility solver
./build/price --chain 1024 --greeks --iv
./build/price --iv 10.450583572185565
```

`price --help` lists all options (including continuous dividend yield `--q`,
`--force-multi-step`, `--no-reference`, `--seed`, and `--threads`).

## Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Entry & Validation Boundary (zero-exception, zero-RTTI)                                │
│ ┌───────────────────────────────┐     ┌──────────────────────────────────────────────┐ │
│ │ apps/price.cpp + cpuid_guard  │────>│ core/types.hpp                               │ │
│ │ runtime AVX2 + FMA check      │     │ MarketParams / HestonParams -> std::expected │ │
│ └───────────────────────────────┘     └───────────────────────┬──────────────────────┘ │
└───────────────────────────────────────────────────────────────┼────────────────────────┘
                            ┌───────────────────────────────────┴─────────┐
                            v                                             v
┌───────────────────────────────────────────────────────┐  ┌─────────────────────────────┐
│ Monte Carlo Pipeline (engines/monte_carlo.hpp)        │  │ Batched Black-Scholes & IV  │
│ ┌─ Compile-Time Policies (core/concepts.hpp) ───────┐ │  │ (engines/black_scholes.hpp) │
│ │ Process  : GBMProcess | HestonProcess             │ │  │                             │
│ │ Payoff   : European | Asian | Barrier | Lookback  │ │  │                             │
│ │ Sampling : Standard | Antithetic                  │ │  │ ┌─────────────────────────┐ │
│ └─────────────────────────┬─────────────────────────┘ │  │ │ OptionChainSoA          │ │
│                           v                           │  │ │ 64B-aligned,            │ │
│ ┌───────────────────────────────────────────────────┐ │  │ │ 4-wide padded layout    │ │
│ │ MonteCarloEngine                                  │ │  │ └────────────┬────────────┘ │
│ │ concept dispatch & requires Greek guards          │ │  │              │              │
│ └────────────┬──────────────────────────┬───────────┘ │  │              │              │
│              v                          v             │  │              v              │
│ ┌─────────────────────────┐ ┌───────────────────────┐ │  │ ┌─────────────────────────┐ │
│ │ engines/thread_pool.hpp │ │ core/dual.hpp         │ │  │ │ AVX2 Black-Scholes      │ │
│ │ Pinned std::jthreads    │ │ Dual / Jet over Vec4d │ │  │ │ 5-Greek kernels &       │ │
│ │ 4096-path chunks        │ │ Pathwise, LR, CRN     │ │  │ │ Safeguarded-Halley IV   │ │
│ └────────────┬────────────┘ └───────────┬───────────┘ │  │ │ (OTM Manaster-Koehler)  │ │
│              v                          v             │  │ └─────────────────────────┘ │
│ ┌───────────────────────────────────────────────────┐ │  │                             │
│ │ core/stats.hpp                                    │ │  │                             │
│ │ Pilot-shifted ChunkAccum -> ordered merge_chan    │ │  │                             │
│ └───────────────────────────────────────────────────┘ │  │                             │
└───────────────────────────┬───────────────────────────┘  └──────────────┬──────────────┘
                            ┆ uses xoshiro_avx2,                          ┆ uses avx2_math
                            ┆ avx2_math, Vec4d                            ┆ & Vec4d
                            v                                             v
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ AVX2 + FMA Compute Core (include/pricer/simd/)                                         │
│ ┌──────────────────────────┐ ┌──────────────────────────┐ ┌──────────────────────────┐ │
│ │ xoshiro_avx2.hpp         │ │ avx2_math.hpp            │ │ vec4d.hpp                │ │
│ │ Xoshiro256++ x4          │ │ exp, log, sincos         │ │ Vec4d (4x double)        │ │
│ │ branchless Box-Muller    │ │ norm_cdf / norm_pdf      │ │ AVX2 + FMA value type    │ │
│ └──────────────────────────┘ └──────────────────────────┘ └──────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Header-only C++23 core (`include/pricer/`)**: zero-exception, zero-RTTI
  contract verified by a dedicated `test_no_exceptions` build
  (`-fno-exceptions -fno-rtti` / `/EHs-c- /GR-`). Inputs are validated once at
  the API boundary via `std::expected<T, ValidationError>` (including negative
  `r` and dividend yield `q`).
- **Static policy-based polymorphism**: `MonteCarloEngine<Process, Payoff, Sampling>`
  dispatches at compile time via C++23 concepts (`ProcessPolicy`, `PayoffPolicy`,
  `SamplingPolicy`). Unsupported Greek combinations (e.g. naive CRN Gamma on
  discontinuous Barriers or GBM `Dual` Vega on Heston) are rejected at compile
  time via `requires`.
- **Hand-tuned AVX2 + FMA (`x86-64-v3`)**: `Vec4d` value type, `< 1e-15`
  vectorized `exp_avx2`, `log_avx2`, `sincos_avx2`, and Cody-rational
  `norm_cdf_avx2` (preserving left-tail relative accuracy to `< 1e-13`), plus
  4-wide transposed `Xoshiro256++ x4` and branchless AVX2 Box-Muller. A
  separate non-AVX2 translation unit (`apps/cpuid_guard.cpp`) verifies AVX2+FMA
  at CLI startup.
- **Models & payoffs**:
  - `GBMProcess` (exact 1-step terminal sampling for European, or multi-step log-space)
  - `HestonProcess` (branchless Full Truncation Euler)
  - `EuropeanPayoff`, `GeometricAsianPayoff`, `ArithmeticAsianPayoff`,
    `BarrierPayoff` (`UpOut`/`DownOut`/`UpIn`/`DownIn` in log-space),
    `LookbackPayoff` (`FixedStrike`/`FloatingStrike` in log-space)
- **Forward-mode AD & Greeks**: `Dual<T>` / `Jet<T, 3>` over `double` and `Vec4d`
  for single-pass pathwise `Delta`/`Vega`/`Rho`, mixed pathwise-LR `Gamma`
  (European & Geometric Asian), exact homogeneity (`Gamma = 0`, Floating
  Lookback), CRN central-difference `Theta`, and safe CRN sensitivities for
  Barriers and Heston (`d_v0`, `d_long_run_var`, `rho`, `theta`).
- **Batched Black-Scholes & AVX2 IV**: 64-byte-aligned `OptionChainSoA` with
  closed-form 5-Greek AVX2 kernels and an OTM-reduced Manaster–Koehler
  safeguarded-Halley AVX2 IV solver converging in sigma-space to `1e-12`.
- **Persistent `ThreadPool` & exact cross-thread determinism**: pre-spawned
  `std::jthread` workers pinned to distinct physical cores (without permanently
  altering the caller's affinity), fixed `4096`-path chunk-keyed RNG seeding,
  pilot-shifted AVX2 chunk accumulators, cache-line-isolated chunk slots, and
  deterministic chunk-ordered Chan variance merging. The pool accepts **one
  caller at a time**: engines may share a `ThreadPool*`, but their `price*()`
  calls must not overlap across threads (use one pool per concurrent caller).

## Evidence (Ryzen 5 5600X, `g++ 16.2.0` Release)

Unless stated otherwise: European call, `S0 = K = 100`, `r = 0.05`, `q = 0.0`,
`sigma = 0.20`, `T = 1.0`, closed-form price **10.4505835722**.

### Before (`mcpricer` C++17 @ `755f55e`) vs. After (`pricer` C++23 / AVX2)

Both columns were measured on the **same AMD Ryzen 5 5600X** (`g++ 16.2.0`,
CMake Release). European speedups are split explicitly into **like-for-like**
(252-step new vs. 252-step legacy) and **algorithmic** (1-step exact European
vs. 252-step legacy):

| Workload | Legacy C++17 (`mcpricer`) | Modern C++23/AVX2 (`pricer`) | Speedup |
|---|---:|---:|---:|
| Black-Scholes price-only (`N = 16,384`) | `124.13 ns/opt` | `10.49 ns/opt` (`95.3 M/s`) | **11.8x** |
| Black-Scholes price + 5 Greeks (`N = 16,384`) | `548.87 ns/opt` | `12.39 ns/opt` (`80.7 M/s`) | **44.3x** |
| Black-Scholes IV Halley AVX2 (`N = 16,384`) | *(not implemented)* | `77.28 ns/opt` (`12.9 M/s`) | — |
| **Like-for-like 252-step** GBM Plain (100k, 1T) | `3001.6 ms` (`0.033 M/s`) | `45.98 ms` (`2.17 M/s`) | **65.3x** |
| **Like-for-like 252-step** GBM Plain (100k, 6T) | `632.7 ms` (`0.158 M/s`) | `7.97 ms` (`12.55 M/s`) | **79.4x** |
| **Like-for-like 252-step** GBM Plain (250k, 6T) | `1495.6 ms` (`0.167 M/s`) | `20.94 ms` (`11.94 M/s`) | **71.4x** |
| **Like-for-like 252-step** GBM Antithetic (250k, 6T) | `878.1 ms` (`0.285 M/s`) | `10.66 ms` (`23.45 M/s`) | **82.4x** |
| **Like-for-like 252-step** GBM 5-Greeks+Theta (250k, 6T) | `2937.6 ms` (`0.085 M/s`) | `23.54 ms` (`10.62 M/s`) | **124.8x** |
| **Algorithmic 1-step** GBM Plain (100k, 6T) | `632.7 ms` (252-step) | `0.093 ms` (`1,073 M/s`) | **~6,800x** |
| **Algorithmic 1-step** GBM Plain (10M, 6T) | — | `6.75 ms` (`1,481 M/s`) | — |
| **Algorithmic 1-step** GBM Antithetic (10M, 6T) | — | `4.52 ms` (`2,215 M/s`) | — |
| **Algorithmic 1-step** GBM 5-Greeks+Theta (250k, 6T) | `2937.6 ms` (252-step) | `0.498 ms` (`501.6 M/s`) | **~5,900x** |

### Multi-core scaling & exact bitwise reproducibility

Round-robin sweep at `N = 1,000,000` paths × `252` steps (1 warm-up + 5 rounds,
pinned 6-core `ThreadPool`):

| Model (252 steps, 1M paths) | 1T p50 (ms) | 2T p50 (ms) | 4T p50 (ms) | 6T p50 (ms) | 6T Speedup | Bitwise identical (1/2/4/6T) |
|---|---:|---:|---:|---:|---:|---|
| GBM Plain | `468.13` | `230.21` (`2.03x`) | `113.23` (`4.13x`) | `76.10` | **6.15x** | **YES** (`10.467102597230`) |
| Heston FTE Plain (BK Case I) | `977.74` | `470.78` (`2.08x`) | `238.10` (`4.11x`) | `160.16` | **6.10x** | **YES** (`6.810687825788`) |

The slight >6.0x ratio (~1–2.5% above linear) reflects Windows core-scheduling
and Zen 3 boost variance on the unpinned 1T inline path (run-to-run spread is
`1.9–6.1%`); in `bench_pricer`'s paired 1T/6T table, GBM 1M × 252 scales at
`5.77x` (`460.5 ms -> 79.9 ms`). Because chunks are fixed at 4096 paths,
seeded deterministically by `(seed, chunk_id)`, and reduced via Chan's merge in
chunk order, both `price` and `std_error` are **bitwise identical** across 1, 2,
4, and 6 threads (zero ULPs difference, resolving the 32-ULP reduction-order
drift of the C++17 baseline).

### Exotics & Heston throughput (`100,000` paths × `252` steps)

| Instrument | 1T p50 (ms) | 6T p50 (ms) | 6T Throughput | 6T Speedup |
|---|---:|---:|---:|---:|
| Geometric Asian Call (GBM) | `47.27` | `7.98` | `12.53 M paths/s` | `5.92x` |
| Arithmetic Asian Call (GBM) | `79.24` | `13.81` | `7.24 M paths/s` | `5.74x` |
| Barrier Up-and-Out Call (`B = 130`, GBM) | `48.22` | `8.24` | `12.14 M paths/s` | `5.85x` |
| Lookback Fixed-Strike Call (GBM) | `48.23` | `8.11` | `12.33 M paths/s` | `5.94x` |
| Lookback Floating-Strike Call (GBM) | `49.04` | `8.01` | `12.48 M paths/s` | `6.12x` |
| Heston FTE Plain (BK Case I) | `97.59` | `16.74` | `5.97 M paths/s` | `5.83x` |
| Heston FTE Antithetic (BK Case I) | `53.41` | `9.22` | `10.85 M paths/s` | `5.80x` |

### Equal-work antithetic variates (`250,000` pairs = `500,000` paths, 6T)

Comparing `250,000` antithetic pairs against `500,000` plain paths (equal
simulated path count, round-robin median of 7):

| Strike | Moneyness | `var_ratio` (1-step) | Implied `rho` | Time ratio (1-step / 252-step) | Efficiency (1-step / 252-step) |
|---|---|---:|---:|---:|---:|
| 60 | deep ITM | `24.19` | `-0.959` | `0.64` / `0.54` | **37.6x** / **44.0x** |
| 80 | ITM | `8.76` | `-0.886` | `0.63` / `0.54` | **14.0x** / **16.0x** |
| 100 | ATM | `2.02` | `-0.504` | `0.60` / `0.53` | **3.35x** / **3.77x** |
| 120 | OTM | `1.18` | `-0.149` | `0.63` / `0.54` | **1.85x** / **2.18x** |
| 140 | deep OTM | `1.06` | `-0.053` | `0.68` / `0.53` | **1.56x** / **1.96x** |

Since `Var((X+Y)/2) = sigma^2 (1 + rho)/2`, `var_ratio = 1 / (1 + rho)`. Deep
ITM payoffs are nearly linear in the driving Gaussian so pairing cuts variance
~24x; deep OTM payoffs are mostly zero so variance reduction collapses to
`1.06x`, while half the RNG work is still saved (`time_ratio ~ 0.53-0.68`).

### Convergence & `N^-0.5` empirical standard error

24 independent replications per `N` (1-step exact European GBM):

| N | Mean MC price | Bias vs. BS | Observed SE (24 reps) | Engine SE |
|---:|---:|---:|---:|---:|
| 1,000 | `10.5331208` | `+0.0825372` | `0.4213513` | `0.4519622` |
| 10,000 | `10.3948431` | `-0.0557405` | `0.1376817` | `0.1489339` |
| 100,000 | `10.4433309` | `-0.0072526` | `0.0320505` | `0.0466049` |
| 1,000,000 | `10.4546710` | `+0.0040874` | `0.0135344` | `0.0147535` |

OLS fit of `log(SE)` on `log(N)`: slope **-0.5113** (theory `-0.5000`),
`R^2 = 0.9914`.

### Forced inlining (`PRICER_FORCE_INLINE`) on vs. off

`PRICER_FORCE_INLINE` expands to `__forceinline` on MSVC and
`[[gnu::always_inline]] inline` on GCC/Clang. Unlike plain `inline`, which
compilers treat only as a hint, these are binding: GCC must inline the call
wherever it can. To measure what the attribute buys, every one of its 226 uses
was removed (free functions that aren't templates kept a plain `inline` to
satisfy the ODR), and the full `bench_pricer` suite was run once with and once
without it on the same Ryzen 5 5600X (`g++ 16.2.0`, Release):

| Workload | With `PRICER_FORCE_INLINE` | Without (plain `inline`) | Slowdown without |
|---|---:|---:|---:|
| Black-Scholes price-only (`N = 16,384`) | `10.46 ns/opt` | `18.10 ns/opt` | **1.73x** |
| Black-Scholes price + 5 Greeks (`N = 16,384`) | `12.21 ns/opt` | `17.84 ns/opt` | **1.46x** |
| Black-Scholes IV Halley AVX2 (`N = 16,384`) | `78.37 ns/opt` | `114.48 ns/opt` | **1.46x** |
| GBM 1-step Plain (10M, 1T) | `40.90 ms` | `63.93 ms` | **1.56x** |
| GBM 1-step 5-Greeks (1M, 1T) | `9.35 ms` | `14.61 ms` | **1.56x** |
| GBM 252-step Plain (1M, 1T) | `495.43 ms` | `719.42 ms` | **1.45x** |
| GBM 252-step 5-Greeks (250k, 1T) | `136.83 ms` | `235.14 ms` | **1.72x** |
| GBM 252-step 5-Greeks (250k, 6T) | `24.39 ms` | `45.52 ms` | **1.87x** |
| Heston FTE Plain (1M × 252, 1T) | `1010.49 ms` | `1705.49 ms` | **1.69x** |
| Heston FTE Antithetic (1M × 252, 1T) | `556.03 ms` | `558.57 ms` | `1.00x` |
| Arithmetic Asian Call (100k × 252, 6T) | `15.34 ms` | `24.92 ms` | **1.62x** |
| Lookback Fixed-Strike Call (100k × 252, 6T) | `8.21 ms` | `12.32 ms` | **1.50x** |

Prices and standard errors were bitwise identical in both runs, so the attribute
changes speed only, never results. The largest losses are in the `Dual` / `Jet`
forward-mode AD paths and the `exp_avx2` / `log_avx2` / `norm_cdf_avx2` kernels:
these are big enough that GCC's inliner declines them once it is no longer
forced to, and each out-of-line call then likely costs `Vec4d` values a
round-trip through memory instead of staying in `ymm` registers. Small `Vec4d` operators are still
inlined either way. The attribute is therefore kept on every hot-path SIMD and
AD function.

Each configuration was measured in a single run. The "with" column was built
from the previous commit, so it also differs from the "without" build in
unrelated uncommitted edits; the consistency of the slowdown across sections
nonetheless points at inlining as the dominant cause. Full raw output for both
runs is in [`bench/inline_comparison.txt`](bench/inline_comparison.txt).

## Repository layout

```text
include/pricer/
├── core/      types.hpp, concepts.hpp, dual.hpp, stats.hpp
├── simd/      vec4d.hpp, avx2_math.hpp, xoshiro_avx2.hpp
├── models/    gbm.hpp, heston.hpp
├── payoffs/   european.hpp, asian.hpp, barrier.hpp, lookback.hpp
└── engines/   black_scholes.hpp, monte_carlo.hpp, thread_pool.hpp
apps/          cpuid_guard.cpp, cpuid_guard.hpp, price.cpp
tests/         check.h, test_pricer.cpp, test_no_exceptions.cpp
bench/         bench_pricer.cpp, REPORT.md, inline_comparison.txt
```

## Build, test & CI

Requires CMake >= 3.25 and a C++23 compiler on `x86-64-v3` (AVX2 + FMA):
GCC 14+, Clang 19+, or MSVC 2022+.

```bash
# Release build & full test suite
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure   # pricer_tests, test_no_exceptions, price_smoke

# Debug + ASan/UBSan (GCC/Clang)
cmake -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DPRICER_ENABLE_SANITIZERS=ON -DPRICER_FAST_TESTS=ON
cmake --build build-asan && ctest --test-dir build-asan --output-on-failure

# Run benchmark suite (full or targeted section)
./build/bench_pricer
./build/bench_pricer --section scaling
```

Optional CMake flags: `-DPRICER_USE_CXX26=ON`, `-DPRICER_MARCH_NATIVE=ON`,
`-DPRICER_USE_GOOGLE_BENCHMARK=ON`.

CI (`.github/workflows/ci.yml`) runs every push and PR across:
- `ubuntu-latest` (`g++-14` Release)
- `ubuntu-latest` (`clang++-19` Release)
- `ubuntu-latest` (`g++-14` Debug + ASan/UBSan with `PRICER_FAST_TESTS=ON`)
- `windows-latest` (MSVC Release)

### Testing methodology

Statistical checks in `tests/test_pricer.cpp` use standard-error-normalized
`within_se` assertions (`|mc - ref| <= 4 * se`) with explicit `NaN` rejection
(`tests/check.h`). Analytic/numerical verification covers:
- AVX2 math kernels (`exp`, `log`, `sincos`, `norm_cdf`) across 100k points
  against `std::cmath` (`< 1e-15` rel/abs; left-tail `norm_cdf < 1e-13` rel)
- `Xoshiro256++ x4` bit-for-bit scalar parity, `jump()`/`long_jump()` vectors,
  4-sigma Box-Muller moments, 20-bin Pearson chi-squared, and extreme tails
- Black-Scholes put-call/Delta/Gamma/Vega parity (`q > 0`, `r < 0`), `Jet<Vec4d, 3>`
  AD oracle, and AVX2 IV round-trip across 16,384 contracts
- Exact discrete Kemna–Vorst Geometric Asian, Jensen inequalities, Reiner–Rubinstein
  + Broadie–Glasserman–Kou continuity-shifted Barriers/Lookbacks with multi-step
  refinement (`m = 64, 252, 1000`), and CRN In-Out parity (`<= 1.8e-15`)
- Self-contained Fang–Oosterlee COS Heston oracle (`Broadie-Kaya Case I` &
  Feller-satisfying benchmarks) + Heston sensitivities (including `v0 = 0`)
- Persistent `ThreadPool` caller-affinity preservation and exact bitwise
  thread-count invariance across `1, 2, 4, 6` threads

## Future extensions

- Sobol / Halton quasi-Monte Carlo sequences via `SamplingPolicy`
- Control variates (e.g. Geometric Asian control for Arithmetic Asian, or BS control for Heston)
- American / Bermudan exercise via Longstaff–Schwartz regression
