# Monte Carlo Option Pricer

[![CI](https://github.com/mmuurrrraayy/Monte-Carlo-Option-Pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/mmuurrrraayy/Monte-Carlo-Option-Pricer/actions/workflows/ci.yml)

A C++17 Monte Carlo engine for pricing European options under risk-neutral
geometric Brownian motion, with full Greeks, antithetic variance reduction,
multithreaded execution, and validation against closed-form Black-Scholes.

Every number in this README is measured on this repository by the harness in
`bench/`, at the parameters stated. Nothing is quoted from theory or
extrapolated. See `bench/REPORT.md` for the full methodology.

## Quick start

```bash
cmake -B build && cmake --build build
./build/price --s0 100 --k 100 --r 0.05 --sigma 0.2 --T 1 --paths 1000000 --greeks
```

```
call  S0 100  K 100  r 0.05  sigma 0.2  T 1
paths 1000000  steps/path 252  threads 8  seed 42

price      10.43991191
std_error  0.01472391
95% CI     [10.41105305, 10.46877078]

black_scholes 10.45058357   diff -0.01067166   (-0.72 se)

greek           value     std_error  black_scholes    err_se
delta      0.63667096    0.00057633     0.63683065     -0.28
gamma      0.01884335    0.00013873     0.01876202      0.59
vega      37.47267562    0.07576323    37.52403469     -0.68
theta     -6.40864318    0.00886991    -6.41402755      0.61
rho       53.22718392    0.04722283    53.23248155     -0.11
```

`--antithetic` switches to the paired estimator, `--threads` sets the worker
count, `--no-reference` drops the closed-form column. `price --help` lists
everything.

## What it does

- European call and put pricing by Monte Carlo simulation of price paths (252 steps per year), under the risk-neutral drift.
- A closed-form Black-Scholes engine used purely as a reference to validate price and Greeks, it never prices anything itself.
- Confidence intervals from the Bessel-corrected sample variance of the discounted payoffs, computed in the same pass as the mean rather than from a second, independently drawn batch.
- Full Greeks — delta, vega and rho by pathwise differentiation, gamma by the likelihood-ratio (Broadie-Glasserman) estimator, theta by a common-random-numbers central difference — each with its own standard error.
- Antithetic variates, measured against the plain estimator at equal work (below).
- A multithreaded engine using a counter-based RNG (splitmix64 mixer feeding Box-Muller) keyed by each path's global index, so the draws a path receives don't depend on thread count or partitioning. No shared RNG state, no lock. Per-thread accumulators are padded to a cache line to avoid false sharing.

## Evidence

Unless stated otherwise: European call, S0 = K = 100, r = 0.05, sigma = 0.20,
T = 1.0, closed-form price **10.4505835722**.

### Convergence to closed form

Mean over 24 independent replications per N (`bench converge 24`):

| N | Mean MC price | Bias vs closed form |
|---|---|---|
| 1,000 | 10.2601848 | -0.1903988 |
| 10,000 | 10.4362023 | -0.0143813 |
| 100,000 | 10.4578672 | +0.0072836 |
| 1,000,000 | 10.4516914 | +0.0011078 |

Bias shrinks ~170x from N=1,000 to N=1,000,000 and changes sign along the way —
consistent with sampling noise around the true value, not a systematic offset.

### Standard error scales as N^-0.5

Measured empirically as the sample standard deviation of price across 24
replications per N, not read off the engine's own error bar:

- Least-squares fit of log(SE) on log(N): fitted exponent **-0.5567**, R² = 0.9970, 1-sigma uncertainty ≈ ±0.029 on the slope.
- That is ~2 sigma from the theoretical -0.5, driven mainly by the smallest N; the three larger sizes sit under the engine's own error estimate. Read as consistent with -0.5, not a clean pass — more replications would settle it.
- No evidence of correlated random-number streams across replications.

### Antithetic variates, at equal work

Measured with `bench anti <pairs> <K>`. The comparison is `pairs` pair-averages
(= 2×pairs simulated paths) against the plain estimator at 2×pairs paths, so
both sides simulate the same number of paths. Round-robin schedule, first round
discarded, median of 5. Measured on the CMake build, on AC power at 100% charge.

- **var_ratio** = (SE_plain / SE_anti)² at equal path count. A value of 1.0 means pairing bought nothing beyond spending those paths on the plain estimator.
- **efficiency** = var_ratio × (t_plain / t_anti) — the work-normalised figure, which charges antithetic for its extra `exp()` per step.

At 250,000 pairs (= 500,000 paths), sweeping the strike:

| Strike | Moneyness | var_ratio | Implied corr. ρ | Time ratio | Efficiency |
|---|---|---|---|---|---|
| 60 | deep ITM | 23.70 | -0.958 | 0.70 | **34.1x** |
| 80 | ITM | 8.61 | -0.884 | 0.65 | **13.3x** |
| 100 | ATM | 2.00 | -0.500 | 0.65 | **3.10x** |
| 120 | OTM | 1.16 | -0.141 | 0.62 | **1.87x** |
| 140 | deep OTM | 1.04 | -0.040 | 0.66 | **1.58x** |

Since Var((X+Y)/2) = σ²(1+ρ)/2, the variance ratio at equal paths is exactly
1/(1+ρ); the ρ column is that identity inverted, and it is the whole story. Deep
in the money the payoff is nearly linear in the driving noise, so the two legs
are almost perfectly anticorrelated and the technique is worth ~34x. Deep out of
the money both legs are usually zero together, the anticorrelation collapses,
and antithetic is close to a waste of the extra arithmetic. **The gain is a
property of the payoff, not of the technique.**

The ATM variance ratio holds steady as N grows, confirming it is structural
rather than a sampling artifact:

| Pairs | Equiv. paths | SE plain | SE anti | var_ratio | Efficiency |
|---|---|---|---|---|---|
| 100,000 | 200,000 | 0.0329501 | 0.0232821 | 2.003 | 3.20x |
| 250,000 | 500,000 | 0.0208659 | 0.0147555 | 2.000 | 3.17x |
| 400,000 | 800,000 | 0.0164653 | 0.0116755 | 1.989 | 3.13x |

The two tables overlap at one configuration (250,000 pairs, K = 100) and report
efficiency as 3.10 and 3.17; separate runs, the difference is timing noise.
Everything not derived from a clock is reproducible: `var_ratio` comes out
bitwise identical (1.9997) in both runs, as do both standard errors. Only the
time-derived efficiency moves. Repeat runs of the K = 60 configuration spanned
34.1x to 37.6x, about 10%, so treat the efficiency column as one significant
figure.

The time ratio deserves a note: antithetic simulates the same number of paths in
~0.65x the time, because the mirror leg negates an already-drawn Z rather than
drawing a new one, and Box-Muller (a `log`, a `sqrt`, a `cos`) is the expensive
part of the inner loop. Roughly half the measured efficiency gain is variance
reduction and half is that saved RNG work.

### Cross-thread reproducibility — not bitwise, and that's expected

`bench repro 200000` on the CMake build (`-O3 -DNDEBUG`), identical across runs:

| Threads | Price | Difference vs 1 thread | In ulps |
|---|---|---|---|
| 1 | 10.448018599844515 | 0 | 0 |
| 2 | 10.448018599844522 | 7.11e-15 | 4 |
| 4 | 10.448018599844572 | 5.68e-14 | 32 |

Every path draws the same random numbers at any thread count, since draws are
keyed by global path index so this is not RNG divergence. It is floating-point
addition order: with p threads the total is a sum of p partial sums, and FP
addition isn't associative. The largest gap is 32 ulps (one ulp at 10.448 is
1.776e-15), roughly twelve orders of magnitude below the Monte Carlo standard
error at this N (0.03295). Exact cross-thread determinism is achievable by fixing
the block count independent of p and reduce in block order.

These low-order digits are build-dependent: a hand-compiled `-O2 -march=native`
binary gives 1 and 31 ulps instead, because `-march=native` licenses different
FP contraction. Both agree to 15 significant figures.

### Parallel scaling

N = 250,000 paths, median of 5 runs, on the CMake build, on AC power at 100%
charge (`bench sweep 250000`):

| Threads | Median time (s) | Speedup | Efficiency | Spread across the 5 runs |
|---|---|---|---|---|
| 1 | 1.8614 | 1.00x | 1.00 | 3.6% |
| 2 | 1.0526 | 1.77x | 0.88 | 13.5% |
| 4 | 0.6339 | 2.94x | 0.73 | 4.5% |

Price is 10.4357897169 at every thread count, so the three rows are timing the
same computation.

Run-to-run spread is 4–14%. The 2-thread row is the noisy one and its 13.5%
spread is wide enough that 1.77x should be read as approximate; the 1- and
4-thread rows, at 3.6% and 4.5%, comfortably support the efficiency drop at 4
threads being real scaling behaviour rather than noise. An earlier measurement
on battery showed 83–88% spread and was discarded rather than reported;
`bench/REPORT.md` documents that diagnosis.

## Layout

```
include/mcpricer/   Public headers (declarations only).
src/                Implementation (.cpp, one per header).
apps/price.cpp      Command-line pricer.
tests/              Correctness checks, run through ctest.
bench/              Performance harness and the benchmark report.
```

`mcpricer` builds as a static library that `price`, `bench`, and the two check
binaries all link against.

The engine splits into: a GBM path simulator; an `Option` holding the payoff and
its derivative (used by the pathwise Greeks); a serial `MonteCarloEngine` with an
antithetic subclass and a `Precision` helper for confidence intervals; a
closed-form `Black_Scholes` reference; and a standalone `ParallelEngine` that
reimplements pricing, antithetic pricing, and the Greeks on the counter-based
RNG with cache-line-padded accumulators. `DataVariables.h` holds the shared
market/option/simulation parameter structs.

## Build / run

```bash
cmake -B build && cmake --build build
ctest --test-dir build --output-on-failure   # check_serial, check_greeks
./build/price --help
./build/bench anti 250000 100
```

Without CMake:

```bash
g++ -std=c++17 -O2 -march=native -DNDEBUG -pthread -Iinclude \
    src/*.cpp apps/price.cpp -o price
```

Substitute `bench/bench.cpp`, `tests/check_serial.cpp`, or
`tests/check_greeks.cpp` for the other binaries.

- `price` — the pricer: value, standard error, 95% CI, optional Greeks, closed-form comparison.
- `check_serial` — 22 checks: closed-form identities (put-call parity, delta parity, gamma/vega call-put equality, sign bounds), the serial and antithetic engines against closed form, and `Precision`'s confidence interval.
- `check_greeks` — 12 checks: every Greek from the parallel engine, error measured in standard errors, plus a theta h-sensitivity scan.
- `bench` — modes `calibrate`, `sweep [N]`, `converge [reps]`, `repro [N]`, `anti [pairs] [K]`.

### Testing approach

Agreement is asserted in **standard errors**, not absolute tolerances: under a
correct estimator `(mc - ref)/se` is ~N(0,1) at any N, so the 4-sigma bound means
the same thing at every sample size and doesn't need re-tuning when N changes —
which is the usual route by which a tolerance gets quietly loosened until it
passes. NaN is rejected explicitly, since `fabs(NaN) > 4.0` is false and would
otherwise be the one result that reported a pass. Seeds are fixed and variates
are keyed by global path index, so the checks are deterministic per build rather
than flaky.

The harness (`tests/check.h`) is ~60 lines rather than Catch2 or GoogleTest,
which would add a configure-time network dependency for what amounts to a
failure counter and an exit status.

The suite is verified by fault injection — each of these was introduced into a
scratch copy and confirmed to fail the build's `ctest`:

| Injected fault | Caught by | Worst deviation |
|---|---|---|
| `exp(-r*K)` discounting (uses strike as time) | `check_serial` | exact-equality check, off by 10.4 |
| GBM drift regressed to real-world `mu` | `check_serial` | 492 sigma |
| `-1.0` term dropped from LR gamma estimator | `check_greeks` | 226 sigma |

## Known limitations

- Path simulation runs 252 steps for a payoff that only depends on S_T. For European options a single draw of the terminal log-return would be exact and ~252x cheaper; the per-step loop is groundwork for the path-dependent payoffs below, which don't exist yet.
- The serial engine (`MonteCarloEngine`/`GBM`/`Precision`) and `ParallelEngine` are separate implementations that share only the payoff and parameter structs.
- `Option` dispatches on an enum, so new payoffs mean editing the class.

## Roadmap

- Sobol and other quasi-Monte Carlo sequences.
- Path-dependent payoffs — Asian and barrier options.
- Heston stochastic volatility.
- Control variates (the closed-form price is already available as the control).
