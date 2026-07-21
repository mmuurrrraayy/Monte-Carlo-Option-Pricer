# Monte Carlo Pricer — Parallel Benchmark & RNG Verification

All figures below come from runs on this machine. Nothing is estimated or extrapolated.

## Headline

The RNG verification **passes**. The timing table below was measured on the CMake
build, **on AC power at 100% charge**, and supersedes both the original
battery-power table and the earlier hand-compiled AC re-run — see *Timing — CMake
build, AC power* for the numbers and *Timing — original battery-power run
(superseded)* for what changed and why.

## Environment

| | |
|---|---|
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz (nominal) |
| Physical cores | 4 (via `GetLogicalProcessorInformationEx` / `RelationProcessorCore`) |
| Logical processors | 8 (SMT enabled) |
| Compiler | g++ 15.2.0 (MSYS2, Rev8) |
| Build | CMake 4.1.0, Ninja generator, `cmake -B build && cmake --build build` |
| Flags | `-O3 -DNDEBUG -std=c++17` (CMake Release; `CMAKE_CXX_EXTENSIONS OFF`) |
| Power state | **On AC, 100% charge**, confirmed via `Win32_Battery` before the sweep |
| Paths (N) | 250,000 |
| Steps per path | 252 (T = 1.0, dt = 1/252) |
| Contract | European call, S0 = K = 100, r = 0.05, sigma = 0.20, T = 1.0 |

All current figures come from the CMake build, so the whole report is one build
configuration. Earlier revisions of this document mixed two: a hand-rolled
`g++ -O2 -march=native` invocation was used while CMake was not yet installed on
this machine. Where that older build produced a different number it is called
out explicitly rather than silently replaced — see *Reproducibility across thread
counts*, where the low-order bits genuinely differ between the two.

## Parallel design (what is being measured)

- **Path distribution:** contiguous blocks, `N/p` paths per thread, remainder spread
  over the first `N mod p` threads.
- **Normal variates:** counter-based, *not* per-thread engines. `Z` is a pure function
  of `(seed, path_index * step_count + step)` via a splitmix64 mixer feeding
  Box-Muller. No shared RNG state, therefore no data race and no reliance on seeding
  luck for stream separation.
- **Accumulation:** one `alignas(64)` cache-line-padded accumulator per thread,
  reduced once at the end in index order. The false-sharing pattern
  (`results[tid] +=` into a `vector<double>`) is specifically avoided.
- **T1 baseline:** `threads == 1` runs `run_block` inline on the calling thread — no
  spawn, no join, same machine code as a parallel worker, same build.

Timing window covers only `eng.price(N, threads)` under `steady_clock`. Engine
construction, the accumulator allocation, and all I/O are outside it. Thread
spawn/join is inside it, since that cost is intrinsic to running with p threads.

## Timing — CMake build, AC power (current)

Same harness, same N, same round-robin-with-warm-up protocol as every previous
run. Power confirmed on AC at 100% charge before the sweep.

| Threads | Median (s) | Speedup T1/Tp | Efficiency | Min–max across the 5 runs |
|---|---|---|---|---|
| 1 | 1.8614 | 1.000 | 1.000 | 1.8095 – 1.8741 (3.6% spread) |
| 2 | 1.0526 | 1.768 | 0.884 | 0.9641 – 1.0938 (13.5% spread) |
| 4 | 0.6339 | 2.936 | 0.734 | 0.6166 – 0.6444 (4.5% spread) |

Price at every thread count: 10.4357897169 — identical to every earlier run,
including the battery one (same seed, deterministic given N; timing is the only
thing power state or build flags affect here).

Spread is 4–14%, against 83–88% on battery. Two of the three rows are tight
(3.6% and 4.5%) and the 4-thread efficiency of 0.73 is a real result: thread
spawn/join overhead plus memory-bandwidth sharing across 4 cores on a
memory-light per-path workload. The 2-thread row is the weak one — 13.5% spread
on a 1.77x speedup means that figure carries roughly ±0.1, so it should be read
as "close to linear" rather than as 1.768. Tightening it would need more rounds;
it is left as measured rather than quietly re-run until it looked better. No
superlinear speedup at any thread count.

These medians supersede the earlier `-O2 -march=native` table (1.8229 / 1.0445 /
0.6297, speedups 1.745 and 2.895). The two builds agree to within the spread of
the measurement, which is the expected outcome — the inner loop is dominated by
`exp` and Box-Muller, not by anything `-march=native` vectorises.

## Timing — original battery-power run (superseded)

Round-robin schedule across thread counts, first round discarded as warm-up,
median of the remaining 5.

| Threads | Median (s) | Speedup T1/Tp | Efficiency | Min–max across the 5 runs |
|---|---|---|---|---|
| 1 | 3.1568 | 1.000 | 1.000 | 2.698 – 3.316 (23% spread) |
| 2 | 1.9941 | 1.583 | 0.792 | 1.617 – 2.962 (83% spread) |
| 4 | 1.1672 | 2.705 | 0.676 | 1.149 – 2.155 (88% spread) |

Price at every thread count: 10.4357897169 (vs closed form 10.4505835722; the
-0.0148 gap is ~0.5 standard errors at N = 250,000, i.e. consistent).

**These medians were not trustworthy** — see *Anomalies and diagnosis* below for why.
Kept here only for comparison against the AC-power re-run above.

## Anomalies and diagnosis (battery-power run)

This section explains what was wrong with the *original* battery-power table above.
None of it applies to the AC-power runs, whose spread (4–14%) is in the range
expected from OS scheduling jitter alone.

**1. Run-to-run spread exceeds the effect being measured.** Within-configuration
spread reaches 88%. Two *identical* sustained 4-thread throughput measurements,
minutes apart, returned 96,776 and 131,841 paths/s — 36% apart. The measurement
uncertainty is larger than the difference between "scales well" and "scales badly",
so the efficiency figures cannot support a conclusion either way.

**Root cause: the laptop is running on battery at 34% charge.** This is a 15 W
U-series part with Intel DPTF (`dptf_helper`) actively managing power. Per-core
`% Processor Performance` sat at ~55% of nominal (~1.3 GHz) *even at idle*. Sustained
all-core load is power-limited, not compute-limited. Firefox was also running.

**2. Schedule ordering was itself a confound.** The first sweep ran all 5 timings of
each thread count consecutively. That put every 1-thread timing on a cool chip and
every 4-thread timing on a hot one, and reported T1 = 2.418 s, 4-thread speedup 2.07.
Re-running round-robin gave T1 = 3.157 s and speedup 2.70. Same binary, same N — a
31% swing in speedup from schedule order alone. The interleaved number is the sound
one; the blocked number was biased against the parallel runs.

**3. Pinning to distinct physical cores made throughput worse, not better.** I tested
SMT-sibling collision as a scaling suspect. Pinned sustained throughput: 29,211
paths/s at 1 thread vs 64,981 unpinned; 85,575 vs 131,841 at 4 threads. Pinning a
thread to one core concentrates heat on that core, whereas an unpinned thread
migrates and spreads it — on a thermally-limited part that costs more than SMT
contention does. **SMT collision is therefore not the scaling limiter here**; power
delivery is. The `pin` flag is left in the harness but should stay off.

No superlinear speedup was observed at any thread count.

## RNG stream independence

### Convergence to closed-form Black-Scholes

Closed form: **10.4505835722**. Mean over 24 independent replications per N:

| N | Mean price | Bias vs closed form |
|---|---|---|
| 1,000 | 10.2601847727 | -0.1903987995 |
| 10,000 | 10.4362022901 | -0.0143812821 |
| 100,000 | 10.4578672098 | +0.0072836376 |
| 1,000,000 | 10.4516913677 | +0.0011077955 |

Bias falls by ~170x over three orders of magnitude in N and changes sign, consistent
with sampling noise around the true value rather than a systematic offset. Converges.

### 1/sqrt(N) scaling

Standard error measured **empirically**: for each N, 24 replications with distinct
seeds, then the sample standard deviation of the 24 resulting prices. This is a real
test of stream independence — correlated streams would not shrink at the right rate.

| N | Observed SE (24 reps) | Engine's internal SE (single run) |
|---|---|---|
| 1,000 | 0.5362428555 | 0.4788488166 |
| 10,000 | 0.1226967380 | 0.1492474666 |
| 100,000 | 0.0363493832 | 0.0466441511 |
| 1,000,000 | 0.0112126454 | 0.0147343895 |

Least-squares fit of log(SE) on log(N):

- **Fitted exponent: -0.5567**
- R^2 = 0.9970
- Approximate 1-sigma on the slope: **+/- 0.029**

Theory is -0.5. The fit is ~2 sigma steep. With 24 replications the SE estimate
itself carries ~15% noise, and the steepening is driven mainly by the N = 1,000
point, whose observed SE (0.536) sits above the engine's own estimate (0.479) while
the three larger-N points sit below theirs. I would call this **consistent with
-0.5**, not evidence of a defect — but it is a 2 sigma deviation, not a clean pass,
and raising the replication count is what would settle it.

Verdict: no evidence of correlated streams.

### Reproducibility across thread counts

**Not bitwise reproducible. Stating that plainly.**

Re-measured under the CMake build (`cmake -B build; cmake --build build`, Ninja
generator, g++ 15.2.0, `-O3 -DNDEBUG -std=gnu++17`) via `./build/bench repro 200000`.
Two consecutive runs produced byte-identical output.

| Threads | Price | Delta vs 1 thread | In ulps | Bitwise equal |
|---|---|---|---|---|
| 1 | 10.448018599844515 | 0 | 0 | yes |
| 2 | 10.448018599844522 | 7.11e-15 | 4 | no |
| 4 | 10.448018599844572 | 5.68e-14 | 32 | no |

The cause is **not** the RNG. Because variates are keyed by global path index, path
`i` draws identical numbers at every thread count. The difference is reduction order:
with p threads the result is a sum of p partial sums, and floating-point addition is
not associative. One ulp at 10.448 is 1.776e-15, so the 2-thread delta is 4 ulps and
the 4-thread delta of 5.68e-14 is 32 ulps — a ratio of 1.7e-12 against the Monte
Carlo standard error at that N (0.0329500586), i.e. ~12 orders of magnitude below it.

These low-order digits are build-dependent. The earlier hand-compiled
`-O2 -march=native` binary gave deltas of 1 and 31 ulps for the same seed and N;
`-march=native` licenses different floating-point contraction, so the partial sums
round differently. Both builds agree to 15 significant figures and both sit ~12
orders of magnitude below the statistical error, so the conclusion is flag-independent
— but exact reproduction of the digits above requires the CMake build.

Exact cross-thread determinism is achievable — partition into a fixed number of
blocks independent of p and reduce in block order — at some throughput cost. Not
implemented, since the discrepancy is far below the statistical error.

## Timing is now defensible

The three original action items are done: this machine was on AC power at 100%
charge for the whole sweep, and the resulting spread (4–14%) confirms the earlier
83–88% spread was a power-throttling artifact, not inherent measurement noise. The
build flags have since changed once — from hand-rolled `-O2 -march=native` to the
CMake Release build — and the sweep was re-run rather than carried over, which is
why the medians here differ slightly from earlier revisions. No further re-run is
needed unless the hardware, build flags, or N change again.

One caveat is left standing rather than buried: at 2 threads the spread is 13.5%,
wider than the 4.4–7.4% seen on the earlier build, so that single row is the least
trustworthy number in this document.

The RNG results were unaffected by power state throughout; they are correctness
measurements, not timing.

## Source changes required to get here

Paths below are pre-restructuring (`src/*.h`); the same fixes now live in
`include/mcpricer/*.h` after the header/implementation split. The pricer did not
compile and would not have converged. Fixed:

- `Black-Scholes.h` — trailing comma in the constructor parameter list (hard compile
  error); missing return after the switch.
- `Black-Scholes.h` `d_plus` — parenthesisation: `r + (sigma^2/2)*T` instead of
  `(r + sigma^2/2)*T`, so `r` was never scaled by maturity. This corrupted the
  reference value the MC is checked against.
- `MonteCarloEngine.h` `discounted` — `exp(-r*K)` used the strike as time to expiry;
  should be `od.T`.
- `GBM.h` — drift was the real-world `mu`; a risk-neutral pricer must diffuse at `r`.
- `DataVariables.h` / `Option.h` — mutually inclusive headers, so whichever was
  included first saw no type definitions. `OptionType` moved into `DataVariables.h`.
- `Precision.h` — held `MonteCarloEngine` by value (slicing any derived engine back to
  base); `path` and `s0` were read uninitialised.
