#ifndef ParallelEngine_H
#define ParallelEngine_H

#include "mcpricer/DataVariables.h"
#include "mcpricer/Option.h"

#include <cstdint>
#include <vector>

// Counter-based normal variates.
//
// The serial engine draws Z from one shared std::mt19937 + normal_distribution.
// Both are stateful, so calling them from several threads is a data race, and
// even with a lock the variate a path receives would depend on scheduling.
//
// Here Z is a pure function of (seed, variate index). The index is derived from
// the GLOBAL path index, so path i sees the same variates regardless of how many
// threads are running or how paths were partitioned. Independence between
// "streams" therefore reduces to the avalanche quality of the mixer rather than
// to seeding luck.
//
// Note what this does NOT buy: the final price is not bitwise identical across
// thread counts. Every path contributes the same payoff, but with p threads the
// total is reduced as a sum of p partial sums, and floating-point addition is
// not associative. Measured at 4 ulps (2 threads) and 32 ulps (4 threads) --
// ~12 orders of magnitude below the Monte Carlo standard error, so immaterial,
// but not zero. The exact counts shift with optimization flags; see
// bench/REPORT.md, "Reproducibility across thread counts".
namespace cbrng {

    std::uint64_t mix64(std::uint64_t x);

    // (0,1) -- never exactly 0, so std::log below is always finite.
    double to_unit(std::uint64_t x);

    // One standard normal per variate index, via Box-Muller.
    // Two independent mixer outputs per Z; the second Box-Muller variate is
    // discarded to keep the index arithmetic trivial.
    double normal(std::uint64_t key, std::uint64_t index);

} // namespace cbrng

struct PriceResult {
    double price;      // discounted mean payoff
    double std_error;  // discounted sample standard error of the mean
    double mean_payoff;
};

// value +/- std_error for a single Greek estimator.
struct GreekEstimate {
    double value;
    double std_error;
};

struct GreeksResult {
    GreekEstimate price;
    GreekEstimate delta;
    GreekEstimate gamma;
    GreekEstimate vega;
    GreekEstimate rho;
};

class ParallelEngine {
private:
    market_data md;
    option_data od;
    sim_data sd;
    Option option;
    int step_count;
    double drift_dt, vol_sqrt_dt, discount;
    std::uint64_t key;

    // Padded to a cache line. Accumulating into adjacent elements of a shared
    // vector<double> is the textbook false-sharing bug: 8 doubles per 64-byte
    // line, so every thread's += invalidates every other thread's line.
    struct alignas(64) Accumulator {
        double sum = 0.0;
        double sumsq = 0.0;
        char pad[64 - 2 * sizeof(double)];
    };

    // Five payoff-derived quantities per path: the base payoff plus the four
    // pathwise/LR Greek estimators (theta is handled separately by theta_crn,
    // since it needs a paired two-maturity simulation, not a single path's
    // payoff). Deliberately oversized past 64 bytes -- alignas rounds the
    // array stride up to a cache-line multiple, so no manual pad is needed.
    struct alignas(64) GreekAccumulator {
        double sum_payoff = 0.0, sumsq_payoff = 0.0;
        double sum_delta = 0.0,  sumsq_delta = 0.0;
        double sum_gamma = 0.0,  sumsq_gamma = 0.0;
        double sum_vega = 0.0,   sumsq_vega = 0.0;
        double sum_rho = 0.0,    sumsq_rho = 0.0;
    };

    struct alignas(64) PairAccumulator {
        double sum = 0.0;
        double sumsq = 0.0;
    };

    // Thread dispatch, factored out of price()/price_and_greeks()/theta_crn()
    // so all three run identical path-generation and threading code and only
    // differ in what they accumulate per path. threads==1 runs block() inline
    // on the calling thread -- no spawn, no join, same code as a worker.
    //
    // Defined in the .cpp (not here): it is only ever instantiated from the
    // three call sites in ParallelEngine.cpp, so the definition only needs to
    // be visible in that one translation unit.
    template <class Acc, class Block>
    std::vector<Acc> run_parallel(std::uint64_t N, int threads, Block&& block) const;

    // The hot loop. Identical code runs for 1 thread and for p threads, so the
    // T1 baseline is the same machine code as each Tp worker.
    void run_block(std::uint64_t begin, std::uint64_t end, Accumulator& acc) const;

    // Antithetic variates. Pair i reuses the SAME variate indices as plain path
    // i (base = i*step_count), so the "+Z" leg is bitwise identical to the plain
    // engine's path i -- the pairing is the only difference, which is what makes
    // the equal-work comparison in `bench anti` a like-for-like one.
    void run_block_anti(std::uint64_t begin, std::uint64_t end, Accumulator& acc) const;

    // Same paths as run_block, extended with the pathwise/LR Greek estimators.
    // Every quantity here is derived from the SAME simulated path used for the
    // base price -- no extra RNG draws, so this costs one pass, not five.
    void run_block_greeks(std::uint64_t begin, std::uint64_t end, GreekAccumulator& acc) const;

public:
    ParallelEngine(market_data md_, option_data od_, sim_data sd_, Option option_);

    int steps() const { return step_count; }

    // Optional hook called on each worker with its index, before it starts work.
    // Used by the benchmark to pin workers to distinct physical cores; null in
    // normal use, so the engine carries no platform dependency.
    void (*on_thread_start)(int) = nullptr;

    PriceResult price(std::uint64_t N, int threads) const;

    // Antithetic-variate price over `pairs` PAIRS, i.e. 2*pairs simulated paths.
    // The returned std_error is the error of the mean over `pairs` independent
    // pair-averages -- so comparing it against price(2*pairs, .) is an
    // equal-work comparison, which is the only kind that says anything useful
    // about a variance-reduction technique.
    PriceResult price_antithetic(std::uint64_t pairs, int threads) const;

    // Price plus delta/gamma/vega/rho, all from one pass over N paths (the
    // same paths price() would use). See run_block_greeks for the estimators.
    GreeksResult price_and_greeks(std::uint64_t N, int threads) const;

    // Theta via common-random-numbers central difference: (P(T-h) - P(T+h)) / (2h).
    //
    // Bumping T naively would change step_count = T/time_step, which shifts
    // every path's variate indices (base = i*step_count) and silently breaks
    // CRN. Instead step_count is held fixed at THIS engine's value and only
    // dt = T'/step_count is rescaled for the +-h legs, so path i draws the
    // IDENTICAL Z sequence at T-h and T+h -- the two legs are simulated in the
    // same loop, sharing every Z, so the difference is pure signal.
    GreekEstimate theta_crn(std::uint64_t N, int threads, double h) const;
};
#endif
