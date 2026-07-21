// Validates the full Greek suite (delta, gamma, vega, theta, rho) computed by
// ParallelEngine against the closed-form Black-Scholes values.
//
// Estimators:
//   delta, vega, rho  -- pathwise (ParallelEngine::price_and_greeks)
//   gamma             -- likelihood ratio / score function (same call, since
//                         it needs the same simulated paths)
//   theta             -- CRN central difference (ParallelEngine::theta_crn),
//                         because bumping T changes step_count and only CRN
//                         with a fixed step_count keeps that bump honest
//
// Every Greek's own standard error is reported, and each closed-form
// comparison is expressed as (MC - closed_form) / std_error, i.e. "how many
// standard errors off is this", not just an absolute gap.
#include "mcpricer/Black-Scholes.h"
#include "mcpricer/ParallelEngine.h"
#include "mcpricer/Option.h"
#include "mcpricer/DataVariables.h"

#include "check.h"

#include <cstdio>
#include <cmath>
#include <thread>

// Every check is "how many standard errors from the closed-form value", and
// must come in under this bound. Expressed in standard errors rather than as an
// absolute tolerance so that changing N rescales the bound automatically.
//
// 4.0 sits in a wide empty gap. Under a correct estimator the ratio is ~N(0,1)
// and the worst row here runs about 1.1; the historical exp(-r*K) discounting
// bug sat roughly 50 standard errors out. The seed is fixed and the variates are
// keyed by global path index, so this is deterministic for a given build -- a
// regression test in statistically meaningful units, not a flaky one.
static constexpr double kMaxErrInSe = 4.0;

struct Setup {
    market_data md;
    option_data od;
    sim_data sd;
    Option option;
};

static Setup make_setup(OptionType type, unsigned int seed) {
    // mu is deliberately NOT equal to r. Nothing in the pricer reads mu, but if
    // GBM ever regresses to diffusing at the real-world drift (the bug this
    // project started with), mu == r would hide it completely. At mu = 0.15 that
    // regression moves the price from ~10.45 to ~16 -- hundreds of standard
    // errors, caught instantly.
    market_data md; md.s0 = 100.0; md.r = 0.05; md.mu = 0.15; md.sigma = 0.20;
    option_data od; od.K = 100.0; od.T = 1.0; od.option_type = type;
    sim_data sd; sd.N = 0; sd.seed = seed;
    return Setup{md, od, sd, Option(md, od, sd)};
}

static void validate(const char* label, OptionType type, std::uint64_t N, int threads, double h) {
    auto s = make_setup(type, 42);
    ParallelEngine eng(s.md, s.od, s.sd, s.option);
    Black_Scholes bs(0.0, s.md, s.od);

    const auto g = eng.price_and_greeks(N, threads);
    const auto th = eng.theta_crn(N, threads, h);

    std::printf("\n-- %s (N=%llu, threads=%d, theta h=%.6f) --\n",
                label, (unsigned long long)N, threads, h);
    check::within_se("price", g.price.value, g.price.std_error, bs.price(s.md.s0), kMaxErrInSe);
    check::within_se("delta", g.delta.value, g.delta.std_error, bs.delta(s.md.s0), kMaxErrInSe);
    check::within_se("gamma", g.gamma.value, g.gamma.std_error, bs.gamma(s.md.s0), kMaxErrInSe);
    check::within_se("vega",  g.vega.value,  g.vega.std_error,  bs.vega(s.md.s0),  kMaxErrInSe);
    check::within_se("theta", th.value,      th.std_error,      bs.theta(s.md.s0), kMaxErrInSe);
    check::within_se("rho",   g.rho.value,   g.rho.std_error,   bs.rho(s.md.s0),   kMaxErrInSe);
}

// Demonstrates the CRN bias/variance point directly: with common random
// numbers, shrinking h should shrink the truncation bias without inflating
// std_error the way an uncorrelated (non-CRN) finite difference would.
static void theta_h_scan(OptionType type, std::uint64_t N, int threads) {
    auto s = make_setup(type, 42);
    ParallelEngine eng(s.md, s.od, s.sd, s.option);
    Black_Scholes bs(0.0, s.md, s.od);
    const auto ref = bs.theta(s.md.s0);

    std::printf("\n-- theta CRN h-sensitivity (%s, N=%llu) --\n",
                type == OptionType::Call ? "call" : "put", (unsigned long long)N);
    const double hs[] = {0.05, 0.005, 0.0005, 0.00005};
    for (double h : hs) {
        const auto th = eng.theta_crn(N, threads, h);
        char name[64];
        std::snprintf(name, sizeof name, "theta h=%.5f", h);
        check::within_se(name, th.value, th.std_error, ref, kMaxErrInSe);
    }
}

int main() {
    const auto hw = std::thread::hardware_concurrency();
    const auto threads = hw > 0 ? static_cast<int>(hw) : 1;
    auto N = std::uint64_t{2000000};
    const auto h = 0.005; // ~1.8 days; justified empirically by the scan below

    validate("call", OptionType::Call, N, threads, h);
    validate("put",  OptionType::Put,  N, threads, h);

    theta_h_scan(OptionType::Call, 500000, threads);

    return check::summary();
}
