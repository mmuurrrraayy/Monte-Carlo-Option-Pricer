#include "mcpricer/ParallelEngine.h"

#include <cmath>
#include <thread>

namespace cbrng {

    std::uint64_t mix64(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    double to_unit(std::uint64_t x) {
        return ((x >> 11) + 0.5) * 0x1.0p-53;
    }

    double normal(std::uint64_t key, std::uint64_t index) {
        const auto u1 = to_unit(mix64(key ^ (2 * index)));
        const auto u2 = to_unit(mix64(key ^ (2 * index + 1)));
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

} // namespace cbrng

/*
 * Definition must precede its uses below (price/price_and_greeks/theta_crn),
 * since this translation unit is the only place it is instantiated.
 */
template <class Acc, class Block>
std::vector<Acc> ParallelEngine::run_parallel(std::uint64_t N, int threads, Block&& block) const {
    std::vector<Acc> acc(threads);
    if (threads == 1) {
        if (on_thread_start) on_thread_start(0);
        block(0, N, acc[0]);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(threads);
        const auto chunk = N / threads;
        const auto rem = N % threads;
        auto begin = std::uint64_t{0};
        for (int t = 0; t < threads; ++t) {
            const auto len = chunk + (static_cast<std::uint64_t>(t) < rem ? 1 : 0);
            pool.emplace_back([this, begin, len, &acc, t, &block] {
                if (on_thread_start) on_thread_start(t);
                block(begin, begin + len, acc[t]);
            });
            begin += len;
        }
        for (auto& th : pool) th.join();
    }
    return acc;
}

void ParallelEngine::run_block(std::uint64_t begin, std::uint64_t end, Accumulator& acc) const {
    auto sum = 0.0, sumsq = 0.0;
    for (std::uint64_t i = begin; i < end; ++i) {
        auto S = md.s0;
        const auto base = i * static_cast<std::uint64_t>(step_count);
        for (int step = 0; step < step_count; ++step) {
            const auto Z = cbrng::normal(key, base + step);
            S *= std::exp(drift_dt + vol_sqrt_dt * Z);
        }
        const auto p = option.payoff(S);
        sum += p;
        sumsq += p * p;
    }
    acc.sum = sum;
    acc.sumsq = sumsq;
}

/*
 * Antithetic variates. The mirror leg negates Z at every step, so the two legs
 * share one RNG draw sequence and cost one extra exp() per step rather than a
 * second set of variates.
 *
 * Why this reduces variance: payoff(S(Z)) and payoff(S(-Z)) are negatively
 * correlated, and Var((X+Y)/2) = (Var X + Var Y + 2Cov(X,Y))/4, so the pair
 * average carries less variance than either leg alone. The reduction is only a
 * win per unit WORK if that shrinkage beats the factor of 2 in paths -- which
 * is exactly what `bench anti` measures rather than assumes.
 */
void ParallelEngine::run_block_anti(std::uint64_t begin, std::uint64_t end, Accumulator& acc) const {
    auto sum = 0.0, sumsq = 0.0;
    for (std::uint64_t i = begin; i < end; ++i) {
        auto Sp = md.s0, Sm = md.s0;
        const auto base = i * static_cast<std::uint64_t>(step_count);
        for (int step = 0; step < step_count; ++step) {
            const auto Z = cbrng::normal(key, base + step);
            Sp *= std::exp(drift_dt + vol_sqrt_dt * Z);
            Sm *= std::exp(drift_dt - vol_sqrt_dt * Z);
        }
        /*
         * One sample = one PAIR AVERAGE, not two samples. Accumulating the two
         * legs as independent draws would be the classic antithetic blunder: it
         * reports the variance of the unpaired estimator and hides the very
         * effect being measured.
         */
        const auto p = 0.5 * (option.payoff(Sp) + option.payoff(Sm));
        sum += p;
        sumsq += p * p;
    }
    acc.sum = sum;
    acc.sumsq = sumsq;
}

/*
 * The whole-path log return is exact: ln(S_T/S0) = (r - sigma^2/2)*T +
 * sigma*sqrt(T)*Z_agg, where Z_agg = (sum of the per-step Z's)/sqrt(step_count)
 * is itself an exact standard normal (sum of iid N(0,1) draws, rescaled).
 * That closed form is what makes the pathwise/LR derivatives below exact
 * rather than approximate, regardless of step_count.
 */
void ParallelEngine::run_block_greeks(std::uint64_t begin, std::uint64_t end, GreekAccumulator& acc) const {
    auto pf = 0.0, pf2 = 0.0;
    auto de = 0.0, de2 = 0.0;
    auto ga = 0.0, ga2 = 0.0;
    auto ve = 0.0, ve2 = 0.0;
    auto rh = 0.0, rh2 = 0.0;
    const auto sqrtT = std::sqrt(od.T);
    const auto s0sq_sigsq_T = md.s0 * md.s0 * md.sigma * md.sigma * od.T;

    for (std::uint64_t i = begin; i < end; ++i) {
        auto S = md.s0;
        auto sumZ = 0.0;
        const auto base = i * static_cast<std::uint64_t>(step_count);
        for (int step = 0; step < step_count; ++step) {
            const auto Z = cbrng::normal(key, base + step);
            sumZ += Z;
            S *= std::exp(drift_dt + vol_sqrt_dt * Z);
        }

        const auto payoff = option.payoff(S);
        const auto prime = option.payoff_prime(S);       // d(payoff)/dS_T, a.e.
        const auto Zagg = sumZ / std::sqrt(static_cast<double>(step_count));

        /*
         * Pathwise: differentiate S_T(S0,sigma,r,Z_agg) analytically with
         * Z_agg held fixed (the same "fix the randomness, vary the
         * parameter" coupling used for all three), then chain through the
         * payoff's a.e. derivative. Valid because max(x,0) is Lipschitz,
         * so d/dtheta and E[.] commute (dominated convergence) even though
         * the payoff itself is only piecewise-smooth.
         */
        const auto dST_dS0 = S / md.s0;                       // d ln S_T/dS0 = 1/S0
        const auto dST_dsigma = S * (sqrtT * Zagg - md.sigma * od.T);
        const auto dST_dr = S * od.T;                         // d ln S_T/dr = T

        const auto delta_i = discount * prime * dST_dS0;
        const auto vega_i  = discount * prime * dST_dsigma;
        /*
         * rho also has an explicit term through the discount factor itself:
         * d(discount*E[payoff])/dr = -T*discount*payoff + discount*prime*dST_dr.
         */
        const auto rho_i   = discount * (prime * dST_dr - od.T * payoff);

        /*
         * Gamma: naive pathwise would require differentiating `prime`
         * itself, i.e. a Dirac delta at S_T=K -- undefined pointwise and
         * not something dominated convergence rescues. Instead this is
         * the likelihood-ratio (score function) estimator: differentiate
         * the log-normal DENSITY of S_T w.r.t. S0 twice, leaving the
         * payoff untouched. Broadie-Glasserman (1996):
         *   Gamma = discount * E[payoff(S_T) * (Z^2 - sigma*sqrt(T)*Z - 1)
         *                        / (S0^2 * sigma^2 * T)]
         */
        const auto gamma_i = discount * payoff
            * (Zagg * Zagg - md.sigma * sqrtT * Zagg - 1.0) / s0sq_sigsq_T;

        pf += payoff;   pf2 += payoff * payoff;
        de += delta_i;  de2 += delta_i * delta_i;
        ga += gamma_i;  ga2 += gamma_i * gamma_i;
        ve += vega_i;   ve2 += vega_i * vega_i;
        rh += rho_i;    rh2 += rho_i * rho_i;
    }

    acc.sum_payoff = pf; acc.sumsq_payoff = pf2;
    acc.sum_delta  = de; acc.sumsq_delta  = de2;
    acc.sum_gamma  = ga; acc.sumsq_gamma  = ga2;
    acc.sum_vega   = ve; acc.sumsq_vega   = ve2;
    acc.sum_rho    = rh; acc.sumsq_rho    = rh2;
}

ParallelEngine::ParallelEngine(market_data md_, option_data od_, sim_data sd_, Option option_)
    : md(md_), od(od_), sd(sd_), option(option_) {
    step_count = static_cast<int>(od.T / sd.time_step);
    const auto m = md.r - 0.5 * md.sigma * md.sigma;
    drift_dt = m * sd.time_step;
    vol_sqrt_dt = md.sigma * std::sqrt(sd.time_step);
    discount = std::exp(-md.r * od.T);
    key = cbrng::mix64(sd.seed);
}

PriceResult ParallelEngine::price(std::uint64_t N, int threads) const {
    auto acc = run_parallel<Accumulator>(N, threads,
        [this](std::uint64_t begin, std::uint64_t end, Accumulator& a) {
            run_block(begin, end, a);
        });

    /*
     * Deterministic reduction order: summing acc[0..p) in index order gives
     * a result that does not depend on thread completion order, but DOES
     * depend on p, since floating-point addition is not associative.
     */
    auto sum = 0.0, sumsq = 0.0;
    for (int t = 0; t < threads; ++t) {
        sum += acc[t].sum;
        sumsq += acc[t].sumsq;
    }

    const auto n = static_cast<double>(N);
    const auto mean = sum / n;
    const auto var = (sumsq / n - mean * mean) * (n / (n - 1.0));
    PriceResult r;
    r.mean_payoff = mean;
    r.price = discount * mean;
    r.std_error = discount * std::sqrt(var / n);
    return r;
}

PriceResult ParallelEngine::price_antithetic(std::uint64_t pairs, int threads) const {
    auto acc = run_parallel<Accumulator>(pairs, threads,
        [this](std::uint64_t begin, std::uint64_t end, Accumulator& a) {
            run_block_anti(begin, end, a);
        });

    auto sum = 0.0, sumsq = 0.0;
    for (int t = 0; t < threads; ++t) {
        sum += acc[t].sum;
        sumsq += acc[t].sumsq;
    }

    /* n is the number of PAIRS -- the number of independent samples. */
    const auto n = static_cast<double>(pairs);
    const auto mean = sum / n;
    const auto var = (sumsq / n - mean * mean) * (n / (n - 1.0));
    PriceResult r;
    r.mean_payoff = mean;
    r.price = discount * mean;
    r.std_error = discount * std::sqrt(var / n);
    return r;
}

GreeksResult ParallelEngine::price_and_greeks(std::uint64_t N, int threads) const {
    auto acc = run_parallel<GreekAccumulator>(N, threads,
        [this](std::uint64_t begin, std::uint64_t end, GreekAccumulator& a) {
            run_block_greeks(begin, end, a);
        });

    auto pf = 0.0, pf2 = 0.0, de = 0.0, de2 = 0.0;
    auto ga = 0.0, ga2 = 0.0, ve = 0.0, ve2 = 0.0, rh = 0.0, rh2 = 0.0;
    for (int t = 0; t < threads; ++t) {
        pf += acc[t].sum_payoff; pf2 += acc[t].sumsq_payoff;
        de += acc[t].sum_delta;  de2 += acc[t].sumsq_delta;
        ga += acc[t].sum_gamma;  ga2 += acc[t].sumsq_gamma;
        ve += acc[t].sum_vega;   ve2 += acc[t].sumsq_vega;
        rh += acc[t].sum_rho;    rh2 += acc[t].sumsq_rho;
    }

    const auto n = static_cast<double>(N);
    auto estimate_price = [&](double s, double s2) {
        const auto mean = s / n;
        const auto var = (s2 / n - mean * mean) * (n / (n - 1.0));
        return GreekEstimate{discount * mean, discount * std::sqrt(var / n)};
    };
    /*
     * delta_i/gamma_i/vega_i/rho_i are already fully-discounted per-path
     * estimators (see run_block_greeks), so no extra discount factor here.
     */
    auto estimate_greek = [&](double s, double s2) {
        const auto mean = s / n;
        const auto var = (s2 / n - mean * mean) * (n / (n - 1.0));
        return GreekEstimate{mean, std::sqrt(var / n)};
    };

    GreeksResult r;
    r.price = estimate_price(pf, pf2);
    r.delta = estimate_greek(de, de2);
    r.gamma = estimate_greek(ga, ga2);
    r.vega  = estimate_greek(ve, ve2);
    r.rho   = estimate_greek(rh, rh2);
    return r;
}

GreekEstimate ParallelEngine::theta_crn(std::uint64_t N, int threads, double h) const {
    const auto T_minus = od.T - h;
    const auto T_plus  = od.T + h;
    const auto dt_minus = T_minus / step_count;
    const auto dt_plus  = T_plus  / step_count;
    const auto m = md.r - 0.5 * md.sigma * md.sigma;
    const auto drift_minus = m * dt_minus;
    const auto drift_plus  = m * dt_plus;
    const auto vol_minus = md.sigma * std::sqrt(dt_minus);
    const auto vol_plus  = md.sigma * std::sqrt(dt_plus);
    const auto disc_minus = std::exp(-md.r * T_minus);
    const auto disc_plus  = std::exp(-md.r * T_plus);
    const auto sc = step_count;
    const auto k = key;

    auto acc = run_parallel<PairAccumulator>(N, threads,
        [this, sc, k, drift_minus, drift_plus, vol_minus, vol_plus,
         disc_minus, disc_plus, h]
        (std::uint64_t begin, std::uint64_t end, PairAccumulator& a) {
            auto sum = 0.0, sumsq = 0.0;
            for (std::uint64_t i = begin; i < end; ++i) {
                auto Sm = md.s0, Sp = md.s0;
                const auto base = i * static_cast<std::uint64_t>(sc);
                for (int step = 0; step < sc; ++step) {
                    const auto Z = cbrng::normal(k, base + step);
                    Sm *= std::exp(drift_minus + vol_minus * Z);
                    Sp *= std::exp(drift_plus  + vol_plus  * Z);
                }
                /* theta = -dPrice/dT ~= (P(T-h) - P(T+h)) / (2h) */
                const auto diff =
                    (disc_minus * option.payoff(Sm) - disc_plus * option.payoff(Sp)) / (2.0 * h);
                sum += diff;
                sumsq += diff * diff;
            }
            a.sum = sum;
            a.sumsq = sumsq;
        });

    auto sum = 0.0, sumsq = 0.0;
    for (int t = 0; t < threads; ++t) {
        sum += acc[t].sum;
        sumsq += acc[t].sumsq;
    }
    const auto n = static_cast<double>(N);
    const auto mean = sum / n;
    const auto var = (sumsq / n - mean * mean) * (n / (n - 1.0));
    return GreekEstimate{mean, std::sqrt(var / n)};
}
