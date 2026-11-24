#include "cpuid_guard.hpp"

#include "pricer/core/types.hpp"
#include "pricer/engines/black_scholes.hpp"
#include "pricer/engines/monte_carlo.hpp"
#include "pricer/engines/thread_pool.hpp"
#include "pricer/models/gbm.hpp"
#include "pricer/models/heston.hpp"
#include "pricer/payoffs/asian.hpp"
#include "pricer/payoffs/barrier.hpp"
#include "pricer/payoffs/european.hpp"
#include "pricer/payoffs/lookback.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace {

using namespace pricer;

enum class ModelKind { GBM, Heston };
enum class PayoffKind { European, AsianGeom, AsianArith, Barrier, Lookback };

void usage() {
    std::printf(
        "usage: price [options]\n"
        "\n"
        "Market & contract options:\n"
        "  --s0 <f>              spot price                              (default 100)\n"
        "  --k <f>               strike price                            (default 100)\n"
        "  --r <f>               risk-free rate, annualised              (default 0.05)\n"
        "  --q <f>               continuous dividend yield               (default 0.0)\n"
        "  --sigma <f>           volatility, annualised (GBM)            (default 0.20)\n"
        "  --T <f>               time to expiry in years                 (default 1.0)\n"
        "  --type <s>            call | put                              (default call)\n"
        "\n"
        "Model & payoff options:\n"
        "  --model <s>           gbm | heston                            (default gbm)\n"
        "  --payoff <s>          european | asian-geom | asian-arith |\n"
        "                        barrier | lookback                      (default european)\n"
        "  --barrier <f>         barrier level (for --payoff barrier)\n"
        "  --barrier-type <s>    up-out | down-out | up-in | down-in     (default auto out)\n"
        "  --lookback-style <s>  fixed | floating                        (default fixed)\n"
        "  --v0 <f>              Heston initial variance                 (default sigma^2)\n"
        "  --kappa <f>           Heston mean-reversion speed             (default 2.0)\n"
        "  --heston-theta <f>    Heston long-run variance                (default sigma^2)\n"
        "  --xi <f>              Heston vol-of-vol                       (default 0.30)\n"
        "  --rho <f>             Heston spot-vol correlation             (default -0.50)\n"
        "\n"
        "Simulation & Greeks options:\n"
        "  --paths <n>           simulated paths                         (default 1000000)\n"
        "  --steps <n>           time steps per path (multi-step)        (default 252)\n"
        "  --force-multi-step    force multi-step loop for European GBM\n"
        "  --threads <n>         worker threads (0 = physical cores)     (default 0)\n"
        "  --seed <n>            RNG seed                                (default 42)\n"
        "  --antithetic          use antithetic variates (paths/2 pairs)\n"
        "  --greeks              compute sensitivities\n"
        "  --theta-h <f>         bump for CRN theta estimate             (default 0.005)\n"
        "  --no-reference        suppress closed-form Black-Scholes comparison\n"
        "\n"
        "Black-Scholes AVX2 chain & Implied Volatility modes:\n"
        "  --chain <n>           price an N-strike AVX2 SoA Black-Scholes chain\n"
        "  --iv [price]          solve Black-Scholes IV for market price (or chain round-trip)\n"
        "  -h, --help            this message\n");
}

bool arg_double(int argc, char** argv, int& i, double& out) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", argv[i]);
        return false;
    }
    char* end = nullptr;
    const double v = std::strtod(argv[i + 1], &end);
    if (end == argv[i + 1] || *end != '\0') {
        std::fprintf(stderr, "error: %s: '%s' is not a number\n", argv[i], argv[i + 1]);
        return false;
    }
    out = v;
    ++i;
    return true;
}

bool arg_u64(int argc, char** argv, int& i, std::uint64_t& out) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", argv[i]);
        return false;
    }
    if (argv[i + 1][0] == '-') {
        std::fprintf(stderr, "error: %s: '%s' must be a non-negative integer\n", argv[i], argv[i + 1]);
        return false;
    }
    char* end = nullptr;
    const unsigned long long v = std::strtoull(argv[i + 1], &end, 10);
    if (end == argv[i + 1] || *end != '\0') {
        std::fprintf(stderr, "error: %s: '%s' is not an integer\n", argv[i], argv[i + 1]);
        return false;
    }
    out = static_cast<std::uint64_t>(v);
    ++i;
    return true;
}

const char* payoff_name(PayoffKind k, BarrierType bt, LookbackStyle ls) {
    switch (k) {
        case PayoffKind::European:   return "european";
        case PayoffKind::AsianGeom:  return "asian-geom";
        case PayoffKind::AsianArith: return "asian-arith";
        case PayoffKind::Barrier:
            switch (bt) {
                case BarrierType::UpOut:   return "barrier(up-out)";
                case BarrierType::DownOut: return "barrier(down-out)";
                case BarrierType::UpIn:    return "barrier(up-in)";
                case BarrierType::DownIn:  return "barrier(down-in)";
            }
            return "barrier";
        case PayoffKind::Lookback:
            return ls == LookbackStyle::FixedStrike ? "lookback(fixed)" : "lookback(floating)";
    }
    return "unknown";
}

void print_pricing_result(const PricingResult& res) {
    std::printf("\nprice      %.8f\n", res.price);
    std::printf("std_error  %.8f\n", res.std_error);
    std::printf("95%% CI     [%.8f, %.8f]\n", res.ci_low, res.ci_high);
}

void print_gbm_greeks(const MCGreeksResult& g, bool want_ref, const GreeksResult* ref) {
    std::printf("\n%-6s %14s %13s", "greek", "value", "std_error");
    if (want_ref && ref != nullptr) {
        std::printf(" %14s %9s", "black_scholes", "err_se");
    }
    std::printf("\n");

    struct Row { const char* name; GreekEstimate e; double r; };
    const Row rows[] = {
        {"delta", g.delta, ref != nullptr ? ref->delta : 0.0},
        {"gamma", g.gamma, ref != nullptr ? ref->gamma : 0.0},
        {"vega",  g.vega,  ref != nullptr ? ref->vega  : 0.0},
        {"theta", g.theta, ref != nullptr ? ref->theta : 0.0},
        {"rho",   g.rho,   ref != nullptr ? ref->rho   : 0.0},
    };
    for (const auto& row : rows) {
        std::printf("%-6s %14.8f %13.8f", row.name, row.e.value, row.e.std_error);
        if (want_ref && ref != nullptr) {
            std::printf(" %14.8f %9.2f", row.r,
                        row.e.std_error > 0.0 ? (row.e.value - row.r) / row.e.std_error : 0.0);
        }
        std::printf("\n");
    }
}

void print_barrier_greeks(const BarrierGreeksResult& g) {
    std::printf("\n%-6s %14s %13s\n", "greek", "value", "std_error");
    struct Row { const char* name; GreekEstimate e; };
    const Row rows[] = {
        {"delta", g.delta},
        {"vega",  g.vega},
        {"theta", g.theta},
        {"rho",   g.rho},
    };
    for (const auto& row : rows) {
        std::printf("%-6s %14.8f %13.8f\n", row.name, row.e.value, row.e.std_error);
    }
    std::printf("(gamma omitted: naive CRN gamma has O(1/eps^3) variance on discontinuous barriers)\n");
}

void print_heston_greeks(const HestonGreeksResult& g) {
    std::printf("\n%-15s %14s %13s\n", "sensitivity", "value", "std_error");
    struct Row { const char* name; GreekEstimate e; };
    const Row rows[] = {
        {"delta",          g.delta},
        {"d_v0",           g.d_v0},
        {"d_long_run_var", g.d_long_run_var},
        {"theta",          g.theta},
        {"rho",            g.rho},
    };
    for (const auto& row : rows) {
        std::printf("%-15s %14.8f %13.8f\n", row.name, row.e.value, row.e.std_error);
    }
}

template <OptionType Type>
int run_chain_mode(double s0, double r, double q, double sigma, double T,
                   std::size_t n, bool want_greeks, bool want_iv) {
    if (n == 0) {
        std::fprintf(stderr, "error: --chain count must be >= 1\n");
        return 2;
    }
    if (auto mp = MarketParams::create(s0, r, sigma, T, q); !mp.has_value()) {
        std::fprintf(stderr, "error: %.*s\n",
                     static_cast<int>(to_string(mp.error()).size()),
                     to_string(mp.error()).data());
        return 2;
    }

    OptionChainSoA chain(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double frac = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.5;
        const double k_i = s0 * (0.70 + 0.60 * frac);
        chain.set(i, s0, k_i, r, q, sigma, T);
    }

    AlignedBuffer prices(chain.padded_n);
    ChainGreeksSoA greeks(want_greeks ? n : 0);
    AlignedBuffer ivs(want_iv ? chain.padded_n : 0);

    const auto t0 = std::chrono::steady_clock::now();
    if (want_greeks) {
        price_and_greeks_chain_avx2<Type>(chain, greeks);
    } else {
        price_chain_avx2<Type>(chain, prices.data());
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double price_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    double iv_us = 0.0;
    double max_iv_err = 0.0;
    std::size_t iv_failed = 0;
    if (want_iv) {
        const double* mkt = want_greeks ? greeks.price.data() : prices.data();
        const auto ti0 = std::chrono::steady_clock::now();
        implied_vol_chain_avx2<Type>(chain, mkt, ivs.data());
        const auto ti1 = std::chrono::steady_clock::now();
        iv_us = std::chrono::duration<double, std::micro>(ti1 - ti0).count();
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isfinite(ivs[i])) {
                max_iv_err = (std::max)(max_iv_err, std::fabs(ivs[i] - sigma));
            } else {
                ++iv_failed;
            }
        }
    }

    std::printf("black-scholes AVX2 chain  %s  N=%zu  S0=%.6g  r=%.6g  q=%.6g  sigma=%.6g  T=%.6g\n",
                Type == OptionType::Call ? "call" : "put", n, s0, r, q, sigma, T);
    std::printf("price%s: %.2f us (%.2f ns/option)\n",
                want_greeks ? "+5greeks" : "",
                price_us, (price_us * 1e3) / static_cast<double>(n));
    if (want_iv) {
        if (iv_failed == 0) {
            std::printf("iv_solve:      %.2f us (%.2f ns/option, max |iv - sigma| = %.3e)\n",
                        iv_us, (iv_us * 1e3) / static_cast<double>(n), max_iv_err);
        } else if (iv_failed == n) {
            std::printf("iv_solve:      %.2f us (%.2f ns/option, %zu/%zu failed/OOB -> nan)\n",
                        iv_us, (iv_us * 1e3) / static_cast<double>(n), iv_failed, n);
        } else {
            std::printf("iv_solve:      %.2f us (%.2f ns/option, max |iv - sigma| = %.3e, %zu/%zu failed/OOB)\n",
                        iv_us, (iv_us * 1e3) / static_cast<double>(n), max_iv_err, iv_failed, n);
        }
    }

    std::printf("\n%10s %14s", "strike", "price");
    if (want_greeks) {
        std::printf(" %11s %11s %11s %11s %11s", "delta", "gamma", "vega", "theta", "rho");
    }
    if (want_iv) {
        std::printf(" %14s", "implied_vol");
    }
    std::printf("\n");

    const std::size_t sample_count = (std::min)(n, std::size_t{5});
    for (std::size_t s = 0; s < sample_count; ++s) {
        const std::size_t idx = (sample_count > 1)
            ? (s * (n - 1)) / (sample_count - 1)
            : 0;
        const double p = want_greeks ? greeks.price[idx] : prices[idx];
        std::printf("%10.4f %14.8f", chain.strike[idx], p);
        if (want_greeks) {
            std::printf(" %11.6f %11.6f %11.6f %11.6f %11.6f",
                        greeks.delta[idx], greeks.gamma[idx], greeks.vega[idx],
                        greeks.theta[idx], greeks.rho[idx]);
        }
        if (want_iv) {
            std::printf(" %14.10f", ivs[idx]);
        }
        std::printf("\n");
    }
    return 0;
}

template <OptionType Type>
int run_single_iv_mode(double s0, double K, double r, double q, double T, double mkt_price) {
    if (!std::isfinite(s0) || s0 <= 0.0 || !std::isfinite(K) || K <= 0.0 ||
        !std::isfinite(r) || !std::isfinite(q) || !std::isfinite(T) || T <= 0.0 ||
        !std::isfinite(mkt_price)) {
        std::fprintf(stderr, "error: invalid inputs for implied volatility solve\n");
        return 2;
    }

    OptionChainSoA chain(1);
    chain.set(0, s0, K, r, q, 0.20, T);
    AlignedBuffer mkt(chain.padded_n);
    AlignedBuffer out_iv(chain.padded_n);
    mkt[0] = mkt_price;

    implied_vol_chain_avx2<Type>(chain, mkt.data(), out_iv.data());
    const double iv = out_iv[0];

    std::printf("black-scholes IV  %s  S0=%.6g  K=%.6g  r=%.6g  q=%.6g  T=%.6g  market_price=%.8f\n",
                Type == OptionType::Call ? "call" : "put", s0, K, r, q, T, mkt_price);
    if (!std::isfinite(iv)) {
        std::fprintf(stderr, "error: market price %.8f lies outside Black-Scholes no-arbitrage bounds\n",
                     mkt_price);
        return 2;
    }
    const double check_p = black_scholes_price<Type>(s0, K, r, q, iv, T);
    std::printf("\nimplied_vol  %.12f\n", iv);
    std::printf("bs_price(iv) %.12f  (residual %+.3e)\n", check_p, check_p - mkt_price);
    return 0;
}

void print_price_summary(double price, double std_error) {
    constexpr double kZ95 = 1.959963984540054;
    std::printf("\nprice      %.8f\n", price);
    std::printf("std_error  %.8f\n", std_error);
    std::printf("95%% CI     [%.8f, %.8f]\n", price - kZ95 * std_error, price + kZ95 * std_error);
}

void print_bs_reference(double mc_price, double mc_se, const GreeksResult& bs_ref) {
    std::printf("\nblack_scholes %.8f   diff %+.8f   (%+.2f se)\n",
                bs_ref.price, mc_price - bs_ref.price,
                mc_se > 0.0 ? (mc_price - bs_ref.price) / mc_se : 0.0);
}

template <typename Engine>
int run_mc_engine(Engine& eng, const SimConfig& cfg, const GreeksConfig& gcfg,
                  const HestonGreeksConfig& hcfg, bool want_greeks, bool want_ref,
                  const MarketParams* gbm_mp, double strike, OptionType opt_type) {
    const bool has_bs_ref = (gbm_mp != nullptr && want_ref);
    GreeksResult bs_ref{};
    if (has_bs_ref) {
        bs_ref = (opt_type == OptionType::Call)
            ? black_scholes_greeks<OptionType::Call>(gbm_mp->s0, strike, gbm_mp->r, gbm_mp->q, gbm_mp->sigma, gbm_mp->T)
            : black_scholes_greeks<OptionType::Put>(gbm_mp->s0, strike, gbm_mp->r, gbm_mp->q, gbm_mp->sigma, gbm_mp->T);
    }

    if (!want_greeks) {
        const auto res = eng.price(cfg);
        if (!res.has_value()) {
            const auto msg = to_string(res.error());
            std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
            return 2;
        }
        print_pricing_result(*res);
        if (has_bs_ref) {
            print_bs_reference(res->price, res->std_error, bs_ref);
        }
        return 0;
    }

    using Proc = typename Engine::process_type;
    using Pay = typename Engine::payoff_type;
    if constexpr (supports_gbm_5_greeks_v<Proc, Pay>) {
        const auto g = eng.price_with_greeks(cfg, gcfg);
        if (!g.has_value()) {
            const auto msg = to_string(g.error());
            std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
            return 2;
        }
        print_price_summary(g->price.value, g->price.std_error);
        if (has_bs_ref) {
            print_bs_reference(g->price.value, g->price.std_error, bs_ref);
        }
        print_gbm_greeks(*g, has_bs_ref, has_bs_ref ? &bs_ref : nullptr);
    } else if constexpr (supports_barrier_greeks_v<Proc, Pay>) {
        const auto g = eng.price_with_barrier_greeks(cfg, gcfg);
        if (!g.has_value()) {
            const auto msg = to_string(g.error());
            std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
            return 2;
        }
        print_price_summary(g->price.value, g->price.std_error);
        print_barrier_greeks(*g);
    } else if constexpr (supports_heston_greeks_v<Proc, Pay>) {
        const auto g = eng.price_with_heston_greeks(cfg, hcfg);
        if (!g.has_value()) {
            const auto msg = to_string(g.error());
            std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
            return 2;
        }
        print_price_summary(g->price.value, g->price.std_error);
        print_heston_greeks(*g);
    } else {
        std::fprintf(stderr, "error: Greeks are not supported for Heston x Barrier (discontinuous payoff)\n");
        return 2;
    }
    return 0;
}

template <ProcessPolicy Process, PayoffPolicy Payoff>
int dispatch_sampling(const Process& proc, const Payoff& payoff, ThreadPool* pool,
                      bool antithetic, const SimConfig& cfg, const GreeksConfig& gcfg,
                      const HestonGreeksConfig& hcfg, bool want_greeks, bool want_ref,
                      const MarketParams* gbm_mp, double strike, OptionType opt_type) {
    if (antithetic) {
        MonteCarloEngine<Process, Payoff, AntitheticSampling> eng(proc, payoff, pool);
        return run_mc_engine(eng, cfg, gcfg, hcfg, want_greeks, want_ref, gbm_mp, strike, opt_type);
    } else {
        MonteCarloEngine<Process, Payoff, StandardSampling> eng(proc, payoff, pool);
        return run_mc_engine(eng, cfg, gcfg, hcfg, want_greeks, want_ref, gbm_mp, strike, opt_type);
    }
}

template <ProcessPolicy Process, OptionType Type>
int dispatch_payoff(const Process& proc, PayoffKind pkind, double K,
                    bool has_barrier, double barrier_val, BarrierType btype,
                    LookbackStyle lb_style, ThreadPool* pool, bool antithetic,
                    const SimConfig& cfg, const GreeksConfig& gcfg,
                    const HestonGreeksConfig& hcfg, bool want_greeks, bool want_ref,
                    const MarketParams* gbm_mp) {
    switch (pkind) {
        case PayoffKind::European: {
            const auto p = EuropeanPayoff<Type>::create(K);
            if (!p.has_value()) {
                const auto msg = to_string(p.error());
                std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
                return 2;
            }
            return dispatch_sampling(proc, *p, pool, antithetic, cfg, gcfg, hcfg,
                                     want_greeks, want_ref, gbm_mp, K, Type);
        }
        case PayoffKind::AsianGeom: {
            const auto p = GeometricAsianPayoff<Type>::create(K);
            if (!p.has_value()) {
                const auto msg = to_string(p.error());
                std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
                return 2;
            }
            return dispatch_sampling(proc, *p, pool, antithetic, cfg, gcfg, hcfg,
                                     want_greeks, false, nullptr, K, Type);
        }
        case PayoffKind::AsianArith: {
            const auto p = ArithmeticAsianPayoff<Type>::create(K);
            if (!p.has_value()) {
                const auto msg = to_string(p.error());
                std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
                return 2;
            }
            return dispatch_sampling(proc, *p, pool, antithetic, cfg, gcfg, hcfg,
                                     want_greeks, false, nullptr, K, Type);
        }
        case PayoffKind::Barrier: {
            if (!has_barrier) {
                std::fprintf(stderr, "error: --payoff barrier requires --barrier <level>\n");
                return 2;
            }
            auto run_b = [&]<BarrierType BT>() -> int {
                const auto p = BarrierPayoff<Type, BT>::create(K, barrier_val, proc.initial_spot());
                if (!p.has_value()) {
                    const auto msg = to_string(p.error());
                    std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
                    return 2;
                }
                return dispatch_sampling(proc, *p, pool, antithetic, cfg, gcfg, hcfg,
                                         want_greeks, false, nullptr, K, Type);
            };
            switch (btype) {
                case BarrierType::UpOut:   return run_b.template operator()<BarrierType::UpOut>();
                case BarrierType::DownOut: return run_b.template operator()<BarrierType::DownOut>();
                case BarrierType::UpIn:    return run_b.template operator()<BarrierType::UpIn>();
                case BarrierType::DownIn:  return run_b.template operator()<BarrierType::DownIn>();
            }
            return 2;
        }
        case PayoffKind::Lookback: {
            if (lb_style == LookbackStyle::FixedStrike) {
                const auto p = LookbackPayoff<Type, LookbackStyle::FixedStrike>::create(K);
                if (!p.has_value()) {
                    const auto msg = to_string(p.error());
                    std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
                    return 2;
                }
                return dispatch_sampling(proc, *p, pool, antithetic, cfg, gcfg, hcfg,
                                         want_greeks, false, nullptr, K, Type);
            } else {
                const auto p = LookbackPayoff<Type, LookbackStyle::FloatingStrike>::create();
                if (!p.has_value()) {
                    const auto msg = to_string(p.error());
                    std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
                    return 2;
                }
                return dispatch_sampling(proc, *p, pool, antithetic, cfg, gcfg, hcfg,
                                         want_greeks, false, nullptr, K, Type);
            }
        }
    }
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    if (!pricer_has_avx2_fma()) {
        std::fprintf(stderr, "error: CPU does not support AVX2 + FMA (x86-64-v3 required)\n");
        return 1;
    }

    double s0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.0;
    double sigma = 0.20;
    double T = 1.0;
    double theta_h = 0.005;

    bool has_v0 = false;
    bool has_htheta = false;
    double v0 = 0.04;
    double kappa = 2.0;
    double htheta = 0.04;
    double xi = 0.30;
    double rho = -0.50;

    bool has_barrier = false;
    bool has_barrier_type = false;
    double barrier = 0.0;
    BarrierType barrier_type = BarrierType::UpOut;
    LookbackStyle lookback_style = LookbackStyle::FixedStrike;

    std::uint64_t paths = 1'000'000;
    std::uint64_t steps = 252;
    std::uint64_t seed = 42;
    std::uint64_t threads = 0;

    OptionType type = OptionType::Call;
    ModelKind model = ModelKind::GBM;
    PayoffKind payoff = PayoffKind::European;

    bool want_greeks = false;
    bool want_anti = false;
    bool want_ref = true;
    bool force_multi_step = false;

    bool want_chain = false;
    std::uint64_t chain_n = 0;
    bool want_iv = false;
    bool has_iv_price = false;
    double iv_market_price = 0.0;

    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--s0")      { if (!arg_double(argc, argv, i, s0)) return 2; }
        else if (a == "--k")       { if (!arg_double(argc, argv, i, K)) return 2; }
        else if (a == "--r")       { if (!arg_double(argc, argv, i, r)) return 2; }
        else if (a == "--q")       { if (!arg_double(argc, argv, i, q)) return 2; }
        else if (a == "--sigma")   { if (!arg_double(argc, argv, i, sigma)) return 2; }
        else if (a == "--T")       { if (!arg_double(argc, argv, i, T)) return 2; }
        else if (a == "--theta-h") { if (!arg_double(argc, argv, i, theta_h)) return 2; }
        else if (a == "--v0")      { if (!arg_double(argc, argv, i, v0)) return 2; has_v0 = true; }
        else if (a == "--kappa")   { if (!arg_double(argc, argv, i, kappa)) return 2; }
        else if (a == "--heston-theta") { if (!arg_double(argc, argv, i, htheta)) return 2; has_htheta = true; }
        else if (a == "--xi")      { if (!arg_double(argc, argv, i, xi)) return 2; }
        else if (a == "--rho")     { if (!arg_double(argc, argv, i, rho)) return 2; }
        else if (a == "--barrier") { if (!arg_double(argc, argv, i, barrier)) return 2; has_barrier = true; }
        else if (a == "--paths")   { if (!arg_u64(argc, argv, i, paths)) return 2; }
        else if (a == "--steps")   { if (!arg_u64(argc, argv, i, steps)) return 2; }
        else if (a == "--seed")    { if (!arg_u64(argc, argv, i, seed)) return 2; }
        else if (a == "--threads") { if (!arg_u64(argc, argv, i, threads)) return 2; }
        else if (a == "--chain")   { if (!arg_u64(argc, argv, i, chain_n)) return 2; want_chain = true; }
        else if (a == "--iv") {
            want_iv = true;
            if (i + 1 < argc) {
                char* end = nullptr;
                const double v = std::strtod(argv[i + 1], &end);
                if (end != argv[i + 1] && *end == '\0') {
                    iv_market_price = v;
                    has_iv_price = true;
                    ++i;
                }
            }
        }
        else if (a == "--greeks")           want_greeks = true;
        else if (a == "--antithetic")       want_anti = true;
        else if (a == "--no-reference")     want_ref = false;
        else if (a == "--force-multi-step") force_multi_step = true;
        else if (a == "--type") {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: --type needs a value\n"); return 2; }
            const std::string_view v(argv[++i]);
            if (v == "call") type = OptionType::Call;
            else if (v == "put") type = OptionType::Put;
            else { std::fprintf(stderr, "error: --type must be 'call' or 'put'\n"); return 2; }
        }
        else if (a == "--model") {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: --model needs a value\n"); return 2; }
            const std::string_view v(argv[++i]);
            if (v == "gbm") model = ModelKind::GBM;
            else if (v == "heston") model = ModelKind::Heston;
            else { std::fprintf(stderr, "error: --model must be 'gbm' or 'heston'\n"); return 2; }
        }
        else if (a == "--payoff") {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: --payoff needs a value\n"); return 2; }
            const std::string_view v(argv[++i]);
            if (v == "european")         payoff = PayoffKind::European;
            else if (v == "asian-geom")  payoff = PayoffKind::AsianGeom;
            else if (v == "asian-arith") payoff = PayoffKind::AsianArith;
            else if (v == "barrier")     payoff = PayoffKind::Barrier;
            else if (v == "lookback")    payoff = PayoffKind::Lookback;
            else {
                std::fprintf(stderr, "error: --payoff must be european|asian-geom|asian-arith|barrier|lookback\n");
                return 2;
            }
        }
        else if (a == "--barrier-type") {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: --barrier-type needs a value\n"); return 2; }
            const std::string_view v(argv[++i]);
            has_barrier_type = true;
            if (v == "up-out")        barrier_type = BarrierType::UpOut;
            else if (v == "down-out") barrier_type = BarrierType::DownOut;
            else if (v == "up-in")    barrier_type = BarrierType::UpIn;
            else if (v == "down-in")  barrier_type = BarrierType::DownIn;
            else {
                std::fprintf(stderr, "error: --barrier-type must be up-out|down-out|up-in|down-in\n");
                return 2;
            }
        }
        else if (a == "--lookback-style") {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: --lookback-style needs a value\n"); return 2; }
            const std::string_view v(argv[++i]);
            if (v == "fixed")         lookback_style = LookbackStyle::FixedStrike;
            else if (v == "floating") lookback_style = LookbackStyle::FloatingStrike;
            else {
                std::fprintf(stderr, "error: --lookback-style must be 'fixed' or 'floating'\n");
                return 2;
            }
        }
        else {
            std::fprintf(stderr, "error: unknown option %s\n", argv[i]);
            usage();
            return 2;
        }
    }

    if (want_chain) {
        return (type == OptionType::Call)
            ? run_chain_mode<OptionType::Call>(s0, r, q, sigma, T, static_cast<std::size_t>(chain_n), want_greeks, want_iv)
            : run_chain_mode<OptionType::Put>(s0, r, q, sigma, T, static_cast<std::size_t>(chain_n), want_greeks, want_iv);
    }

    if (want_iv) {
        if (!has_iv_price) {
            std::fprintf(stderr, "error: --iv requires a market price (or combine with --chain <n>)\n");
            return 2;
        }
        return (type == OptionType::Call)
            ? run_single_iv_mode<OptionType::Call>(s0, K, r, q, T, iv_market_price)
            : run_single_iv_mode<OptionType::Put>(s0, K, r, q, T, iv_market_price);
    }

    if (payoff == PayoffKind::Barrier && has_barrier && !has_barrier_type) {
        barrier_type = (barrier > s0) ? BarrierType::UpOut : BarrierType::DownOut;
    }

    // Validate before printing headers or spawning pool threads.
    if (want_greeks && model == ModelKind::Heston && payoff == PayoffKind::Barrier) {
        std::fprintf(stderr, "error: Greeks are not supported for Heston x Barrier (discontinuous payoff)\n");
        return 2;
    }

    const std::size_t active_threads = (threads == 0)
        ? physical_core_count()
        : static_cast<std::size_t>(threads);

    SimConfig cfg{};
    cfg.paths = static_cast<std::size_t>(paths);
    cfg.steps = static_cast<std::size_t>(steps);
    cfg.seed = seed;
    cfg.threads = active_threads;
    cfg.force_multi_step = force_multi_step;

    GreeksConfig gcfg{};
    gcfg.theta_h = theta_h;
    HestonGreeksConfig hcfg{};
    hcfg.theta_h = theta_h;

    const bool exact_1step = (model == ModelKind::GBM && payoff == PayoffKind::European && !force_multi_step);
    const std::size_t min_paths = want_anti ? 4 : 2;
    if (cfg.paths < min_paths) {
        std::fprintf(stderr, "error: path count must be >= %zu%s\n",
                     min_paths, want_anti ? " for antithetic variates" : "");
        return 2;
    }
    if (!exact_1step && cfg.steps < 1) {
        const auto msg = to_string(ValidationError::InvalidSteps);
        std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
        return 2;
    }
    if (want_greeks && (!std::isfinite(theta_h) || theta_h <= 0.0)) {
        const auto msg = to_string(ValidationError::InvalidBump);
        std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
        return 2;
    }

    if (payoff == PayoffKind::Barrier) {
        if (!has_barrier) {
            std::fprintf(stderr, "error: --payoff barrier requires --barrier <level>\n");
            return 2;
        }
        auto check_barrier = [&]<BarrierType BT>() -> std::expected<void, ValidationError> {
            const auto p = BarrierPayoff<OptionType::Call, BT>::create(K, barrier, s0);
            if (!p.has_value()) {
                return std::unexpected(p.error());
            }
            return {};
        };
        std::expected<void, ValidationError> b_ok{};
        switch (barrier_type) {
            case BarrierType::UpOut:   b_ok = check_barrier.template operator()<BarrierType::UpOut>(); break;
            case BarrierType::DownOut: b_ok = check_barrier.template operator()<BarrierType::DownOut>(); break;
            case BarrierType::UpIn:    b_ok = check_barrier.template operator()<BarrierType::UpIn>(); break;
            case BarrierType::DownIn:  b_ok = check_barrier.template operator()<BarrierType::DownIn>(); break;
        }
        if (!b_ok.has_value()) {
            const auto msg = to_string(b_ok.error());
            std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
            return 2;
        }
    } else if (!(payoff == PayoffKind::Lookback && lookback_style == LookbackStyle::FloatingStrike)) {
        if (const auto p = EuropeanPayoff<OptionType::Call>::create(K); !p.has_value()) {
            const auto msg = to_string(p.error());
            std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
            return 2;
        }
    }

    const std::size_t effective_paths = want_anti ? (cfg.paths / 2) * 2 : cfg.paths;
    const std::size_t reported_steps = exact_1step ? 1 : cfg.steps;

    if (model == ModelKind::GBM) {
        const auto mp = MarketParams::create(s0, r, sigma, T, q);
        if (!mp.has_value()) {
            const auto msg = to_string(mp.error());
            std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
            return 2;
        }
        ThreadPool pool(active_threads, true);
        GBMProcess proc(*mp, cfg.steps > 0 ? cfg.steps : 1);

        std::printf("%s  %s  %s  S0 %.6g  K %.6g  r %.6g  q %.6g  sigma %.6g  T %.6g",
                    type == OptionType::Call ? "call" : "put",
                    "gbm",
                    payoff_name(payoff, barrier_type, lookback_style),
                    s0, K, r, q, sigma, T);
        if (payoff == PayoffKind::Barrier && has_barrier) {
            std::printf("  B %.6g", barrier);
        }
        std::printf("\npaths %zu  steps/path %zu%s  threads %zu  seed %llu%s\n",
                    effective_paths, reported_steps,
                    exact_1step ? " (exact)" : "",
                    active_threads, static_cast<unsigned long long>(seed),
                    want_anti ? "  [antithetic]" : "");

        return (type == OptionType::Call)
            ? dispatch_payoff<GBMProcess, OptionType::Call>(
                  proc, payoff, K, has_barrier, barrier, barrier_type, lookback_style,
                  &pool, want_anti, cfg, gcfg, hcfg, want_greeks, want_ref, &(*mp))
            : dispatch_payoff<GBMProcess, OptionType::Put>(
                  proc, payoff, K, has_barrier, barrier, barrier_type, lookback_style,
                  &pool, want_anti, cfg, gcfg, hcfg, want_greeks, want_ref, &(*mp));
    } else {
        if (!has_v0) v0 = sigma * sigma;
        if (!has_htheta) htheta = sigma * sigma;
        const auto hp = HestonParams::create(s0, r, v0, kappa, htheta, xi, rho, T, q);
        if (!hp.has_value()) {
            const auto msg = to_string(hp.error());
            std::fprintf(stderr, "error: %.*s\n", static_cast<int>(msg.size()), msg.data());
            return 2;
        }
        ThreadPool pool(active_threads, true);
        HestonProcess proc(*hp, cfg.steps > 0 ? cfg.steps : 1);

        std::printf("%s  heston  %s  S0 %.6g  K %.6g  r %.6g  q %.6g  v0 %.6g  kappa %.6g  theta %.6g  xi %.6g  rho %.6g  T %.6g",
                    type == OptionType::Call ? "call" : "put",
                    payoff_name(payoff, barrier_type, lookback_style),
                    s0, K, r, q, v0, kappa, htheta, xi, rho, T);
        if (payoff == PayoffKind::Barrier && has_barrier) {
            std::printf("  B %.6g", barrier);
        }
        std::printf("\npaths %zu  steps/path %zu  threads %zu  seed %llu%s\n",
                    effective_paths, cfg.steps, active_threads,
                    static_cast<unsigned long long>(seed),
                    want_anti ? "  [antithetic]" : "");

        return (type == OptionType::Call)
            ? dispatch_payoff<HestonProcess, OptionType::Call>(
                  proc, payoff, K, has_barrier, barrier, barrier_type, lookback_style,
                  &pool, want_anti, cfg, gcfg, hcfg, want_greeks, false, nullptr)
            : dispatch_payoff<HestonProcess, OptionType::Put>(
                  proc, payoff, K, has_barrier, barrier, barrier_type, lookback_style,
                  &pool, want_anti, cfg, gcfg, hcfg, want_greeks, false, nullptr);
    }
}
