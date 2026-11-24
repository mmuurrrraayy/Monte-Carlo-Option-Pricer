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
#include "pricer/simd/xoshiro_avx2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#if defined(PRICER_USE_GOOGLE_BENCHMARK)
#include <benchmark/benchmark.h>
#endif

namespace {

using namespace pricer;

// Compiler barrier / sink to prevent dead-code elimination without fencing.
template <typename T>
PRICER_FORCE_INLINE void do_not_optimize(const T& value) noexcept {
#if defined(_MSC_VER)
    static volatile double sink = 0.0;
    sink = static_cast<double>(value);
    _ReadWriteBarrier();
#else
    asm volatile("" : : "r,m"(value) : "memory");
#endif
}

PRICER_FORCE_INLINE std::uint64_t read_tsc_end() noexcept {
    unsigned int aux = 0;
    return __rdtscp(&aux);
}

double calibrate_ns_per_cycle() {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t c0 = read_tsc_end();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.025) {
        _mm_pause();
    }
    const std::uint64_t c1 = read_tsc_end();
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / static_cast<double>(c1 - c0);
}

std::string cpu_brand() {
#if defined(_WIN32) && defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    char brand[49] = {0};
    __cpuid(regs, static_cast<int>(0x80000000u));
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return "unknown";
    for (int i = 0; i < 3; ++i) {
        __cpuid(regs, static_cast<int>(0x80000002u) + i);
        std::memcpy(brand + i * 16, regs, 16);
    }
    std::string s(brand);
#elif defined(__x86_64__) || defined(_M_X64)
    unsigned int regs[4] = {0, 0, 0, 0};
    char brand[49] = {0};
    asm volatile("cpuid"
                 : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                 : "a"(0x80000000u), "c"(0u));
    if (regs[0] < 0x80000004u) return "unknown";
    for (unsigned int i = 0; i < 3; ++i) {
        asm volatile("cpuid"
                     : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                     : "a"(0x80000002u + i), "c"(0u));
        std::memcpy(brand + i * 16, regs, 16);
    }
    std::string s(brand);
#else
    std::string s("unknown");
#endif
    while (!s.empty() && s.back() == ' ') s.pop_back();
    const auto b = s.find_first_not_of(' ');
    return b == std::string::npos ? s : s.substr(b);
}

struct LatencyStats {
    double min_ns = 0.0;
    double max_ns = 0.0;
    double p50_ns = 0.0;
    double p95_ns = 0.0;
    double p99_ns = 0.0;
    double p999_ns = 0.0;
    double ns_per_item = 0.0;
    double m_items_per_sec = 0.0;
};

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double idx = p * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(idx);
    const auto hi = (std::min)(lo + 1, sorted.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

LatencyStats summarize_ns(std::vector<double> samples_ns, std::size_t items_per_call) {
    std::sort(samples_ns.begin(), samples_ns.end());
    LatencyStats st{};
    st.min_ns = samples_ns.front();
    st.max_ns = samples_ns.back();
    st.p50_ns = percentile(samples_ns, 0.50);
    st.p95_ns = percentile(samples_ns, 0.95);
    st.p99_ns = percentile(samples_ns, 0.99);
    st.p999_ns = percentile(samples_ns, 0.999);
    st.ns_per_item = st.p50_ns / static_cast<double>(items_per_call);
    st.m_items_per_sec = 1e3 / st.ns_per_item;
    return st;
}

template <typename Fn>
LatencyStats measure_tsc(std::size_t items_per_call, int warmup, int samples,
                         double ns_per_cycle, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> call_ns;
    call_ns.reserve(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const std::uint64_t c0 = read_tsc_end();
        fn();
        const std::uint64_t c1 = read_tsc_end();
        call_ns.push_back(static_cast<double>(c1 - c0) * ns_per_cycle);
    }
    return summarize_ns(std::move(call_ns), items_per_call);
}

template <typename Fn>
double measure_once_ns(Fn&& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

void bench_black_scholes_chain(double ns_per_cycle) {
    std::printf("=== [1] Black-Scholes OptionChainSoA & AVX2 IV Solver ===\n");
    std::printf("%-24s %7s %9s %9s %10s %10s %10s %10s\n",
                "Kernel", "N", "ns/opt", "M opt/s", "p50 (us)", "p95 (us)", "p99 (us)", "p99.9(us)");

    constexpr std::size_t sizes[] = {64, 1024, 16384};
    for (std::size_t N : sizes) {
        OptionChainSoA chain(N);
        ChainGreeksSoA greeks(N);
        AlignedBuffer prices(chain.padded_n);
        AlignedBuffer ivs(chain.padded_n);

        Xoshiro256PlusPlus rng(0xB5C4A1ULL ^ static_cast<std::uint64_t>(N));
        for (std::size_t i = 0; i < N; ++i) {
            const double S = 100.0;
            const double K = 70.0 + 60.0 * rng.next_uniform_open01();
            const double r = -0.01 + 0.08 * rng.next_uniform_open01();
            const double q = 0.04 * rng.next_uniform_open01();
            const double sigma = 0.10 + 0.50 * rng.next_uniform_open01();
            const double tau = 0.10 + 2.00 * rng.next_uniform_open01();
            chain.set(i, S, K, r, q, sigma, tau);
        }
        price_chain_avx2<OptionType::Call>(chain, prices);

        const int warmup = (N <= 1024) ? 500 : 100;
        const int samples = (N <= 1024) ? 2000 : 500;

        for (int w = 0; w < 20; ++w) {
            price_chain_avx2<OptionType::Call>(chain, prices);
            price_and_greeks_chain_avx2<OptionType::Call>(chain, greeks);
            implied_vol_chain_avx2<OptionType::Call>(chain, prices.data(), ivs.data());
        }

        const auto st_price = measure_tsc(N, warmup, samples, ns_per_cycle, [&]() {
            price_chain_avx2<OptionType::Call>(chain, prices);
            do_not_optimize(prices[N - 1]);
        });
        const auto st_greeks = measure_tsc(N, warmup, samples, ns_per_cycle, [&]() {
            price_and_greeks_chain_avx2<OptionType::Call>(chain, greeks);
            do_not_optimize(greeks.price[N - 1] + greeks.theta[N - 1]);
        });
        const auto st_iv = measure_tsc(N, warmup / 2, samples / 2, ns_per_cycle, [&]() {
            implied_vol_chain_avx2<OptionType::Call>(chain, prices.data(), ivs.data());
            do_not_optimize(ivs[N - 1]);
        });

        auto print_row = [&](const char* name, const LatencyStats& st) {
            std::printf("%-24s %7zu %9.2f %9.2f %10.2f %10.2f %10.2f %10.2f\n",
                        name, N, st.ns_per_item, st.m_items_per_sec,
                        st.p50_ns * 1e-3, st.p95_ns * 1e-3, st.p99_ns * 1e-3, st.p999_ns * 1e-3);
        };
        print_row("BS price (AVX2 SoA)", st_price);
        print_row("BS price+5G (AVX2 SoA)", st_greeks);
        print_row("BS IV Halley (AVX2 SoA)", st_iv);
        std::printf("\n");
    }
}

void bench_european_gbm(ThreadPool& pool, std::size_t phys_cores) {
    std::printf("=== [2] European GBM Monte Carlo (1T vs %zuT Pinned) ===\n", phys_cores);
    std::printf("Params: Call S0=100 K=100 r=0.05 q=0.0 sigma=0.20 T=1.0\n");
    std::printf("Protocol: round-robin warm-up (2 rounds) + 9 measured rounds\n\n");

    const auto mp = *MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.0);
    const auto payoff = *EuropeanPayoff<OptionType::Call>::create(100.0);

    MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
        eng_std(GBMProcess(mp, 252), payoff, &pool);
    MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
        eng_anti(GBMProcess(mp, 252), payoff, &pool);

    std::printf("%-30s %9s %4s %9s %9s %9s %9s %10s\n",
                "Workload", "Paths", "Thr", "p50 (ms)", "p95 (ms)", "p99 (ms)", "ns/path", "M paths/s");

    auto run_pair_1t_mt = [&](const char* label, std::size_t paths, std::size_t steps,
                              bool force_ms, bool anti, bool greeks, int rounds) {
        const std::size_t thrs[2] = {1, phys_cores};
        std::vector<double> samples[2];
        samples[0].reserve(static_cast<std::size_t>(rounds));
        samples[1].reserve(static_cast<std::size_t>(rounds));
        constexpr int kWarmup = 2;

        for (int r = 0; r < kWarmup + rounds; ++r) {
            for (int ti = 0; ti < 2; ++ti) {
                SimConfig cfg{
                    .paths = paths,
                    .steps = steps,
                    .seed = 42,
                    .threads = thrs[ti],
                    .force_multi_step = force_ms,
                };
                const double ns = measure_once_ns([&]() {
                    if (greeks) {
                        const auto res = anti ? eng_anti.price_with_greeks(cfg)
                                              : eng_std.price_with_greeks(cfg);
                        do_not_optimize(res->price.value + res->gamma.value + res->theta.value);
                    } else {
                        const auto res = anti ? eng_anti.price(cfg) : eng_std.price(cfg);
                        do_not_optimize(res->price);
                    }
                });
                if (r >= kWarmup) {
                    samples[ti].push_back(ns);
                }
            }
        }

        for (int ti = 0; ti < 2; ++ti) {
            const LatencyStats st = summarize_ns(std::move(samples[ti]), paths);
            std::printf("%-30s %9zu %3zuT %9.3f %9.3f %9.3f %9.2f %10.2f\n",
                        label, paths, thrs[ti],
                        st.p50_ns * 1e-6, st.p95_ns * 1e-6, st.p99_ns * 1e-6,
                        st.ns_per_item, st.m_items_per_sec);
        }
    };

    for (std::size_t N : {std::size_t{100'000}, std::size_t{1'000'000}, std::size_t{10'000'000}}) {
        const int rounds = (N <= 1'000'000) ? 15 : 9;
        run_pair_1t_mt("GBM Euro 1-step Plain", N, 1, false, false, false, rounds);
        run_pair_1t_mt("GBM Euro 1-step Antithetic", N, 1, false, true, false, rounds);
    }
    run_pair_1t_mt("GBM Euro 1-step 5-Greeks", 100'000, 1, false, false, true, 15);
    run_pair_1t_mt("GBM Euro 1-step 5-Greeks", 250'000, 1, false, false, true, 15);
    run_pair_1t_mt("GBM Euro 1-step 5-Greeks", 1'000'000, 1, false, false, true, 9);
    std::printf("\n");

    for (std::size_t N : {std::size_t{100'000}, std::size_t{250'000}, std::size_t{1'000'000}}) {
        const int rounds = (N <= 250'000) ? 9 : 5;
        run_pair_1t_mt("GBM Euro 252-step Plain", N, 252, true, false, false, rounds);
        run_pair_1t_mt("GBM Euro 252-step Antithetic", N, 252, true, true, false, rounds);
    }
    run_pair_1t_mt("GBM Euro 252-step 5-Greeks", 100'000, 252, true, false, true, 5);
    run_pair_1t_mt("GBM Euro 252-step 5-Greeks", 250'000, 252, true, false, true, 5);
    std::printf("\n");
}

void bench_thread_scaling(ThreadPool& pool, std::size_t phys_cores) {
    std::printf("=== [3] Multi-Core Scaling Sweep & Bitwise Reproducibility (N = 1,000,000 x 252 steps) ===\n");

    std::vector<std::size_t> thread_counts = {1, 2, 4};
    if (phys_cores > 4) {
        thread_counts.push_back(phys_cores);
    }

    const std::size_t N = 1'000'000;
    const std::size_t steps = 252;
    constexpr int kWarmup = 1;
    constexpr int kRounds = 5;

    auto run_sweep = [&](const char* title, auto& eng, bool force_ms) {
        std::printf("\n-- %s --\n", title);
        std::vector<std::vector<double>> times(thread_counts.size());
        std::vector<PricingResult> results(thread_counts.size());

        // Interleave thread counts each round to reduce thermal drift bias.
        for (int r = 0; r < kWarmup + kRounds; ++r) {
            for (std::size_t ti = 0; ti < thread_counts.size(); ++ti) {
                SimConfig cfg{
                    .paths = N,
                    .steps = steps,
                    .seed = 42,
                    .threads = thread_counts[ti],
                    .force_multi_step = force_ms,
                };
                PricingResult res{};
                const double ns = measure_once_ns([&]() {
                    res = *eng.price(cfg);
                    do_not_optimize(res.price);
                });
                results[ti] = res;
                if (r >= kWarmup) {
                    times[ti].push_back(ns);
                }
            }
        }

        double t1_ms = 0.0;
        std::printf("%7s %10s %9s %10s %10s %16s %10s\n",
                    "Threads", "p50 (ms)", "Speedup", "Eff.", "Spread %", "Price", "BitwiseEq");
        for (std::size_t ti = 0; ti < thread_counts.size(); ++ti) {
            const LatencyStats st = summarize_ns(times[ti], N);
            const double med_ms = st.p50_ns * 1e-6;
            if (ti == 0) t1_ms = med_ms;
            const double speedup = t1_ms / med_ms;
            const double eff = speedup / static_cast<double>(thread_counts[ti]);
            const double spread_pct = 100.0 * (st.max_ns - st.min_ns) / st.p50_ns;
            const bool eq = (std::memcmp(&results[ti].price, &results[0].price, sizeof(double)) == 0)
                         && (std::memcmp(&results[ti].std_error, &results[0].std_error, sizeof(double)) == 0);
            std::printf("%7zu %10.3f %8.2fx %10.3f %9.2f%% %16.12f %10s\n",
                        thread_counts[ti], med_ms, speedup, eff, spread_pct,
                        results[ti].price, eq ? "YES" : "NO");
        }
    };

    const auto mp = *MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.0);
    const auto hp = *HestonParams::create(100.0, 0.0319, 0.010201, 6.21, 0.019, 0.61, -0.7, 1.0, 0.0);
    const auto payoff = *EuropeanPayoff<OptionType::Call>::create(100.0);

    MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
        eng_gbm(GBMProcess(mp, steps), payoff, &pool);
    MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
        eng_heston(HestonProcess(hp, steps), payoff, &pool);

    run_sweep("GBM 252-step Plain (N=1,000,000)", eng_gbm, true);
    run_sweep("Heston Full Truncation Euler 252-step Plain (N=1,000,000)", eng_heston, false);
    std::printf("\n");
}

void bench_antithetic_sweep(ThreadPool& pool, std::size_t phys_cores) {
    std::printf("=== [4] Equal-Work Antithetic Strike Sweep (250,000 pairs = 500,000 paths, %zuT) ===\n",
                phys_cores);

    const auto mp = *MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.0);
    constexpr std::size_t kPairs = 250'000;
    constexpr std::size_t kPaths = 2 * kPairs;
    constexpr int kWarmup = 2;
    constexpr int kRounds = 7;

    auto run_table = [&](const char* title, bool force_ms) {
        std::printf("\n-- %s --\n", title);
        std::printf("%6s %10s %10s %10s %10s %10s %10s\n",
                    "Strike", "SE plain", "SE anti", "var_ratio", "imp. rho", "time_ratio", "Efficiency");

        for (double K : {60.0, 80.0, 100.0, 120.0, 140.0}) {
            const auto pay = *EuropeanPayoff<OptionType::Call>::create(K);
            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
                eng_p(GBMProcess(mp, 252), pay, &pool);
            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                eng_a(GBMProcess(mp, 252), pay, &pool);

            SimConfig cfg{
                .paths = kPaths,
                .steps = force_ms ? std::size_t{252} : std::size_t{1},
                .seed = 42,
                .threads = phys_cores,
                .force_multi_step = force_ms,
            };

            std::vector<double> tp, ta;
            PricingResult rp{}, ra{};
            for (int r = 0; r < kWarmup + kRounds; ++r) {
                const double ns_p = measure_once_ns([&]() {
                    rp = *eng_p.price(cfg);
                    do_not_optimize(rp.price);
                });
                const double ns_a = measure_once_ns([&]() {
                    ra = *eng_a.price(cfg);
                    do_not_optimize(ra.price);
                });
                if (r >= kWarmup) {
                    tp.push_back(ns_p);
                    ta.push_back(ns_a);
                }
            }

            const double med_p = summarize_ns(std::move(tp), kPaths).p50_ns;
            const double med_a = summarize_ns(std::move(ta), kPaths).p50_ns;
            const double var_ratio = (rp.std_error / ra.std_error) * (rp.std_error / ra.std_error);
            const double implied_rho = (1.0 / var_ratio) - 1.0;
            const double time_ratio = med_a / med_p;
            const double eff = var_ratio * (med_p / med_a);
            std::printf("%6.0f %10.6f %10.6f %10.2f %10.3f %10.2f %9.2fx\n",
                        K, rp.std_error, ra.std_error, var_ratio, implied_rho, time_ratio, eff);
        }
    };

    run_table("Exact 1-step European GBM", false);
    run_table("252-step European GBM", true);
    std::printf("\n");
}

void bench_exotics(ThreadPool& pool, std::size_t phys_cores) {
    std::printf("=== [5] Path-Dependent Exotics under GBM (100,000 paths x 252 steps) ===\n");
    std::printf("%-26s %4s %10s %9s %9s %9s %9s %10s %8s\n",
                "Exotic", "Thr", "Price", "SE", "p50 (ms)", "p99 (ms)", "ns/path", "M paths/s", "Speedup");

    const auto mp = *MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.0);
    constexpr std::size_t N = 100'000;
    constexpr std::size_t steps = 252;
    constexpr int kWarmup = 2;
    constexpr int kRounds = 7;

    auto bench_one = [&](const char* name, auto& eng) {
        const std::size_t thrs[2] = {1, phys_cores};
        std::vector<double> samples[2];
        PricingResult res[2]{};
        for (int r = 0; r < kWarmup + kRounds; ++r) {
            for (int ti = 0; ti < 2; ++ti) {
                SimConfig cfg{.paths = N, .steps = steps, .seed = 42, .threads = thrs[ti]};
                const double ns = measure_once_ns([&]() {
                    res[ti] = *eng.price(cfg);
                    do_not_optimize(res[ti].price);
                });
                if (r >= kWarmup) {
                    samples[ti].push_back(ns);
                }
            }
        }
        const LatencyStats st1 = summarize_ns(std::move(samples[0]), N);
        const LatencyStats stm = summarize_ns(std::move(samples[1]), N);
        std::printf("%-26s %3zuT %10.5f %9.5f %9.3f %9.3f %9.1f %10.3f %7.2fx\n",
                    name, thrs[0], res[0].price, res[0].std_error,
                    st1.p50_ns * 1e-6, st1.p99_ns * 1e-6, st1.ns_per_item, st1.m_items_per_sec, 1.0);
        std::printf("%-26s %3zuT %10.5f %9.5f %9.3f %9.3f %9.1f %10.3f %7.2fx\n",
                    name, thrs[1], res[1].price, res[1].std_error,
                    stm.p50_ns * 1e-6, stm.p99_ns * 1e-6, stm.ns_per_item, stm.m_items_per_sec,
                    st1.p50_ns / stm.p50_ns);
    };

    MonteCarloEngine<GBMProcess, GeometricAsianPayoff<OptionType::Call>, StandardSampling>
        eng_ga(GBMProcess(mp, steps), *GeometricAsianPayoff<OptionType::Call>::create(100.0), &pool);
    MonteCarloEngine<GBMProcess, ArithmeticAsianPayoff<OptionType::Call>, StandardSampling>
        eng_aa(GBMProcess(mp, steps), *ArithmeticAsianPayoff<OptionType::Call>::create(100.0), &pool);
    MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>, StandardSampling>
        eng_bar(GBMProcess(mp, steps), *BarrierPayoff<OptionType::Call, BarrierType::UpOut>::create(100.0, 130.0, 100.0), &pool);
    MonteCarloEngine<GBMProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FixedStrike>, StandardSampling>
        eng_lbfix(GBMProcess(mp, steps), *LookbackPayoff<OptionType::Call, LookbackStyle::FixedStrike>::create(100.0), &pool);
    MonteCarloEngine<GBMProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>, StandardSampling>
        eng_lbflt(GBMProcess(mp, steps), *LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>::create(), &pool);

    bench_one("Geometric Asian Call", eng_ga);
    bench_one("Arithmetic Asian Call", eng_aa);
    bench_one("Barrier UpOut Call (B=130)", eng_bar);
    bench_one("Lookback Fixed Call", eng_lbfix);
    bench_one("Lookback Floating Call", eng_lbflt);
    std::printf("\n");
}

void bench_heston(ThreadPool& pool, std::size_t phys_cores) {
    std::printf("=== [6] Heston Stochastic Volatility (Full Truncation Euler, 252 steps) ===\n");
    std::printf("Params: Broadie-Kaya Case I (S0=K=100 r=0.0319 v0=0.010201 kappa=6.21 theta=0.019 xi=0.61 rho=-0.7 T=1)\n");
    std::printf("%-26s %9s %4s %10s %9s %9s %9s %9s %10s %8s\n",
                "Mode", "Paths", "Thr", "Price", "SE", "p50 (ms)", "p99 (ms)", "ns/path", "M paths/s", "Speedup");

    const auto hp = *HestonParams::create(100.0, 0.0319, 0.010201, 6.21, 0.019, 0.61, -0.7, 1.0, 0.0);
    const auto pay = *EuropeanPayoff<OptionType::Call>::create(100.0);
    constexpr std::size_t steps = 252;

    MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
        eng_std(HestonProcess(hp, steps), pay, &pool);
    MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
        eng_anti(HestonProcess(hp, steps), pay, &pool);

    auto run_heston = [&](const char* name, auto& eng, std::size_t N, int rounds) {
        const std::size_t thrs[2] = {1, phys_cores};
        std::vector<double> samples[2];
        PricingResult res[2]{};
        constexpr int kWarmup = 1;
        for (int r = 0; r < kWarmup + rounds; ++r) {
            for (int ti = 0; ti < 2; ++ti) {
                SimConfig cfg{.paths = N, .steps = steps, .seed = 42, .threads = thrs[ti]};
                const double ns = measure_once_ns([&]() {
                    res[ti] = *eng.price(cfg);
                    do_not_optimize(res[ti].price);
                });
                if (r >= kWarmup) {
                    samples[ti].push_back(ns);
                }
            }
        }
        const LatencyStats st1 = summarize_ns(std::move(samples[0]), N);
        const LatencyStats stm = summarize_ns(std::move(samples[1]), N);
        std::printf("%-26s %9zu %3zuT %10.5f %9.5f %9.3f %9.3f %9.1f %10.3f %7.2fx\n",
                    name, N, thrs[0], res[0].price, res[0].std_error,
                    st1.p50_ns * 1e-6, st1.p99_ns * 1e-6, st1.ns_per_item, st1.m_items_per_sec, 1.0);
        std::printf("%-26s %9zu %3zuT %10.5f %9.5f %9.3f %9.3f %9.1f %10.3f %7.2fx\n",
                    name, N, thrs[1], res[1].price, res[1].std_error,
                    stm.p50_ns * 1e-6, stm.p99_ns * 1e-6, stm.ns_per_item, stm.m_items_per_sec,
                    st1.p50_ns / stm.p50_ns);
    };

    run_heston("Heston Plain", eng_std, 100'000, 7);
    run_heston("Heston Antithetic", eng_anti, 100'000, 7);
    run_heston("Heston Plain", eng_std, 1'000'000, 5);
    run_heston("Heston Antithetic", eng_anti, 1'000'000, 5);
    std::printf("\n");
}

void bench_convergence(ThreadPool& pool, std::size_t phys_cores) {
    std::printf("=== [7] Convergence & Empirical N^-0.5 SE Scaling (24 independent replications) ===\n");
    const auto mp = *MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.0);
    const auto pay = *EuropeanPayoff<OptionType::Call>::create(100.0);
    const double bs_ref = black_scholes_price<OptionType::Call>(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
    std::printf("Black-Scholes closed form: %.10f\n", bs_ref);
    std::printf("%10s %14s %14s %14s %14s\n",
                "N", "Mean MC price", "Bias vs BS", "Observed SE", "Engine SE");

    MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
        eng(GBMProcess(mp, 1), pay, &pool);

    constexpr int kReps = 24;
    const std::size_t ns[] = {1'000, 10'000, 100'000, 1'000'000};
    std::vector<double> log_n, log_se;

    for (std::size_t N : ns) {
        std::vector<double> prices;
        prices.reserve(kReps);
        double single_se = 0.0;
        for (int rep = 0; rep < kReps; ++rep) {
            SimConfig cfg{
                .paths = N,
                .steps = 1,
                .seed = 1000ULL + static_cast<std::uint64_t>(rep) * 7919ULL + static_cast<std::uint64_t>(N),
                .threads = phys_cores,
            };
            const auto res = *eng.price(cfg);
            prices.push_back(res.price);
            if (rep == 0) single_se = res.std_error;
        }
        double mean = 0.0;
        for (double p : prices) mean += p;
        mean /= static_cast<double>(kReps);
        double m2 = 0.0;
        for (double p : prices) m2 += (p - mean) * (p - mean);
        const double obs_se = std::sqrt(m2 / static_cast<double>(kReps - 1));

        log_n.push_back(std::log(static_cast<double>(N)));
        log_se.push_back(std::log(obs_se));

        std::printf("%10zu %14.7f %+14.7f %14.7f %14.7f\n",
                    N, mean, mean - bs_ref, obs_se, single_se);
    }

    const double k = static_cast<double>(log_n.size());
    double mx = 0.0, my = 0.0;
    for (std::size_t i = 0; i < log_n.size(); ++i) {
        mx += log_n[i];
        my += log_se[i];
    }
    mx /= k;
    my /= k;
    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (std::size_t i = 0; i < log_n.size(); ++i) {
        const double dx = log_n[i] - mx;
        const double dy = log_se[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    const double slope = sxy / sxx;
    const double r2 = (sxy * sxy) / (sxx * syy);
    std::printf("OLS log(SE) vs log(N): slope = %.4f (theory -0.5000), R^2 = %.6f\n\n", slope, r2);
}

} // namespace

#if defined(PRICER_USE_GOOGLE_BENCHMARK)

static void BM_BS_PriceChain(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    OptionChainSoA chain(N);
    AlignedBuffer prices(chain.padded_n);
    for (std::size_t i = 0; i < N; ++i) {
        chain.set(i, 100.0, 70.0 + 60.0 * static_cast<double>(i) / static_cast<double>(N),
                  0.05, 0.01, 0.20, 1.0);
    }
    for (auto _ : state) {
        price_chain_avx2<OptionType::Call>(chain, prices.data());
        benchmark::DoNotOptimize(prices[N - 1]);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}
BENCHMARK(BM_BS_PriceChain)->Arg(64)->Arg(1024)->Arg(16384);

static void BM_BS_GreeksChain(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    OptionChainSoA chain(N);
    ChainGreeksSoA greeks(N);
    for (std::size_t i = 0; i < N; ++i) {
        chain.set(i, 100.0, 70.0 + 60.0 * static_cast<double>(i) / static_cast<double>(N),
                  0.05, 0.01, 0.20, 1.0);
    }
    for (auto _ : state) {
        price_and_greeks_chain_avx2<OptionType::Call>(chain, greeks);
        benchmark::DoNotOptimize(greeks.price[N - 1]);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}
BENCHMARK(BM_BS_GreeksChain)->Arg(64)->Arg(1024)->Arg(16384);

static void BM_BS_IVChain(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    OptionChainSoA chain(N);
    AlignedBuffer prices(chain.padded_n);
    AlignedBuffer ivs(chain.padded_n);
    for (std::size_t i = 0; i < N; ++i) {
        chain.set(i, 100.0, 70.0 + 60.0 * static_cast<double>(i) / static_cast<double>(N),
                  0.05, 0.01, 0.20, 1.0);
    }
    price_chain_avx2<OptionType::Call>(chain, prices.data());
    for (auto _ : state) {
        implied_vol_chain_avx2<OptionType::Call>(chain, prices.data(), ivs.data());
        benchmark::DoNotOptimize(ivs[N - 1]);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}
BENCHMARK(BM_BS_IVChain)->Arg(64)->Arg(1024)->Arg(16384);

static void BM_MC_GBM_Euro1Step(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    const auto thr = static_cast<std::size_t>(state.range(1));
    ThreadPool pool(thr, true);
    const auto mp = *MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.0);
    MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
        eng(GBMProcess(mp, 1), EuropeanPayoff<OptionType::Call>(100.0), &pool);
    SimConfig cfg{.paths = N, .steps = 1, .seed = 42, .threads = thr};
    for (auto _ : state) {
        auto r = eng.price(cfg);
        benchmark::DoNotOptimize(r->price);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}
BENCHMARK(BM_MC_GBM_Euro1Step)->Args({100000, 1})->Args({1000000, 6});

static void BM_MC_Heston252(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    const auto thr = static_cast<std::size_t>(state.range(1));
    ThreadPool pool(thr, true);
    const auto hp = *HestonParams::create(100.0, 0.0319, 0.010201, 6.21, 0.019, 0.61, -0.7, 1.0, 0.0);
    MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
        eng(HestonProcess(hp, 252), EuropeanPayoff<OptionType::Call>(100.0), &pool);
    SimConfig cfg{.paths = N, .steps = 252, .seed = 42, .threads = thr};
    for (auto _ : state) {
        auto r = eng.price(cfg);
        benchmark::DoNotOptimize(r->price);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}
BENCHMARK(BM_MC_Heston252)->Args({100000, 1})->Args({100000, 6});

int main(int argc, char** argv) {
    if (!pricer_has_avx2_fma()) {
        std::fprintf(stderr, "error: CPU does not support AVX2 + FMA\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}

#else

int main(int argc, char** argv) {
    if (!pricer_has_avx2_fma()) {
        std::fprintf(stderr, "error: CPU does not support AVX2 + FMA\n");
        return 1;
    }

    std::string_view section = "all";
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        if (a == "--section" && i + 1 < argc) {
            section = argv[++i];
        } else if (a == "-h" || a == "--help") {
            std::printf("usage: bench_pricer [--section all|bs|euro|scaling|anti|exotic|heston|converge]\n");
            return 0;
        }
    }

    const double ns_per_cycle = calibrate_ns_per_cycle();
    const std::size_t phys_cores = physical_core_count();
    const unsigned logical = std::thread::hardware_concurrency();

    std::printf("=== C++23 / AVX2 `pricer` Benchmark Harness ===\n");
    std::printf("CPU:            %s\n", cpu_brand().c_str());
    std::printf("Physical cores: %zu (logical: %u)\n", phys_cores, logical);
    std::printf("TSC calibrated: %.4f GHz (%.5f ns/cycle)\n\n",
                1.0 / ns_per_cycle, ns_per_cycle);

    ThreadPool pool(phys_cores, true);

    if (section == "all" || section == "bs")      bench_black_scholes_chain(ns_per_cycle);
    if (section == "all" || section == "euro")    bench_european_gbm(pool, phys_cores);
    if (section == "all" || section == "scaling") bench_thread_scaling(pool, phys_cores);
    if (section == "all" || section == "anti")    bench_antithetic_sweep(pool, phys_cores);
    if (section == "all" || section == "exotic")  bench_exotics(pool, phys_cores);
    if (section == "all" || section == "heston")  bench_heston(pool, phys_cores);
    if (section == "all" || section == "converge") bench_convergence(pool, phys_cores);

    return 0;
}

#endif
