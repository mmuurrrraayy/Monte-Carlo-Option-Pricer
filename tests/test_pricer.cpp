#include "check.h"
#include "cpuid_guard.hpp"

#include "pricer/core/concepts.hpp"
#include "pricer/core/dual.hpp"
#include "pricer/core/stats.hpp"
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
#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"
#include "pricer/simd/xoshiro_avx2.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>
#include <vector>

namespace {

using namespace pricer;
using pricer::OptionType;

// Maps IEEE-754 sign-magnitude bits to monotonic unsigned order for ULP distance.
std::uint64_t ulp_distance(double a, double b) noexcept {
    if (a == b) return 0;
    auto to_ordered = [](double x) noexcept -> std::uint64_t {
        const auto bits = std::bit_cast<std::uint64_t>(x);
        return (bits & (1ULL << 63)) ? (~bits + 1ULL) : (bits | (1ULL << 63));
    };
    const std::uint64_t oa = to_ordered(a);
    const std::uint64_t ob = to_ordered(b);
    return oa > ob ? (oa - ob) : (ob - oa);
}

void test_toolchain_and_cpuid() {
    std::printf("== Toolchain & CPUID guard ==\n");

    check::is_true("pricer_has_avx2_fma", pricer_has_avx2_fma());

#if defined(__AVX2__)
    constexpr bool has_avx2_macro = true;
#else
    constexpr bool has_avx2_macro = false;
#endif
    check::is_true("compiled_with_avx2", has_avx2_macro);

#if defined(_MSC_VER)
    constexpr long cxx_std = _MSVC_LANG;
#else
    constexpr long cxx_std = __cplusplus;
#endif
    check::is_true("cxx23_or_newer", cxx_std >= 202302L);

#if defined(PRICER_FAST_TESTS)
    std::printf("note: PRICER_FAST_TESTS is enabled\n");
#endif
}

struct MockGBM {
    static constexpr bool supports_exact_terminal = true;
    static constexpr bool is_gbm = true;
    double initial_spot() const noexcept { return 100.0; }
    double maturity() const noexcept { return 1.0; }
    double discount_factor() const noexcept { return 0.95; }
    Vec4d terminal_spot(Vec4d z) const noexcept { return z; }
    Vec4d step_log(Vec4d log_s, Vec4d z) const noexcept { return log_s + z; }
};

struct MockHeston {
    static constexpr bool supports_exact_terminal = false;
    static constexpr bool is_gbm = false;
    double initial_spot() const noexcept { return 100.0; }
    double maturity() const noexcept { return 1.0; }
    double discount_factor() const noexcept { return 0.95; }
};

struct MockEuropean {
    static constexpr bool is_path_dependent = false;
    static constexpr bool needs_spot_per_step = false;
    static constexpr bool is_smooth = true;
    Vec4d evaluate(Vec4d st) const noexcept { return st; }
};

struct MockAsian {
    static constexpr bool is_path_dependent = true;
    static constexpr bool needs_spot_per_step = false;
    static constexpr bool is_smooth = true;
    static constexpr bool is_geometric_asian = true;
    template <typename T> struct State { T acc; };
    State<Vec4d> init_state(Vec4d) const noexcept { return {Vec4d::zero()}; }
    void update(State<Vec4d>& st, Vec4d log_s) const noexcept { st.acc += log_s; }
    Vec4d evaluate(const State<Vec4d>& st, Vec4d, double inv_steps) const noexcept {
        return st.acc * inv_steps;
    }
};

struct MockCustomSmoothPayoff {
    static constexpr bool is_path_dependent = true;
    static constexpr bool needs_spot_per_step = false;
    static constexpr bool is_smooth = true;
    template <typename T> struct State { T acc; };
    State<Vec4d> init_state(Vec4d) const noexcept { return {Vec4d::zero()}; }
    void update(State<Vec4d>& st, Vec4d log_s) const noexcept { st.acc += log_s; }
    Vec4d evaluate(const State<Vec4d>& st, Vec4d, double inv_steps) const noexcept {
        return st.acc * inv_steps;
    }
};

struct MockBarrier {
    static constexpr bool is_path_dependent = true;
    static constexpr bool needs_spot_per_step = false;
    static constexpr bool is_smooth = false;
    template <typename T> struct State { Vec4d mask; };
    State<Vec4d> init_state(Vec4d) const noexcept { return {Vec4d::zero()}; }
    void update(State<Vec4d>& st, Vec4d log_s) const noexcept { st.mask = bit_or(st.mask, log_s); }
    Vec4d evaluate(const State<Vec4d>&, Vec4d log_s, double) const noexcept { return log_s; }
};

static_assert(NumericType<double>);
static_assert(NumericType<Vec4d>);
static_assert(NumericType<Dual<double>>);
static_assert(NumericType<Dual<Vec4d>>);
static_assert(NumericType<Jet<double, 3>>);
static_assert(NumericType<Jet<Vec4d, 3>>);

static_assert(SamplingPolicy<StandardSampling>);
static_assert(SamplingPolicy<AntitheticSampling>);
static_assert(!StandardSampling::is_antithetic);
static_assert(AntitheticSampling::is_antithetic);

static_assert(ProcessPolicy<MockGBM>);
static_assert(ProcessPolicy<MockHeston>);
static_assert(PayoffPolicy<MockEuropean>);
static_assert(PayoffPolicy<MockAsian>);
static_assert(PayoffPolicy<MockCustomSmoothPayoff>);
static_assert(PayoffPolicy<MockBarrier>);

static_assert(supports_gbm_5_greeks_v<MockGBM, MockEuropean>);
static_assert(supports_gbm_5_greeks_v<MockGBM, MockAsian>);
static_assert(!supports_gbm_5_greeks_v<MockGBM, MockCustomSmoothPayoff>);
static_assert(!supports_gbm_5_greeks_v<MockGBM, MockBarrier>);
static_assert(!supports_gbm_5_greeks_v<MockHeston, MockEuropean>);

static_assert(supports_barrier_greeks_v<MockGBM, MockBarrier>);
static_assert(!supports_barrier_greeks_v<MockGBM, MockEuropean>);
static_assert(!supports_barrier_greeks_v<MockHeston, MockBarrier>);

static_assert(supports_heston_greeks_v<MockHeston, MockEuropean>);
static_assert(supports_heston_greeks_v<MockHeston, MockAsian>);
static_assert(supports_heston_greeks_v<MockHeston, MockCustomSmoothPayoff>);
static_assert(!supports_heston_greeks_v<MockHeston, MockBarrier>);
static_assert(!supports_heston_greeks_v<MockGBM, MockEuropean>);

void test_types_and_validation() {
    std::printf("\n== Types & Parameter Validation ==\n");

    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double pinf = std::numeric_limits<double>::infinity();
    const double ninf = -pinf;

    {
        auto ok = MarketParams::create(100.0, -0.015, 0.20, 1.0, -0.005);
        check::is_true("market_neg_r_q_valid", ok.has_value() && ok->r == -0.015 && ok->q == -0.005);
        auto z_vol = MarketParams::create(100.0, 0.0, 0.0, 0.5, 0.02);
        check::is_true("market_zero_sigma_valid", z_vol.has_value() && z_vol->sigma == 0.0 && z_vol->q == 0.02);
    }

    check::is_true("market_s0_zero",
        !MarketParams::create(0.0, 0.05, 0.2, 1.0).has_value() &&
        MarketParams::create(0.0, 0.05, 0.2, 1.0).error() == ValidationError::NonPositiveSpot);
    check::is_true("market_s0_nan",
        MarketParams::create(qnan, 0.05, 0.2, 1.0).error() == ValidationError::NonPositiveSpot);
    check::is_true("market_s0_inf",
        MarketParams::create(pinf, 0.05, 0.2, 1.0).error() == ValidationError::NonPositiveSpot);
    check::is_true("market_r_nan",
        MarketParams::create(100.0, qnan, 0.2, 1.0).error() == ValidationError::InvalidRate);
    check::is_true("market_r_ninf",
        MarketParams::create(100.0, ninf, 0.2, 1.0).error() == ValidationError::InvalidRate);
    check::is_true("market_sigma_neg",
        MarketParams::create(100.0, 0.05, -0.01, 1.0).error() == ValidationError::NegativeVolatility);
    check::is_true("market_sigma_nan",
        MarketParams::create(100.0, 0.05, qnan, 1.0).error() == ValidationError::NegativeVolatility);
    check::is_true("market_T_zero",
        MarketParams::create(100.0, 0.05, 0.2, 0.0).error() == ValidationError::NonPositiveMaturity);
    check::is_true("market_T_inf",
        MarketParams::create(100.0, 0.05, 0.2, pinf).error() == ValidationError::NonPositiveMaturity);
    check::is_true("market_q_nan",
        MarketParams::create(100.0, 0.05, 0.2, 1.0, qnan).error() == ValidationError::InvalidDividendYield);
    check::is_true("market_q_inf",
        MarketParams::create(100.0, 0.05, 0.2, 1.0, pinf).error() == ValidationError::InvalidDividendYield);

    {
        auto ok = HestonParams::create(100.0, -0.02, 0.0, 2.0, 0.04, 0.3, -1.0, 1.0, -0.01);
        check::is_true("heston_valid_neg_rq_zero_v0", ok.has_value() && ok->v0 == 0.0 && ok->rho == -1.0);
        auto ok2 = HestonParams::create(100.0, 0.03, 0.04, 2.0, 0.04, 0.3, 1.0, 1.0, 0.02);
        check::is_true("heston_valid_rho_pos1", ok2.has_value() && ok2->q == 0.02);
    }

    check::is_true("heston_s0_neg",
        HestonParams::create(-1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5, 1.0).error() == ValidationError::NonPositiveSpot);
    check::is_true("heston_r_inf",
        HestonParams::create(100.0, pinf, 0.04, 2.0, 0.04, 0.3, -0.5, 1.0).error() == ValidationError::InvalidRate);
    check::is_true("heston_v0_neg",
        HestonParams::create(100.0, 0.05, -1e-6, 2.0, 0.04, 0.3, -0.5, 1.0).error() == ValidationError::InvalidInitialVariance);
    check::is_true("heston_v0_nan",
        HestonParams::create(100.0, 0.05, qnan, 2.0, 0.04, 0.3, -0.5, 1.0).error() == ValidationError::InvalidInitialVariance);
    check::is_true("heston_kappa_zero",
        HestonParams::create(100.0, 0.05, 0.04, 0.0, 0.04, 0.3, -0.5, 1.0).error() == ValidationError::NonPositiveKappaOrTheta);
    check::is_true("heston_theta_neg",
        HestonParams::create(100.0, 0.05, 0.04, 2.0, -0.04, 0.3, -0.5, 1.0).error() == ValidationError::NonPositiveKappaOrTheta);
    check::is_true("heston_xi_zero",
        HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04, 0.0, -0.5, 1.0).error() == ValidationError::NonPositiveVolOfVol);
    check::is_true("heston_rho_gt1",
        HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, 1.01, 1.0).error() == ValidationError::InvalidCorrelation);
    check::is_true("heston_rho_lt_m1",
        HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -1.01, 1.0).error() == ValidationError::InvalidCorrelation);
    check::is_true("heston_rho_nan",
        HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, qnan, 1.0).error() == ValidationError::InvalidCorrelation);
    check::is_true("heston_T_neg",
        HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5, -1.0).error() == ValidationError::NonPositiveMaturity);
    check::is_true("heston_q_nan",
        HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5, 1.0, qnan).error() == ValidationError::InvalidDividendYield);
}

void test_avx2_math() {
    std::printf("\n== Hand-Tuned AVX2 Math Kernels ==\n");
    constexpr std::size_t N = 100'000;

    {
        double max_rel = 0.0;
        std::uint64_t max_ulp = 0;
        alignas(32) double in[4];
        alignas(32) double out[4];

        for (std::size_t i = 0; i < N; i += 4) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double t = static_cast<double>(i + k) / static_cast<double>(N - 1);
                in[k] = -50.0 + 100.0 * t;
            }
            const Vec4d res = exp_avx2(Vec4d::load_aligned(in));
            res.store_aligned(out);
            for (std::size_t k = 0; k < 4; ++k) {
                const double ref = std::exp(in[k]);
                const double rel = std::fabs(out[k] - ref) / ref;
                max_rel = std::max(max_rel, rel);
                max_ulp = std::max(max_ulp, ulp_distance(out[k], ref));
            }
        }
        std::printf("  exp_avx2 [-50, 50]: max_rel = %.3e, max_ulp = %llu\n",
                    max_rel, static_cast<unsigned long long>(max_ulp));
        check::is_true("exp_avx2_rel_lt_1e-15", max_rel < 1e-15);
        check::is_true("exp_avx2_ulp_le_4", max_ulp <= 4);

        bool all_finite = true;
        for (std::size_t i = 0; i < N; i += 4) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double t = static_cast<double>(i + k) / static_cast<double>(N - 1);
                in[k] = -708.396 + (709.08 + 708.396) * t;
            }
            exp_avx2(Vec4d::load_aligned(in)).store_aligned(out);
            for (std::size_t k = 0; k < 4; ++k) {
                if (!std::isfinite(out[k]) || out[k] <= 0.0) {
                    all_finite = false;
                }
            }
        }
        check::is_true("exp_avx2_finite_wide_range", all_finite);

        const Vec4d edge_in = Vec4d::set(-1000.0, -708.396, 0.0, 1000.0);
        alignas(32) double edge_out[4];
        exp_avx2(edge_in).store_aligned(edge_out);
        alignas(32) double hi_clamp[4];
        exp_avx2(Vec4d::splat(709.08)).store_aligned(hi_clamp);
        check::is_true("exp_avx2_clamp_neg1000",
            std::isfinite(edge_out[0]) && edge_out[0] > 0.0 && edge_out[0] == edge_out[1]);
        check::is_true("exp_avx2_zero_exact", edge_out[2] == 1.0);
        check::is_true("exp_avx2_clamp_pos1000",
            std::isfinite(edge_out[3]) && edge_out[3] == hi_clamp[0]);
    }

    {
        const double log_lo = std::log(0x1.0p-53);
        const double log_hi = std::log(1e12);
        double max_rel = 0.0;
        std::uint64_t max_ulp = 0;
        alignas(32) double in[4];
        alignas(32) double out[4];

        for (std::size_t i = 0; i < N; i += 4) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double t = static_cast<double>(i + k) / static_cast<double>(N - 1);
                in[k] = std::exp(log_lo + (log_hi - log_lo) * t);
            }
            log_avx2(Vec4d::load_aligned(in)).store_aligned(out);
            for (std::size_t k = 0; k < 4; ++k) {
                const double ref = std::log(in[k]);
                if (std::fabs(in[k] - 1.0) > 1e-3) {
                    const double rel = std::fabs(out[k] - ref) / std::fabs(ref);
                    max_rel = std::max(max_rel, rel);
                    max_ulp = std::max(max_ulp, ulp_distance(out[k], ref));
                }
            }
        }
        std::printf("  log_avx2 [2^-53, 1e12]: max_rel = %.3e, max_ulp = %llu\n",
                    max_rel, static_cast<unsigned long long>(max_ulp));
        check::is_true("log_avx2_rel_lt_1e-15", max_rel < 1e-15);
        check::is_true("log_avx2_ulp_le_4", max_ulp <= 4);

        double max_abs_near1 = 0.0;
        for (std::size_t i = 0; i < N; i += 4) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double t = static_cast<double>(i + k) / static_cast<double>(N - 1);
                in[k] = 0.5 + 1.5 * t;
            }
            log_avx2(Vec4d::load_aligned(in)).store_aligned(out);
            for (std::size_t k = 0; k < 4; ++k) {
                const double ref = std::log(in[k]);
                max_abs_near1 = std::max(max_abs_near1, std::fabs(out[k] - ref));
            }
        }
        std::printf("  log_avx2 [0.5, 2.0]:    max_abs = %.3e\n", max_abs_near1);
        check::is_true("log_avx2_near1_abs_lt_1e-15", max_abs_near1 < 1e-15);

        const Vec4d edge_in = Vec4d::set(0x1.0p-53, 1.0 - 0x1.0p-53, 1.0, 1e12);
        alignas(32) double edge_out[4];
        log_avx2(edge_in).store_aligned(edge_out);
        check::close("log_avx2_edge_2m53", edge_out[0], std::log(0x1.0p-53), 1e-14);
        check::close("log_avx2_edge_1m2m53", edge_out[1], std::log(1.0 - 0x1.0p-53), 1e-16);
        check::is_true("log_avx2_edge_one_exact", edge_out[2] == 0.0);
        check::close("log_avx2_edge_1e12", edge_out[3], std::log(1e12), 1e-14);
    }

    {
        constexpr double two_pi = 2.0 * std::numbers::pi;
        double max_sin_abs = 0.0;
        double max_cos_abs = 0.0;
        alignas(32) double in[4];
        alignas(32) double out_s[4];
        alignas(32) double out_c[4];

        for (std::size_t i = 0; i < N; i += 4) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double t = static_cast<double>(i + k) / static_cast<double>(N - 1);
                in[k] = -two_pi + 2.0 * two_pi * t;
            }
            Vec4d vs, vc;
            sincos_avx2(Vec4d::load_aligned(in), vs, vc);
            vs.store_aligned(out_s);
            vc.store_aligned(out_c);
            for (std::size_t k = 0; k < 4; ++k) {
                max_sin_abs = std::max(max_sin_abs, std::fabs(out_s[k] - std::sin(in[k])));
                max_cos_abs = std::max(max_cos_abs, std::fabs(out_c[k] - std::cos(in[k])));
            }
        }
        std::printf("  sincos_avx2 [-2pi, 2pi]: max_sin_abs = %.3e, max_cos_abs = %.3e\n",
                    max_sin_abs, max_cos_abs);
        check::is_true("sincos_sin_abs_lt_1e-15", max_sin_abs < 1e-15);
        check::is_true("sincos_cos_abs_lt_1e-15", max_cos_abs < 1e-15);
    }

    {
        double max_abs = 0.0;
        alignas(32) double in[4];
        alignas(32) double out[4];

        for (std::size_t i = 0; i < N; i += 4) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double t = static_cast<double>(i + k) / static_cast<double>(N - 1);
                in[k] = -8.0 + 16.0 * t;
            }
            norm_cdf_avx2(Vec4d::load_aligned(in)).store_aligned(out);
            for (std::size_t k = 0; k < 4; ++k) {
                const double ref = norm_cdf(in[k]);
                max_abs = std::max(max_abs, std::fabs(out[k] - ref));
            }
        }

        double max_rel_left = 0.0;
        for (std::size_t i = 0; i < N; i += 4) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double t = static_cast<double>(i + k) / static_cast<double>(N - 1);
                in[k] = -20.0 + 20.0 * t;
            }
            norm_cdf_avx2(Vec4d::load_aligned(in)).store_aligned(out);
            for (std::size_t k = 0; k < 4; ++k) {
                const double ref = norm_cdf(in[k]);
                const double rel = std::fabs(out[k] - ref) / ref;
                max_rel_left = std::max(max_rel_left, rel);
            }
        }
        std::printf("  norm_cdf_avx2 [-8, 8] abs = %.3e, [-20, 0] rel = %.3e\n",
                    max_abs, max_rel_left);
        check::is_true("norm_cdf_abs_lt_1e-14", max_abs < 1e-14);
        check::is_true("norm_cdf_left_rel_lt_1e-13", max_rel_left < 1e-13);

        alignas(32) double edge[4];
        norm_cdf_avx2(Vec4d::set(-50.0, 0.0, 50.0, -1e100)).store_aligned(edge);
        check::is_true("norm_cdf_edge_neg50", edge[0] == 0.0);
        check::is_true("norm_cdf_edge_zero", edge[1] == 0.5);
        check::is_true("norm_cdf_edge_pos50", edge[2] == 1.0);
        check::is_true("norm_cdf_edge_huge_neg", edge[3] == 0.0);

        const double qnan = std::numeric_limits<double>::quiet_NaN();
        alignas(32) double nan_exp[4], nan_log[4], nan_cdf[4], nan_pdf[4];
        exp_avx2(Vec4d::set(qnan, 0.0, -1000.0, 1000.0)).store_aligned(nan_exp);
        log_avx2(Vec4d::set(qnan, 0.0, -1.0, 1.0)).store_aligned(nan_log);
        Vec4d v_cdf, v_pdf;
        norm_cdf_and_pdf_avx2(Vec4d::set(qnan, 0.0, -50.0, 50.0), v_cdf, v_pdf);
        v_cdf.store_aligned(nan_cdf);
        v_pdf.store_aligned(nan_pdf);
        check::is_true("exp_avx2_propagates_nan",
            std::isnan(nan_exp[0]) && nan_exp[1] == 1.0 && std::isfinite(nan_exp[2]) && std::isfinite(nan_exp[3]));
        check::is_true("log_avx2_propagates_nan_nonpos",
            std::isnan(nan_log[0]) && std::isnan(nan_log[1]) && std::isnan(nan_log[2]) && nan_log[3] == 0.0);
        check::is_true("norm_cdf_pdf_propagates_nan",
            std::isnan(nan_cdf[0]) && std::isnan(nan_pdf[0]) &&
            nan_cdf[1] == 0.5 && nan_cdf[2] == 0.0 && nan_cdf[3] == 1.0);
    }
}

void test_xoshiro_and_box_muller() {
    std::printf("\n== Xoshiro256++ x4 & Box-Muller ==\n");

    {
        constexpr std::uint64_t seed = 123456789ULL;
        constexpr std::uint64_t chunk_id = 7ULL;
        Xoshiro256PlusPlusX4 vec_rng = Xoshiro256PlusPlusX4::from_chunk(seed, chunk_id);

        std::array<Xoshiro256PlusPlus, 4> scalar_rngs;
        for (std::uint64_t i = 0; i < 4; ++i) {
            scalar_rngs[static_cast<std::size_t>(i)] =
                Xoshiro256PlusPlus(seed ^ mix64((chunk_id << 2) | i));
        }

        bool bits_match = true;
        bool u01_match = true;
        alignas(32) std::uint64_t v_words[4];
        alignas(32) double v_u01[4];

        for (int step = 0; step < 5000; ++step) {
            const __m256i vw = vec_rng.next_u64x4();
            _mm256_store_si256(reinterpret_cast<__m256i*>(v_words), vw);
            for (std::size_t lane = 0; lane < 4; ++lane) {
                if (v_words[lane] != scalar_rngs[lane].next()) {
                    bits_match = false;
                }
            }
            const Vec4d vu = vec_rng.next_uniform_open01();
            vu.store_aligned(v_u01);
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const double su = scalar_rngs[lane].next_uniform_open01();
                if (std::bit_cast<std::uint64_t>(v_u01[lane]) != std::bit_cast<std::uint64_t>(su)) {
                    u01_match = false;
                }
                if (!(v_u01[lane] > 0.0 && v_u01[lane] < 1.0)) {
                    u01_match = false;
                }
            }
        }
        check::is_true("xoshiro_x4_bit_exact_u64", bits_match);
        check::is_true("xoshiro_x4_bit_exact_u01", u01_match);
    }

    {
        Xoshiro256PlusPlus a({1ULL, 2ULL, 3ULL, 4ULL});
        Xoshiro256PlusPlus b({1ULL, 2ULL, 3ULL, 4ULL});
        for (int i = 0; i < 17; ++i) (void)a.next();
        a.jump();
        b.jump();
        for (int i = 0; i < 17; ++i) (void)b.next();
        check::is_true("xoshiro_jump_commutes_next", a.s == b.s);

        Xoshiro256PlusPlus c({0x123456789abcdef0ULL, 0xfedcba9876543210ULL, 0x0f0f0f0f0f0f0f0fULL, 0xaaaaaaaaaaaaaaaaULL});
        Xoshiro256PlusPlus d = c;
        c.jump();
        c.long_jump();
        d.long_jump();
        d.jump();
        check::is_true("xoshiro_long_jump_commutes", c.s == d.s);

        Xoshiro256PlusPlus x({0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x4444444444444444ULL});
        Xoshiro256PlusPlus y({0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL, 0x0123456789abcdefULL});
        Xoshiro256PlusPlus xy({x.s[0] ^ y.s[0], x.s[1] ^ y.s[1], x.s[2] ^ y.s[2], x.s[3] ^ y.s[3]});
        x.jump();
        y.jump();
        xy.jump();
        const bool jump_linear =
            (xy.s[0] == (x.s[0] ^ y.s[0])) && (xy.s[1] == (x.s[1] ^ y.s[1])) &&
            (xy.s[2] == (x.s[2] ^ y.s[2])) && (xy.s[3] == (x.s[3] ^ y.s[3]));
        check::is_true("xoshiro_jump_gf2_linear", jump_linear);

        x.long_jump();
        y.long_jump();
        xy.long_jump();
        const bool lj_linear =
            (xy.s[0] == (x.s[0] ^ y.s[0])) && (xy.s[1] == (x.s[1] ^ y.s[1])) &&
            (xy.s[2] == (x.s[2] ^ y.s[2])) && (xy.s[3] == (x.s[3] ^ y.s[3]));
        check::is_true("xoshiro_long_jump_gf2_linear", lj_linear);

        Xoshiro256PlusPlus ref({1ULL, 0ULL, 0ULL, 0ULL});
        check::is_true("xoshiro_first_output_exact", ref.next() == ((1ULL << 23) + 1ULL));
        Xoshiro256PlusPlus zstate({0ULL, 0ULL, 0ULL, 0ULL});
        zstate.jump();
        zstate.long_jump();
        check::is_true("xoshiro_zero_invariant",
            zstate.s[0] == 0 && zstate.s[1] == 0 && zstate.s[2] == 0 && zstate.s[3] == 0);
    }

    {
        const __m256i extreme_bits = _mm256_setr_epi64x(
            0LL,
            static_cast<long long>(~0ULL),
            static_cast<long long>(1ULL << 12),
            static_cast<long long>(1ULL << 63));
        alignas(32) double u[4];
        Xoshiro256PlusPlusX4::bits_to_open01(extreme_bits).store_aligned(u);
        check::is_true("u01_zero_bits_strictly_pos", u[0] == 0x1.0p-53 && u[0] > 0.0);
        check::is_true("u01_allones_strictly_lt1", u[1] == (1.0 - 0x1.0p-53) && u[1] < 1.0);
        check::is_true("u01_symmetric_endpoints", (u[0] + u[1]) == 1.0);
        check::is_true("u01_interior_open01", u[2] > 0.0 && u[2] < 1.0 && u[3] > 0.0 && u[3] < 1.0);
    }

    {
        constexpr double u1_ext = 0x1.0p-53;
        constexpr double u2_vals[4] = {0.0, 0.125, 0.37, 1.0 - 0x1.0p-53};
        const Vec4d vu1 = Vec4d::splat(u1_ext);
        const Vec4d vu2 = Vec4d::set(u2_vals[0], u2_vals[1], u2_vals[2], u2_vals[3]);
        Vec4d z0, z1;
        BoxMullerAVX2::from_uniforms(vu1, vu2, z0, z1);
        alignas(32) double out0[4], out1[4];
        z0.store_aligned(out0);
        z1.store_aligned(out1);

        double max_err = 0.0;
        bool all_ok = true;
        const double r_ref = std::sqrt(-2.0 * std::log(u1_ext));
        for (int k = 0; k < 4; ++k) {
            const double th = 2.0 * std::numbers::pi * u2_vals[k];
            const double ref0 = r_ref * std::cos(th);
            const double ref1 = r_ref * std::sin(th);
            if (!std::isfinite(out0[k]) || !std::isfinite(out1[k])) all_ok = false;
            max_err = std::max(max_err, std::fabs(out0[k] - ref0));
            max_err = std::max(max_err, std::fabs(out1[k] - ref1));
        }
        std::printf("  BoxMuller extreme u1=2^-53: |Z|=%.5f, max_err=%.3e\n", r_ref, max_err);
        check::is_true("box_muller_extreme_finite", all_ok && std::fabs(out0[0] - r_ref) < 1e-14);
        check::is_true("box_muller_extreme_lt_1e-14", max_err < 1e-14);
    }

    {
        constexpr std::size_t kSamples = 1'000'000;
        constexpr int kBins = 20;
        std::array<std::size_t, kBins> bins{};
        std::size_t tail_gt3 = 0;
        std::size_t tail_gt4 = 0;

        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        double sum4 = 0.0;

        Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(42ULL, 0ULL);
        alignas(32) double buf[8];
        for (std::size_t i = 0; i < kSamples; i += 8) {
            Vec4d z0, z1;
            rng.next_normals_pair(z0, z1);
            z0.store_aligned(buf);
            z1.store_aligned(buf + 4);
            for (int k = 0; k < 8; ++k) {
                const double z = buf[k];
                const double z2 = z * z;
                sum1 += z;
                sum2 += z2;
                sum3 += z2 * z;
                sum4 += z2 * z2;

                const double az = std::fabs(z);
                if (az > 3.0) ++tail_gt3;
                if (az > 4.0) ++tail_gt4;

                const double u = norm_cdf(z);
                int b = static_cast<int>(u * static_cast<double>(kBins));
                if (b < 0) b = 0;
                if (b >= kBins) b = kBins - 1;
                ++bins[static_cast<std::size_t>(b)];
            }
        }

        const double dn = static_cast<double>(kSamples);
        const double mean = sum1 / dn;
        const double m2 = sum2 / dn - mean * mean;
        const double m3 = sum3 / dn - 3.0 * mean * (sum2 / dn) + 2.0 * mean * mean * mean;
        const double m4 = sum4 / dn - 4.0 * mean * (sum3 / dn)
                        + 6.0 * mean * mean * (sum2 / dn) - 3.0 * mean * mean * mean * mean;
        const double sd = std::sqrt(m2);
        const double skew = m3 / (sd * sd * sd);
        const double kurt = m4 / (m2 * m2);

        const double expected = dn / static_cast<double>(kBins);
        double chi2 = 0.0;
        for (std::size_t count : bins) {
            const double diff = static_cast<double>(count) - expected;
            chi2 += (diff * diff) / expected;
        }

        const double p3_exact = 2.0 * norm_cdf(-3.0);
        const double p4_exact = 2.0 * norm_cdf(-4.0);
        const double p3_emp = static_cast<double>(tail_gt3) / dn;
        const double p4_emp = static_cast<double>(tail_gt4) / dn;
        const double se3 = std::sqrt(p3_exact * (1.0 - p3_exact) / dn);
        const double se4 = std::sqrt(p4_exact * (1.0 - p4_exact) / dn);

        std::printf("  BoxMuller (N=1M): mean=%+.5f var=%.5f skew=%+.5f kurt=%.5f chi2_19=%.2f\n",
                    mean, m2, skew, kurt, chi2);
        check::close("bm_mean_4sigma", mean, 0.0, 0.004);
        check::close("bm_var_4sigma", m2, 1.0, 0.0057);
        check::close("bm_skew_4sigma", skew, 0.0, 0.0098);
        check::close("bm_kurt_4sigma", kurt, 3.0, 0.0196);
        check::is_true("bm_chi2_lt_48", chi2 < 48.0);
        check::within_se("bm_tail_gt3", p3_emp, se3, p3_exact, 4.0);
        check::within_se("bm_tail_gt4", p4_emp, se4, p4_exact, 4.0);
    }
}

template <typename T>
T test_composite_fn(T s, T sigma, T r, double K, double tau) noexcept {
    using std::exp;
    using std::log;
    using std::sqrt;
    const T sqrt_t = sqrt(T(tau));
    const T vol_sqrt_t = sigma * sqrt_t;
    const T d1 = (log(s / K) + (r + 0.5 * sigma * sigma) * tau) / vol_sqrt_t;
    const T d2 = d1 - vol_sqrt_t;
    const T bs = s * norm_cdf(d1) - K * exp(-r * tau) * norm_cdf(d2);
    return bs;
}

void test_dual_and_jet() {
    std::printf("\n== Dual & Jet Automatic Differentiation ==\n");

    constexpr double K = 100.0;
    constexpr double tau = 1.25;
    // 4-point O(h^4) central difference stencil.
    auto central_fd4 = [](auto fn, double x, double h) noexcept -> double {
        return (-fn(x + 2.0 * h) + 8.0 * fn(x + h) - 8.0 * fn(x - h) + fn(x - 2.0 * h))
               / (12.0 * h);
    };
    constexpr double h_s = 1e-3;
    constexpr double h_v = 1e-4;
    constexpr double h_r = 1e-4;

    {
        constexpr double s0 = 105.0;
        constexpr double sig = 0.23;
        constexpr double r0 = 0.045;

        const double fd_s = central_fd4([&](double s) { return test_composite_fn(s, sig, r0, K, tau); }, s0, h_s);
        const double fd_v = central_fd4([&](double v) { return test_composite_fn(s0, v, r0, K, tau); }, sig, h_v);
        const double fd_r = central_fd4([&](double r) { return test_composite_fn(s0, sig, r, K, tau); }, r0, h_r);

        const auto ds = test_composite_fn(Dual<double>::variable(s0), Dual<double>::constant(sig), Dual<double>::constant(r0), K, tau);
        const auto dv = test_composite_fn(Dual<double>::constant(s0), Dual<double>::variable(sig), Dual<double>::constant(r0), K, tau);
        const auto dr = test_composite_fn(Dual<double>::constant(s0), Dual<double>::constant(sig), Dual<double>::variable(r0), K, tau);

        const double sqrt_t = std::sqrt(tau);
        const double d1 = (std::log(s0 / K) + (r0 + 0.5 * sig * sig) * tau) / (sig * sqrt_t);
        const double d2 = d1 - sig * sqrt_t;
        const double phi_d1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * std::numbers::pi);
        const double exact_delta = norm_cdf(d1);
        const double exact_vega  = s0 * phi_d1 * sqrt_t;
        const double exact_rho   = K * tau * std::exp(-r0 * tau) * norm_cdf(d2);

        check::close("dual_double_val", ds.val, test_composite_fn(s0, sig, r0, K, tau), 1e-13);
        check::close("dual_double_dS_fd", ds.d, fd_s, 1e-9);
        check::close("dual_double_dSigma_fd", dv.d, fd_v, 1e-9);
        check::close("dual_double_dR_fd", dr.d, fd_r, 1e-9);
        check::close("dual_double_dS_exact", ds.d, exact_delta, 1e-13);
        check::close("dual_double_dSigma_exact", dv.d, exact_vega, 1e-12);
        check::close("dual_double_dR_exact", dr.d, exact_rho, 1e-12);

        const auto itm = max(Dual<double>::variable(110.0) - K, 0.0);
        const auto otm = max(Dual<double>::variable(90.0) - K, 0.0);
        check::close("dual_double_max_itm_d", itm.d, 1.0, 1e-15);
        check::close("dual_double_max_otm_d", otm.d, 0.0, 1e-15);
    }

    {
        const double spots[4] = {80.0, 95.0, 105.0, 130.0};
        const double sigs[4]  = {0.15, 0.25, 0.30, 0.40};
        const double rates[4] = {-0.01, 0.02, 0.05, 0.08};

        const Vec4d vS = Vec4d::set(spots[0], spots[1], spots[2], spots[3]);
        const Vec4d vV = Vec4d::set(sigs[0], sigs[1], sigs[2], sigs[3]);
        const Vec4d vR = Vec4d::set(rates[0], rates[1], rates[2], rates[3]);

        const auto dual_vec = test_composite_fn(
            Dual<Vec4d>::variable(vS),
            Dual<Vec4d>::constant(vV),
            Dual<Vec4d>::constant(vR),
            K, tau);

        const auto jet_vec = test_composite_fn(
            Jet<Vec4d, 3>::variable<0>(vS),
            Jet<Vec4d, 3>::variable<1>(vV),
            Jet<Vec4d, 3>::variable<2>(vR),
            K, tau);

        alignas(32) double dv_val[4], dv_d[4];
        alignas(32) double j_val[4], j_d0[4], j_d1[4], j_d2[4];
        dual_vec.val.store_aligned(dv_val);
        dual_vec.d.store_aligned(dv_d);
        jet_vec.val.store_aligned(j_val);
        jet_vec.d[0].store_aligned(j_d0);
        jet_vec.d[1].store_aligned(j_d1);
        jet_vec.d[2].store_aligned(j_d2);

        double max_dual_err = 0.0;
        double max_jet_err = 0.0;
        for (int k = 0; k < 4; ++k) {
            const double ref = test_composite_fn(spots[k], sigs[k], rates[k], K, tau);
            const double fd0 = central_fd4([&](double s) { return test_composite_fn(s, sigs[k], rates[k], K, tau); }, spots[k], h_s);
            const double fd1 = central_fd4([&](double v) { return test_composite_fn(spots[k], v, rates[k], K, tau); }, sigs[k], h_v);
            const double fd2 = central_fd4([&](double r) { return test_composite_fn(spots[k], sigs[k], r, K, tau); }, rates[k], h_r);

            max_dual_err = std::max(max_dual_err, std::fabs(dv_val[k] - ref));
            max_dual_err = std::max(max_dual_err, std::fabs(dv_d[k] - fd0));
            max_jet_err = std::max(max_jet_err, std::fabs(j_val[k] - ref));
            max_jet_err = std::max(max_jet_err, std::fabs(j_d0[k] - fd0));
            max_jet_err = std::max(max_jet_err, std::fabs(j_d1[k] - fd1));
            max_jet_err = std::max(max_jet_err, std::fabs(j_d2[k] - fd2));
        }
        std::printf("  Dual<Vec4d> max_err = %.3e, Jet<Vec4d,3> max_err = %.3e\n",
                    max_dual_err, max_jet_err);
        check::is_true("dual_vec4d_lt_1e-9", max_dual_err < 1e-9);
        check::is_true("jet_vec4d_3_lt_1e-9", max_jet_err < 1e-9);

        const auto s_jet = Jet<Vec4d, 3>::variable<0>(vS);
        const auto payoff = max(s_jet - K, 0.0);
        alignas(32) double p_val[4], p_d0[4];
        payoff.val.store_aligned(p_val);
        payoff.d[0].store_aligned(p_d0);
        check::is_true("jet_vec4d_max_branchless",
            p_val[0] == 0.0 && p_d0[0] == 0.0 &&
            p_val[1] == 0.0 && p_d0[1] == 0.0 &&
            p_val[2] == 5.0 && p_d0[2] == 1.0 &&
            p_val[3] == 30.0 && p_d0[3] == 1.0);
    }
}

void test_black_scholes_chain_and_iv() {
    std::printf("\n== Black-Scholes SoA Chain, Closed-Form Greeks & AVX2 IV ==\n");

    using OptType = pricer::OptionType;

    {
        OptionChainSoA chain13(13);
        ChainGreeksSoA greeks13(13);
        check::is_true("soa_padded_n_13_is_16", chain13.padded_n == 16 && greeks13.padded_n == 16);

        auto is_64b_aligned = [](const void* p) noexcept -> bool {
            return (reinterpret_cast<std::uintptr_t>(p) % 64U) == 0U;
        };
        const bool all_aligned =
            is_64b_aligned(chain13.spot.data()) && is_64b_aligned(chain13.strike.data()) &&
            is_64b_aligned(chain13.rate.data()) && is_64b_aligned(chain13.div_yield.data()) &&
            is_64b_aligned(chain13.vol.data()) && is_64b_aligned(chain13.tau.data()) &&
            is_64b_aligned(greeks13.price.data()) && is_64b_aligned(greeks13.delta.data()) &&
            is_64b_aligned(greeks13.gamma.data()) && is_64b_aligned(greeks13.vega.data()) &&
            is_64b_aligned(greeks13.theta.data()) && is_64b_aligned(greeks13.rho.data());
        check::is_true("soa_64byte_aligned", all_aligned);

        bool tail_ok = true;
        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{3},
                              std::size_t{5}, std::size_t{7}, std::size_t{13}}) {
            OptionChainSoA c(n);
            ChainGreeksSoA g(n);
            std::vector<double> p(n);
            std::vector<double> iv(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double K = 85.0 + 3.0 * static_cast<double>(i);
                const double sig = 0.15 + 0.02 * static_cast<double>(i);
                c.set(i, 100.0, K, 0.03, 0.01, sig, 0.75);
            }
            price_chain_avx2<OptType::Call>(c, p.data());
            price_and_greeks_chain_avx2<OptType::Call>(c, g);
            implied_vol_chain_avx2<OptType::Call>(c, p.data(), iv.data());
            for (std::size_t i = 0; i < n; ++i) {
                const double ref_p = black_scholes_price<OptType::Call>(
                    c.spot[i], c.strike[i], c.rate[i], c.div_yield[i], c.vol[i], c.tau[i]);
                if (!std::isfinite(p[i]) || std::fabs(p[i] - ref_p) > 1e-12 ||
                    std::fabs(g.price[i] - ref_p) > 1e-12 ||
                    !std::isfinite(iv[i]) || std::fabs(iv[i] - c.vol[i]) > 1e-11) {
                    tail_ok = false;
                }
            }
        }
        check::is_true("soa_non_multiple_of_4_tails", tail_ok);
    }

    {
        constexpr std::size_t N = 16'384;
        OptionChainSoA chain(N);
        ChainGreeksSoA g_call(N);
        ChainGreeksSoA g_put(N);
        AlignedBuffer p_call(chain.padded_n);
        AlignedBuffer p_put(chain.padded_n);

        Xoshiro256PlusPlus rng(0x31415926ULL);
        for (std::size_t i = 0; i < N; ++i) {
            const double S = 50.0 + 150.0 * rng.next_uniform_open01();
            const double K = S * (0.5 + 1.0 * rng.next_uniform_open01());
            const double r = -0.03 + 0.12 * rng.next_uniform_open01();
            const double q = -0.01 + 0.07 * rng.next_uniform_open01();
            const double sigma = 0.05 + 0.95 * rng.next_uniform_open01();
            const double tau = 0.02 + 3.0 * rng.next_uniform_open01();
            chain.set(i, S, K, r, q, sigma, tau);
        }

        price_chain_avx2<OptType::Call>(chain, p_call);
        price_chain_avx2<OptType::Put>(chain, p_put);
        price_and_greeks_chain_avx2<OptType::Call>(chain, g_call);
        price_and_greeks_chain_avx2<OptType::Put>(chain, g_put);

        double max_pcp = 0.0;
        double max_pcp_g = 0.0;
        double max_delta_par = 0.0;
        double max_gamma_eq = 0.0;
        double max_vega_eq = 0.0;
        double max_theta_par = 0.0;
        double max_rho_par = 0.0;

        for (std::size_t i = 0; i < N; ++i) {
            const double S = chain.spot[i];
            const double K = chain.strike[i];
            const double r = chain.rate[i];
            const double q = chain.div_yield[i];
            const double tau = chain.tau[i];
            const double disc_r = std::exp(-r * tau);
            const double disc_q = std::exp(-q * tau);
            const double fwd_parity = S * disc_q - K * disc_r;

            max_pcp = std::max(max_pcp, std::fabs((p_call[i] - p_put[i]) - fwd_parity));
            max_pcp_g = std::max(max_pcp_g, std::fabs((g_call.price[i] - g_put.price[i]) - fwd_parity));
            max_delta_par = std::max(max_delta_par, std::fabs((g_call.delta[i] - g_put.delta[i]) - disc_q));
            max_gamma_eq = std::max(max_gamma_eq, std::fabs(g_call.gamma[i] - g_put.gamma[i]));
            max_vega_eq = std::max(max_vega_eq, std::fabs(g_call.vega[i] - g_put.vega[i]));
            const double theta_rhs = q * S * disc_q - r * K * disc_r;
            max_theta_par = std::max(max_theta_par, std::fabs((g_call.theta[i] - g_put.theta[i]) - theta_rhs));
            const double rho_rhs = K * tau * disc_r;
            max_rho_par = std::max(max_rho_par, std::fabs((g_call.rho[i] - g_put.rho[i]) - rho_rhs));
        }

        std::printf("  Parity (N=16384): PCP=%.2e Delta=%.2e Gamma=%.2e Vega=%.2e Theta=%.2e Rho=%.2e\n",
                    max_pcp, max_delta_par, max_gamma_eq, max_vega_eq, max_theta_par, max_rho_par);
        check::is_true("bs_put_call_parity_lt_1e-12", max_pcp < 1e-12 && max_pcp_g < 1e-12);
        check::is_true("bs_delta_parity_lt_1e-12", max_delta_par < 1e-12);
        check::is_true("bs_gamma_call_eq_put_lt_1e-12", max_gamma_eq < 1e-12);
        check::is_true("bs_vega_call_eq_put_lt_1e-12", max_vega_eq < 1e-12);
        check::is_true("bs_theta_parity_lt_1e-12", max_theta_par < 1e-12);
        check::is_true("bs_rho_parity_lt_1e-12", max_rho_par < 1e-12);
    }

    {
        constexpr std::size_t N = 1024;
        OptionChainSoA chain(N);
        ChainGreeksSoA g_call(N);
        ChainGreeksSoA g_put(N);

        Xoshiro256PlusPlus rng(0x27182818ULL);
        for (std::size_t i = 0; i < N; ++i) {
            const double S = 60.0 + 100.0 * rng.next_uniform_open01();
            const double K = S * (0.65 + 0.70 * rng.next_uniform_open01());
            const double r = -0.02 + 0.10 * rng.next_uniform_open01();
            const double q = 0.05 * rng.next_uniform_open01();
            const double sigma = 0.10 + 0.60 * rng.next_uniform_open01();
            const double tau = 0.10 + 2.50 * rng.next_uniform_open01();
            chain.set(i, S, K, r, q, sigma, tau);
        }

        price_and_greeks_chain_avx2<OptType::Call>(chain, g_call);
        price_and_greeks_chain_avx2<OptType::Put>(chain, g_put);

        double max_jet_err = 0.0;
        double max_dual_theta_err = 0.0;
        for (std::size_t i = 0; i < N; i += 4) {
            const Vec4d vS = Vec4d::load_aligned(chain.spot.data() + i);
            const Vec4d vK = Vec4d::load_aligned(chain.strike.data() + i);
            const Vec4d vr = Vec4d::load_aligned(chain.rate.data() + i);
            const Vec4d vq = Vec4d::load_aligned(chain.div_yield.data() + i);
            const Vec4d vs = Vec4d::load_aligned(chain.vol.data() + i);
            const Vec4d vt = Vec4d::load_aligned(chain.tau.data() + i);

            auto check_ad_type = [&]<OptType Type>(const ChainGreeksSoA& g) {
                const auto jet = black_scholes_price<Type>(
                    Jet<Vec4d, 3>::variable<0>(vS),
                    Jet<Vec4d, 3>::constant(vK),
                    Jet<Vec4d, 3>::variable<2>(vr),
                    Jet<Vec4d, 3>::constant(vq),
                    Jet<Vec4d, 3>::variable<1>(vs),
                    Jet<Vec4d, 3>::constant(vt));

                const auto dual_t = black_scholes_price<Type>(
                    Dual<Vec4d>::constant(vS),
                    Dual<Vec4d>::constant(vK),
                    Dual<Vec4d>::constant(vr),
                    Dual<Vec4d>::constant(vq),
                    Dual<Vec4d>::constant(vs),
                    Dual<Vec4d>::variable(vt));

                alignas(32) double jp[4], jd[4], jv[4], jr[4], dt[4];
                jet.val.store_aligned(jp);
                jet.d[0].store_aligned(jd);
                jet.d[1].store_aligned(jv);
                jet.d[2].store_aligned(jr);
                (-dual_t.d).store_aligned(dt);

                for (std::size_t k = 0; k < 4; ++k) {
                    const std::size_t idx = i + k;
                    max_jet_err = std::max(max_jet_err, std::fabs(g.price[idx] - jp[k]));
                    max_jet_err = std::max(max_jet_err, std::fabs(g.delta[idx] - jd[k]));
                    max_jet_err = std::max(max_jet_err, std::fabs(g.vega[idx] - jv[k]));
                    max_jet_err = std::max(max_jet_err, std::fabs(g.rho[idx] - jr[k]));
                    max_dual_theta_err = std::max(max_dual_theta_err, std::fabs(g.theta[idx] - dt[k]));
                }
            };
            check_ad_type.template operator()<OptType::Call>(g_call);
            check_ad_type.template operator()<OptType::Put>(g_put);
        }

        // Step sizes scaled to variable magnitude (h_gamma ~ eps^(1/4) * S) to balance truncation vs roundoff.
        double max_fd_delta = 0.0, max_fd_gamma = 0.0, max_fd_vega = 0.0;
        double max_fd_theta = 0.0, max_fd_rho = 0.0;
        for (std::size_t i = 0; i < 128; ++i) {
            const double S = chain.spot[i];
            const double K = chain.strike[i];
            const double r = chain.rate[i];
            const double q = chain.div_yield[i];
            const double sig = chain.vol[i];
            const double tau = chain.tau[i];

            const double h_s = 1e-5 * S;
            const double h_g = 1.5e-4 * S;
            const double h_v = 1e-5 * sig;
            const double h_r = 2e-6;
            const double h_t = 1e-5 * tau;

            auto check_fd_type = [&]<OptType Type>(const ChainGreeksSoA& g) {
                auto P = [&](double s_, double r_, double sig_, double t_) {
                    return black_scholes_price<Type>(s_, K, r_, q, sig_, t_);
                };
                const double p0 = P(S, r, sig, tau);
                const double fd_delta = (P(S + h_s, r, sig, tau) - P(S - h_s, r, sig, tau)) / (2.0 * h_s);
                const double fd_gamma = (P(S + h_g, r, sig, tau) - 2.0 * p0 + P(S - h_g, r, sig, tau)) / (h_g * h_g);
                const double fd_vega  = (P(S, r, sig + h_v, tau) - P(S, r, sig - h_v, tau)) / (2.0 * h_v);
                const double fd_theta = (P(S, r, sig, tau - h_t) - P(S, r, sig, tau + h_t)) / (2.0 * h_t);
                const double fd_rho   = (P(S, r + h_r, sig, tau) - P(S, r - h_r, sig, tau)) / (2.0 * h_r);

                max_fd_delta = std::max(max_fd_delta, std::fabs(g.delta[i] - fd_delta));
                max_fd_gamma = std::max(max_fd_gamma, std::fabs(g.gamma[i] - fd_gamma));
                max_fd_vega  = std::max(max_fd_vega,  std::fabs(g.vega[i]  - fd_vega));
                max_fd_theta = std::max(max_fd_theta, std::fabs(g.theta[i] - fd_theta));
                max_fd_rho   = std::max(max_fd_rho,   std::fabs(g.rho[i]   - fd_rho));
            };
            check_fd_type.template operator()<OptType::Call>(g_call);
            check_fd_type.template operator()<OptType::Put>(g_put);
        }
        const double max_fd_err = std::max({max_fd_delta, max_fd_gamma, max_fd_vega, max_fd_theta, max_fd_rho});

        std::printf("  Greeks oracle: max_jet_err=%.2e, max_dual_theta=%.2e, max_fd_err=%.2e "
                    "(D=%.1e G=%.1e V=%.1e T=%.1e R=%.1e)\n",
                    max_jet_err, max_dual_theta_err, max_fd_err,
                    max_fd_delta, max_fd_gamma, max_fd_vega, max_fd_theta, max_fd_rho);
        check::is_true("bs_greeks_vs_jet_lt_1e-12", max_jet_err < 1e-12);
        check::is_true("bs_theta_vs_dual_lt_1e-12", max_dual_theta_err < 1e-12);
        check::is_true("bs_greeks_vs_fd_lt_1e-7", max_fd_err < 1e-7);
    }

    {
        OptionChainSoA exp_chain(4);
        exp_chain.set(0, 120.0, 100.0, 0.05, 0.02, 0.25, 0.0);
        exp_chain.set(1, 100.0, 100.0, 0.05, 0.02, 0.25, 0.0);
        exp_chain.set(2,  80.0, 100.0, 0.05, 0.02, 0.25, 0.0);
        exp_chain.set(3, 110.0, 100.0, 0.05, 0.02, 0.00, 1.0);

        ChainGreeksSoA gc(4), gp(4);
        AlignedBuffer pc(4), pp(4);
        price_chain_avx2<OptType::Call>(exp_chain, pc);
        price_chain_avx2<OptType::Put>(exp_chain, pp);
        price_and_greeks_chain_avx2<OptType::Call>(exp_chain, gc);
        price_and_greeks_chain_avx2<OptType::Put>(exp_chain, gp);

        const bool call_exp_ok =
            pc[0] == 20.0 && pc[1] == 0.0 && pc[2] == 0.0 &&
            gc.price[0] == 20.0 && gc.delta[0] == 1.0 && gc.gamma[0] == 0.0 &&
            gc.vega[0] == 0.0 && gc.theta[0] == 0.0 && gc.rho[0] == 0.0 &&
            gc.price[1] == 0.0 && gc.delta[1] == 0.0 &&
            gc.price[2] == 0.0 && gc.delta[2] == 0.0;
        const bool put_exp_ok =
            pp[0] == 0.0 && pp[1] == 0.0 && pp[2] == 20.0 &&
            gp.price[0] == 0.0 && gp.delta[0] == 0.0 &&
            gp.price[1] == 0.0 && gp.delta[1] == 0.0 &&
            gp.price[2] == 20.0 && gp.delta[2] == -1.0 && gp.gamma[2] == 0.0 &&
            gp.vega[2] == 0.0 && gp.theta[2] == 0.0 && gp.rho[2] == 0.0;

        const double disc_intrinsic = std::max(110.0 * std::exp(-0.02) - 100.0 * std::exp(-0.05), 0.0);
        const bool zvol_ok =
            std::fabs(pc[3] - disc_intrinsic) < 1e-12 &&
            std::fabs(gc.price[3] - disc_intrinsic) < 1e-12 &&
            std::fabs(gc.delta[3] - std::exp(-0.02)) < 1e-12 &&
            gc.gamma[3] == 0.0 && gc.vega[3] == 0.0 &&
            pp[3] == 0.0 && gp.price[3] == 0.0 && gp.delta[3] == 0.0;

        const double sc_c = black_scholes_price<OptType::Call>(120.0, 100.0, 0.05, 0.02, 0.25, 0.0);
        const double sc_p = black_scholes_price<OptType::Put>(80.0, 100.0, 0.05, 0.02, 0.25, 0.0);
        const auto jet_exp = black_scholes_price<OptType::Call>(
            Jet<Vec4d, 3>::variable<0>(Vec4d::set(120.0, 100.0, 80.0, 110.0)),
            Jet<Vec4d, 3>::constant(Vec4d::splat(100.0)),
            Jet<Vec4d, 3>::variable<2>(Vec4d::splat(0.05)),
            Jet<Vec4d, 3>::constant(Vec4d::splat(0.02)),
            Jet<Vec4d, 3>::variable<1>(Vec4d::splat(0.25)),
            Jet<Vec4d, 3>::constant(Vec4d::zero()));
        alignas(32) double jv[4], jd0[4], jd1[4], jd2[4];
        jet_exp.val.store_aligned(jv);
        jet_exp.d[0].store_aligned(jd0);
        jet_exp.d[1].store_aligned(jd1);
        jet_exp.d[2].store_aligned(jd2);
        const bool jet_exp_ok =
            sc_c == 20.0 && sc_p == 20.0 &&
            jv[0] == 20.0 && jd0[0] == 1.0 && jd1[0] == 0.0 && jd2[0] == 0.0 &&
            jv[2] == 0.0 && jd0[2] == 0.0 && jd1[2] == 0.0 && jd2[2] == 0.0 &&
            std::isfinite(hsum(jet_exp.val)) && std::isfinite(hsum(jet_exp.d[0])) &&
            std::isfinite(hsum(jet_exp.d[1])) && std::isfinite(hsum(jet_exp.d[2]));

        check::is_true("bs_tau_zero_call_exact", call_exp_ok);
        check::is_true("bs_tau_zero_put_exact", put_exp_ok);
        check::is_true("bs_sigma_zero_exact", zvol_ok);
        check::is_true("bs_tau_zero_jet_nan_free", jet_exp_ok);

        const double qnan = std::numeric_limits<double>::quiet_NaN();
        OptionChainSoA nan_chain(4);
        nan_chain.set(0, qnan,  100.0, 0.05, 0.01, 0.20, 1.0);
        nan_chain.set(1, 100.0, qnan,  0.05, 0.01, 0.20, 1.0);
        nan_chain.set(2, 100.0, 100.0, 0.05, 0.01, qnan, 1.0);
        nan_chain.set(3, 100.0, 100.0, 0.05, 0.01, 0.20, qnan);

        AlignedBuffer pnc(4), pnp(4);
        ChainGreeksSoA gnc(4), gnp(4);
        price_chain_avx2<OptType::Call>(nan_chain, pnc);
        price_chain_avx2<OptType::Put>(nan_chain, pnp);
        price_and_greeks_chain_avx2<OptType::Call>(nan_chain, gnc);
        price_and_greeks_chain_avx2<OptType::Put>(nan_chain, gnp);

        bool bs_nan_ok = true;
        for (std::size_t i = 0; i < 4; ++i) {
            bs_nan_ok = bs_nan_ok && std::isnan(pnc[i]) && std::isnan(pnp[i])
                && std::isnan(gnc.price[i]) && std::isnan(gnc.delta[i]) && std::isnan(gnc.gamma[i])
                && std::isnan(gnc.vega[i])  && std::isnan(gnc.theta[i]) && std::isnan(gnc.rho[i])
                && std::isnan(gnp.price[i]) && std::isnan(gnp.delta[i]) && std::isnan(gnp.gamma[i])
                && std::isnan(gnp.vega[i])  && std::isnan(gnp.theta[i]) && std::isnan(gnp.rho[i]);
        }
        check::is_true("bs_chain_propagates_nan", bs_nan_ok);
    }

    {
        constexpr std::size_t N = 16'384;
        OptionChainSoA chain(N);
        Xoshiro256PlusPlus rng(0x16180339ULL);

        std::size_t filled = 0;
        while (filled < N) {
            const double S = 50.0 + 150.0 * rng.next_uniform_open01();
            const double K = S * std::exp(-0.9 + 1.8 * rng.next_uniform_open01());
            const double r = -0.02 + 0.10 * rng.next_uniform_open01();
            const double q = 0.05 * rng.next_uniform_open01();
            const double sigma = 0.05 + 1.45 * rng.next_uniform_open01();
            const double tau = 0.05 + 3.0 * rng.next_uniform_open01();

            const double sqrt_t = std::sqrt(tau);
            const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_t);
            const double phi_d1 = 0.39894228040143267794 * std::exp(-0.5 * d1 * d1);
            const double vega_over_s = std::exp(-q * tau) * phi_d1 * sqrt_t;
            if (std::fabs(d1) <= 5.0 && vega_over_s >= 1e-6) {
                chain.set(filled, S, K, r, q, sigma, tau);
                ++filled;
            }
        }

        AlignedBuffer call_prices(chain.padded_n);
        AlignedBuffer put_prices(chain.padded_n);
        AlignedBuffer call_iv(chain.padded_n);
        AlignedBuffer put_iv(chain.padded_n);

        price_chain_avx2<OptType::Call>(chain, call_prices);
        price_chain_avx2<OptType::Put>(chain, put_prices);
        implied_vol_chain_avx2<OptType::Call>(chain, call_prices.data(), call_iv.data());
        implied_vol_chain_avx2<OptType::Put>(chain, put_prices.data(), put_iv.data());

        double max_otm_err = 0.0;
        double max_call_iv_err = 0.0;
        double max_put_iv_err = 0.0;
        double max_call_ulp_ratio = 0.0;
        double max_put_ulp_ratio = 0.0;
        bool all_finite = true;
        for (std::size_t i = 0; i < N; ++i) {
            if (!std::isfinite(call_iv[i]) || !std::isfinite(put_iv[i])) {
                all_finite = false;
            }
            const double S = chain.spot[i];
            const double K = chain.strike[i];
            const double r = chain.rate[i];
            const double q = chain.div_yield[i];
            const double sig = chain.vol[i];
            const double tau = chain.tau[i];
            const double sqrt_t = std::sqrt(tau);
            const double d1 = (std::log(S / K) + (r - q + 0.5 * sig * sig) * tau) / (sig * sqrt_t);
            const double vega = S * std::exp(-q * tau) * 0.39894228040143267794 * std::exp(-0.5 * d1 * d1) * sqrt_t;
            const double fwd_s = S * std::exp(-q * tau);
            const double fwd_k = K * std::exp(-r * tau);

            const double ec = std::fabs(call_iv[i] - sig);
            const double ep = std::fabs(put_iv[i] - sig);
            // Conditioning floor in sigma space is ulp(P) / Vega.
            const double ulp_c = std::nextafter(call_prices[i], std::numeric_limits<double>::infinity()) - call_prices[i];
            const double ulp_p = std::nextafter(put_prices[i], std::numeric_limits<double>::infinity()) - put_prices[i];
            if (vega > 0.0) {
                if (ec > 1e-12) max_call_ulp_ratio = std::max(max_call_ulp_ratio, ec / (ulp_c / vega));
                if (ep > 1e-12) max_put_ulp_ratio  = std::max(max_put_ulp_ratio,  ep / (ulp_p / vega));
            }
            if (fwd_s <= fwd_k) {
                max_otm_err = std::max(max_otm_err, ec);
            } else {
                max_otm_err = std::max(max_otm_err, ep);
            }
            if (vega >= 1e-4 * std::max(S, call_prices[i])) {
                max_call_iv_err = std::max(max_call_iv_err, ec);
            }
            if (vega >= 1e-4 * std::max(S, put_prices[i])) {
                max_put_iv_err = std::max(max_put_iv_err, ep);
            }
        }
        std::printf("  IV round-trip (N=16384): OTM(Vega/S>=1e-6)=%.2e, Call(Vega/max(S,P)>=1e-4)=%.2e, "
                    "Put(Vega/max(S,P)>=1e-4)=%.2e, max_ITM_ulps=(%.1f, %.1f)\n",
                    max_otm_err, max_call_iv_err, max_put_iv_err, max_call_ulp_ratio, max_put_ulp_ratio);
        check::is_true("iv_well_cond_finite", all_finite);
        check::is_true("iv_otm_vega_1e-6_lt_1e-11", max_otm_err < 1e-11);
        check::is_true("iv_well_cond_call_lt_1e-11", max_call_iv_err < 1e-11);
        check::is_true("iv_well_cond_put_lt_1e-11", max_put_iv_err < 1e-11);
        check::is_true("iv_itm_near_ieee754_floor", max_call_ulp_ratio <= 12.0 && max_put_ulp_ratio <= 12.0);
    }

    {
        constexpr std::size_t N = 1024;
        OptionChainSoA chain_call(N);
        OptionChainSoA chain_put(N);
        Xoshiro256PlusPlus rng(0xDEEF1234ULL);

        for (std::size_t i = 0; i < N; ++i) {
            const double S = 100.0;
            const double r = -0.01 + 0.08 * rng.next_uniform_open01();
            const double q = 0.03 * rng.next_uniform_open01();
            const double sigma = 0.15 + 0.45 * rng.next_uniform_open01();
            const double tau = 0.25 + 1.50 * rng.next_uniform_open01();
            const double sqrt_t = std::sqrt(tau);
            const double mag_d1 = 5.05 + 2.90 * rng.next_uniform_open01();

            const double log_sk_c = (-mag_d1) * sigma * sqrt_t - (r - q + 0.5 * sigma * sigma) * tau;
            const double K_c = S * std::exp(-log_sk_c);
            chain_call.set(i, S, K_c, r, q, sigma, tau);

            const double log_sk_p = (+mag_d1) * sigma * sqrt_t - (r - q + 0.5 * sigma * sigma) * tau;
            const double K_p = S * std::exp(-log_sk_p);
            chain_put.set(i, S, K_p, r, q, sigma, tau);
        }

        AlignedBuffer pc(chain_call.padded_n), pp(chain_put.padded_n);
        AlignedBuffer ivc(chain_call.padded_n), ivp(chain_put.padded_n);
        price_chain_avx2<OptType::Call>(chain_call, pc);
        price_chain_avx2<OptType::Put>(chain_put, pp);
        implied_vol_chain_avx2<OptType::Call>(chain_call, pc.data(), ivc.data());
        implied_vol_chain_avx2<OptType::Put>(chain_put, pp.data(), ivp.data());

        double max_rel_price_err = 0.0;
        bool all_finite = true;
        for (std::size_t i = 0; i < N; ++i) {
            if (!std::isfinite(ivc[i]) || !std::isfinite(ivp[i])) {
                all_finite = false;
                continue;
            }
            const double rc = black_scholes_price<OptType::Call>(
                chain_call.spot[i], chain_call.strike[i], chain_call.rate[i],
                chain_call.div_yield[i], ivc[i], chain_call.tau[i]);
            const double rp = black_scholes_price<OptType::Put>(
                chain_put.spot[i], chain_put.strike[i], chain_put.rate[i],
                chain_put.div_yield[i], ivp[i], chain_put.tau[i]);
            max_rel_price_err = std::max(max_rel_price_err, std::fabs(rc - pc[i]) / chain_call.spot[i]);
            max_rel_price_err = std::max(max_rel_price_err, std::fabs(rp - pp[i]) / chain_put.spot[i]);
        }
        std::printf("  IV deep wings (5<|d1|<=8, N=%zu): max |BS(iv)-P|/S = %.2e\n",
                    N, max_rel_price_err);
        check::is_true("iv_deep_wings_finite", all_finite);
        check::is_true("iv_deep_wings_price_lt_1e-13S", max_rel_price_err < 1e-13);
    }

    {
        OptionChainSoA ext(4);
        ext.set(0, 100.0, 100.0, 0.05, 0.01, 12.5,   0.02);
        ext.set(1, 100.0, 110.0, 0.03, 0.00, 15.0,   0.01);
        ext.set(2, 100.0, 100.0, 0.02, 0.02, 5.0e-4, 1.00);
        ext.set(3, 100.0, 100.0, 0.00, 0.00, 5.0e-7, 1.00);

        AlignedBuffer p_ext(4), iv_ext(4);
        price_chain_avx2<OptType::Call>(ext, p_ext);
        implied_vol_chain_avx2<OptType::Call>(ext, p_ext.data(), iv_ext.data());
        bool ext_ok = true;
        for (std::size_t k = 0; k < 4; ++k) {
            const double rel_err = std::fabs(iv_ext[k] - ext.vol[k]) / std::max(1.0, ext.vol[k]);
            if (!std::isfinite(iv_ext[k]) || rel_err > 1e-11) {
                ext_ok = false;
            }
        }
        check::is_true("iv_bracket_expansion_hi_lo", ext_ok);

        OptionChainSoA bad(8);
        for (std::size_t i = 0; i < 8; ++i) {
            bad.set(i, 100.0, 95.0, 0.05, 0.01, 0.30, 1.0);
        }
        bad.tau[7] = 0.0;

        const double valid_call_p = black_scholes_price<OptType::Call>(100.0, 95.0, 0.05, 0.01, 0.30, 1.0);
        const double disc_r = std::exp(-0.05);
        const double disc_q = std::exp(-0.01);
        const double intrinsic_c = 100.0 * disc_q - 95.0 * disc_r;
        const double upper_c = 100.0 * disc_q;
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        const double pinf = std::numeric_limits<double>::infinity();

        alignas(32) double mkt_bad[8] = {
            valid_call_p,
            intrinsic_c,
            intrinsic_c - 1.0,
            upper_c,
            upper_c + 1.0,
            qnan,
            valid_call_p,
            pinf,
        };
        alignas(32) double iv_bad[8];
        implied_vol_chain_avx2<OptType::Call>(bad, mkt_bad, iv_bad);

        const bool oob_ok =
            std::isfinite(iv_bad[0]) && std::fabs(iv_bad[0] - 0.30) < 1e-11 &&
            std::isnan(iv_bad[1]) && std::isnan(iv_bad[2]) && std::isnan(iv_bad[3]) &&
            std::isnan(iv_bad[4]) && std::isnan(iv_bad[5]) &&
            std::isfinite(iv_bad[6]) && std::fabs(iv_bad[6] - 0.30) < 1e-11 &&
            std::isnan(iv_bad[7]);
        check::is_true("iv_no_arb_oob_nan_isolated", oob_ok);

        const double put_lo = std::max(105.0 * disc_r - 100.0 * disc_q, 0.0);
        const double put_hi = 105.0 * disc_r;
        const bool put_oob_ok =
            std::isnan(black_scholes_implied_vol<OptType::Put>(100.0, 105.0, 0.05, 0.01, 1.0, put_lo)) &&
            std::isnan(black_scholes_implied_vol<OptType::Put>(100.0, 105.0, 0.05, 0.01, 1.0, put_hi)) &&
            std::isnan(black_scholes_implied_vol<OptType::Put>(100.0, 105.0, 0.05, 0.01, 1.0, -0.5));
        check::is_true("iv_put_oob_nan", put_oob_ok);
    }
}

static_assert(ProcessPolicy<GBMProcess>);
static_assert(ProcessPolicy<HestonProcess>);
static_assert(PayoffPolicy<EuropeanPayoff<OptionType::Call>>);
static_assert(PayoffPolicy<EuropeanPayoff<OptionType::Put>>);
static_assert(PayoffPolicy<GeometricAsianPayoff<OptionType::Call>>);
static_assert(PayoffPolicy<ArithmeticAsianPayoff<OptionType::Put>>);
static_assert(PayoffPolicy<BarrierPayoff<OptionType::Call, BarrierType::UpOut>>);
static_assert(PayoffPolicy<LookbackPayoff<OptionType::Call, LookbackStyle::FixedStrike>>);
static_assert(PayoffPolicy<LookbackPayoff<OptionType::Put, LookbackStyle::FloatingStrike>>);

template <typename Eng>
concept HasGbm5Greeks = requires(Eng& e, const SimConfig& cfg) {
    e.price_with_greeks(cfg);
};

template <typename Eng>
concept HasBarrierGreeks = requires(Eng& e, const SimConfig& cfg) {
    e.price_with_barrier_greeks(cfg);
};

template <typename Eng>
concept HasHestonGreeks = requires(Eng& e, const SimConfig& cfg) {
    e.price_with_heston_greeks(cfg);
};

static_assert(HasGbm5Greeks<MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>>>);
static_assert(HasGbm5Greeks<MonteCarloEngine<GBMProcess, GeometricAsianPayoff<OptionType::Call>>>);
static_assert(HasGbm5Greeks<MonteCarloEngine<GBMProcess, ArithmeticAsianPayoff<OptionType::Call>>>);
static_assert(HasGbm5Greeks<MonteCarloEngine<GBMProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FixedStrike>>>);
static_assert(HasGbm5Greeks<MonteCarloEngine<GBMProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>>>);
static_assert(!HasGbm5Greeks<MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>>>);
static_assert(!HasGbm5Greeks<MonteCarloEngine<GBMProcess, MockCustomSmoothPayoff>>);
static_assert(!HasGbm5Greeks<MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>>>);

static_assert(HasBarrierGreeks<MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>>>);
static_assert(!HasBarrierGreeks<MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>>>);
static_assert(!HasBarrierGreeks<MonteCarloEngine<HestonProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>>>);

static_assert(HasHestonGreeks<MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>>>);
static_assert(HasHestonGreeks<MonteCarloEngine<HestonProcess, GeometricAsianPayoff<OptionType::Call>>>);
static_assert(HasHestonGreeks<MonteCarloEngine<HestonProcess, ArithmeticAsianPayoff<OptionType::Put>>>);
static_assert(HasHestonGreeks<MonteCarloEngine<HestonProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>>>);
static_assert(!HasHestonGreeks<MonteCarloEngine<HestonProcess, BarrierPayoff<OptionType::Call, BarrierType::DownOut>>>);
static_assert(!HasHestonGreeks<MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>>>);

// Discrete Kemna-Vorst Geometric Asian closed form (monitored at t_1..t_m).
template <OptionType Type>
double kemna_vorst_discrete(double s0, double K, double r, double q,
                            double sigma, double T, std::size_t m) noexcept {
    const double dm = static_cast<double>(m);
    const double sigma_g = sigma * std::sqrt(((dm + 1.0) * (2.0 * dm + 1.0)) / (6.0 * dm * dm));
    const double mu_g = (r - q - 0.5 * sigma * sigma) * ((dm + 1.0) / (2.0 * dm))
                      + 0.5 * sigma_g * sigma_g;
    const double q_g = r - mu_g;
    return black_scholes_price<Type>(s0, K, r, q_g, sigma_g, T);
}

// Reiner-Rubinstein (1991) continuous Barrier closed form (zero rebate).
template <OptionType Type, BarrierType BType>
double reiner_rubinstein_continuous(double S, double K, double H,
                                    double r, double q, double sigma, double T) noexcept {
    const double b = r - q;
    const double sig_sqrt_t = sigma * std::sqrt(T);
    const double mu = (b - 0.5 * sigma * sigma) / (sigma * sigma);
    const double x1 = std::log(S / K) / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    const double x2 = std::log(S / H) / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    const double y1 = std::log((H * H) / (S * K)) / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    const double y2 = std::log(H / S) / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;

    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);
    const double hs_2mu = std::pow(H / S, 2.0 * mu);
    const double hs_2mu2 = hs_2mu * (H / S) * (H / S);

    auto A = [&](double phi) noexcept {
        return phi * S * disc_q * norm_cdf(phi * x1)
             - phi * K * disc_r * norm_cdf(phi * x1 - phi * sig_sqrt_t);
    };
    auto B = [&](double phi) noexcept {
        return phi * S * disc_q * norm_cdf(phi * x2)
             - phi * K * disc_r * norm_cdf(phi * x2 - phi * sig_sqrt_t);
    };
    auto C = [&](double phi, double eta) noexcept {
        return phi * S * disc_q * hs_2mu2 * norm_cdf(eta * y1)
             - phi * K * disc_r * hs_2mu  * norm_cdf(eta * y1 - eta * sig_sqrt_t);
    };
    auto D = [&](double phi, double eta) noexcept {
        return phi * S * disc_q * hs_2mu2 * norm_cdf(eta * y2)
             - phi * K * disc_r * hs_2mu  * norm_cdf(eta * y2 - eta * sig_sqrt_t);
    };

    double knock_in = 0.0;
    if constexpr (Type == OptionType::Call && (BType == BarrierType::DownIn || BType == BarrierType::DownOut)) {
        knock_in = (K > H) ? C(+1.0, +1.0) : (A(+1.0) - B(+1.0) + D(+1.0, +1.0));
    } else if constexpr (Type == OptionType::Call && (BType == BarrierType::UpIn || BType == BarrierType::UpOut)) {
        knock_in = (K > H) ? A(+1.0) : (B(+1.0) - C(+1.0, -1.0) + D(+1.0, -1.0));
    } else if constexpr (Type == OptionType::Put && (BType == BarrierType::DownIn || BType == BarrierType::DownOut)) {
        knock_in = (K > H) ? (B(-1.0) - C(-1.0, +1.0) + D(-1.0, +1.0)) : A(-1.0);
    } else {
        knock_in = (K > H) ? (A(-1.0) - B(-1.0) + D(-1.0, -1.0)) : C(-1.0, -1.0);
    }

    if constexpr (BType == BarrierType::UpIn || BType == BarrierType::DownIn) {
        return knock_in;
    } else {
        return black_scholes_price<Type>(S, K, r, q, sigma, T) - knock_in;
    }
}

// Broadie-Glasserman-Kou (1997) continuity shift for discrete barriers.
template <OptionType Type, BarrierType BType>
double reiner_rubinstein_bgk(double S, double K, double H,
                             double r, double q, double sigma, double T,
                             std::size_t m) noexcept {
    constexpr double kBetaBGK = 0.5825971579390107; // -zeta(1/2) / sqrt(2*pi)
    const double dt = T / static_cast<double>(m);
    const double shift = kBetaBGK * sigma * std::sqrt(dt);
    const double h_adj = (BType == BarrierType::UpOut || BType == BarrierType::UpIn)
        ? (H * std::exp(shift))
        : (H * std::exp(-shift));
    return reiner_rubinstein_continuous<Type, BType>(S, K, h_adj, r, q, sigma, T);
}

// Goldman-Sosin-Gatto (1979) / Conze-Viswanathan (1991) Lookback closed forms
// (r != q; M_0 = m_0 = S_0; FixedStrike Call K >= S0, Put K <= S0).
template <OptionType Type, LookbackStyle Style>
double lookback_continuous(double S, double K, double r, double q, double sigma, double T) noexcept {
    const double b = r - q;
    const double sqrt_t = std::sqrt(T);
    const double sig_sqrt_t = sigma * sqrt_t;
    const double disc_r = std::exp(-r * T);
    const double disc_q = std::exp(-q * T);
    const double coef = (sigma * sigma) / (2.0 * b);

    if constexpr (Style == LookbackStyle::FloatingStrike) {
        const double a1 = (b + 0.5 * sigma * sigma) * T / sig_sqrt_t;
        const double a2 = a1 - sig_sqrt_t;
        const double a3 = a1 - (2.0 * b / sigma) * sqrt_t;
        if constexpr (Type == OptionType::Call) {
            return S * disc_q * norm_cdf(a1) - S * disc_r * norm_cdf(a2)
                 + S * coef * (disc_r * norm_cdf(-a3) - disc_q * norm_cdf(-a1));
        } else {
            return S * disc_r * norm_cdf(-a2) - S * disc_q * norm_cdf(-a1)
                 + S * coef * (disc_q * norm_cdf(a1) - disc_r * norm_cdf(a3));
        }
    } else {
        const double d1 = (std::log(S / K) + (b + 0.5 * sigma * sigma) * T) / sig_sqrt_t;
        const double d2 = d1 - sig_sqrt_t;
        const double pow_sk = std::pow(S / K, -2.0 * b / (sigma * sigma));
        if constexpr (Type == OptionType::Call) {
            return S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
                 + S * disc_r * coef * (-pow_sk * norm_cdf(d1 - (2.0 * b / sigma) * sqrt_t)
                                        + std::exp(b * T) * norm_cdf(d1));
        } else {
            return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)
                 + S * disc_r * coef * (pow_sk * norm_cdf(-d1 + (2.0 * b / sigma) * sqrt_t)
                                        - std::exp(b * T) * norm_cdf(-d1));
        }
    }
}

// Broadie-Glasserman-Kou (1999) continuity shift for discrete lookbacks.
template <OptionType Type, LookbackStyle Style>
double lookback_bgk(double S, double K, double r, double q, double sigma, double T,
                    std::size_t m) noexcept {
    constexpr double kBetaBGK = 0.5825971579390107;
    const double shift = kBetaBGK * sigma * std::sqrt(T / static_cast<double>(m));
    const double disc_q = std::exp(-q * T);
    if constexpr (Style == LookbackStyle::FloatingStrike && Type == OptionType::Call) {
        const double c_cont = lookback_continuous<Type, Style>(S, K, r, q, sigma, T);
        const double ep = std::exp(shift);
        return ep * c_cont - (ep - 1.0) * S * disc_q;
    } else if constexpr (Style == LookbackStyle::FloatingStrike && Type == OptionType::Put) {
        const double p_cont = lookback_continuous<Type, Style>(S, K, r, q, sigma, T);
        const double em = std::exp(-shift);
        return em * p_cont - (1.0 - em) * S * disc_q;
    } else if constexpr (Style == LookbackStyle::FixedStrike && Type == OptionType::Call) {
        const double ep = std::exp(shift);
        return std::exp(-shift) * lookback_continuous<Type, Style>(S, K * ep, r, q, sigma, T);
    } else {
        const double em = std::exp(-shift);
        return std::exp(shift) * lookback_continuous<Type, Style>(S, K * em, r, q, sigma, T);
    }
}

// Fang-Oosterlee (2008) COS Heston pricer with Albrecher et al. (2007) CF.
template <OptionType Type>
double heston_cos_price(const HestonParams& p, double K,
                        std::size_t n_cos = 128, double L = 10.0) noexcept {
    using cpx = std::complex<double>;
    const double x = std::log(p.s0 / K);
    const double T = p.T;
    const double k = p.kappa;
    const double th = p.theta;
    const double xi = p.xi;
    const double rho = p.rho;
    const double v0 = p.v0;
    const double ekT = std::exp(-k * T);

    // Cumulants of Y_T = ln(S_T / S_0) (Fang & Oosterlee 2008, Table 11).
    const double c1 = (p.r - p.q) * T
                    + (1.0 - ekT) * (th - v0) / (2.0 * k)
                    - 0.5 * th * T;
    const double c2 = (1.0 / (8.0 * k * k * k)) * (
        xi * T * k * ekT * (v0 - th) * (8.0 * k * rho - 4.0 * xi)
        + k * rho * xi * (1.0 - ekT) * (16.0 * th - 8.0 * v0)
        + 2.0 * th * k * T * (-4.0 * k * rho * xi + xi * xi + 4.0 * k * k)
        + xi * xi * ((th - 2.0 * v0) * ekT * ekT + th * (6.0 * ekT - 7.0) + 2.0 * v0)
        + 8.0 * k * k * (v0 - th) * (1.0 - ekT)
    );

    const double width = L * std::sqrt(std::fabs(c2));
    const double a = x + c1 - width;
    const double b = x + c1 + width;
    const double bma = b - a;

    auto cf_heston = [&](double u) noexcept -> cpx {
        const cpx iu(0.0, u);
        const cpx beta = k - rho * xi * iu;
        const cpx d = std::sqrt(beta * beta + xi * xi * (u * u + iu));
        const cpx g = (beta - d) / (beta + d);
        const cpx exp_mdT = std::exp(-d * T);
        const cpx one_m_ge = cpx(1.0, 0.0) - g * exp_mdT;
        const cpx one_m_g = cpx(1.0, 0.0) - g;
        const cpx C = iu * (p.r - p.q) * T
                    + (k * th / (xi * xi)) * ((beta - d) * T - 2.0 * std::log(one_m_ge / one_m_g));
        const cpx D = ((beta - d) / (xi * xi)) * ((cpx(1.0, 0.0) - exp_mdT) / one_m_ge);
        return std::exp(C + D * v0);
    };

    // Price Put on [a, 0] so exp(y) <= 1 avoids overflow; recover Call via put-call parity.
    double sum = 0.0;
    for (std::size_t n = 0; n < n_cos; ++n) {
        const double dn = static_cast<double>(n);
        const double omega = dn * std::numbers::pi / bma;
        const double c_lo = a;
        const double c_hi = 0.0;

        const double cos_hi = std::cos(omega * (c_hi - a));
        const double sin_hi = std::sin(omega * (c_hi - a));
        const double cos_lo = 1.0;
        const double sin_lo = 0.0;
        const double exp_hi = 1.0;
        const double exp_lo = std::exp(c_lo);

        const double chi = (1.0 / (1.0 + omega * omega)) * (
            cos_hi * exp_hi - cos_lo * exp_lo + omega * (sin_hi * exp_hi - sin_lo * exp_lo)
        );
        const double psi = (n == 0)
            ? (c_hi - c_lo)
            : ((sin_hi - sin_lo) / omega);
        const double u_put = (2.0 / bma) * (-chi + psi);

        const cpx phi = cf_heston(omega) * std::exp(cpx(0.0, omega * (x - a)));
        const double weight = (n == 0) ? 0.5 : 1.0;
        sum += weight * phi.real() * u_put;
    }

    const double put_price = K * std::exp(-p.r * T) * sum;
    if constexpr (Type == OptionType::Put) {
        return put_price;
    } else {
        return put_price + p.s0 * std::exp(-p.q * T) - K * std::exp(-p.r * T);
    }
}

void test_stats_and_european() {
    std::printf("\n== Chunk Stats, European GBM MC & 5-Greek Parity ==\n");

    {
        auto verify_vs_welford = [](const char* label, const std::vector<double>& samples,
                                    std::size_t chunk_size) {
            double w_mean = 0.0;
            double w_m2 = 0.0;
            for (std::size_t i = 0; i < samples.size(); ++i) {
                const double x = samples[i];
                const double delta = x - w_mean;
                w_mean += delta / static_cast<double>(i + 1);
                w_m2 += delta * (x - w_mean);
            }
            const double w_var = w_m2 / static_cast<double>(samples.size() - 1);

            std::vector<WorkerAccum> chunks;
            for (std::size_t start = 0; start < samples.size(); start += chunk_size) {
                const std::size_t end = std::min(samples.size(), start + chunk_size);
                ChunkAccum ca;
                std::size_t i = start;
                while (i + 4 <= end) {
                    ca.add_vec(Vec4d::load_unaligned(samples.data() + i));
                    i += 4;
                }
                if (i < end) {
                    alignas(32) double tmp[4] = {0.0, 0.0, 0.0, 0.0};
                    const std::size_t rem = end - i;
                    for (std::size_t k = 0; k < rem; ++k) tmp[k] = samples[i + k];
                    ca.add_partial(Vec4d::load_aligned(tmp), rem);
                }
                chunks.push_back(ca.finalize());
            }
            const PricingResult res = reduce_chunks(chunks);
            const double mean_rel = std::fabs(res.price - w_mean) / std::max(1.0, std::fabs(w_mean));
            const double var_rel = std::fabs(res.variance - w_var) / std::max(1e-12, w_var);
            std::printf("  %-26s mean_rel=%.2e  var_rel=%.2e\n", label, mean_rel, var_rel);
            check::is_true(label, res.paths == samples.size() && mean_rel < 1e-11 && var_rel < 1e-11);
        };

        Xoshiro256PlusPlus rng(0x57A75ULL);
        std::vector<double> normal_s(10'003);
        std::vector<double> cancel_s(10'003);
        for (std::size_t i = 0; i < normal_s.size(); ++i) {
            const double u = rng.next_uniform_open01() - 0.5;
            normal_s[i] = 10.0 + 3.5 * u;
            // mean^2 / var ~ 1.2e9 stresses catastrophic cancellation in unshifted variance.
            cancel_s[i] = 10'000.0 + 1.0 * u;
        }
        verify_vs_welford("chan_normal_vs_welford", normal_s, 1024);
        verify_vs_welford("chan_deep_itm_cancel", cancel_s, 1024);

        ChunkAccum nan_ca;
        nan_ca.add_scalar(1.0);
        nan_ca.add_scalar(std::numeric_limits<double>::quiet_NaN());
        nan_ca.add_scalar(2.0);
        const WorkerAccum nan_w = nan_ca.finalize();
        const PricingResult nan_pr = to_pricing_result(nan_w);
        check::is_true("chunk_accum_preserves_nan_m2",
            std::isnan(nan_w.mean) && std::isnan(nan_w.m2) && std::isnan(nan_pr.std_error));
    }

    {
        const auto mp = MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.0);
        const auto payoff = EuropeanPayoff<OptionType::Call>::create(100.0);
        const double bs_ref = black_scholes_price<OptionType::Call>(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);

        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
            eng_std(GBMProcess(*mp), *payoff);
        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            eng_anti(GBMProcess(*mp), *payoff);

        SimConfig bad_paths{};
        bad_paths.paths = 1;
        check::is_true("mc_std_rejects_paths_1",
            !eng_std.price(bad_paths).has_value() &&
            eng_std.price(bad_paths).error() == ValidationError::InvalidPaths);
        bad_paths.paths = 3;
        check::is_true("mc_anti_rejects_paths_3",
            !eng_anti.price(bad_paths).has_value() &&
            eng_anti.price(bad_paths).error() == ValidationError::InvalidPaths);
        SimConfig bad_steps{};
        bad_steps.paths = 100;
        bad_steps.steps = 0;
        bad_steps.force_multi_step = true;
        check::is_true("mc_rejects_zero_steps",
            !eng_std.price(bad_steps).has_value() &&
            eng_std.price(bad_steps).error() == ValidationError::InvalidSteps);

        bool partial_ok = true;
        for (std::size_t n : {std::size_t{17}, std::size_t{1000}, std::size_t{4095},
                              std::size_t{4097}, std::size_t{20000}}) {
            SimConfig cfg{};
            cfg.paths = n;
            cfg.seed = 101;
            const auto rs = eng_std.price(cfg);
            const auto ra = eng_anti.price(cfg);
            if (!rs.has_value() || rs->paths != n || !std::isfinite(rs->price) ||
                !std::isfinite(rs->std_error) || rs->std_error <= 0.0) {
                partial_ok = false;
            }
            if (!ra.has_value() || ra->paths != 2 * (n / 2) || !std::isfinite(ra->price) ||
                !std::isfinite(ra->std_error) || ra->std_error <= 0.0) {
                partial_ok = false;
            }
            if (n >= 4095) {
                if (std::fabs((rs->price - bs_ref) / rs->std_error) > 4.0 ||
                    std::fabs((ra->price - bs_ref) / ra->std_error) > 4.0) {
                    partial_ok = false;
                }
            }
        }
        check::is_true("partial_last_chunk_exact", partial_ok);
    }

    {
        const std::size_t n_1step = 200'000;
#if defined(PRICER_FAST_TESTS)
        const std::size_t n_multi = 50'000;
#else
        const std::size_t n_multi = 100'000;
#endif
        const auto mp_q = MarketParams::create(100.0, 0.05, 0.25, 1.0, 0.03);
        const auto mp_neg_r = MarketParams::create(100.0, -0.02, 0.20, 1.5, 0.01);

        const double ref_c_q = black_scholes_price<OptionType::Call>(100.0, 105.0, 0.05, 0.03, 0.25, 1.0);
        const double ref_p_q = black_scholes_price<OptionType::Put>(100.0, 105.0, 0.05, 0.03, 0.25, 1.0);
        const double ref_c_nr = black_scholes_price<OptionType::Call>(100.0, 95.0, -0.02, 0.01, 0.20, 1.5);
        const double ref_p_nr = black_scholes_price<OptionType::Put>(100.0, 95.0, -0.02, 0.01, 0.20, 1.5);

        SimConfig cfg1{.paths = n_1step, .steps = 1, .seed = 42, .threads = 1, .force_multi_step = false};
        SimConfig cfgm{.paths = n_multi, .steps = 64, .seed = 43, .threads = 1, .force_multi_step = true};

        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
            ec_std(GBMProcess(*mp_q), EuropeanPayoff<OptionType::Call>(105.0));
        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            ec_anti(GBMProcess(*mp_q), EuropeanPayoff<OptionType::Call>(105.0));
        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Put>, AntitheticSampling>
            ep_anti(GBMProcess(*mp_q), EuropeanPayoff<OptionType::Put>(105.0));

        const auto rc_std = ec_std.price(cfg1);
        const auto rc_anti = ec_anti.price(cfg1);
        const auto rp_anti = ep_anti.price(cfg1);
        const auto rc_multi = ec_anti.price(cfgm);

        check::within_se("euro_call_q_1step_std", rc_std->price, rc_std->std_error, ref_c_q, 4.0);
        check::within_se("euro_call_q_1step_anti", rc_anti->price, rc_anti->std_error, ref_c_q, 4.0);
        check::within_se("euro_put_q_1step_anti", rp_anti->price, rp_anti->std_error, ref_p_q, 4.0);
        check::within_se("euro_call_q_multistep", rc_multi->price, rc_multi->std_error, ref_c_q, 4.0);

        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            ec_nr(GBMProcess(*mp_neg_r), EuropeanPayoff<OptionType::Call>(95.0));
        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Put>, StandardSampling>
            ep_nr(GBMProcess(*mp_neg_r), EuropeanPayoff<OptionType::Put>(95.0));
        const auto rc_nr = ec_nr.price(cfg1);
        const auto rp_nr = ep_nr.price(cfg1);
        check::within_se("euro_call_neg_r_anti", rc_nr->price, rc_nr->std_error, ref_c_nr, 4.0);
        check::within_se("euro_put_neg_r_std", rp_nr->price, rp_nr->std_error, ref_p_nr, 4.0);
    }

    {
        const auto mp = MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.0);
        SimConfig cfg{.paths = 200'000, .steps = 1, .seed = 777};
        auto ratio_at_K = [&](double K) -> double {
            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
                es{GBMProcess(*mp), EuropeanPayoff<OptionType::Call>(K)};
            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                ea{GBMProcess(*mp), EuropeanPayoff<OptionType::Call>(K)};
            const auto rs = es.price(cfg);
            const auto ra = ea.price(cfg);
            const double r = rs->std_error / ra->std_error;
            return r * r;
        };
        const double r60 = ratio_at_K(60.0);
        const double r100 = ratio_at_K(100.0);
        const double r140 = ratio_at_K(140.0);
        std::printf("  Antithetic equal-work var_ratio: K=60: %.2fx, K=100: %.2fx, K=140: %.2fx\n",
                    r60, r100, r140);
        check::is_true("anti_var_ratio_atm_gt_1.5", r100 > 1.5);
        check::is_true("anti_var_ratio_itm_gt_15", r60 > 15.0);
        check::is_true("anti_var_ratio_monotonic", r60 > r100 && r100 > r140 && r140 > 1.0);
    }

    {
        const auto mp = MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.02);
        const double K = 100.0;
        const GreeksResult ref_c = black_scholes_greeks<OptionType::Call>(100.0, K, 0.05, 0.02, 0.20, 1.0);
        const GreeksResult ref_p = black_scholes_greeks<OptionType::Put>(100.0, K, 0.05, 0.02, 0.20, 1.0);

        SimConfig cfg{.paths = 200'000, .steps = 1, .seed = 2025};
        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            ec{GBMProcess(*mp), EuropeanPayoff<OptionType::Call>(K)};
        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Put>, AntitheticSampling>
            ep{GBMProcess(*mp), EuropeanPayoff<OptionType::Put>(K)};

        const auto gc = ec.price_with_greeks(cfg);
        const auto gp = ep.price_with_greeks(cfg);
        check::within_se("euro_call_mc_price", gc->price.value, gc->price.std_error, ref_c.price, 4.0);
        check::within_se("euro_call_mc_delta", gc->delta.value, gc->delta.std_error, ref_c.delta, 4.0);
        check::within_se("euro_call_mc_gamma", gc->gamma.value, gc->gamma.std_error, ref_c.gamma, 4.0);
        check::within_se("euro_call_mc_vega",  gc->vega.value,  gc->vega.std_error,  ref_c.vega,  4.0);
        check::within_se("euro_call_mc_theta", gc->theta.value, gc->theta.std_error, ref_c.theta, 4.0);
        check::within_se("euro_call_mc_rho",   gc->rho.value,   gc->rho.std_error,   ref_c.rho,   4.0);

        check::within_se("euro_put_mc_price",  gp->price.value, gp->price.std_error, ref_p.price, 4.0);
        check::within_se("euro_put_mc_delta",  gp->delta.value, gp->delta.std_error, ref_p.delta, 4.0);
        check::within_se("euro_put_mc_gamma",  gp->gamma.value, gp->gamma.std_error, ref_p.gamma, 4.0);
        check::within_se("euro_put_mc_vega",   gp->vega.value,  gp->vega.std_error,  ref_p.vega,  4.0);
        check::within_se("euro_put_mc_theta",  gp->theta.value, gp->theta.std_error, ref_p.theta, 4.0);
        check::within_se("euro_put_mc_rho",    gp->rho.value,   gp->rho.std_error,   ref_p.rho,   4.0);

        bool theta_scan_ok = true;
        for (double h : {0.001, 0.005, 0.02}) {
            GreeksConfig gcfg{};
            gcfg.theta_h = h;
            const auto gt = ec.price_with_greeks(cfg, gcfg);
            if (!gt.has_value() || !(gt->theta.value < 0.0) ||
                std::fabs((gt->theta.value - ref_c.theta) / gt->theta.std_error) > 4.0) {
                theta_scan_ok = false;
            }
        }
        check::is_true("euro_theta_h_scan_stable", theta_scan_ok);
    }
}

void test_exotics_and_greeks() {
    std::printf("\n== Exotics (Asian, Barrier, Lookback) & Exotic Greeks ==\n");

#if defined(PRICER_FAST_TESTS)
    constexpr std::size_t kMultiPaths = 50'000;
    constexpr std::size_t kRefinePaths = 50'000;
#else
    constexpr std::size_t kMultiPaths = 120'000;
    constexpr std::size_t kRefinePaths = 150'000;
#endif

    {
        const auto mp = MarketParams::create(100.0, 0.05, 0.25, 1.0, 0.02);
        const double K = 100.0;
        for (std::size_t m : {std::size_t{12}, std::size_t{252}}) {
            SimConfig cfg{.paths = kMultiPaths, .steps = m, .seed = 400 + m};
            MonteCarloEngine<GBMProcess, GeometricAsianPayoff<OptionType::Call>, AntitheticSampling>
                gc{GBMProcess(*mp), GeometricAsianPayoff<OptionType::Call>(K)};
            MonteCarloEngine<GBMProcess, GeometricAsianPayoff<OptionType::Put>, AntitheticSampling>
                gp{GBMProcess(*mp), GeometricAsianPayoff<OptionType::Put>(K)};

            const auto rc = gc.price(cfg);
            const auto rp = gp.price(cfg);
            const double ref_c = kemna_vorst_discrete<OptionType::Call>(100.0, K, 0.05, 0.02, 0.25, 1.0, m);
            const double ref_p = kemna_vorst_discrete<OptionType::Put>(100.0, K, 0.05, 0.02, 0.25, 1.0, m);

            check::within_se(m == 12 ? "geom_asian_call_m12" : "geom_asian_call_m252",
                             rc->price, rc->std_error, ref_c, 4.0);
            check::within_se(m == 12 ? "geom_asian_put_m12" : "geom_asian_put_m252",
                             rp->price, rp->std_error, ref_p, 4.0);
        }
    }

    {
        const auto mp_pos = MarketParams::create(100.0, 0.05, 0.25, 1.0, 0.0);
        const auto mp_r0  = MarketParams::create(100.0, 0.00, 0.25, 1.0, 0.0);
        SimConfig cfg{.paths = kMultiPaths, .steps = 64, .seed = 505, .threads = 1, .force_multi_step = true};

        MonteCarloEngine<GBMProcess, GeometricAsianPayoff<OptionType::Call>, StandardSampling>
            gc(GBMProcess(*mp_pos), GeometricAsianPayoff<OptionType::Call>(100.0));
        MonteCarloEngine<GBMProcess, ArithmeticAsianPayoff<OptionType::Call>, StandardSampling>
            ac(GBMProcess(*mp_pos), ArithmeticAsianPayoff<OptionType::Call>(100.0));
        const double euro_c = black_scholes_price<OptionType::Call>(100.0, 100.0, 0.05, 0.0, 0.25, 1.0);

        const auto rgc = gc.price(cfg);
        const auto rac = ac.price(cfg);
        check::is_true("asian_call_jensen_bounds",
            rgc->price <= rac->price && rac->price <= euro_c);

        MonteCarloEngine<GBMProcess, GeometricAsianPayoff<OptionType::Put>, StandardSampling>
            gp(GBMProcess(*mp_r0), GeometricAsianPayoff<OptionType::Put>(100.0));
        MonteCarloEngine<GBMProcess, ArithmeticAsianPayoff<OptionType::Put>, StandardSampling>
            ap(GBMProcess(*mp_r0), ArithmeticAsianPayoff<OptionType::Put>(100.0));
        const double euro_p = black_scholes_price<OptionType::Put>(100.0, 100.0, 0.0, 0.0, 0.25, 1.0);

        const auto rgp = gp.price(cfg);
        const auto rap = ap.price(cfg);
        check::is_true("asian_put_jensen_bounds",
            rap->price <= rgp->price && rap->price <= euro_p);
    }

    {
        check::is_true("barrier_up_rejects_s0_ge_b",
            BarrierPayoff<OptionType::Call, BarrierType::UpOut>::create(100.0, 100.0, 100.0).error()
                == ValidationError::InvalidBarrier);
        check::is_true("barrier_down_rejects_s0_le_b",
            BarrierPayoff<OptionType::Put, BarrierType::DownIn>::create(100.0, 100.0, 100.0).error()
                == ValidationError::InvalidBarrier);

        const auto mp = MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.01);
        SimConfig cfg_bar{.paths = kMultiPaths, .steps = 252, .seed = 601, .threads = 1, .force_multi_step = true};

        MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
            euro_m(GBMProcess(*mp), EuropeanPayoff<OptionType::Call>(100.0));
        MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>, StandardSampling>
            uo_c(GBMProcess(*mp), BarrierPayoff<OptionType::Call, BarrierType::UpOut>(100.0, 125.0));
        MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpIn>, StandardSampling>
            ui_c(GBMProcess(*mp), BarrierPayoff<OptionType::Call, BarrierType::UpIn>(100.0, 125.0));
        MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::DownOut>, StandardSampling>
            do_c(GBMProcess(*mp), BarrierPayoff<OptionType::Call, BarrierType::DownOut>(100.0, 80.0));
        MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::DownIn>, StandardSampling>
            di_c(GBMProcess(*mp), BarrierPayoff<OptionType::Call, BarrierType::DownIn>(100.0, 80.0));

        const auto r_euro = euro_m.price(cfg_bar);
        const auto r_uo = uo_c.price(cfg_bar);
        const auto r_ui = ui_c.price(cfg_bar);
        const auto r_do = do_c.price(cfg_bar);
        const auto r_di = di_c.price(cfg_bar);

        check::close("barrier_crn_up_inout_parity", r_uo->price + r_ui->price, r_euro->price, 1e-12);
        check::close("barrier_crn_dn_inout_parity", r_do->price + r_di->price, r_euro->price, 1e-12);

        MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>, StandardSampling>
            uo_far(GBMProcess(*mp), BarrierPayoff<OptionType::Call, BarrierType::UpOut>(100.0, 1e6));
        MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::DownOut>, StandardSampling>
            do_far(GBMProcess(*mp), BarrierPayoff<OptionType::Call, BarrierType::DownOut>(100.0, 1e-6));
        check::close("barrier_up_far_eq_euro", uo_far.price(cfg_bar)->price, r_euro->price, 1e-12);
        check::close("barrier_dn_far_eq_euro", do_far.price(cfg_bar)->price, r_euro->price, 1e-12);

        auto check_bgk = [&]<OptionType OT, BarrierType BT>(const char* name, double H) {
            MonteCarloEngine<GBMProcess, BarrierPayoff<OT, BT>, AntitheticSampling>
                eng(GBMProcess(*mp), BarrierPayoff<OT, BT>(100.0, H));
            const auto r = eng.price(cfg_bar);
            const double ref = reiner_rubinstein_bgk<OT, BT>(100.0, 100.0, H, 0.05, 0.01, 0.20, 1.0, 252);
            // 0.025 slack covers O(1/m) BGK continuity-shift residual.
            const double diff = std::fabs(r->price - ref);
            const bool ok = std::isfinite(r->price) && std::isfinite(r->std_error)
                         && diff <= 4.0 * r->std_error + 0.025;
            std::printf("  %-26s mc=%10.5f  bgk=%10.5f  diff=%+.5f  se=%.5f  %s\n",
                        name, r->price, ref, r->price - ref, r->std_error, ok ? "ok" : "FAIL");
            check::record(ok);
        };
        check_bgk.template operator()<OptionType::Call, BarrierType::UpOut>("bar_call_upout_bgk", 125.0);
        check_bgk.template operator()<OptionType::Call, BarrierType::UpIn>("bar_call_upin_bgk", 125.0);
        check_bgk.template operator()<OptionType::Call, BarrierType::DownOut>("bar_call_dnout_bgk", 80.0);
        check_bgk.template operator()<OptionType::Call, BarrierType::DownIn>("bar_call_dnin_bgk", 80.0);
        check_bgk.template operator()<OptionType::Put,  BarrierType::UpOut>("bar_put_upout_bgk", 125.0);
        check_bgk.template operator()<OptionType::Put,  BarrierType::UpIn>("bar_put_upin_bgk", 125.0);
        check_bgk.template operator()<OptionType::Put,  BarrierType::DownOut>("bar_put_dnout_bgk", 80.0);
        check_bgk.template operator()<OptionType::Put,  BarrierType::DownIn>("bar_put_dnin_bgk", 80.0);

        const double rr_cont = reiner_rubinstein_continuous<OptionType::Call, BarrierType::UpOut>(
            100.0, 100.0, 125.0, 0.05, 0.01, 0.20, 1.0);
        double errs[3] = {0.0, 0.0, 0.0};
        const std::size_t ms[3] = {64, 252, 1000};
        for (int k = 0; k < 3; ++k) {
            SimConfig cfg_r{.paths = kRefinePaths, .steps = ms[k], .seed = 710 + static_cast<std::uint64_t>(k)};
            MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>, AntitheticSampling>
                eng(GBMProcess(*mp), BarrierPayoff<OptionType::Call, BarrierType::UpOut>(100.0, 125.0));
            const auto r = eng.price(cfg_r);
            const double bgk = reiner_rubinstein_bgk<OptionType::Call, BarrierType::UpOut>(
                100.0, 100.0, 125.0, 0.05, 0.01, 0.20, 1.0, ms[k]);
            errs[k] = std::fabs(r->price - rr_cont);
            std::printf("  Barrier UpOut refinement m=%4zu: mc=%.5f cont=%.5f |mc-cont|=%.5f |mc-bgk|=%.5f (se=%.5f)\n",
                        ms[k], r->price, rr_cont, errs[k], std::fabs(r->price - bgk), r->std_error);
        }
        check::is_true("barrier_monotonic_refinement", errs[0] > errs[1] && errs[1] > errs[2]);
    }

    {
        const auto mp = MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.01);
        SimConfig cfg_lb{.paths = kMultiPaths, .steps = 252, .seed = 801};

        auto check_lb_bgk = [&]<OptionType OT, LookbackStyle ST>(const char* name, double K) {
            LookbackPayoff<OT, ST> pf = [&]() {
                if constexpr (ST == LookbackStyle::FloatingStrike) {
                    return LookbackPayoff<OT, ST>();
                } else {
                    return LookbackPayoff<OT, ST>(K);
                }
            }();
            MonteCarloEngine<GBMProcess, LookbackPayoff<OT, ST>, AntitheticSampling>
                eng(GBMProcess(*mp), pf);
            const auto r = eng.price(cfg_lb);
            const double bgk = lookback_bgk<OT, ST>(100.0, K, 0.05, 0.01, 0.20, 1.0, 252);
            const double diff = std::fabs(r->price - bgk);
            const bool ok = std::isfinite(r->price) && std::isfinite(r->std_error)
                         && diff <= 4.0 * r->std_error + 0.05;
            std::printf("  %-26s mc=%10.5f  bgk=%10.5f  diff=%+.5f  se=%.5f  %s\n",
                        name, r->price, bgk, r->price - bgk, r->std_error, ok ? "ok" : "FAIL");
            check::record(ok);
            return *r;
        };

        const auto r_fix_c = check_lb_bgk.template operator()<OptionType::Call, LookbackStyle::FixedStrike>("lb_fixed_call_bgk", 100.0);
        (void)check_lb_bgk.template operator()<OptionType::Put, LookbackStyle::FixedStrike>("lb_fixed_put_bgk", 100.0);
        (void)check_lb_bgk.template operator()<OptionType::Call, LookbackStyle::FloatingStrike>("lb_float_call_bgk", 100.0);
        (void)check_lb_bgk.template operator()<OptionType::Put, LookbackStyle::FloatingStrike>("lb_float_put_bgk", 100.0);

        const double euro_c = black_scholes_price<OptionType::Call>(100.0, 100.0, 0.05, 0.01, 0.20, 1.0);
        check::is_true("lb_fixed_call_ge_euro", r_fix_c.price >= euro_c);

        const double lb_cont = lookback_continuous<OptionType::Call, LookbackStyle::FloatingStrike>(
            100.0, 100.0, 0.05, 0.01, 0.20, 1.0);
        double lb_errs[3] = {0.0, 0.0, 0.0};
        const std::size_t ms[3] = {64, 252, 1000};
        for (int k = 0; k < 3; ++k) {
            SimConfig cfg_r{.paths = kRefinePaths, .steps = ms[k], .seed = 820 + static_cast<std::uint64_t>(k)};
            MonteCarloEngine<GBMProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>, AntitheticSampling>
                eng{GBMProcess(*mp), LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>()};
            const auto r = eng.price(cfg_r);
            const double bgk = lookback_bgk<OptionType::Call, LookbackStyle::FloatingStrike>(
                100.0, 100.0, 0.05, 0.01, 0.20, 1.0, ms[k]);
            lb_errs[k] = std::fabs(r->price - lb_cont);
            std::printf("  Lookback FloatCall refinement m=%4zu: mc=%.5f cont=%.5f |mc-cont|=%.5f |mc-bgk|=%.5f (se=%.5f)\n",
                        ms[k], r->price, lb_cont, lb_errs[k], std::fabs(r->price - bgk), r->std_error);
        }
        check::is_true("lookback_monotonic_refinement", lb_errs[0] > lb_errs[1] && lb_errs[1] > lb_errs[2]);
    }

    {
        const auto mp = MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.02);
        const double K = 100.0;
        const std::size_t m = 64;
        SimConfig cfg_g{.paths = kMultiPaths, .steps = m, .seed = 901};

        {
            MonteCarloEngine<GBMProcess, GeometricAsianPayoff<OptionType::Call>, AntitheticSampling>
                eng{GBMProcess(*mp), GeometricAsianPayoff<OptionType::Call>(K)};
            const auto g = eng.price_with_greeks(cfg_g);

            auto kv = [&](double s_, double r_, double sig_, double t_) {
                return kemna_vorst_discrete<OptionType::Call>(s_, K, r_, 0.02, sig_, t_, m);
            };
            const double h_s = 1e-3, h_v = 1e-4, h_r = 1e-4, h_t = 1e-4;
            const double p0 = kv(100.0, 0.05, 0.20, 1.0);
            const double ref_d = (kv(100.0 + h_s, 0.05, 0.20, 1.0) - kv(100.0 - h_s, 0.05, 0.20, 1.0)) / (2.0 * h_s);
            const double ref_g = (kv(100.0 + 1e-2, 0.05, 0.20, 1.0) - 2.0 * p0 + kv(100.0 - 1e-2, 0.05, 0.20, 1.0)) / 1e-4;
            const double ref_v = (kv(100.0, 0.05, 0.20 + h_v, 1.0) - kv(100.0, 0.05, 0.20 - h_v, 1.0)) / (2.0 * h_v);
            const double ref_t = (kv(100.0, 0.05, 0.20, 1.0 - h_t) - kv(100.0, 0.05, 0.20, 1.0 + h_t)) / (2.0 * h_t);
            const double ref_r = (kv(100.0, 0.05 + h_r, 0.20, 1.0) - kv(100.0, 0.05 - h_r, 0.20, 1.0)) / (2.0 * h_r);

            check::within_se("geom_asian_delta", g->delta.value, g->delta.std_error, ref_d, 4.0);
            check::within_se("geom_asian_gamma", g->gamma.value, g->gamma.std_error, ref_g, 4.0);
            check::within_se("geom_asian_vega",  g->vega.value,  g->vega.std_error,  ref_v, 4.0);
            check::within_se("geom_asian_theta", g->theta.value, g->theta.std_error, ref_t, 4.0);
            check::within_se("geom_asian_rho",   g->rho.value,   g->rho.std_error,   ref_r, 4.0);
        }

        {
            MonteCarloEngine<GBMProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>, AntitheticSampling>
                eng{GBMProcess(*mp), LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>()};
            const auto g = eng.price_with_greeks(cfg_g);
            check::close("lb_float_homogeneity_delta", g->delta.value, g->price.value / mp->s0, 1e-12);
            check::is_true("lb_float_homogeneity_gamma", g->gamma.value == 0.0 && g->gamma.std_error == 0.0);
        }

        {
            auto check_exotic_fd = [&]<typename PayoffT>(const char* tag, PayoffT pf) {
                MonteCarloEngine<GBMProcess, PayoffT, AntitheticSampling> eng(GBMProcess(*mp), pf);
                const auto g = eng.price_with_greeks(cfg_g);

                auto price_at = [&](MarketParams p_mod) {
                    MonteCarloEngine<GBMProcess, PayoffT, AntitheticSampling> e(GBMProcess(p_mod), pf);
                    return e.price(cfg_g)->price;
                };
                MarketParams sp = *mp, sm = *mp;
                sp.s0 += 0.5; sm.s0 -= 0.5;
                const double fd_d = (price_at(sp) - price_at(sm)) / 1.0;

                MarketParams gp_p = *mp, gp_m = *mp;
                gp_p.s0 = mp->s0 * 1.01; gp_m.s0 = mp->s0 * 0.99;
                MonteCarloEngine<GBMProcess, PayoffT, AntitheticSampling> ep_d(GBMProcess(gp_p), pf);
                MonteCarloEngine<GBMProcess, PayoffT, AntitheticSampling> em_d(GBMProcess(gp_m), pf);
                const double fd_g = (ep_d.price_with_greeks(cfg_g)->delta.value -
                                     em_d.price_with_greeks(cfg_g)->delta.value) / (0.02 * mp->s0);

                MarketParams vp = *mp, vm = *mp;
                vp.sigma += 1e-3; vm.sigma -= 1e-3;
                const double fd_v = (price_at(vp) - price_at(vm)) / 2e-3;

                MarketParams rp = *mp, rm = *mp;
                rp.r += 1e-4; rm.r -= 1e-4;
                const double fd_r = (price_at(rp) - price_at(rm)) / 2e-4;

                std::printf("  %-14s D=%.4f(fd %.4f) G=%.5f(fd %.5f) V=%.4f(fd %.4f) R=%.4f(fd %.4f)\n",
                            tag, g->delta.value, fd_d, g->gamma.value, fd_g,
                            g->vega.value, fd_v, g->rho.value, fd_r);
                const bool ok =
                    std::isfinite(g->delta.value) && std::fabs((g->delta.value - fd_d) / g->delta.std_error) <= 4.0 &&
                    std::isfinite(g->gamma.value) && std::fabs((g->gamma.value - fd_g) / std::max(g->gamma.std_error, 1e-6)) <= 4.0 &&
                    std::isfinite(g->vega.value)  && std::fabs((g->vega.value  - fd_v) / g->vega.std_error)  <= 4.0 &&
                    std::isfinite(g->rho.value)   && std::fabs((g->rho.value   - fd_r) / g->rho.std_error)   <= 4.0 &&
                    std::isfinite(g->theta.value);
                check::is_true(tag, ok);
            };
            check_exotic_fd("arith_asian_5g", ArithmeticAsianPayoff<OptionType::Call>(K));
            // K = 105 > S0 avoids the t_0 kink at M_0 = S0 = K.
            check_exotic_fd("fixed_lb_5g",    LookbackPayoff<OptionType::Call, LookbackStyle::FixedStrike>(105.0));
        }

        {
            const double H = 85.0;
            const std::size_t m_bar = 128;
            SimConfig cfg_b{.paths = kMultiPaths, .steps = m_bar, .seed = 955};
            GreeksConfig gcfg{};
            gcfg.eps_s0_rel = 1e-2;

            BarrierPayoff<OptionType::Call, BarrierType::DownOut> bar_pf(K, H);
            MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::DownOut>, StandardSampling>
                eng_bar(GBMProcess(*mp), bar_pf);
            const auto gb = eng_bar.price_with_barrier_greeks(cfg_b, gcfg);

            MarketParams mp_p = *mp, mp_m = *mp;
            mp_p.s0 = mp->s0 * (1.0 + gcfg.eps_s0_rel);
            mp_m.s0 = mp->s0 * (1.0 - gcfg.eps_s0_rel);
            const double two_eps = mp_p.s0 - mp_m.s0;

            MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::DownOut>, StandardSampling>
                eng_p(GBMProcess(mp_p), bar_pf);
            MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::DownOut>, StandardSampling>
                eng_m(GBMProcess(mp_m), bar_pf);
            SimConfig cfg_p = cfg_b; cfg_p.seed = 956;
            SimConfig cfg_m = cfg_b; cfg_m.seed = 957;
            const auto rp = eng_p.price(cfg_p);
            const auto rm = eng_m.price(cfg_m);
            const double indep_delta = (rp->price - rm->price) / two_eps;
            const double indep_se = std::sqrt(rp->std_error * rp->std_error +
                                              rm->std_error * rm->std_error) / two_eps;
            const double crn_var_ratio = (indep_se / gb->delta.std_error) * (indep_se / gb->delta.std_error);

            const double bgk_dp = reiner_rubinstein_bgk<OptionType::Call, BarrierType::DownOut>(
                mp_p.s0, K, H, mp->r, mp->q, mp->sigma, mp->T, m_bar);
            const double bgk_dm = reiner_rubinstein_bgk<OptionType::Call, BarrierType::DownOut>(
                mp_m.s0, K, H, mp->r, mp->q, mp->sigma, mp->T, m_bar);
            const double bgk_delta = (bgk_dp - bgk_dm) / two_eps;

            std::printf("  Barrier DownOut CRN Delta: %.5f (se=%.5f) vs indep=%.5f (se=%.5f) BGK=%.5f var_ratio=%.1fx\n",
                        gb->delta.value, gb->delta.std_error, indep_delta, indep_se, bgk_delta, crn_var_ratio);
            check::within_se("barrier_crn_delta_vs_indep", gb->delta.value, indep_se, indep_delta, 4.0);
            check::is_true("barrier_crn_delta_vs_bgk",
                std::fabs(gb->delta.value - bgk_delta) <= 4.0 * gb->delta.std_error + 0.02);
            check::is_true("barrier_crn_var_ratio_ge_10", crn_var_ratio >= 10.0);
            check::is_true("barrier_other_greeks_finite",
                std::isfinite(gb->vega.value) && std::isfinite(gb->theta.value) && std::isfinite(gb->rho.value));

            BarrierPayoff<OptionType::Call, BarrierType::UpOut> uo_pf(K, 125.0);
            MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>, StandardSampling>
                eng_uo(GBMProcess(*mp), uo_pf);
            const auto guo = eng_uo.price_with_barrier_greeks(cfg_b, gcfg);
            const double bgk_uo_d = (
                reiner_rubinstein_bgk<OptionType::Call, BarrierType::UpOut>(mp_p.s0, K, 125.0, mp->r, mp->q, mp->sigma, mp->T, m_bar) -
                reiner_rubinstein_bgk<OptionType::Call, BarrierType::UpOut>(mp_m.s0, K, 125.0, mp->r, mp->q, mp->sigma, mp->T, m_bar)
            ) / two_eps;
            check::is_true("barrier_upout_crn_delta_bgk",
                std::fabs(guo->delta.value - bgk_uo_d) <= 4.0 * guo->delta.std_error + 0.02);
        }
    }
}

void test_heston() {
    std::printf("\n== Heston Full Truncation Euler, COS Oracle & Sensitivities ==\n");

#if defined(PRICER_FAST_TESTS)
    constexpr std::size_t kHestonPaths = 50'000;
#else
    constexpr std::size_t kHestonPaths = 150'000;
#endif

    {
        const double sigma = 0.22;
        const auto hp = HestonParams::create(100.0, 0.04, sigma * sigma, 2.0, sigma * sigma,
                                             1e-8, -0.5, 1.0, 0.015);
        SimConfig cfg{.paths = kHestonPaths, .steps = 64, .seed = 1101};

        MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            hc(HestonProcess(*hp), EuropeanPayoff<OptionType::Call>(100.0));
        MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Put>, AntitheticSampling>
            hp_eng(HestonProcess(*hp), EuropeanPayoff<OptionType::Put>(100.0));

        const double bs_c = black_scholes_price<OptionType::Call>(100.0, 100.0, 0.04, 0.015, sigma, 1.0);
        const double bs_p = black_scholes_price<OptionType::Put>(100.0, 100.0, 0.04, 0.015, sigma, 1.0);
        const auto rc = hc.price(cfg);
        const auto rp = hp_eng.price(cfg);
        check::within_se("heston_gbm_limit_call", rc->price, rc->std_error, bs_c, 4.0);
        check::within_se("heston_gbm_limit_put",  rp->price, rp->std_error, bs_p, 4.0);
    }

    // Broadie-Kaya (2006) Case I (Feller-violating); published exact price: 6.8061.
    {
        const auto hp_bk = HestonParams::create(100.0, 0.0319, 0.010201, 6.21, 0.019,
                                                0.61, -0.7, 1.0, 0.0);
        const double cos_bk = heston_cos_price<OptionType::Call>(*hp_bk, 100.0, 128, 10.0);
        check::close("heston_cos_bk_case1", cos_bk, 6.8061, 1e-4);

        const std::size_t ms[4] = {8, 32, 128, 252};
        double bias[4] = {0.0, 0.0, 0.0, 0.0};
        double ses[4] = {0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < 4; ++k) {
            SimConfig cfg{.paths = kHestonPaths, .steps = ms[k], .seed = 1200 + static_cast<std::uint64_t>(k)};
            MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                eng(HestonProcess(*hp_bk), EuropeanPayoff<OptionType::Call>(100.0));
            const auto r = eng.price(cfg);
            bias[k] = std::fabs(r->price - cos_bk);
            ses[k] = r->std_error;
            std::printf("  Heston Case I refinement m=%3zu: mc=%.5f cos=%.5f |bias|=%.5f (se=%.5f)\n",
                        ms[k], r->price, cos_bk, bias[k], r->std_error);
        }
        check::is_true("heston_case1_monotonic_bias",
            bias[0] > bias[1] && bias[1] > bias[2] && bias[1] > bias[3]);
        check::is_true("heston_case1_m252_accurate", bias[3] <= 4.0 * ses[3] + 0.02);
    }

    {
        const auto hp_fel = HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04,
                                                 0.3, -0.5, 1.0, 0.0);
        const double cos_fel = heston_cos_price<OptionType::Call>(*hp_fel, 100.0, 128, 10.0);
        const std::size_t ms[3] = {32, 128, 252};
        double bias[3] = {0.0, 0.0, 0.0};
        double se_252 = 0.0;
        for (int k = 0; k < 3; ++k) {
            SimConfig cfg{.paths = kHestonPaths, .steps = ms[k], .seed = 1300 + static_cast<std::uint64_t>(k)};
            MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                eng(HestonProcess(*hp_fel), EuropeanPayoff<OptionType::Call>(100.0));
            const auto r = eng.price(cfg);
            bias[k] = std::fabs(r->price - cos_fel);
            if (k == 2) se_252 = r->std_error;
            std::printf("  Heston Feller refinement m=%3zu: mc=%.5f cos=%.5f |bias|=%.5f (se=%.5f)\n",
                        ms[k], r->price, cos_fel, bias[k], r->std_error);
        }
        check::is_true("heston_feller_m252_close", bias[2] <= 4.0 * se_252 + 0.02);
        check::is_true("heston_feller_m252_le_m32", bias[2] <= bias[0] + 4.0 * se_252);
    }

    {
        const auto hp = HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5, 1.0, 0.0);
        SimConfig cfg{.paths = 20'000, .steps = 64, .seed = 1401};

        MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
            he_euro(HestonProcess(*hp), EuropeanPayoff<OptionType::Call>(100.0));
        MonteCarloEngine<HestonProcess, GeometricAsianPayoff<OptionType::Call>, StandardSampling>
            he_ga(HestonProcess(*hp), GeometricAsianPayoff<OptionType::Call>(100.0));
        MonteCarloEngine<HestonProcess, ArithmeticAsianPayoff<OptionType::Call>, StandardSampling>
            he_aa(HestonProcess(*hp), ArithmeticAsianPayoff<OptionType::Call>(100.0));
        MonteCarloEngine<HestonProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>, StandardSampling>
            he_uo(HestonProcess(*hp), BarrierPayoff<OptionType::Call, BarrierType::UpOut>(100.0, 125.0));
        MonteCarloEngine<HestonProcess, BarrierPayoff<OptionType::Call, BarrierType::UpIn>, StandardSampling>
            he_ui(HestonProcess(*hp), BarrierPayoff<OptionType::Call, BarrierType::UpIn>(100.0, 125.0));
        MonteCarloEngine<HestonProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FixedStrike>, StandardSampling>
            he_lb(HestonProcess(*hp), LookbackPayoff<OptionType::Call, LookbackStyle::FixedStrike>(100.0));

        const auto r_eu = he_euro.price(cfg);
        const auto r_ga = he_ga.price(cfg);
        const auto r_aa = he_aa.price(cfg);
        const auto r_uo = he_uo.price(cfg);
        const auto r_ui = he_ui.price(cfg);
        const auto r_lb = he_lb.price(cfg);

        check::is_true("heston_asian_jensen",
            std::isfinite(r_ga->price) && std::isfinite(r_aa->price) && r_ga->price <= r_aa->price);
        check::close("heston_barrier_crn_parity", r_uo->price + r_ui->price, r_eu->price, 1e-12);
        check::is_true("heston_lookback_ge_euro",
            std::isfinite(r_lb->price) && r_lb->price >= r_eu->price);
    }

    {
        const auto hp_bk = HestonParams::create(100.0, 0.0319, 0.010201, 6.21, 0.019,
                                                0.61, -0.7, 1.0, 0.0);
        SimConfig cfg{.paths = kHestonPaths, .steps = 128, .seed = 1501};
        MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            eng_bk(HestonProcess(*hp_bk), EuropeanPayoff<OptionType::Call>(100.0));
        const auto g_bk = eng_bk.price_with_heston_greeks(cfg);

        HestonParams sp = *hp_bk, sm = *hp_bk;
        sp.s0 += 0.5; sm.s0 -= 0.5;
        MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            ep(HestonProcess(sp), EuropeanPayoff<OptionType::Call>(100.0));
        MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            em(HestonProcess(sm), EuropeanPayoff<OptionType::Call>(100.0));
        const double crn_delta = (ep.price(cfg)->price - em.price(cfg)->price) / 1.0;

        check::within_se("heston_case1_pw_delta", g_bk->delta.value, g_bk->delta.std_error, crn_delta, 4.0);
        check::is_true("heston_case1_greeks_finite",
            std::isfinite(g_bk->price.value) && std::isfinite(g_bk->delta.value) &&
            std::isfinite(g_bk->d_v0.value) && std::isfinite(g_bk->d_long_run_var.value) &&
            std::isfinite(g_bk->theta.value) && std::isfinite(g_bk->rho.value));

        const auto hp_fel = HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04,
                                                 0.3, -0.5, 1.0, 0.0);
        MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            eng_fel(HestonProcess(*hp_fel), EuropeanPayoff<OptionType::Call>(100.0));
        const auto g_fel = eng_fel.price_with_heston_greeks(cfg);

        auto cos_fd = [&](auto mutator, double h) {
            HestonParams p1 = *hp_fel, p2 = *hp_fel;
            mutator(p1, +h);
            mutator(p2, -h);
            return (heston_cos_price<OptionType::Call>(p1, 100.0) -
                    heston_cos_price<OptionType::Call>(p2, 100.0)) / (2.0 * h);
        };
        const double ref_delta = cos_fd([](HestonParams& p, double h) { p.s0 += h; }, 1e-3);
        const double ref_dv0   = cos_fd([](HestonParams& p, double h) { p.v0 += h; }, 1e-5);
        const double ref_dlrv  = cos_fd([](HestonParams& p, double h) { p.theta += h; }, 1e-5);
        const double ref_rho   = cos_fd([](HestonParams& p, double h) { p.r += h; }, 1e-5);
        const double ref_theta = -cos_fd([](HestonParams& p, double h) { p.T += h; }, 1e-4);

        std::printf("  Heston Feller Greeks vs COS FD: D=%.4f(%.4f) dv0=%.2f(%.2f) dlrv=%.2f(%.2f) T=%.4f(%.4f) R=%.3f(%.3f)\n",
                    g_fel->delta.value, ref_delta, g_fel->d_v0.value, ref_dv0,
                    g_fel->d_long_run_var.value, ref_dlrv, g_fel->theta.value, ref_theta,
                    g_fel->rho.value, ref_rho);

        auto close_heston_greek = [](const char* name, GreekEstimate g, double ref, double disc_tol) {
            const bool ok = std::isfinite(g.value) && std::isfinite(g.std_error) && g.std_error > 0.0
                         && std::fabs(g.value - ref) <= 4.0 * g.std_error + disc_tol;
            std::printf("  %-26s mc=%10.5f  ref=%10.5f  diff=%+.5f  se=%.5f  %s\n",
                        name, g.value, ref, g.value - ref, g.std_error, ok ? "ok" : "FAIL");
            check::record(ok);
        };
        close_heston_greek("heston_feller_delta", g_fel->delta, ref_delta, 0.005);
        close_heston_greek("heston_feller_dv0",   g_fel->d_v0,  ref_dv0,   1.5);
        close_heston_greek("heston_feller_dlrv",  g_fel->d_long_run_var, ref_dlrv, 1.5);
        close_heston_greek("heston_feller_theta", g_fel->theta, ref_theta, 0.10);
        close_heston_greek("heston_feller_rho",   g_fel->rho,   ref_rho,   0.15);

        const auto hp_v0z = HestonParams::create(100.0, 0.05, 0.0, 2.0, 0.04, 0.3, -0.5, 1.0, 0.0);
        SimConfig cfg_z{.paths = 40'000, .steps = 128, .seed = 1555};
        MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
            eng_z(HestonProcess(*hp_v0z), EuropeanPayoff<OptionType::Call>(100.0));
        const auto g_z = eng_z.price_with_heston_greeks(cfg_z);

        HestonParams hp_z_eps = *hp_v0z;
        hp_z_eps.v0 = 1e-4;
        const double ref_dv0_z = (heston_cos_price<OptionType::Call>(hp_z_eps, 100.0) -
                                  heston_cos_price<OptionType::Call>(*hp_v0z, 100.0)) / 1e-4;
        close_heston_greek("heston_v0_zero_dv0_fwd", g_z->d_v0, ref_dv0_z, 3.0);
    }
}

void test_thread_pool() {
    std::printf("\n== Persistent ThreadPool & Bitwise Thread-Count Invariance ==\n");

    const std::size_t phys_cores = physical_core_count();
    std::printf("  detected physical cores: %zu\n", phys_cores);
    check::is_true("physical_core_count_ge_1", phys_cores >= 1);

    {
#if defined(_WIN32)
        auto get_current_affinity = []() -> DWORD_PTR {
            DWORD_PTR proc_mask = 0, sys_mask = 0;
            if (!GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) || proc_mask == 0) {
                return 0;
            }
            const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), proc_mask);
            if (prev != 0) {
                (void)SetThreadAffinityMask(GetCurrentThread(), prev);
            }
            return prev;
        };
        const DWORD_PTR before_mask = get_current_affinity();
#elif defined(__linux__)
        cpu_set_t before_set{};
        const bool have_affinity =
            (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &before_set) == 0);
#endif
        ThreadPool pool((std::max)(phys_cores, std::size_t{6}), true);

        std::atomic<std::size_t> count{0};
        pool.parallel_for(0, 6, [&](std::size_t) noexcept { count.fetch_add(1); });
        check::is_true("tp_zero_chunks", count.load() == 0);

        pool.parallel_for(1, 6, [&](std::size_t c) noexcept { count.fetch_add(c + 1); });
        check::is_true("tp_one_chunk", count.load() == 1);

        std::vector<std::size_t> seen(128, 0);
        for (std::size_t thr : {std::size_t{1}, std::size_t{6}, std::size_t{2}, std::size_t{4}, std::size_t{0}, std::size_t{99}}) {
            std::fill(seen.begin(), seen.end(), 0);
            pool.parallel_for(seen.size(), thr, [&](std::size_t c) noexcept {
                seen[c] = c + 1;
            });
            bool all_ok = true;
            for (std::size_t i = 0; i < seen.size(); ++i) {
                if (seen[i] != i + 1) {
                    all_ok = false;
                    break;
                }
            }
            check::is_true("tp_all_chunks_executed", all_ok);
        }

#if defined(_WIN32)
        const DWORD_PTR after_mask = get_current_affinity();
        check::is_true("caller_affinity_preserved_win32", before_mask == after_mask);
#elif defined(__linux__)
        if (have_affinity) {
            cpu_set_t after_set{};
            const bool got_after =
                (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &after_set) == 0);
            check::is_true("caller_affinity_preserved_linux",
                           got_after && CPU_EQUAL(&before_set, &after_set));
        }
#endif
    }

    {
        ThreadPool unpinned(4, false);
        check::is_true("tp_unpinned_size", unpinned.size() == 4 && !unpinned.pin_to_physical_cores());
        std::vector<std::uint64_t> out(64, 0);
        unpinned.parallel_for(out.size(), 4, [&](std::size_t c) noexcept { out[c] = c * c; });
        bool ok = true;
        for (std::size_t i = 0; i < out.size(); ++i) {
            ok = ok && (out[i] == i * i);
        }
        check::is_true("tp_unpinned_exec", ok);
    }

    {
        auto bits_eq = [](double a, double b) noexcept -> bool {
            return std::memcmp(&a, &b, sizeof(double)) == 0;
        };
        auto ge_bits_eq = [&](const GreekEstimate& a, const GreekEstimate& b) noexcept -> bool {
            return bits_eq(a.value, b.value) && bits_eq(a.std_error, b.std_error);
        };

        ThreadPool pool(6, true);
        const std::size_t thread_counts[] = {1, 2, 4, 6};

        const auto mp = MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.01);
        const auto hp = HestonParams::create(100.0, 0.0319, 0.010201, 6.21, 0.019, 0.61, -0.7, 1.0, 0.0);

        {
            const std::size_t n_paths = 65'553;
            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
                eng_std(GBMProcess(*mp), EuropeanPayoff<OptionType::Call>(100.0), &pool);
            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                eng_anti(GBMProcess(*mp), EuropeanPayoff<OptionType::Call>(100.0), &pool);

            SimConfig base_cfg{.paths = n_paths, .steps = 1, .seed = 777, .threads = 1};
            const auto ref_std = eng_std.price(base_cfg);
            const auto ref_anti = eng_anti.price(base_cfg);
            const auto ref_g = eng_std.price_with_greeks(base_cfg);

            bool std_inv = ref_std.has_value();
            bool anti_inv = ref_anti.has_value();
            bool greeks_inv = ref_g.has_value();

            for (std::size_t thr : thread_counts) {
                SimConfig cfg = base_cfg;
                cfg.threads = thr;
                const auto rs = eng_std.price(cfg);
                const auto ra = eng_anti.price(cfg);
                const auto rg = eng_std.price_with_greeks(cfg);

                std_inv = std_inv && rs.has_value()
                    && bits_eq(rs->price, ref_std->price)
                    && bits_eq(rs->std_error, ref_std->std_error)
                    && rs->paths == ref_std->paths;
                anti_inv = anti_inv && ra.has_value()
                    && bits_eq(ra->price, ref_anti->price)
                    && bits_eq(ra->std_error, ref_anti->std_error)
                    && ra->paths == ref_anti->paths;
                greeks_inv = greeks_inv && rg.has_value()
                    && ge_bits_eq(rg->price, ref_g->price)
                    && ge_bits_eq(rg->delta, ref_g->delta)
                    && ge_bits_eq(rg->gamma, ref_g->gamma)
                    && ge_bits_eq(rg->vega, ref_g->vega)
                    && ge_bits_eq(rg->theta, ref_g->theta)
                    && ge_bits_eq(rg->rho, ref_g->rho);
            }
            check::is_true("bitwise_invariance_euro_1step_std_1_2_4_6T", std_inv);
            check::is_true("bitwise_invariance_euro_1step_anti_1_2_4_6T", anti_inv);
            check::is_true("bitwise_invariance_euro_5greeks_1_2_4_6T", greeks_inv);
        }

        {
            const std::size_t n_paths = 32'777;
            SimConfig base_cfg{.paths = n_paths, .steps = 32, .seed = 888, .threads = 1, .force_multi_step = true};

            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Put>, StandardSampling>
                eng_ms(GBMProcess(*mp), EuropeanPayoff<OptionType::Put>(100.0), &pool);
            MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>, StandardSampling>
                eng_bar(GBMProcess(*mp), BarrierPayoff<OptionType::Call, BarrierType::UpOut>(100.0, 125.0), &pool);
            MonteCarloEngine<GBMProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>, AntitheticSampling>
                eng_lb(GBMProcess(*mp), LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>(), &pool);

            const auto ref_ms = eng_ms.price(base_cfg);
            const auto ref_bar_g = eng_bar.price_with_barrier_greeks(base_cfg);
            const auto ref_lb = eng_lb.price(base_cfg);

            bool ms_inv = ref_ms.has_value();
            bool bar_inv = ref_bar_g.has_value();
            bool lb_inv = ref_lb.has_value();

            for (std::size_t thr : thread_counts) {
                SimConfig cfg = base_cfg;
                cfg.threads = thr;
                const auto rms = eng_ms.price(cfg);
                const auto rbg = eng_bar.price_with_barrier_greeks(cfg);
                const auto rlb = eng_lb.price(cfg);

                ms_inv = ms_inv && rms.has_value()
                    && bits_eq(rms->price, ref_ms->price)
                    && bits_eq(rms->std_error, ref_ms->std_error);
                bar_inv = bar_inv && rbg.has_value()
                    && ge_bits_eq(rbg->price, ref_bar_g->price)
                    && ge_bits_eq(rbg->delta, ref_bar_g->delta)
                    && ge_bits_eq(rbg->vega, ref_bar_g->vega)
                    && ge_bits_eq(rbg->theta, ref_bar_g->theta)
                    && ge_bits_eq(rbg->rho, ref_bar_g->rho);
                lb_inv = lb_inv && rlb.has_value()
                    && bits_eq(rlb->price, ref_lb->price)
                    && bits_eq(rlb->std_error, ref_lb->std_error);
            }
            check::is_true("bitwise_invariance_gbm_multistep_1_2_4_6T", ms_inv);
            check::is_true("bitwise_invariance_barrier_greeks_1_2_4_6T", bar_inv);
            check::is_true("bitwise_invariance_lookback_anti_1_2_4_6T", lb_inv);
        }

        {
            const std::size_t n_paths = 24'580;
            SimConfig base_cfg{.paths = n_paths, .steps = 24, .seed = 999, .threads = 1};

            MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                eng_null(HestonProcess(*hp), EuropeanPayoff<OptionType::Call>(100.0), nullptr);
            MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                eng_pool(HestonProcess(*hp), EuropeanPayoff<OptionType::Call>(100.0), &pool);

            const auto ref_null = eng_null.price(base_cfg);
            const auto ref_hg = eng_pool.price_with_heston_greeks(base_cfg);

            bool hest_inv = ref_null.has_value();
            bool hest_g_inv = ref_hg.has_value();
            for (std::size_t thr : thread_counts) {
                SimConfig cfg = base_cfg;
                cfg.threads = thr;
                const auto rh = eng_pool.price(cfg);
                const auto rhg = eng_pool.price_with_heston_greeks(cfg);
                hest_inv = hest_inv && rh.has_value()
                    && bits_eq(rh->price, ref_null->price)
                    && bits_eq(rh->std_error, ref_null->std_error);
                hest_g_inv = hest_g_inv && rhg.has_value()
                    && ge_bits_eq(rhg->price, ref_hg->price)
                    && ge_bits_eq(rhg->delta, ref_hg->delta)
                    && ge_bits_eq(rhg->d_v0, ref_hg->d_v0)
                    && ge_bits_eq(rhg->d_long_run_var, ref_hg->d_long_run_var)
                    && ge_bits_eq(rhg->theta, ref_hg->theta)
                    && ge_bits_eq(rhg->rho, ref_hg->rho);
            }
            check::is_true("bitwise_invariance_heston_price_1_2_4_6T", hest_inv);
            check::is_true("bitwise_invariance_heston_greeks_1_2_4_6T", hest_g_inv);
        }
    }
}

} // namespace

int main() {
    test_toolchain_and_cpuid();
    test_types_and_validation();
    test_avx2_math();
    test_xoshiro_and_box_muller();
    test_dual_and_jet();
    test_black_scholes_chain_and_iv();
    test_stats_and_european();
    test_exotics_and_greeks();
    test_heston();
    test_thread_pool();
    return check::summary();
}
