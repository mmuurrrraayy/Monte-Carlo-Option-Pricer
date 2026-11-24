// Verifies all public headers build and run with exceptions and RTTI disabled.

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#error "test_no_exceptions must be compiled with C++ exceptions disabled"
#endif

#if defined(__cpp_rtti) || defined(__GXX_RTTI) || defined(_CPPRTTI)
#error "test_no_exceptions must be compiled with RTTI disabled"
#endif

#include "cpuid_guard.hpp"

#include "pricer/core/concepts.hpp"
#include "pricer/core/dual.hpp"
#include "pricer/core/stats.hpp"
#include "pricer/core/types.hpp"
#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"
#include "pricer/simd/xoshiro_avx2.hpp"

#include "pricer/engines/black_scholes.hpp"
#include "pricer/engines/monte_carlo.hpp"
#include "pricer/engines/thread_pool.hpp"
#include "pricer/models/gbm.hpp"
#include "pricer/models/heston.hpp"
#include "pricer/payoffs/asian.hpp"
#include "pricer/payoffs/barrier.hpp"
#include "pricer/payoffs/european.hpp"
#include "pricer/payoffs/lookback.hpp"

#include <cmath>
#include <cstdio>

int main() {
    using namespace pricer;

    bool ok = pricer_has_avx2_fma();

    const auto m_ok = MarketParams::create(100.0, -0.01, 0.2, 1.0, 0.02);
    const auto m_bad = MarketParams::create(-5.0, 0.05, 0.2, 1.0);
    const auto h_ok = HestonParams::create(100.0, 0.03, 0.04, 2.0, 0.04, 0.3, -0.5, 1.0, 0.01);
    ok = ok && m_ok.has_value() && !m_bad.has_value()
            && m_bad.error() == ValidationError::NonPositiveSpot
            && h_ok.has_value();

    const Vec4d x = Vec4d::set(-2.0, 0.0, 1.0, 2.0);
    const Vec4d ex = exp_avx2(x);
    const Vec4d lx = log_avx2(ex);
    Vec4d s, c;
    sincos_avx2(x, s, c);
    const Vec4d nc = norm_cdf_avx2(x);
    ok = ok && std::isfinite(hsum(lx)) && std::isfinite(hsum(s + c)) && std::isfinite(hsum(nc));

    Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(42ULL, 0ULL);
    Vec4d z0, z1;
    rng.next_normals_pair(z0, z1);
    ok = ok && std::isfinite(hsum(z0)) && std::isfinite(hsum(z1));

    const auto d = exp(Dual<Vec4d>::variable(Vec4d::splat(1.0)));
    const auto j = max(Jet<Vec4d, 3>::variable<0>(Vec4d::splat(110.0)) - 100.0, 0.0);
    ok = ok && std::isfinite(hsum(d.d)) && std::fabs(hsum(j.d[0]) - 4.0) < 1e-12;

    OptionChainSoA chain(5);
    ChainGreeksSoA greeks(5);
    AlignedBuffer prices(chain.padded_n);
    AlignedBuffer ivs(chain.padded_n);
    ok = ok && chain.is_allocated() && greeks.is_allocated();
    for (std::size_t i = 0; i < chain.count; ++i) {
        chain.set(i, 100.0, 90.0 + 5.0 * static_cast<double>(i), 0.03, 0.01, 0.25, 1.0);
    }
    price_chain_avx2<OptionType::Call>(chain, prices.data());
    price_and_greeks_chain_avx2<OptionType::Call>(chain, greeks);
    implied_vol_chain_avx2<OptionType::Call>(chain, prices.data(), ivs.data());
    const double scalar_p = black_scholes_price<OptionType::Call>(100.0, 100.0, 0.03, 0.01, 0.25, 1.0);
    const auto jet_p = black_scholes_price<OptionType::Call>(
        Jet<Vec4d, 3>::variable<0>(Vec4d::splat(100.0)),
        Jet<Vec4d, 3>::constant(Vec4d::splat(100.0)),
        Jet<Vec4d, 3>::variable<2>(Vec4d::splat(0.03)),
        Jet<Vec4d, 3>::constant(Vec4d::splat(0.01)),
        Jet<Vec4d, 3>::variable<1>(Vec4d::splat(0.25)),
        Jet<Vec4d, 3>::constant(Vec4d::splat(1.0)));
    ok = ok && std::isfinite(scalar_p)
            && std::fabs(prices[2] - scalar_p) < 1e-12
            && std::fabs(greeks.price[2] - scalar_p) < 1e-12
            && std::fabs(ivs[2] - 0.25) < 1e-11
            && std::isfinite(hsum(jet_p.d[0]));

    {
        const auto mp = MarketParams::create(100.0, 0.05, 0.20, 1.0, 0.01);
        const auto hp = HestonParams::create(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5, 1.0, 0.01);
        const auto euro_call = EuropeanPayoff<OptionType::Call>::create(100.0);
        const auto geom_asian = GeometricAsianPayoff<OptionType::Call>::create(100.0);
        const auto arith_asian = ArithmeticAsianPayoff<OptionType::Put>::create(100.0);
        const auto barrier_uo = BarrierPayoff<OptionType::Call, BarrierType::UpOut>::create(100.0, 130.0, 100.0);
        const auto lb_float = LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>::create();

        ok = ok && mp.has_value() && hp.has_value() && euro_call.has_value()
                && geom_asian.has_value() && arith_asian.has_value()
                && barrier_uo.has_value() && lb_float.has_value();

        if (ok) {
            SimConfig cfg{};
            cfg.paths = 256;
            cfg.steps = 16;
            cfg.seed = 7;

            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                eng_euro(GBMProcess(*mp), *euro_call);
            const auto r_euro = eng_euro.price(cfg);
            const auto g_euro = eng_euro.price_with_greeks(cfg);

            MonteCarloEngine<GBMProcess, GeometricAsianPayoff<OptionType::Call>, StandardSampling>
                eng_gasian(GBMProcess(*mp), *geom_asian);
            const auto r_gasian = eng_gasian.price(cfg);

            MonteCarloEngine<GBMProcess, ArithmeticAsianPayoff<OptionType::Put>, StandardSampling>
                eng_aasian(GBMProcess(*mp), *arith_asian);
            const auto r_aasian = eng_aasian.price(cfg);

            MonteCarloEngine<GBMProcess, BarrierPayoff<OptionType::Call, BarrierType::UpOut>, StandardSampling>
                eng_bar(GBMProcess(*mp), *barrier_uo);
            const auto g_bar = eng_bar.price_with_barrier_greeks(cfg);

            MonteCarloEngine<GBMProcess, LookbackPayoff<OptionType::Call, LookbackStyle::FloatingStrike>, StandardSampling>
                eng_lb(GBMProcess(*mp), *lb_float);
            const auto r_lb = eng_lb.price(cfg);

            MonteCarloEngine<HestonProcess, EuropeanPayoff<OptionType::Call>, AntitheticSampling>
                eng_hest(HestonProcess(*hp), *euro_call);
            const auto g_hest = eng_hest.price_with_heston_greeks(cfg);

            ThreadPool pool(2, true);
            SimConfig cfg_mt{};
            cfg_mt.paths = 16384;
            cfg_mt.steps = 1;
            cfg_mt.seed = 11;
            cfg_mt.threads = 2;
            MonteCarloEngine<GBMProcess, EuropeanPayoff<OptionType::Call>, StandardSampling>
                eng_mt(GBMProcess(*mp), *euro_call, &pool);
            const auto r_mt = eng_mt.price(cfg_mt);
            SimConfig cfg_st = cfg_mt;
            cfg_st.threads = 1;
            const auto r_st = eng_mt.price(cfg_st);

            ok = ok && r_euro.has_value() && std::isfinite(r_euro->price)
                    && g_euro.has_value() && std::isfinite(g_euro->gamma.value)
                    && r_gasian.has_value() && std::isfinite(r_gasian->price)
                    && r_aasian.has_value() && std::isfinite(r_aasian->price)
                    && g_bar.has_value() && std::isfinite(g_bar->delta.value)
                    && r_lb.has_value() && std::isfinite(r_lb->price)
                    && g_hest.has_value() && std::isfinite(g_hest->delta.value)
                    && r_mt.has_value() && r_st.has_value()
                    && r_mt->price == r_st->price
                    && r_mt->std_error == r_st->std_error;
        }
    }

    std::printf("test_no_exceptions: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
