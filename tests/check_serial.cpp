// Regression checks for the SERIAL engine (MonteCarloEngine + GBM + Precision)
// and for the closed-form Black-Scholes reference itself.
//
// The parallel engine is covered by check_greeks; this binary exists because the
// serial path is a separate implementation and would otherwise be untested. It
// also covers Black_Scholes directly, which matters more than it looks: every
// other check in the suite compares against it, so a silent error there would
// make every other test agree on the wrong answer.
#include "mcpricer/Black-Scholes.h"
#include "mcpricer/MonteCarloEngine.h"
#include "mcpricer/AntitheticEngine.h"
#include "mcpricer/Precision.h"
#include "check.h"

#include <cstdio>
#include <cmath>

static constexpr double kMaxErrInSe = 4.0;

static market_data make_md() {
    market_data md;
    md.s0 = 100.0; md.r = 0.05; md.sigma = 0.20;
    // mu deliberately != r. Nothing in the pricer should read it; if GBM ever
    // regresses to the real-world drift (the bug this project started with),
    // mu == r would hide it completely. At 0.15 that regression moves the price
    // from ~10.45 to ~16 -- hundreds of standard errors, caught instantly.
    md.mu = 0.15;
    return md;
}

static option_data make_od(OptionType type) {
    option_data od;
    od.K = 100.0; od.T = 1.0; od.option_type = type;
    return od;
}

// Mean and standard error from a single pass. total_payoff() returns the sum and
// the sum of squares together, so the variance is provably built from the same
// pass over the same paths as the mean -- there is no way to pair them wrongly.
struct Est { double price, se; };

static Est estimate(MonteCarloEngine& eng, int N, double s0) {
    const Precision p(eng.total_payoff(N, s0), eng.discount_factor());
    return Est{p.price(), p.std_error()};
}

// The reference the whole suite is measured against. Hard-coded from the
// analytic formula rather than recomputed here, so that an error introduced into
// Black_Scholes cannot quietly move the target it is checked against.
static constexpr double kCallRef = 10.4505835722;
static constexpr double kPutRef  = 5.5735260222;

static void check_black_scholes() {
    const auto md = make_md();
    Black_Scholes call(0.0, md, make_od(OptionType::Call));
    Black_Scholes put(0.0, md, make_od(OptionType::Put));
    const auto S = md.s0;

    std::printf("\n-- closed-form Black-Scholes --\n");
    check::close("call price", call.price(S), kCallRef, 1e-9);
    check::close("put price", put.price(S), kPutRef, 1e-9);

    // Put-call parity: C - P = S - K*exp(-rT). An identity, so the only
    // tolerance needed is for floating-point rounding.
    const auto od = make_od(OptionType::Call);
    const auto parity = S - od.K * std::exp(-md.r * od.T);
    check::close("put-call parity", call.price(S) - put.price(S), parity, 1e-12);

    // d^2(C-P)/dS^2 = 0 and d(C-P)/dsigma = 0, both by parity, so gamma and vega
    // are shared between call and put.
    check::close("gamma call == put", call.gamma(S), put.gamma(S), 1e-12);
    check::close("vega call == put", call.vega(S), put.vega(S), 1e-12);

    // delta_call - delta_put = 1, again by parity.
    check::close("delta parity", call.delta(S) - put.delta(S), 1.0, 1e-12);

    // Sign and bound sanity: a call delta is in (0,1), a put delta in (-1,0),
    // and both gamma and vega are strictly positive.
    check::is_true("call delta in (0,1)", call.delta(S) > 0.0 && call.delta(S) < 1.0);
    check::is_true("put delta in (-1,0)", put.delta(S) < 0.0 && put.delta(S) > -1.0);
    check::is_true("gamma > 0", call.gamma(S) > 0.0);
    check::is_true("vega > 0", call.vega(S) > 0.0);
}

static void check_engines(OptionType type, double ref, const char* label) {
    const auto md = make_md();
    const auto od = make_od(type);
    sim_data sd; sd.N = 0; sd.seed = 42;
    const auto N = 200000;

    Option opt(md, od, sd);

    std::printf("\n-- serial engines, %s (N=%d) --\n", label, N);

    // Each engine gets its own GBM: std::mt19937 is stateful, so sharing one
    // would make the second engine's result depend on how many draws the first
    // consumed.
    GBM g1(md, sd);
    MonteCarloEngine mc(g1, opt, md, od, sd);
    const auto plain = estimate(mc, N, md.s0);
    check::within_se("serial mc", plain.price, plain.se, ref, kMaxErrInSe);

    GBM g2(md, sd);
    AntitheticEngine ae(g2, opt, md, od, sd);
    const auto anti = estimate(ae, N, md.s0);
    check::within_se("antithetic", anti.price, anti.se, ref, kMaxErrInSe);

    // MonteCarloEngine computes the discount factor in TWO places: price() goes
    // through the private discounted(), while estimate() above uses the public
    // discount_factor(). Two expressions for one quantity can drift apart
    // silently -- the historical exp(-r*K) bug lived in exactly one of them --
    // so pin them together. A fresh GBM on the same seed replays the same paths,
    // making this an exact equality rather than a statistical one.
    GBM g4(md, sd);
    MonteCarloEngine mc4(g4, opt, md, od, sd);
    check::close("price() == estimate()", mc4.price(N, md.s0), plain.price, 1e-12);

    // Antithetic must reduce the standard error per SAMPLE (each sample being a
    // pair average). This is the property that makes the pairing worth its extra
    // cost; if it ever stops holding, the pairing has been broken.
    check::is_true("antithetic se < plain se", anti.se < plain.se);

    // Precision is the component that turns accumulated variance into an
    // interval, so exercise it rather than only the raw estimate. A 95% CI from
    // a correct estimator brackets the true value; at 200k paths a failure here
    // is a real defect, not bad luck.
    GBM g3(md, sd);
    MonteCarloEngine mc3(g3, opt, md, od, sd);
    Precision prec(mc3.total_payoff(N, md.s0), mc3.discount_factor());
    const auto ci = prec.calc_CI();
    std::printf("%-28s [%.8f, %.8f]  ref %.8f\n", "precision 95% CI",
                ci.ci_minus, ci.ci_plus, ref);
    check::is_true("CI brackets closed form", ci.ci_minus < ref && ref < ci.ci_plus);
    check::is_true("CI ordered", ci.ci_minus < ci.ci_plus);
}

int main() {
    check_black_scholes();
    check_engines(OptionType::Call, kCallRef, "call");
    check_engines(OptionType::Put, kPutRef, "put");
    return check::summary();
}
