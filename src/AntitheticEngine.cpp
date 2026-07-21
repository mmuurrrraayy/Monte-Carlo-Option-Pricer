#include "mcpricer/AntitheticEngine.h"

AntitheticEngine::AntitheticEngine(GBM& gbm, Option& option, market_data md,
    option_data od, sim_data sd):
    MonteCarloEngine(gbm, option, md,od,sd){}

totals AntitheticEngine::total_payoff(int path, double s0){
    totals t;
    for (int i = 0; i<path; i++){
        auto S = s0;
        auto minus_S = s0;
        for (int step = 0; step<step_count; step++){
            auto Z = gbm.calc_Z();
            S = gbm.calc_St(S, Z);
            minus_S =  gbm.calc_St(minus_S, -Z);
        }
        // One sample is the PAIR average, so sumsq accumulates the square of
        // that average -- the variance this yields is the variance per pair,
        // which is what makes it directly comparable to the plain engine's.
        const auto p = (option.payoff(S)+option.payoff(minus_S))/2.0;
        t.sum += p;
        t.sumsq += p * p;
    }
    t.paths = path;
    return t;
}
