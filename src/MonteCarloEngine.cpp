#include "mcpricer/MonteCarloEngine.h"

#include <cmath>

MonteCarloEngine::MonteCarloEngine(GBM& gbm, Option& option,
    market_data md, option_data od, sim_data sd):
    gbm(gbm),
    option(option),
    md(md),
    sd(sd),
    od(od)
{
    step_count = (int) (od.T/sd.time_step);
}

double MonteCarloEngine::discounted(double average){
    // was exp(-r*K): discounted by the strike instead of the maturity.
    return std::exp(-md.r*od.T)*average;
}

double MonteCarloEngine::price(int N, double s0){
    return discounted(average_payoff(total_payoff(N, s0)));
}

double MonteCarloEngine::average_payoff(const totals& t) const {
    return t.sum / (double) t.paths;
}

double MonteCarloEngine::discount_factor() const { return std::exp(-md.r*od.T); }

totals MonteCarloEngine::total_payoff(int path, double s0){
    totals t;
    for (int i = 0; i<path; i++){
        auto S = s0;
        for (int step = 0; step<step_count; step++){
            auto Z = gbm.calc_Z();
            S = gbm.calc_St(S, Z);
        }
        const auto p = option.payoff(S);
        t.sum += p;
        t.sumsq += p * p;
    }
    t.paths = path;
    return t;
}
