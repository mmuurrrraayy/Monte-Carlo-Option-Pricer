#ifndef AntitheticEngine_H
#define AntitheticEngine_H

#include "mcpricer/MonteCarloEngine.h"
#include "mcpricer/GBM.h"
#include "mcpricer/Option.h"
#include "mcpricer/DataVariables.h"

// Antithetic variates: each path is paired with its mirror image, generated
// from the same step-by-step Z's negated (S(Z) alongside S(-Z)), and the two
// payoffs are averaged before being accumulated. Since S(Z) and S(-Z) are
// negatively correlated, averaging them cancels part of the sampling noise a
// lone path would carry, so for the same path count the estimate comes out
// tighter than plain Monte Carlo -- at roughly double the per-path work.
class AntitheticEngine: public MonteCarloEngine{

    public:
        AntitheticEngine(GBM& gbm, Option& option, market_data md,
            option_data od, sim_data sd);

        totals total_payoff(int path, double s0) override;

};
#endif
