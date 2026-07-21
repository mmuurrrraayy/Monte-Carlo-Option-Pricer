#ifndef MonteCarloEngine_H
#define MonteCarloEngine_H
#include "mcpricer/GBM.h"
#include "mcpricer/Option.h"
#include "mcpricer/DataVariables.h"

/*
 * Accumulated sums from one pass over the simulated paths.
 *
 * Returned by value rather than left in a member. The earlier design wrote the
 * sum of squares into a `last_sumsq` field as a side effect of total_payoff(),
 * which forced every caller to depend on an ordering the type system could not
 * enforce -- read it before the pass and you got zero, read it after a second
 * engine had run and you got that engine's data. Handing the totals back
 * explicitly makes it impossible to pair a mean with a variance drawn from a
 * different pass.
 */
struct totals {
    double sum = 0.0;    // sum of undiscounted payoffs
    double sumsq = 0.0;  // sum of squared undiscounted payoffs
    int paths = 0;
};

/*
 * Baseline serial engine: simulates independent asset paths via GBM,
 * one step at a time, and discounts the average payoff back to today.
 */
class MonteCarloEngine{

    private:

        double discounted(double average);

    protected:

        GBM gbm;
        Option option;
        market_data md;
        option_data od;
        sim_data sd;
        int step_count;

    public:
        MonteCarloEngine(GBM& gbm, Option& option,
            market_data md, option_data od, sim_data sd);

        virtual ~MonteCarloEngine() = default;

        virtual double price(int N, double s0);

        double average_payoff(const totals& t) const;

        double discount_factor() const;

        virtual totals total_payoff(int path, double s0);
};
#endif
