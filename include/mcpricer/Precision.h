#ifndef Precision_H
#define Precision_H

#include "mcpricer/MonteCarloEngine.h"

/*
 * Turns one pass of accumulated payoff sums into a confidence interval.
 *
 * Takes the totals and a discount factor rather than an engine reference. That
 * narrowing is the point: this class simulates nothing, so it cannot re-run a
 * batch behind the caller's back, cannot pair a mean with a variance drawn from
 * a different pass, and -- holding no engine at all -- cannot reintroduce the
 * object slicing that an earlier by-value MonteCarloEngine member caused.
 */
class Precision{
    public:
        struct interval{
            double ci_minus, ci_plus;
        };

        // z defaults to 1.96, the two-sided normal quantile for 95%.
        Precision(const totals& t, double discount, double z = 1.96);

        // Discounted mean payoff.
        double price() const;

        /*
         * Standard error of the DISCOUNTED mean, from the Bessel-corrected
         * sample variance of the simulated payoffs.
         *
         * An earlier version used md.sigma -- the market's volatility parameter
         * -- as though it were the standard deviation of the payoff
         * distribution. It is not: the payoff's spread depends on the option's
         * convexity and moneyness too, and it is not even in the same units once
         * discounting is applied. The correct quantity is the sample standard
         * deviation of the payoffs themselves.
         */
        double std_error() const;

        interval calc_CI() const;

    private:
        totals t;
        double discount;
        double z;
};
#endif
