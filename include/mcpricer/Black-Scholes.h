#ifndef Black_Scholes_H
#define Black_Scholes_H
#include "mcpricer/DataVariables.h"

/*
 *Closed-form reference price and Greeks, used to validate the Monte Carlo
 */
class Black_Scholes{
    private:
        market_data md;
        option_data od;
        double time;
        double remaining_time =0.0;

        double d_plus(double S);
        double d_minus(double S);
        double normalise(double d);
        double pdf(double d);

    public:
        Black_Scholes(double time, market_data md, option_data od);

        double price(double S);

        double delta(double S);

        /*
         *Same expression for calls and puts: d^2(call-put)/dS^2 = 0 by put-call parity. 
         */
        double gamma(double S);

        /*
         *Same expression for calls and puts, for the same put-call-parity reason as gamma.       
         */
        double vega(double S);

        /*
         *Calendar-time convention: theta = dV/dt = -dV/d(remaining_time).
         */
        double theta(double S);

        double rho(double S);
};

#endif
