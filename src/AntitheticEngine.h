#ifndef AntitheticEngine_H
#define AntitheticEngine_H

#include "MonteCarloEngine.h"
#include "GBM.h"
#include "Option.h"
#include "DataVariables.h"


/*
Reduced variance vs monte carlo using mirrored pricing
Reduce variance by introducing negative correlation
between pairs of simulation runs
*/
class AntitheticEngine: public MonteCarloEngine{
        
    public:
        AntitheticEngine(GBM& gbm, Option& option, market_data md, 
            option_data od, sim_data sd):
            MonteCarloEngine(gbm, option, md,od,sd){}

            
        totals total_payoff(int path, double s0)override {
            totals t;
            double total = 0.0;
            double squared_total = 0.0;
            //calculates asset price of each path
            for (int i = 0; i<path; i++){
                double S= s0;
                double minus_S = s0;
                 //calculates asset price for every step
                for (int step = 0; step<step_count; step++){
                    double Z= gbm.calc_Z();
                    S = gbm.calc_St(S, Z);
                    minus_S =  gbm.calc_St(minus_S, -Z);
                }
                double antithetic = (option.payoff(S)+option.payoff(minus_S)/2.0);
                squared_total += antithetic*antithetic;
                total += antithetic;
                }
                t.squared_total = squared_total;
                t.total = total;
                return t;
            }
};
#endif