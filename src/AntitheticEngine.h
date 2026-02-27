#ifndef AntitheticEngine_H
#define AntitheticEngine_H

#include "MonteCarloEngine.h"
#include "GBM.h"
#include "Option.h"
#include "DataVariables.h"

class AntitheticEngine: public MonteCarloEngine{
        
    public:
        AntitheticEngine(GBM& gbm, Option& option, market_data md, 
            option_data od, sim_data sd):
            MonteCarloEngine(gbm, option, md,od,sd){}

            
        double total_payoff(int path, double s0)override {
             double total = 0.0;
            for (int i = 0; i<path; i++){
                double S= s0;
                double minus_S = s0;
                for (int step = 0; step<step_count; step++){
                    double Z= gbm.calc_Z();
                    S = gbm.calc_St(S, Z);
                    minus_S =  gbm.calc_St(minus_S, -Z);
                }
                total += (option.payoff(S)+option.payoff(minus_S))/2.0;
            }
            return total;
        }
        
};
#endif