// get gbm and options 
// find 
#ifndef MonteCarloEngine_H
#define MonteCarloEngine_H
#include "GBM.h"
#include "Option.h"
#include "DataVariables.h"

#include <cmath>

class MonteCarloEngine{

    private:

        double discounted(double average){
            return std::exp(-md.r*od.K)*average;
        }

    protected:

        GBM gbm;
        Option option;
        market_data md;
        option_data od;
        sim_data sd;
        int step_count, path;



    public:
        MonteCarloEngine(GBM& gbm, Option& option, 
            market_data md, option_data od, sim_data sd):
        gbm(gbm),
        option(option),
        md(md), 
        sd(sd),
        od(od)
        {
            step_count =(int) (od.T/sd.time_step);
    
        }

         virtual double price(int N, double s0){
            double total = total_payoff(N,s0);
            double average = average_payoff(total, N);
            return discounted(average);
        }
        
         double average_payoff(double total, int N){
            return total / (double) N;
        }
        
         virtual double total_payoff(int path, double s0){
                double total = 0.0;
                for (int i = 0; i<path; i++){
                    double S= s0;
                    for (int step = 0; step<step_count; step++){
                        double Z= gbm.calc_Z();
                        S = gbm.calc_St(S, Z);
                    }
                    total += option.payoff(S);
                }
                return total;
            }
};
#endif