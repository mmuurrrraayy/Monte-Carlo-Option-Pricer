#ifndef MonteCarloEngine_H
#define MonteCarloEngine_H

#include "GBM.h"
#include "Option.h"
#include "DataVariables.h"
#include "Validator.h"

#include <iostream>
#include <cmath>

class MonteCarloEngine{

    private:
        
     
    
    protected:

        GBM gbm;
        Option option;
        market_data md;
        option_data od;
        sim_data sd;
        int step_count;

        /*
        Calculates the total and squared total of an option
        */
        virtual totals total_payoff(int paths, double s0){
               
                totals t;
                double total = 0.0;
                double squared_total = 0.0;
                //calculates asset price of each path
                for (int i = 0; i<paths; i++){
                    double S= s0;
                    //calculates asset price for every step
                    for (int step = 0; step<step_count; step++){
                        double Z= gbm.calc_Z();
                        S = gbm.calc_St(S, Z);
                    }
                    double payoff = option.payoff(S);
                    squared_total += std::pow(payoff,2.0);
                    total += payoff;
                }
                t.squared_total = squared_total;
                t.total = total;
                return t;
            }

    public:

        MonteCarloEngine(GBM& gbm, Option& option, 
            market_data md, option_data od, sim_data sd):
                gbm(gbm),
                option(option),
                md(md), 
                od(od),
                sd(sd),
                step_count((int)((od.T)/sd.time_step)){}

        /*
        Calculates average payoff of all paths for an option
        */
        double average_payoff(const totals& t, int paths){
            Validator::is_valid_positive_integer(paths);
            return t.total / (double) paths;
        }

        /*
       Calculates present price of option
       based on future average multiplied by
       discount factor
       */
        double discounted(double average) {
            return std::exp(-md.mu * od.T) * average;
        }


        /*
        Calculates variance of a stock opition
        */
        double calc_variance(double squared_payoffs_summed, double average, int paths){
            Validator::is_valid_positive_double(squared_payoffs_summed);
            Validator::is_valid_positive_double(average);
            Validator::is_valid_positive_integer(paths);
            return (squared_payoffs_summed/ (double)paths) - std::pow(average,2.0);
        }
    
        /*
        Calculates total price, average price, and present price
        of an option given paths and initial asset price
        */
        payoff_info price(int paths, double s0){
            Validator::is_valid_positive_integer(paths);
            Validator::is_valid_positive_double(s0);
            payoff_info p_i;
            p_i.t= total_payoff(paths,s0);
            p_i.average =  average_payoff(p_i.t, paths);
            p_i.discounted = discounted(p_i.average);
            return p_i;
        }
            
};
#endif