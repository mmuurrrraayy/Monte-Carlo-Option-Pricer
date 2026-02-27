#ifndef Precision_H
#define Precision_H

#include "Option.h"
#include "DataVariables.h"
#include "MonteCarloEngine.h"
#include <cmath>

class Precision{
    private:
        Option option;
        market_data md;
        option_data od;
        sim_data sd;
        MonteCarloEngine mc;
        const double c = 1.96;
        int N, path;
        double s0;
        struct interval{
            double ci_minus, ci_plus;
        };

        double get_average(){
           return mc.average_payoff(mc.total_payoff(path, s0), N);
        }
    
    public:
        Precision(MonteCarloEngine& mc, Option& option,market_data md,
            option_data od, sim_data sd, int size):
            mc(mc),
            option(option),
            md(md), 
            sd(sd),
            od(od),
            N(size) {}
        
        interval calc_CI(){
            interval ci;
            double mean = get_average();
            ci.ci_minus = mean - (c*((md.sigma)/std::sqrt(N)));
            ci.ci_plus = mean + (c*((md.sigma)/std::sqrt(N)));
            return ci;
        }
    
        
};
#endif