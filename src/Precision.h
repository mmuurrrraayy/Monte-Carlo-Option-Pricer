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
        int N;
        double s0;
        totals t;
         
      
        double get_average(){
           return mc.average_payoff(t.total, N);
        }

        
    public:
        Precision(MonteCarloEngine& mc, Option& option,market_data md,
            option_data od, sim_data sd, int path,  double s0,totals t):
            mc(mc),
            option(option),
            md(md), 
            sd(sd),
            od(od),
            N(path),
            s0(s0),
            t(t)
            {}

        struct interval{
            double ci_minus, ci_plus;
        };

        
        interval calc_CI(){
            interval ci;
            double mean = get_average();
            double variance = mc.calc_variance(t.squared_total, mean, N);
            double standard_error = (c*(std::sqrt(variance/(double)N)));
            ci.ci_minus = mean - standard_error;
            ci.ci_plus = mean + standard_error;
            return ci;
        }
    
        
};
#endif