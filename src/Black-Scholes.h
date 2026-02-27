#ifndef Black_Scholes_H
#define Black_Scholes_H
#include <cmath>
#include "DataVariables.h"
#include "Option.h"

class Black_Scholes{
    private:
        market_data md;
        option_data od;
        OptionType option_type;
        double time;
        double remaining_time =0.0; 

        double d_plus(double S){
            return ((std::log(S/od.K)+(md.r+((md.sigma*md.sigma)/2.0)
            *(remaining_time)))/(md.sigma*std::sqrt(remaining_time)));
        }

        double d_minus(double S){
            return d_plus(S) - md.sigma * std::sqrt(remaining_time);
        }

        double normalise(double d){
            return 0.5 * std::erfc(-d/ sqrt(2));
        }

    public:
        Black_Scholes(double time, market_data md,option_data od, 
           ):
        md(md), 
        od(od),
        time(time)
        {
            remaining_time = od.T - time;
        }
    
        double price(double S){
            double d1 = d_plus(S);
            double d2 = d_minus(S);
            switch(od.option_type){
                case OptionType::Call:
                    return S*normalise(d1) - 
                        (std::exp(-md.r*(remaining_time))
                        *od.K*normalise(d2));
                case OptionType::Put:
                    return od.K*std::exp(-md.r*remaining_time) *
                    normalise(-d2)-S*normalise(-d1);
            }
        }
};

#endif