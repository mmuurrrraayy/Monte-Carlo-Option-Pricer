#ifndef Black_Scholes_H
#define Black_Scholes_H
#include <cmath>
#include "DataVariables.h"
#include "Option.h"
#include "Validator.h"

/*
Validates against monte carlo engine using black-scholes formula
*/
class Black_Scholes{
    private:
        market_data md;
        option_data od;
        double time;
        double remaining_time; 

        double d_plus(double S){
            return ((std::log(S/od.K)+(md.r+((md.sigma*md.sigma)/2.0)
            *(remaining_time)))/(md.sigma*std::sqrt(remaining_time)));
        }

        double d_minus(double S){
            return d_plus(S) - md.sigma * std::sqrt(remaining_time);
        }

        double normalise(double d){
            return 0.5 * std::erfc(-d/ std::sqrt(2.0));
        }

    public:
        Black_Scholes(double time, market_data md,option_data od):
        md(md), 
        od(od),
        time(time)
        {
            remaining_time = od.T - time;
        }
    
        /*
        Calculates price given asset price
        */
        double price(double S){
            Validator::is_valid_positive_double(S);
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
                default:
                    throw std::invalid_argument("Invalid option type");
            }
        }
};

#endif