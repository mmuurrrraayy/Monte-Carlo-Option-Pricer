#ifndef Black_Scholes_H
#define Black_Scholes_H
#include <cmath>
#include "OptionType.h"
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

        double d_plus(double S) {
            double numerator = std::log(S / od.K) + (md.mu + (md.sigma * md.sigma) / 2.0)
                * remaining_time;
            double denominator = md.sigma * std::sqrt(remaining_time);
            return  numerator / denominator;
        }

        double d_minus(double S, double d_p){
            return d_p - md.sigma * std::sqrt(remaining_time);
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
            if (remaining_time <= 0){
                throw std::invalid_argument("time remaining must be greater than 0");
            }
        }
    
        /*
        Calculates price given asset price
        */
        double price(double S){
            Validator::is_valid_positive_double(S);
            double d1 = d_plus(S);
            double d2 = d_minus(S, d1);
            double price;
            switch(od.option_type){
                case OptionType::Call:
                     price =  S*normalise(d1) - 
                        (std::exp(-md.mu*(remaining_time))
                        *od.K*normalise(d2));
                    return price;
                case OptionType::Put:
                    return od.K*std::exp(-md.mu*remaining_time) *
                    normalise(-d2)-S*normalise(-d1);
                default:
                    throw std::invalid_argument("Invalid option type");
            }
        }
};

#endif