// need St(asset price), u drift, sigma - volatility (from option), Wt(brownian  motion), t, Z
//discrete change in time = volatility * discrete t + volatility *root(discrete time *Z) Z~N(0,t)
// discrete time is one day or 1/365
//Z single standard normal random number per time step
// drift = annualised average return of asset in percentage
// discrete t * drift = u/365
// volatility (sigma) (in %) 
//discrete t *  volatlity = sigma%/365
//m := u - 1/2*(sigma squared)
// s:= sigma
//S0 intial price
// St = S0 *exp(mt + sigma*Wt) --> logSt-logS0 ~ N(mt, sigma^2 * t)
#ifndef GBM_H
#define GBM_H

#include "DataVariables.h"
#include <cmath>
#include <random>
#include "Validator.h"

class GBM{
    private:
        market_data market_data_;
        sim_data data_simul;
        double m;
        std::mt19937 rng; 
        std::normal_distribution <double>normal;

        double calc_Wt(double Z) {
            return Z * std::sqrt(data_simul.time_step);
        }


    public:
        GBM(market_data market_data_,sim_data data_simul_): 
        market_data_(market_data_), 
        data_simul(data_simul_),
        m(market_data_.r - 0.5*(market_data_.sigma*market_data_.sigma)),
        rng(data_simul.seed),
        normal(0.0,1.0){}

        double calc_Z() {
            return normal(rng);
        }


        /*
        Calculates price given initial asset price and and 
        random value
        */
        double calc_St(double s0, double Z) {
            Validator::is_valid_positive_double(s0);
            Validator::is_valid_positive_double(Z);
            return s0* std::exp(m*data_simul.time_step+
                 market_data_.sigma*calc_Wt(Z));
        }

};
#endif