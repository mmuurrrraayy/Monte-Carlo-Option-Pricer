#ifndef GBM_H
#define GBM_H

#include "mcpricer/DataVariables.h"
#include <random>

class GBM{
    private:
        market_data market_data_;
        sim_data data_simul;
        double m;
        std::mt19937 rng;
        std::normal_distribution<double> normal;

        double calc_Wt(double Z);

    public:
        GBM(market_data market_data_,sim_data data_simul_);

        double calc_Z();

        double calc_St(double s0, double Z);

};
#endif
