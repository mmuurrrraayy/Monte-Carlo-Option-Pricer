#include "mcpricer/GBM.h"
#include <cmath>

GBM::GBM(market_data market_data_,sim_data data_simul_):
    market_data_(market_data_),
    data_simul(data_simul_),
    /*
     * risk-neutral drift: a pricer must diffuse at r, not the real-world mu,
     * or it converges to something that is not the Black-Scholes price.
     */
    m(market_data_.r - 0.5*(market_data_.sigma*market_data_.sigma)),
    rng(data_simul.seed),
    normal(0.0,1.0){}

double GBM::calc_Wt(double Z) {
    return Z * std::sqrt(data_simul.time_step);
}

double GBM::calc_Z() {
    return normal(rng);
}

double GBM::calc_St(double s0, double Z) {
    return s0* std::exp(m*data_simul.time_step+
           market_data_.sigma*calc_Wt(Z));
}
