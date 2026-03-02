#include "GBM.h"
#include "Option.h"
#include "MonteCarloEngine.h"
#include "Black-Scholes.h"
#include "DataVariables.h"
#include <iostream>
#include <iomanip>

int main() {
    market_data md;
        md.s0 = 100;
        md.mu = 0.05;
        md.sigma = 0.2;

    option_data od;
        od.K = 100;
        od.T = 1.0;
        od.option_type = OptionType::Call;

    sim_data sd;
        sd.N = 1000;
    
    GBM gbm = GBM(md, sd);
    Option option(md, od, sd, od.option_type);
    MonteCarloEngine engine(gbm, option, md, od, sd);
    Black_Scholes bs_engine(od.T,  md, od);

    payoff_info result = engine.price(10000, md.s0);
    double bs_price = bs_engine.price(md.s0);

    std::cout << "MC Price: " << result.discounted << std::endl;
    std::cout << "Average Payoff: " << result.average << std::endl;
    std::cout << "Black-Scholes Price: " << bs_price << std::endl;


    return 0;
}