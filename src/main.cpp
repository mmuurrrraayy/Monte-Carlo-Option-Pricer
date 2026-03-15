#include "GBM.h"
#include "Option.h"
#include "MonteCarloEngine.h"
#include "Black-Scholes.h"
#include "AntitheticEngine.h"
#include "OptionType.h"
#include "DataVariables.h"
#include "Precision.h"
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
    Option option(md, od, sd);
    MonteCarloEngine engine(gbm, option, md, od, sd);
    Black_Scholes bs_engine(0.0,  md, od);
    AntitheticEngine anti_engine(gbm, option, md, od, sd);
   

    payoff_info result = engine.price(sd.N, md.s0);
    double bs_price = bs_engine.price(md.s0);
    payoff_info anti_result = anti_engine.price(sd.N, md.s0);

    Precision prec(engine, sd.N, result.t);
    Precision anti_prec(anti_engine, sd.N, anti_result.t);

    std::cout << "MC Price: " << result.discounted << std::endl;
    std::cout << "Antithetic Price: " << anti_result.discounted << std::endl;
    std::cout << "Average Payoff: " << result.average << std::endl;
    std::cout << "Black-Scholes Price: " << bs_price << std::endl;

    Precision::interval mc_interval = prec.calc_CI();
    Precision::interval anti_interval = anti_prec.calc_CI();

    std::cout << "Confidence Interval for MC: " << mc_interval.ci_minus<< " "<< mc_interval.ci_plus<< std::endl;
    std::cout << "Confidence Interval for Anti: " << anti_interval.ci_minus << " " << anti_interval.ci_plus << std::endl;


    return 0;
}