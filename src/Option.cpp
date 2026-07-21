#include "mcpricer/Option.h"
#include <algorithm>

Option::Option(market_data md,option_data od, sim_data sd):
    md(md),
    sd(sd),
    od(od) {}

double Option::payoff(double St) const {
    switch(od.option_type){
        case OptionType::Call:
            return std::max(St - od.K, 0.0);
        case OptionType::Put:
            return std::max(od.K - St, 0.0);
    }
    return 0.0;
}

double Option::payoff_prime(double St) const {
    switch(od.option_type){
        case OptionType::Call:
            return St > od.K ? 1.0 : 0.0;
        case OptionType::Put:
            return St < od.K ? -1.0 : 0.0;
    }
    return 0.0;
}

