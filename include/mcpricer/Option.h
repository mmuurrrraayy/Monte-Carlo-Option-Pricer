#ifndef Option_H
#define Option_H

#include "mcpricer/DataVariables.h"

class Option{

    private:
        market_data md;
        option_data od;
        sim_data sd;

    public:
        Option(market_data md,option_data od, sim_data sd);

        double payoff(double St) const;
     
         double payoff_prime(double St) const;

};
#endif
