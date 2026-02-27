#ifndef DataVariables_H
#define DataVariables_H
#include "Option.h"

    struct market_data{
        double s0, r, mu, sigma;
    };

    struct option_data{
        double K, T;
        OptionType option_type;
    };

    struct sim_data{
        int N;
        unsigned int seed = 42;
        const double time_step = 1.0/252.0;
    };
#endif

