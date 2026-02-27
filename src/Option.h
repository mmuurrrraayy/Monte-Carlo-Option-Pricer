#ifndef Option_H
#define Option_H

#include <string>
#include <algorithm>
#include "DataVariables.h"

enum class OptionType {Call, Put};

class Option{

    private:
        market_data md;
        option_data od;
        sim_data sd;
        OptionType option_type;
    
    public:
        Option(market_data md,option_data od, sim_data sd,
            OptionType option_type):
            md(md), 
            sd(sd),
            od(od),
            option_type(option_type) {}


        double payoff(double St){
            switch(option_type){
                case OptionType::Call:
                    return std::max(St - od.K, 0.0);
                case OptionType::Put:
                    return std::max(od.K - St, 0.0);
            }
        }

};
#endif