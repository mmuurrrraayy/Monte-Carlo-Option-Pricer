#ifndef Option_H
#define Option_H

#include <string>
#include <algorithm>
#include <stdexcept>
#include "DataVariables.h"
#include "Validator.h"


enum class OptionType {Call, Put};

class Option{

    private:
        market_data md;
        option_data od;
        sim_data sd;
        OptionType option_type;
    
    public:
        Option(const market_data& md, const option_data& od, const sim_data& sd,
            OptionType option_type):
            md(md), 
            od(od),
            sd(sd),
            option_type(option_type) {}


        /*
        Calcaulates payoff of an option given price at a giiven time
        */
        double payoff(double St){
            Validator::is_valid_positive_double(St);
            switch(option_type){
                case OptionType::Call:
                    return std::max(St - od.K, 0.0);
                case OptionType::Put:
                    return std::max(od.K - St, 0.0);
                default:
                    throw std::invalid_argument("Invalid OptionType in payoff");
            }
        }

};
#endif