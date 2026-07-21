#include "mcpricer/Black-Scholes.h"

#include <cmath>

Black_Scholes::Black_Scholes(double time, market_data md, option_data od):
    md(md),
    od(od),
    time(time)
{
    remaining_time = od.T - time;
}

double Black_Scholes::d_plus(double S){
    // was r + (sigma^2/2)*T; the whole bracket must be scaled by T.
    return (std::log(S/od.K)+(md.r+(md.sigma*md.sigma)/2.0)*remaining_time)
           /(md.sigma*std::sqrt(remaining_time));
}

double Black_Scholes::d_minus(double S){
    return d_plus(S) - md.sigma * std::sqrt(remaining_time);
}

double Black_Scholes::normalise(double d){
    return 0.5 * std::erfc(-d/ sqrt(2));
}

double Black_Scholes::pdf(double d){
    static const auto inv_sqrt_2pi = 0.3989422804014327; // 1/sqrt(2*pi)
    return inv_sqrt_2pi * std::exp(-0.5*d*d);
}

double Black_Scholes::price(double S){
    auto d1 = d_plus(S);
    auto d2 = d_minus(S);
    switch(od.option_type){
        case OptionType::Call:
            return S*normalise(d1) -
                (std::exp(-md.r*(remaining_time))
                *od.K*normalise(d2));
        case OptionType::Put:
            return od.K*std::exp(-md.r*remaining_time) *
            normalise(-d2)-S*normalise(-d1);
    }
    return 0.0; // unreachable; silences -Wreturn-type
}

double Black_Scholes::delta(double S){
    const auto d1 = d_plus(S);
    switch(od.option_type){
        case OptionType::Call: return normalise(d1);
        case OptionType::Put:  return normalise(d1) - 1.0;
    }
    return 0.0;
}

double Black_Scholes::gamma(double S){
    const auto d1 = d_plus(S);
    return pdf(d1) / (S*md.sigma*std::sqrt(remaining_time));
}

double Black_Scholes::vega(double S){
    const auto d1 = d_plus(S);
    return S*pdf(d1)*std::sqrt(remaining_time);
}

double Black_Scholes::theta(double S){
    const auto d1 = d_plus(S);
    const auto d2 = d_minus(S);
    const auto diffusion_term = -(S*pdf(d1)*md.sigma) / (2.0*std::sqrt(remaining_time));
    switch(od.option_type){
        case OptionType::Call:
            return diffusion_term - md.r*od.K*std::exp(-md.r*remaining_time)*normalise(d2);
        case OptionType::Put:
            return diffusion_term + md.r*od.K*std::exp(-md.r*remaining_time)*normalise(-d2);
    }
    return 0.0;
}

double Black_Scholes::rho(double S){
    const auto d2 = d_minus(S);
    switch(od.option_type){
        case OptionType::Call:
            return od.K*remaining_time*std::exp(-md.r*remaining_time)*normalise(d2);
        case OptionType::Put:
            return -od.K*remaining_time*std::exp(-md.r*remaining_time)*normalise(-d2);
    }
    return 0.0;
}
