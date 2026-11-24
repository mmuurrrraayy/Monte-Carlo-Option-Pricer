#ifndef PRICER_CORE_TYPES_HPP
#define PRICER_CORE_TYPES_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>

namespace pricer {

enum class OptionType : std::uint8_t {
    Call,
    Put,
};

enum class BarrierType : std::uint8_t {
    UpOut,
    DownOut,
    UpIn,
    DownIn,
};

enum class LookbackStyle : std::uint8_t {
    FixedStrike,
    FloatingStrike,
};

enum class ValidationError : std::uint8_t {
    NonPositiveSpot,
    NonPositiveStrike,
    InvalidRate,
    InvalidDividendYield,
    NegativeVolatility,
    NonPositiveMaturity,
    InvalidSteps,
    InvalidPaths,
    InvalidCorrelation,
    NonPositiveKappaOrTheta,
    NonPositiveVolOfVol,
    InvalidBarrier,
    InvalidInitialVariance,
    InvalidBump,
};

[[nodiscard]] constexpr std::string_view to_string(ValidationError err) noexcept {
    switch (err) {
        case ValidationError::NonPositiveSpot:         return "spot price must be finite and > 0";
        case ValidationError::NonPositiveStrike:       return "strike price must be finite and > 0";
        case ValidationError::InvalidRate:             return "interest rate must be finite";
        case ValidationError::InvalidDividendYield:    return "dividend yield must be finite";
        case ValidationError::NegativeVolatility:      return "volatility must be finite and >= 0";
        case ValidationError::NonPositiveMaturity:     return "maturity must be finite and > 0";
        case ValidationError::InvalidSteps:            return "time steps must be >= 1";
        case ValidationError::InvalidPaths:            return "path count must be >= 2 (>= 4 for antithetic)";
        case ValidationError::InvalidCorrelation:      return "correlation must be finite and in [-1, 1]";
        case ValidationError::NonPositiveKappaOrTheta: return "kappa and theta must be finite and > 0";
        case ValidationError::NonPositiveVolOfVol:     return "vol-of-vol (xi) must be finite and > 0";
        case ValidationError::InvalidBarrier:          return "barrier level is invalid for spot/type";
        case ValidationError::InvalidInitialVariance:  return "initial variance (v0) must be finite and >= 0";
        case ValidationError::InvalidBump:             return "bump size must be finite and > 0";
    }
    return "unknown validation error";
}

struct MarketParams {
    double s0 = 0.0;
    double r = 0.0;
    double sigma = 0.0;
    double T = 0.0;
    double q = 0.0;

    [[nodiscard]] static std::expected<MarketParams, ValidationError>
    create(double s0, double r, double sigma, double T, double q = 0.0) noexcept {
        if (!std::isfinite(s0) || s0 <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveSpot);
        }
        if (!std::isfinite(r)) {
            return std::unexpected(ValidationError::InvalidRate);
        }
        if (!std::isfinite(sigma) || sigma < 0.0) {
            return std::unexpected(ValidationError::NegativeVolatility);
        }
        if (!std::isfinite(T) || T <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveMaturity);
        }
        if (!std::isfinite(q)) {
            return std::unexpected(ValidationError::InvalidDividendYield);
        }
        return MarketParams{s0, r, sigma, T, q};
    }
};

struct HestonParams {
    double s0 = 0.0;
    double r = 0.0;
    double v0 = 0.0;
    double kappa = 0.0;
    double theta = 0.0;
    double xi = 0.0;
    double rho = 0.0;
    double T = 0.0;
    double q = 0.0;

    [[nodiscard]] static std::expected<HestonParams, ValidationError>
    create(double s0, double r, double v0, double kappa, double theta,
           double xi, double rho, double T, double q = 0.0) noexcept {
        if (!std::isfinite(s0) || s0 <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveSpot);
        }
        if (!std::isfinite(r)) {
            return std::unexpected(ValidationError::InvalidRate);
        }
        if (!std::isfinite(v0) || v0 < 0.0) {
            return std::unexpected(ValidationError::InvalidInitialVariance);
        }
        if (!std::isfinite(kappa) || kappa <= 0.0 || !std::isfinite(theta) || theta <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveKappaOrTheta);
        }
        if (!std::isfinite(xi) || xi <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveVolOfVol);
        }
        if (!std::isfinite(rho) || rho < -1.0 || rho > 1.0) {
            return std::unexpected(ValidationError::InvalidCorrelation);
        }
        if (!std::isfinite(T) || T <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveMaturity);
        }
        if (!std::isfinite(q)) {
            return std::unexpected(ValidationError::InvalidDividendYield);
        }
        return HestonParams{s0, r, v0, kappa, theta, xi, rho, T, q};
    }
};

struct PricingResult {
    double price = 0.0;
    double std_error = 0.0;
    double ci_low = 0.0;
    double ci_high = 0.0;
    double variance = 0.0;
    std::size_t paths = 0;
};

struct GreekEstimate {
    double value = 0.0;
    double std_error = 0.0;
};

struct GreeksResult {
    double price = 0.0;
    double delta = 0.0;
    double gamma = 0.0;
    double vega = 0.0;
    double theta = 0.0;
    double rho = 0.0;
};

struct MCGreeksResult {
    GreekEstimate price{};
    GreekEstimate delta{};
    GreekEstimate gamma{};
    GreekEstimate vega{};
    GreekEstimate theta{};
    GreekEstimate rho{};
    std::size_t paths = 0;
};

struct BarrierGreeksResult {
    GreekEstimate price{};
    GreekEstimate delta{};
    GreekEstimate vega{};
    GreekEstimate theta{};
    GreekEstimate rho{};
    std::size_t paths = 0;
};

// d_long_run_var = dV/d(theta_heston); theta = -dV/dT.
struct HestonGreeksResult {
    GreekEstimate price{};
    GreekEstimate delta{};
    GreekEstimate d_v0{};
    GreekEstimate d_long_run_var{};
    GreekEstimate theta{};
    GreekEstimate rho{};
    std::size_t paths = 0;
};

} // namespace pricer

#endif // PRICER_CORE_TYPES_HPP
