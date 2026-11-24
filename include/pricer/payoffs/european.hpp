#ifndef PRICER_PAYOFFS_EUROPEAN_HPP
#define PRICER_PAYOFFS_EUROPEAN_HPP

#include "pricer/core/dual.hpp"
#include "pricer/core/types.hpp"
#include "pricer/simd/vec4d.hpp"

#include <cmath>
#include <expected>
#include <type_traits>

namespace pricer {

template <OptionType Type>
class EuropeanPayoff {
public:
    static constexpr bool is_path_dependent = false;
    static constexpr bool needs_spot_per_step = false;
    static constexpr bool is_smooth = true;
    static constexpr OptionType option_type = Type;

    constexpr explicit EuropeanPayoff(double strike) noexcept : strike_(strike) {}

    [[nodiscard]] static std::expected<EuropeanPayoff, ValidationError>
    create(double strike) noexcept {
        if (!std::isfinite(strike) || strike <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveStrike);
        }
        return EuropeanPayoff(strike);
    }

    [[nodiscard]] constexpr double strike() const noexcept { return strike_; }

    template <typename T>
    [[nodiscard]] PRICER_FORCE_INLINE T evaluate(const T& terminal_spot) const noexcept {
        if constexpr (std::is_same_v<T, Vec4d>) {
            const Vec4d k = Vec4d::splat(strike_);
            if constexpr (Type == OptionType::Call) {
                return max(terminal_spot - k, Vec4d::zero());
            } else {
                return max(k - terminal_spot, Vec4d::zero());
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if constexpr (Type == OptionType::Call) {
                const double diff = terminal_spot - strike_;
                return diff > 0.0 ? diff : 0.0;
            } else {
                const double diff = strike_ - terminal_spot;
                return diff > 0.0 ? diff : 0.0;
            }
        } else {
            if constexpr (Type == OptionType::Call) {
                return max(terminal_spot - strike_, 0.0);
            } else {
                return max(strike_ - terminal_spot, 0.0);
            }
        }
    }

private:
    double strike_ = 0.0;
};

} // namespace pricer

#endif // PRICER_PAYOFFS_EUROPEAN_HPP
