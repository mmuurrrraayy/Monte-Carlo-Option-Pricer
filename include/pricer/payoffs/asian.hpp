#ifndef PRICER_PAYOFFS_ASIAN_HPP
#define PRICER_PAYOFFS_ASIAN_HPP

#include "pricer/core/dual.hpp"
#include "pricer/core/types.hpp"
#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"

#include <cmath>
#include <expected>
#include <type_traits>

namespace pricer {

namespace detail {

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T payoff_zero() noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return Vec4d::zero();
    } else if constexpr (std::is_same_v<T, double>) {
        return 0.0;
    } else {
        return T::constant(0.0);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T payoff_exp(const T& x) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return exp_avx2(x);
    } else if constexpr (std::is_same_v<T, double>) {
        return std::exp(x);
    } else {
        return exp(x);
    }
}

template <OptionType Type, typename T>
[[nodiscard]] PRICER_FORCE_INLINE T vanilla_payoff(const T& spot, double strike) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        const Vec4d k = Vec4d::splat(strike);
        if constexpr (Type == OptionType::Call) {
            return max(spot - k, Vec4d::zero());
        } else {
            return max(k - spot, Vec4d::zero());
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if constexpr (Type == OptionType::Call) {
            const double d = spot - strike;
            return d > 0.0 ? d : 0.0;
        } else {
            const double d = strike - spot;
            return d > 0.0 ? d : 0.0;
        }
    } else {
        if constexpr (Type == OptionType::Call) {
            return max(spot - strike, 0.0);
        } else {
            return max(strike - spot, 0.0);
        }
    }
}

} // namespace detail

// Monitored at t_1..t_m (excludes S_0).
template <OptionType Type>
class GeometricAsianPayoff {
public:
    static constexpr bool is_path_dependent = true;
    static constexpr bool needs_spot_per_step = false;
    static constexpr bool is_smooth = true;
    static constexpr bool is_geometric_asian = true;
    static constexpr OptionType option_type = Type;

    template <typename T>
    struct State {
        T sum_log_s;
    };

    constexpr explicit GeometricAsianPayoff(double strike) noexcept : strike_(strike) {}

    [[nodiscard]] static std::expected<GeometricAsianPayoff, ValidationError>
    create(double strike) noexcept {
        if (!std::isfinite(strike) || strike <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveStrike);
        }
        return GeometricAsianPayoff(strike);
    }

    [[nodiscard]] constexpr double strike() const noexcept { return strike_; }

    template <typename T>
    [[nodiscard]] PRICER_FORCE_INLINE State<T> init_state(const T& /*log_s0*/) const noexcept {
        return State<T>{detail::payoff_zero<T>()};
    }

    template <typename T>
    PRICER_FORCE_INLINE void update(State<T>& st, const T& log_s) const noexcept {
        st.sum_log_s += log_s;
    }

    template <typename T>
    [[nodiscard]] PRICER_FORCE_INLINE T
    evaluate(const State<T>& st, const T& /*final_log_s*/, double inv_steps) const noexcept {
        const T geom_avg = detail::payoff_exp(st.sum_log_s * inv_steps);
        return detail::vanilla_payoff<Type>(geom_avg, strike_);
    }

private:
    double strike_ = 0.0;
};

// Monitored at t_1..t_m (excludes S_0).
template <OptionType Type>
class ArithmeticAsianPayoff {
public:
    static constexpr bool is_path_dependent = true;
    static constexpr bool needs_spot_per_step = true;
    static constexpr bool is_smooth = true;
    static constexpr bool is_arithmetic_asian = true;
    static constexpr OptionType option_type = Type;

    template <typename T>
    struct State {
        T sum_s;
    };

    constexpr explicit ArithmeticAsianPayoff(double strike) noexcept : strike_(strike) {}

    [[nodiscard]] static std::expected<ArithmeticAsianPayoff, ValidationError>
    create(double strike) noexcept {
        if (!std::isfinite(strike) || strike <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveStrike);
        }
        return ArithmeticAsianPayoff(strike);
    }

    [[nodiscard]] constexpr double strike() const noexcept { return strike_; }

    template <typename T>
    [[nodiscard]] PRICER_FORCE_INLINE State<T> init_state(const T& /*log_s0*/) const noexcept {
        return State<T>{detail::payoff_zero<T>()};
    }

    template <typename T>
    PRICER_FORCE_INLINE void update(State<T>& st, const T& log_s) const noexcept {
        st.sum_s += detail::payoff_exp(log_s);
    }

    template <typename T>
    [[nodiscard]] PRICER_FORCE_INLINE T
    evaluate(const State<T>& st, const T& /*final_log_s*/, double inv_steps) const noexcept {
        const T arith_avg = st.sum_s * inv_steps;
        return detail::vanilla_payoff<Type>(arith_avg, strike_);
    }

private:
    double strike_ = 0.0;
};

} // namespace pricer

#endif // PRICER_PAYOFFS_ASIAN_HPP
