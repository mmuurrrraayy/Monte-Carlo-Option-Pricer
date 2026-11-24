#ifndef PRICER_PAYOFFS_LOOKBACK_HPP
#define PRICER_PAYOFFS_LOOKBACK_HPP

#include "pricer/core/dual.hpp"
#include "pricer/core/types.hpp"
#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"

#include <cmath>
#include <expected>
#include <type_traits>

namespace pricer {

// Monitored at t_0..t_m in log-space (includes S_0).
template <OptionType Type, LookbackStyle Style>
class LookbackPayoff {
public:
    static constexpr bool is_path_dependent = true;
    static constexpr bool needs_spot_per_step = false;
    static constexpr bool is_smooth = true;
    static constexpr bool is_lookback = true;
    static constexpr bool is_floating_lookback = (Style == LookbackStyle::FloatingStrike);
    static constexpr bool is_fixed_lookback = (Style == LookbackStyle::FixedStrike);
    static constexpr OptionType option_type = Type;
    static constexpr LookbackStyle lookback_style = Style;

    static constexpr bool tracks_max =
        (Style == LookbackStyle::FixedStrike && Type == OptionType::Call) ||
        (Style == LookbackStyle::FloatingStrike && Type == OptionType::Put);

    template <typename T>
    struct State {
        T extremum_log_s;
    };

    constexpr LookbackPayoff() noexcept
        requires (Style == LookbackStyle::FloatingStrike)
        : strike_(0.0) {}

    constexpr explicit LookbackPayoff(double strike) noexcept : strike_(strike) {}

    [[nodiscard]] static std::expected<LookbackPayoff, ValidationError>
    create(double strike) noexcept
        requires (Style == LookbackStyle::FixedStrike) {
        if (!std::isfinite(strike) || strike <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveStrike);
        }
        return LookbackPayoff(strike);
    }

    [[nodiscard]] static std::expected<LookbackPayoff, ValidationError>
    create() noexcept
        requires (Style == LookbackStyle::FloatingStrike) {
        return LookbackPayoff();
    }

    [[nodiscard]] constexpr double strike() const noexcept { return strike_; }

    template <typename T>
    [[nodiscard]] PRICER_FORCE_INLINE State<T> init_state(const T& log_s0) const noexcept {
        return State<T>{log_s0};
    }

    template <typename T>
    PRICER_FORCE_INLINE void update(State<T>& st, const T& log_s) const noexcept {
        if constexpr (tracks_max) {
            if constexpr (std::is_same_v<T, double>) {
                st.extremum_log_s = (log_s > st.extremum_log_s) ? log_s : st.extremum_log_s;
            } else {
                st.extremum_log_s = max(st.extremum_log_s, log_s);
            }
        } else {
            if constexpr (std::is_same_v<T, double>) {
                st.extremum_log_s = (log_s < st.extremum_log_s) ? log_s : st.extremum_log_s;
            } else {
                st.extremum_log_s = min(st.extremum_log_s, log_s);
            }
        }
    }

    template <typename T>
    [[nodiscard]] PRICER_FORCE_INLINE T
    evaluate(const State<T>& st, const T& final_log_s, double /*inv_steps*/) const noexcept {
        auto eval_exp = [](const T& x) noexcept -> T {
            if constexpr (std::is_same_v<T, Vec4d>) {
                return exp_avx2(x);
            } else if constexpr (std::is_same_v<T, double>) {
                return std::exp(x);
            } else {
                return exp(x);
            }
        };

        const T extremum_spot = eval_exp(st.extremum_log_s);

        if constexpr (Style == LookbackStyle::FixedStrike) {
            if constexpr (std::is_same_v<T, Vec4d>) {
                const Vec4d k = Vec4d::splat(strike_);
                if constexpr (Type == OptionType::Call) {
                    return max(extremum_spot - k, Vec4d::zero());
                } else {
                    return max(k - extremum_spot, Vec4d::zero());
                }
            } else if constexpr (std::is_same_v<T, double>) {
                if constexpr (Type == OptionType::Call) {
                    const double d = extremum_spot - strike_;
                    return d > 0.0 ? d : 0.0;
                } else {
                    const double d = strike_ - extremum_spot;
                    return d > 0.0 ? d : 0.0;
                }
            } else {
                if constexpr (Type == OptionType::Call) {
                    return max(extremum_spot - strike_, 0.0);
                } else {
                    return max(strike_ - extremum_spot, 0.0);
                }
            }
        } else {
            const T terminal_spot = eval_exp(final_log_s);
            if constexpr (Type == OptionType::Call) {
                return terminal_spot - extremum_spot;
            } else {
                return extremum_spot - terminal_spot;
            }
        }
    }

private:
    double strike_ = 0.0;
};

} // namespace pricer

#endif // PRICER_PAYOFFS_LOOKBACK_HPP
