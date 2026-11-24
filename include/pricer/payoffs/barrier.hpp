#ifndef PRICER_PAYOFFS_BARRIER_HPP
#define PRICER_PAYOFFS_BARRIER_HPP

#include "pricer/core/types.hpp"
#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"

#include <cmath>
#include <expected>

namespace pricer {

// Monitored at t_1..t_m in log-space; requires S_0 < B (Up) or S_0 > B (Down).
template <OptionType Type, BarrierType BType>
class BarrierPayoff {
public:
    static constexpr bool is_path_dependent = true;
    static constexpr bool needs_spot_per_step = false;
    static constexpr bool is_smooth = false;
    static constexpr OptionType option_type = Type;
    static constexpr BarrierType barrier_type = BType;

    static constexpr bool is_up = (BType == BarrierType::UpOut || BType == BarrierType::UpIn);
    static constexpr bool is_knock_out = (BType == BarrierType::UpOut || BType == BarrierType::DownOut);

    template <typename T>
    struct State {
        Vec4d knocked_mask;
    };

    BarrierPayoff(double strike, double barrier) noexcept
        : strike_(strike),
          barrier_(barrier),
          log_barrier_(std::log(barrier)) {}

    [[nodiscard]] static std::expected<BarrierPayoff, ValidationError>
    create(double strike, double barrier, double s0) noexcept {
        if (!std::isfinite(strike) || strike <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveStrike);
        }
        if (!std::isfinite(s0) || s0 <= 0.0) {
            return std::unexpected(ValidationError::NonPositiveSpot);
        }
        if (!std::isfinite(barrier) || barrier <= 0.0) {
            return std::unexpected(ValidationError::InvalidBarrier);
        }
        if constexpr (is_up) {
            if (!(s0 < barrier)) {
                return std::unexpected(ValidationError::InvalidBarrier);
            }
        } else {
            if (!(s0 > barrier)) {
                return std::unexpected(ValidationError::InvalidBarrier);
            }
        }
        return BarrierPayoff(strike, barrier);
    }

    [[nodiscard]] constexpr double strike() const noexcept { return strike_; }
    [[nodiscard]] constexpr double barrier() const noexcept { return barrier_; }
    [[nodiscard]] constexpr double log_barrier() const noexcept { return log_barrier_; }

    [[nodiscard]] constexpr bool is_valid_for_spot(double s0) const noexcept {
        if (!std::isfinite(s0) || s0 <= 0.0 || !std::isfinite(barrier_) || barrier_ <= 0.0) {
            return false;
        }
        if constexpr (is_up) {
            return s0 < barrier_;
        } else {
            return s0 > barrier_;
        }
    }

    [[nodiscard]] PRICER_FORCE_INLINE State<Vec4d> init_state(Vec4d /*log_s0*/) const noexcept {
        return State<Vec4d>{Vec4d::zero()};
    }

    PRICER_FORCE_INLINE void update(State<Vec4d>& st, Vec4d log_s) const noexcept {
        update_threshold(st.knocked_mask, log_s, Vec4d::splat(log_barrier_));
    }

    PRICER_FORCE_INLINE static void PRICER_VEC_CALL
    update_threshold(Vec4d& knocked_mask, Vec4d log_s, Vec4d log_b) noexcept {
        Vec4d hit;
        if constexpr (is_up) {
            hit = cmp_ge(log_s, log_b);
        } else {
            hit = cmp_le(log_s, log_b);
        }
        knocked_mask = bit_or(knocked_mask, hit);
    }

    [[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL
    evaluate_from_spot(Vec4d knocked_mask, Vec4d terminal_spot) const noexcept {
        const Vec4d k = Vec4d::splat(strike_);
        Vec4d raw;
        if constexpr (Type == OptionType::Call) {
            raw = max(terminal_spot - k, Vec4d::zero());
        } else {
            raw = max(k - terminal_spot, Vec4d::zero());
        }
        if constexpr (is_knock_out) {
            return bit_andnot(knocked_mask, raw);
        } else {
            return bit_and(knocked_mask, raw);
        }
    }

    [[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL
    evaluate(const State<Vec4d>& st, Vec4d final_log_s, double /*inv_steps*/) const noexcept {
        return evaluate_from_spot(st.knocked_mask, exp_avx2(final_log_s));
    }

private:
    double strike_ = 0.0;
    double barrier_ = 0.0;
    double log_barrier_ = 0.0;
};

} // namespace pricer

#endif // PRICER_PAYOFFS_BARRIER_HPP
