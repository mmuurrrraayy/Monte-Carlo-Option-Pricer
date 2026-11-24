#ifndef PRICER_MODELS_HESTON_HPP
#define PRICER_MODELS_HESTON_HPP

#include "pricer/core/types.hpp"
#include "pricer/simd/vec4d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pricer {

class HestonProcess {
public:
    static constexpr bool supports_exact_terminal = false;
    static constexpr bool is_gbm = false;

    explicit HestonProcess(const HestonParams& params, std::size_t steps = 252) noexcept
        : params_(params),
          steps_(steps > 0 ? steps : 1) {
        recompute();
    }

    [[nodiscard]] HestonProcess with_steps(std::size_t steps) const noexcept {
        return HestonProcess(params_, steps);
    }

    [[nodiscard]] HestonProcess with_params(const HestonParams& params,
                                            std::size_t steps) const noexcept {
        return HestonProcess(params, steps);
    }

    [[nodiscard]] const HestonParams& params() const noexcept { return params_; }
    [[nodiscard]] std::size_t steps() const noexcept { return steps_; }
    [[nodiscard]] double initial_spot() const noexcept { return params_.s0; }
    [[nodiscard]] double log_initial_spot() const noexcept { return log_s0_; }
    [[nodiscard]] double initial_variance() const noexcept { return params_.v0; }
    [[nodiscard]] double maturity() const noexcept { return params_.T; }
    [[nodiscard]] double discount_factor() const noexcept { return disc_; }
    [[nodiscard]] double dt() const noexcept { return dt_; }
    [[nodiscard]] double sqrt_dt() const noexcept { return sqrt_dt_; }

    // Full Truncation Euler (Lord, Koekkoek & Van Dijk 2010): v_pos in drift/vol, raw v in update.
    PRICER_FORCE_INLINE void PRICER_VEC_CALL
    step(Vec4d& log_s, Vec4d& v, Vec4d z1, Vec4d z2) const noexcept {
        const Vec4d v_pos = max(v, Vec4d::zero());
        const Vec4d sqrt_v = sqrt(v_pos);
        const Vec4d z_v = fma(Vec4d::splat(params_.rho), z1,
                              Vec4d::splat(sqrt_1_m_rho2_) * z2);
        const Vec4d drift = fnma(Vec4d::splat(half_dt_), v_pos, Vec4d::splat(rq_dt_));
        log_s = fma(sqrt_v * Vec4d::splat(sqrt_dt_), z1, log_s + drift);
        v = fma(Vec4d::splat(xi_sqrt_dt_) * sqrt_v, z_v,
                fma(Vec4d::splat(kappa_dt_), Vec4d::splat(params_.theta) - v_pos, v));
    }

private:
    void recompute() noexcept {
        log_s0_ = std::log(params_.s0);
        disc_ = std::exp(-params_.r * params_.T);
        dt_ = params_.T / static_cast<double>(steps_);
        sqrt_dt_ = std::sqrt(dt_);
        half_dt_ = 0.5 * dt_;
        rq_dt_ = (params_.r - params_.q) * dt_;
        kappa_dt_ = params_.kappa * dt_;
        xi_sqrt_dt_ = params_.xi * sqrt_dt_;
        sqrt_1_m_rho2_ = std::sqrt(std::max(0.0, 1.0 - params_.rho * params_.rho));
    }

    HestonParams params_{};
    std::size_t steps_ = 252;
    double log_s0_ = 0.0;
    double disc_ = 1.0;
    double dt_ = 0.0;
    double sqrt_dt_ = 0.0;
    double half_dt_ = 0.0;
    double rq_dt_ = 0.0;
    double kappa_dt_ = 0.0;
    double xi_sqrt_dt_ = 0.0;
    double sqrt_1_m_rho2_ = 1.0;
};

} // namespace pricer

#endif // PRICER_MODELS_HESTON_HPP
