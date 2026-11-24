#ifndef PRICER_MODELS_GBM_HPP
#define PRICER_MODELS_GBM_HPP

#include "pricer/core/dual.hpp"
#include "pricer/core/types.hpp"
#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"

#include <cmath>
#include <cstddef>

namespace pricer {

class GBMProcess {
public:
    static constexpr bool supports_exact_terminal = true;
    static constexpr bool is_gbm = true;

    explicit GBMProcess(const MarketParams& params, std::size_t steps = 252) noexcept
        : params_(params),
          steps_(steps > 0 ? steps : 1) {
        recompute();
    }

    [[nodiscard]] GBMProcess with_steps(std::size_t steps) const noexcept {
        return GBMProcess(params_, steps);
    }

    [[nodiscard]] GBMProcess with_params(const MarketParams& params,
                                         std::size_t steps) const noexcept {
        return GBMProcess(params, steps);
    }

    [[nodiscard]] const MarketParams& params() const noexcept { return params_; }
    [[nodiscard]] std::size_t steps() const noexcept { return steps_; }
    [[nodiscard]] double initial_spot() const noexcept { return params_.s0; }
    [[nodiscard]] double log_initial_spot() const noexcept { return log_s0_; }
    [[nodiscard]] double maturity() const noexcept { return params_.T; }
    [[nodiscard]] double discount_factor() const noexcept { return disc_; }
    [[nodiscard]] double sqrt_T() const noexcept { return sqrt_T_; }
    [[nodiscard]] double drift_T() const noexcept { return drift_T_; }
    [[nodiscard]] double vol_sqrt_T() const noexcept { return vol_sqrt_T_; }
    [[nodiscard]] double s0_exp_drift() const noexcept { return s0_exp_drift_; }
    [[nodiscard]] double dt() const noexcept { return dt_; }
    [[nodiscard]] double sqrt_dt() const noexcept { return sqrt_dt_; }
    [[nodiscard]] double drift_dt() const noexcept { return drift_dt_; }
    [[nodiscard]] double vol_sqrt_dt() const noexcept { return vol_sqrt_dt_; }

    [[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL
    terminal_spot(Vec4d z) const noexcept {
        return Vec4d::splat(s0_exp_drift_) * exp_avx2(Vec4d::splat(vol_sqrt_T_) * z);
    }

    PRICER_FORCE_INLINE void PRICER_VEC_CALL
    terminal_spot_antithetic(Vec4d z, Vec4d& st_pos, Vec4d& st_neg) const noexcept {
        const Vec4d factor = exp_avx2(Vec4d::splat(vol_sqrt_T_) * z);
        const Vec4d base = Vec4d::splat(s0_exp_drift_);
        st_pos = base * factor;
        st_neg = base / factor;
    }

    // Jet derivative slots: [0] = d/dS0, [1] = d/dsigma, [2] = d/dr.
    [[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, 3>
    terminal_spot_jet(Vec4d z) const noexcept {
        const Vec4d st = terminal_spot(z);
        const Vec4d inv_s0 = Vec4d::splat(inv_s0_);
        const Vec4d d_log_dsig = fma(Vec4d::splat(sqrt_T_), z, Vec4d::splat(-sigma_T_));
        const Vec4d t_v = Vec4d::splat(params_.T);
        return Jet<Vec4d, 3>{
            st,
            {st * inv_s0, st * d_log_dsig, st * t_v},
        };
    }

    PRICER_FORCE_INLINE void
    terminal_spot_jet_antithetic(Vec4d z,
                                 Jet<Vec4d, 3>& st_pos,
                                 Jet<Vec4d, 3>& st_neg) const noexcept {
        Vec4d sp, sn;
        terminal_spot_antithetic(z, sp, sn);
        const Vec4d inv_s0 = Vec4d::splat(inv_s0_);
        const Vec4d sqrt_t = Vec4d::splat(sqrt_T_);
        const Vec4d neg_sig_t = Vec4d::splat(-sigma_T_);
        const Vec4d t_v = Vec4d::splat(params_.T);

        const Vec4d d_log_dsig_pos = fma(sqrt_t, z, neg_sig_t);
        const Vec4d d_log_dsig_neg = fnma(sqrt_t, z, neg_sig_t);
        st_pos = Jet<Vec4d, 3>{sp, {sp * inv_s0, sp * d_log_dsig_pos, sp * t_v}};
        st_neg = Jet<Vec4d, 3>{sn, {sn * inv_s0, sn * d_log_dsig_neg, sn * t_v}};
    }

    [[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL
    step_log(Vec4d log_s, Vec4d z) const noexcept {
        return fma(Vec4d::splat(vol_sqrt_dt_), z, log_s + Vec4d::splat(drift_dt_));
    }

    [[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, 3> initial_log_spot_jet() const noexcept {
        return Jet<Vec4d, 3>{
            Vec4d::splat(log_s0_),
            {Vec4d::splat(inv_s0_), Vec4d::zero(), Vec4d::zero()},
        };
    }

    [[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, 3>
    step_log_jet(const Jet<Vec4d, 3>& log_s, Vec4d z) const noexcept {
        return Jet<Vec4d, 3>{
            fma(Vec4d::splat(vol_sqrt_dt_), z, log_s.val + Vec4d::splat(drift_dt_)),
            {
                log_s.d[0],
                fma(Vec4d::splat(sqrt_dt_), z, log_s.d[1] + Vec4d::splat(-sigma_dt_)),
                log_s.d[2] + Vec4d::splat(dt_),
            },
        };
    }

    [[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, 3> discount_factor_jet() const noexcept {
        return Jet<Vec4d, 3>{
            Vec4d::splat(disc_),
            {Vec4d::zero(), Vec4d::zero(), Vec4d::splat(-params_.T * disc_)},
        };
    }

private:
    void recompute() noexcept {
        log_s0_ = std::log(params_.s0);
        inv_s0_ = 1.0 / params_.s0;
        sqrt_T_ = std::sqrt(params_.T);
        const double drift_rate = params_.r - params_.q - 0.5 * params_.sigma * params_.sigma;
        drift_T_ = drift_rate * params_.T;
        vol_sqrt_T_ = params_.sigma * sqrt_T_;
        sigma_T_ = params_.sigma * params_.T;
        s0_exp_drift_ = params_.s0 * std::exp(drift_T_);
        disc_ = std::exp(-params_.r * params_.T);

        dt_ = params_.T / static_cast<double>(steps_);
        sqrt_dt_ = std::sqrt(dt_);
        drift_dt_ = drift_rate * dt_;
        vol_sqrt_dt_ = params_.sigma * sqrt_dt_;
        sigma_dt_ = params_.sigma * dt_;
    }

    MarketParams params_{};
    std::size_t steps_ = 252;
    double log_s0_ = 0.0;
    double inv_s0_ = 0.0;
    double sqrt_T_ = 0.0;
    double drift_T_ = 0.0;
    double vol_sqrt_T_ = 0.0;
    double sigma_T_ = 0.0;
    double s0_exp_drift_ = 0.0;
    double disc_ = 1.0;
    double dt_ = 0.0;
    double sqrt_dt_ = 0.0;
    double drift_dt_ = 0.0;
    double vol_sqrt_dt_ = 0.0;
    double sigma_dt_ = 0.0;
};

} // namespace pricer

#endif // PRICER_MODELS_GBM_HPP
