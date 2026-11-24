#ifndef PRICER_ENGINES_MONTE_CARLO_HPP
#define PRICER_ENGINES_MONTE_CARLO_HPP

#include "pricer/core/concepts.hpp"
#include "pricer/core/dual.hpp"
#include "pricer/core/stats.hpp"
#include "pricer/core/types.hpp"
#include "pricer/engines/thread_pool.hpp"
#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"
#include "pricer/simd/xoshiro_avx2.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <vector>

namespace pricer {

struct SimConfig {
    std::size_t paths = 100'000;
    std::size_t steps = 252;          // ignored for 1-step exact European GBM unless force_multi_step
    std::uint64_t seed = 42;
    std::size_t threads = 1;          // 0 = pool default
    bool force_multi_step = false;
};

struct GreeksConfig {
    double eps_s0_rel = 1e-2;
    double eps_sigma = 1e-3;
    double eps_r = 1e-4;
    double theta_h = 0.005;
};

struct HestonGreeksConfig {
    double eps_v0 = 1e-4;             // clamped to <= 0.25 * v0 when v0 > 0
    double eps_long_run_var = 1e-4;   // clamped to <= 0.25 * theta
    double eps_r = 1e-4;
    double theta_h = 0.005;
};

namespace detail {

template <typename P>
inline constexpr bool has_spot_barrier_check_v = requires(const P& p, double s0) {
    { p.is_valid_for_spot(s0) } -> std::convertible_to<bool>;
};

} // namespace detail

template <ProcessPolicy Process, PayoffPolicy Payoff, SamplingPolicy Sampling = StandardSampling>
class MonteCarloEngine {
public:
    using process_type = Process;
    using payoff_type = Payoff;
    using sampling_type = Sampling;

    static constexpr std::size_t kChunkPaths = 4096;
    static constexpr std::size_t kChunkPairs = 2048;

    // Not thread-safe: neither an engine instance nor a shared ThreadPool
    // may have overlapping price*() calls from concurrent threads.
    MonteCarloEngine(Process process, Payoff payoff, ThreadPool* pool = nullptr) noexcept
        : process_(std::move(process)),
          payoff_(std::move(payoff)),
          pool_(pool) {}

    [[nodiscard]] ThreadPool* thread_pool() const noexcept { return pool_; }

    [[nodiscard]] std::expected<PricingResult, ValidationError>
    price(const SimConfig& cfg) {
        if (auto err = validate_sim_config(cfg); !err.has_value()) {
            return std::unexpected(err.error());
        }

        const std::size_t total_obs = total_observations(cfg.paths);
        const std::size_t obs_per_chunk = chunk_observations();
        const std::size_t num_chunks = (total_obs + obs_per_chunk - 1) / obs_per_chunk;
        const std::size_t reported_paths = Sampling::is_antithetic ? (2 * total_obs) : total_obs;

        ensure_chunk_capacity(num_chunks);

        const bool use_exact_1step =
            Process::supports_exact_terminal && !Payoff::is_path_dependent && !cfg.force_multi_step;

        if constexpr (Process::is_gbm && !Payoff::is_path_dependent) {
            if (use_exact_1step) {
                dispatch_chunks(num_chunks, cfg.threads, [&](std::size_t c) noexcept {
                    const std::size_t chunk_obs = chunk_obs_count(c, total_obs, obs_per_chunk);
                    chunk_accums_[c].value = run_gbm_exact_1step_chunk(process_, cfg.seed, c, chunk_obs);
                });
                return reduce_padded_chunks(num_chunks, reported_paths);
            }
        }

        const Process proc_m = process_.with_steps(cfg.steps);
        dispatch_chunks(num_chunks, cfg.threads, [&](std::size_t c) noexcept {
            const std::size_t chunk_obs = chunk_obs_count(c, total_obs, obs_per_chunk);
            if constexpr (Process::is_gbm) {
                chunk_accums_[c].value = run_gbm_multistep_chunk(proc_m, cfg.seed, c, chunk_obs);
            } else {
                chunk_accums_[c].value = run_heston_multistep_chunk(proc_m, cfg.seed, c, chunk_obs);
            }
        });
        return reduce_padded_chunks(num_chunks, reported_paths);
    }

    [[nodiscard]] std::expected<MCGreeksResult, ValidationError>
    price_with_greeks(const SimConfig& cfg, const GreeksConfig& gcfg = {})
        requires supports_gbm_5_greeks_v<Process, Payoff> {
        if (auto err = validate_sim_config(cfg); !err.has_value()) {
            return std::unexpected(err.error());
        }
        if (auto err = validate_greeks_config(gcfg); !err.has_value()) {
            return std::unexpected(err.error());
        }

        const std::size_t total_obs = total_observations(cfg.paths);
        const std::size_t obs_per_chunk = chunk_observations();
        const std::size_t num_chunks = (total_obs + obs_per_chunk - 1) / obs_per_chunk;
        const std::size_t reported_paths = Sampling::is_antithetic ? (2 * total_obs) : total_obs;

        ensure_greeks_capacity(num_chunks);

        const auto p_base = process_.params();
        const double h_theta = (std::min)(gcfg.theta_h, 0.49 * p_base.T);
        MarketParams p_tm = p_base;
        MarketParams p_tp = p_base;
        p_tm.T = p_base.T - h_theta;
        p_tp.T = p_base.T + h_theta;

        const bool use_exact_1step =
            Process::supports_exact_terminal && !Payoff::is_path_dependent && !cfg.force_multi_step;

        const std::size_t steps = use_exact_1step ? 1 : cfg.steps;
        const Process proc = process_.with_steps(steps);
        const Process proc_tm = process_.with_params(p_tm, steps);
        const Process proc_tp = process_.with_params(p_tp, steps);

        dispatch_chunks(num_chunks, cfg.threads, [&](std::size_t c) noexcept {
            const std::size_t chunk_obs = chunk_obs_count(c, total_obs, obs_per_chunk);
            if constexpr (!Payoff::is_path_dependent) {
                if (use_exact_1step) {
                    greeks_accums_[c].slots = run_gbm_euro_1step_greeks_chunk(
                        proc, proc_tm, proc_tp, h_theta, cfg.seed, c, chunk_obs);
                } else {
                    greeks_accums_[c].slots = run_gbm_multistep_greeks_chunk(
                        proc, proc_tm, proc_tp, h_theta, gcfg.eps_s0_rel, cfg.seed, c, chunk_obs);
                }
            } else {
                greeks_accums_[c].slots = run_gbm_multistep_greeks_chunk(
                    proc, proc_tm, proc_tp, h_theta, gcfg.eps_s0_rel, cfg.seed, c, chunk_obs);
            }
        });

        const auto totals = reduce_greeks_slots<6>(num_chunks);

        MCGreeksResult out{};
        out.price = to_greek_estimate(totals[0]);
        out.delta = to_greek_estimate(totals[1]);
        out.gamma = detail::is_floating_lookback_payoff_v<Payoff>
                        ? GreekEstimate{0.0, 0.0}
                        : to_greek_estimate(totals[2]);
        out.vega = to_greek_estimate(totals[3]);
        out.theta = to_greek_estimate(totals[4]);
        out.rho = to_greek_estimate(totals[5]);
        out.paths = reported_paths;
        return out;
    }

    [[nodiscard]] std::expected<BarrierGreeksResult, ValidationError>
    price_with_barrier_greeks(const SimConfig& cfg, const GreeksConfig& gcfg = {})
        requires supports_barrier_greeks_v<Process, Payoff> {
        if (auto err = validate_sim_config(cfg); !err.has_value()) {
            return std::unexpected(err.error());
        }
        if (auto err = validate_greeks_config(gcfg); !err.has_value()) {
            return std::unexpected(err.error());
        }

        const std::size_t total_obs = total_observations(cfg.paths);
        const std::size_t obs_per_chunk = chunk_observations();
        const std::size_t num_chunks = (total_obs + obs_per_chunk - 1) / obs_per_chunk;
        const std::size_t reported_paths = Sampling::is_antithetic ? (2 * total_obs) : total_obs;

        ensure_greeks_capacity(num_chunks);

        const MarketParams base = process_.params();
        const Process proc = process_.with_steps(cfg.steps);

        const bool vega_forward = (base.sigma == 0.0);
        const double eps_sig = vega_forward
            ? gcfg.eps_sigma
            : (std::min)(gcfg.eps_sigma, 0.5 * base.sigma);
        MarketParams p_vp = base;
        MarketParams p_vm = base;
        p_vp.sigma = base.sigma + eps_sig;
        p_vm.sigma = vega_forward ? 0.0 : (base.sigma - eps_sig);
        const Process proc_vp = process_.with_params(p_vp, cfg.steps);
        const Process proc_vm = process_.with_params(p_vm, cfg.steps);

        const double eps_r = gcfg.eps_r;
        MarketParams p_rp = base;
        MarketParams p_rm = base;
        p_rp.r = base.r + eps_r;
        p_rm.r = base.r - eps_r;
        const Process proc_rp = process_.with_params(p_rp, cfg.steps);
        const Process proc_rm = process_.with_params(p_rm, cfg.steps);

        const double h_theta = (std::min)(gcfg.theta_h, 0.49 * base.T);
        MarketParams p_tm = base;
        MarketParams p_tp = base;
        p_tm.T = base.T - h_theta;
        p_tp.T = base.T + h_theta;
        const Process proc_tm = process_.with_params(p_tm, cfg.steps);
        const Process proc_tp = process_.with_params(p_tp, cfg.steps);

        dispatch_chunks(num_chunks, cfg.threads, [&](std::size_t c) noexcept {
            const std::size_t chunk_obs = chunk_obs_count(c, total_obs, obs_per_chunk);
            const auto acc = run_barrier_greeks_chunk(
                proc, proc_vp, proc_vm, vega_forward, eps_sig,
                proc_rp, proc_rm, eps_r,
                proc_tm, proc_tp, h_theta,
                gcfg.eps_s0_rel, cfg.seed, c, chunk_obs);
            for (std::size_t k = 0; k < 5; ++k) {
                greeks_accums_[c].slots[k] = acc[k];
            }
        });

        const auto totals = reduce_greeks_slots<5>(num_chunks);

        BarrierGreeksResult out{};
        out.price = to_greek_estimate(totals[0]);
        out.delta = to_greek_estimate(totals[1]);
        out.vega = to_greek_estimate(totals[2]);
        out.theta = to_greek_estimate(totals[3]);
        out.rho = to_greek_estimate(totals[4]);
        out.paths = reported_paths;
        return out;
    }

    [[nodiscard]] std::expected<HestonGreeksResult, ValidationError>
    price_with_heston_greeks(const SimConfig& cfg, const HestonGreeksConfig& hcfg = {})
        requires supports_heston_greeks_v<Process, Payoff> {
        if (auto err = validate_sim_config(cfg); !err.has_value()) {
            return std::unexpected(err.error());
        }
        if (auto err = validate_heston_greeks_config(hcfg); !err.has_value()) {
            return std::unexpected(err.error());
        }

        const std::size_t total_obs = total_observations(cfg.paths);
        const std::size_t obs_per_chunk = chunk_observations();
        const std::size_t num_chunks = (total_obs + obs_per_chunk - 1) / obs_per_chunk;
        const std::size_t reported_paths = Sampling::is_antithetic ? (2 * total_obs) : total_obs;

        ensure_greeks_capacity(num_chunks);

        const HestonParams base = process_.params();
        const Process proc = process_.with_steps(cfg.steps);

        const bool v0_forward = (base.v0 == 0.0);
        const double eps_v0 = v0_forward
            ? hcfg.eps_v0
            : (std::min)(hcfg.eps_v0, 0.25 * base.v0);
        HestonParams p_v0p = base;
        HestonParams p_v0m = base;
        p_v0p.v0 = base.v0 + eps_v0;
        p_v0m.v0 = v0_forward ? 0.0 : (base.v0 - eps_v0);
        const Process proc_v0p = process_.with_params(p_v0p, cfg.steps);
        const Process proc_v0m = process_.with_params(p_v0m, cfg.steps);

        const double eps_lrv = (std::min)(hcfg.eps_long_run_var, 0.25 * base.theta);
        HestonParams p_lrvp = base;
        HestonParams p_lrvm = base;
        p_lrvp.theta = base.theta + eps_lrv;
        p_lrvm.theta = base.theta - eps_lrv;
        const Process proc_lrvp = process_.with_params(p_lrvp, cfg.steps);
        const Process proc_lrvm = process_.with_params(p_lrvm, cfg.steps);

        const double eps_r = hcfg.eps_r;
        HestonParams p_rp = base;
        HestonParams p_rm = base;
        p_rp.r = base.r + eps_r;
        p_rm.r = base.r - eps_r;
        const Process proc_rp = process_.with_params(p_rp, cfg.steps);
        const Process proc_rm = process_.with_params(p_rm, cfg.steps);

        const double h_theta = (std::min)(hcfg.theta_h, 0.49 * base.T);
        HestonParams p_tm = base;
        HestonParams p_tp = base;
        p_tm.T = base.T - h_theta;
        p_tp.T = base.T + h_theta;
        const Process proc_tm = process_.with_params(p_tm, cfg.steps);
        const Process proc_tp = process_.with_params(p_tp, cfg.steps);

        dispatch_chunks(num_chunks, cfg.threads, [&](std::size_t c) noexcept {
            const std::size_t chunk_obs = chunk_obs_count(c, total_obs, obs_per_chunk);
            greeks_accums_[c].slots = run_heston_greeks_chunk(
                proc,
                proc_v0p, proc_v0m, v0_forward, eps_v0,
                proc_lrvp, proc_lrvm, eps_lrv,
                proc_rp, proc_rm, eps_r,
                proc_tm, proc_tp, h_theta,
                cfg.seed, c, chunk_obs);
        });

        const auto totals = reduce_greeks_slots<6>(num_chunks);

        HestonGreeksResult out{};
        out.price = to_greek_estimate(totals[0]);
        out.delta = to_greek_estimate(totals[1]);
        out.d_v0 = to_greek_estimate(totals[2]);
        out.d_long_run_var = to_greek_estimate(totals[3]);
        out.theta = to_greek_estimate(totals[4]);
        out.rho = to_greek_estimate(totals[5]);
        out.paths = reported_paths;
        return out;
    }

private:
    [[nodiscard]] std::expected<void, ValidationError>
    validate_sim_config(const SimConfig& cfg) const noexcept {
        if constexpr (Sampling::is_antithetic) {
            if (cfg.paths < 4) {
                return std::unexpected(ValidationError::InvalidPaths);
            }
        } else {
            if (cfg.paths < 2) {
                return std::unexpected(ValidationError::InvalidPaths);
            }
        }
        const bool needs_steps =
            !Process::supports_exact_terminal || Payoff::is_path_dependent || cfg.force_multi_step;
        if (needs_steps && cfg.steps < 1) {
            return std::unexpected(ValidationError::InvalidSteps);
        }
        if constexpr (detail::has_spot_barrier_check_v<Payoff>) {
            if (!payoff_.is_valid_for_spot(process_.initial_spot())) {
                return std::unexpected(ValidationError::InvalidBarrier);
            }
        }
        return {};
    }

    [[nodiscard]] static std::expected<void, ValidationError>
    validate_greeks_config(const GreeksConfig& gcfg) noexcept {
        if (!std::isfinite(gcfg.eps_s0_rel) || gcfg.eps_s0_rel <= 0.0 || gcfg.eps_s0_rel >= 1.0) {
            return std::unexpected(ValidationError::InvalidBump);
        }
        if (!std::isfinite(gcfg.eps_sigma) || gcfg.eps_sigma <= 0.0) {
            return std::unexpected(ValidationError::InvalidBump);
        }
        if (!std::isfinite(gcfg.eps_r) || gcfg.eps_r <= 0.0) {
            return std::unexpected(ValidationError::InvalidBump);
        }
        if (!std::isfinite(gcfg.theta_h) || gcfg.theta_h <= 0.0) {
            return std::unexpected(ValidationError::InvalidBump);
        }
        return {};
    }

    [[nodiscard]] static std::expected<void, ValidationError>
    validate_heston_greeks_config(const HestonGreeksConfig& hcfg) noexcept {
        if (!std::isfinite(hcfg.eps_v0) || hcfg.eps_v0 <= 0.0) {
            return std::unexpected(ValidationError::InvalidBump);
        }
        if (!std::isfinite(hcfg.eps_long_run_var) || hcfg.eps_long_run_var <= 0.0) {
            return std::unexpected(ValidationError::InvalidBump);
        }
        if (!std::isfinite(hcfg.eps_r) || hcfg.eps_r <= 0.0) {
            return std::unexpected(ValidationError::InvalidBump);
        }
        if (!std::isfinite(hcfg.theta_h) || hcfg.theta_h <= 0.0) {
            return std::unexpected(ValidationError::InvalidBump);
        }
        return {};
    }

    [[nodiscard]] static constexpr std::size_t total_observations(std::size_t paths) noexcept {
        if constexpr (Sampling::is_antithetic) {
            return paths / 2;
        } else {
            return paths;
        }
    }

    [[nodiscard]] static constexpr std::size_t chunk_observations() noexcept {
        if constexpr (Sampling::is_antithetic) {
            return kChunkPairs;
        } else {
            return kChunkPaths;
        }
    }

    [[nodiscard]] static constexpr std::size_t
    chunk_obs_count(std::size_t chunk_id, std::size_t total_obs, std::size_t obs_per_chunk) noexcept {
        const std::size_t start = chunk_id * obs_per_chunk;
        return std::min(obs_per_chunk, total_obs - start);
    }

    struct alignas(64) PaddedAccum {
        WorkerAccum value{};
    };

    struct alignas(64) PaddedGreeksAccum {
        std::array<WorkerAccum, 6> slots{};
    };

    template <typename Fn>
    void dispatch_chunks(std::size_t num_chunks, std::size_t requested_threads, Fn&& fn) const noexcept {
        if (pool_ != nullptr && requested_threads != 1 && num_chunks > 1) {
            pool_->parallel_for(num_chunks, requested_threads, std::forward<Fn>(fn));
        } else {
            for (std::size_t c = 0; c < num_chunks; ++c) {
                fn(c);
            }
        }
    }

    [[nodiscard]] PricingResult reduce_padded_chunks(std::size_t num_chunks,
                                                     std::size_t reported_paths) const noexcept {
        WorkerAccum total{};
        for (std::size_t c = 0; c < num_chunks; ++c) {
            total = merge_chan(total, chunk_accums_[c].value);
        }
        return to_pricing_result(total, reported_paths);
    }

    template <std::size_t K>
    [[nodiscard]] std::array<WorkerAccum, K> reduce_greeks_slots(std::size_t num_chunks) const noexcept {
        std::array<WorkerAccum, K> totals{};
        for (std::size_t c = 0; c < num_chunks; ++c) {
            for (std::size_t k = 0; k < K; ++k) {
                totals[k] = merge_chan(totals[k], greeks_accums_[c].slots[k]);
            }
        }
        return totals;
    }

    void ensure_chunk_capacity(std::size_t num_chunks) {
        if (chunk_accums_.size() < num_chunks) {
            chunk_accums_.resize(num_chunks);
        }
    }

    void ensure_greeks_capacity(std::size_t num_chunks) {
        if (greeks_accums_.size() < num_chunks) {
            greeks_accums_.resize(num_chunks);
        }
    }

    [[nodiscard]] WorkerAccum
    run_gbm_exact_1step_chunk(const Process& proc, std::uint64_t seed,
                              std::size_t chunk_id, std::size_t chunk_obs) const noexcept
        requires (Process::is_gbm && !Payoff::is_path_dependent) {
        Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(seed, chunk_id);
        ChunkAccum accum;
        const Vec4d disc = Vec4d::splat(proc.discount_factor());
        const Vec4d half_disc = Vec4d::splat(0.5 * proc.discount_factor());

        auto eval_vec = [&](Vec4d z) noexcept -> Vec4d {
            if constexpr (Sampling::is_antithetic) {
                Vec4d sp, sn;
                proc.terminal_spot_antithetic(z, sp, sn);
                return half_disc * (payoff_.evaluate(sp) + payoff_.evaluate(sn));
            } else {
                return disc * payoff_.evaluate(proc.terminal_spot(z));
            }
        };

        std::size_t rem = chunk_obs;
        while (rem >= 8) {
            Vec4d z0, z1;
            rng.next_normals_pair(z0, z1);
            accum.add_vec(eval_vec(z0));
            accum.add_vec(eval_vec(z1));
            rem -= 8;
        }
        if (rem > 0) {
            Vec4d z0, z1;
            rng.next_normals_pair(z0, z1);
            if (rem >= 4) {
                accum.add_vec(eval_vec(z0));
                rem -= 4;
                if (rem > 0) {
                    accum.add_partial(eval_vec(z1), rem);
                }
            } else {
                accum.add_partial(eval_vec(z0), rem);
            }
        }
        return accum.finalize();
    }

    // Advances 2 Vec4d lanes per step so each Box-Muller (z0, z1) pair is fully consumed.
    [[nodiscard]] WorkerAccum
    run_gbm_multistep_chunk(const Process& proc, std::uint64_t seed,
                            std::size_t chunk_id, std::size_t chunk_obs) const noexcept
        requires (Process::is_gbm) {
        Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(seed, chunk_id);
        ChunkAccum accum;

        const std::size_t m = proc.steps();
        [[maybe_unused]] const double inv_steps = 1.0 / static_cast<double>(m);
        const Vec4d log_s0 = Vec4d::splat(proc.log_initial_spot());
        const Vec4d disc = Vec4d::splat(proc.discount_factor());
        const Vec4d half_disc = Vec4d::splat(0.5 * proc.discount_factor());

        auto run_pair8 = [&](Vec4d& out0, Vec4d& out1) noexcept {
            Vec4d log_s0_pos = log_s0;
            Vec4d log_s1_pos = log_s0;
            if constexpr (Sampling::is_antithetic) {
                Vec4d log_s0_neg = log_s0;
                Vec4d log_s1_neg = log_s0;
                if constexpr (Payoff::is_path_dependent) {
                    auto st0_pos = payoff_.init_state(log_s0);
                    auto st1_pos = payoff_.init_state(log_s0);
                    auto st0_neg = payoff_.init_state(log_s0);
                    auto st1_neg = payoff_.init_state(log_s0);
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z0, z1;
                        rng.next_normals_pair(z0, z1);
                        log_s0_pos = proc.step_log(log_s0_pos, z0);
                        log_s1_pos = proc.step_log(log_s1_pos, z1);
                        log_s0_neg = proc.step_log(log_s0_neg, -z0);
                        log_s1_neg = proc.step_log(log_s1_neg, -z1);
                        payoff_.update(st0_pos, log_s0_pos);
                        payoff_.update(st1_pos, log_s1_pos);
                        payoff_.update(st0_neg, log_s0_neg);
                        payoff_.update(st1_neg, log_s1_neg);
                    }
                    out0 = half_disc * (payoff_.evaluate(st0_pos, log_s0_pos, inv_steps) +
                                        payoff_.evaluate(st0_neg, log_s0_neg, inv_steps));
                    out1 = half_disc * (payoff_.evaluate(st1_pos, log_s1_pos, inv_steps) +
                                        payoff_.evaluate(st1_neg, log_s1_neg, inv_steps));
                } else {
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z0, z1;
                        rng.next_normals_pair(z0, z1);
                        log_s0_pos = proc.step_log(log_s0_pos, z0);
                        log_s1_pos = proc.step_log(log_s1_pos, z1);
                        log_s0_neg = proc.step_log(log_s0_neg, -z0);
                        log_s1_neg = proc.step_log(log_s1_neg, -z1);
                    }
                    out0 = half_disc * (payoff_.evaluate(exp_avx2(log_s0_pos)) +
                                        payoff_.evaluate(exp_avx2(log_s0_neg)));
                    out1 = half_disc * (payoff_.evaluate(exp_avx2(log_s1_pos)) +
                                        payoff_.evaluate(exp_avx2(log_s1_neg)));
                }
            } else {
                if constexpr (Payoff::is_path_dependent) {
                    auto st0 = payoff_.init_state(log_s0);
                    auto st1 = payoff_.init_state(log_s0);
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z0, z1;
                        rng.next_normals_pair(z0, z1);
                        log_s0_pos = proc.step_log(log_s0_pos, z0);
                        log_s1_pos = proc.step_log(log_s1_pos, z1);
                        payoff_.update(st0, log_s0_pos);
                        payoff_.update(st1, log_s1_pos);
                    }
                    out0 = disc * payoff_.evaluate(st0, log_s0_pos, inv_steps);
                    out1 = disc * payoff_.evaluate(st1, log_s1_pos, inv_steps);
                } else {
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z0, z1;
                        rng.next_normals_pair(z0, z1);
                        log_s0_pos = proc.step_log(log_s0_pos, z0);
                        log_s1_pos = proc.step_log(log_s1_pos, z1);
                    }
                    out0 = disc * payoff_.evaluate(exp_avx2(log_s0_pos));
                    out1 = disc * payoff_.evaluate(exp_avx2(log_s1_pos));
                }
            }
        };

        std::size_t rem = chunk_obs;
        while (rem >= 8) {
            Vec4d p0, p1;
            run_pair8(p0, p1);
            accum.add_vec(p0);
            accum.add_vec(p1);
            rem -= 8;
        }
        if (rem > 0) {
            Vec4d p0, p1;
            run_pair8(p0, p1);
            if (rem >= 4) {
                accum.add_vec(p0);
                rem -= 4;
                if (rem > 0) {
                    accum.add_partial(p1, rem);
                }
            } else {
                accum.add_partial(p0, rem);
            }
        }
        return accum.finalize();
    }

    [[nodiscard]] WorkerAccum
    run_heston_multistep_chunk(const Process& proc, std::uint64_t seed,
                               std::size_t chunk_id, std::size_t chunk_obs) const noexcept
        requires (!Process::is_gbm) {
        Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(seed, chunk_id);
        ChunkAccum accum;

        const std::size_t m = proc.steps();
        [[maybe_unused]] const double inv_steps = 1.0 / static_cast<double>(m);
        const Vec4d log_s0 = Vec4d::splat(proc.log_initial_spot());
        const Vec4d v0 = Vec4d::splat(proc.initial_variance());
        const Vec4d disc = Vec4d::splat(proc.discount_factor());
        const Vec4d half_disc = Vec4d::splat(0.5 * proc.discount_factor());

        auto run_block4 = [&]() noexcept -> Vec4d {
            Vec4d log_s_pos = log_s0;
            Vec4d v_pos = v0;
            if constexpr (Sampling::is_antithetic) {
                Vec4d log_s_neg = log_s0;
                Vec4d v_neg = v0;
                if constexpr (Payoff::is_path_dependent) {
                    auto st_pos = payoff_.init_state(log_s0);
                    auto st_neg = payoff_.init_state(log_s0);
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z1, z2;
                        rng.next_normals_pair(z1, z2);
                        proc.step(log_s_pos, v_pos, z1, z2);
                        proc.step(log_s_neg, v_neg, -z1, -z2);
                        payoff_.update(st_pos, log_s_pos);
                        payoff_.update(st_neg, log_s_neg);
                    }
                    return half_disc * (payoff_.evaluate(st_pos, log_s_pos, inv_steps) +
                                        payoff_.evaluate(st_neg, log_s_neg, inv_steps));
                } else {
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z1, z2;
                        rng.next_normals_pair(z1, z2);
                        proc.step(log_s_pos, v_pos, z1, z2);
                        proc.step(log_s_neg, v_neg, -z1, -z2);
                    }
                    return half_disc * (payoff_.evaluate(exp_avx2(log_s_pos)) +
                                        payoff_.evaluate(exp_avx2(log_s_neg)));
                }
            } else {
                if constexpr (Payoff::is_path_dependent) {
                    auto st = payoff_.init_state(log_s0);
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z1, z2;
                        rng.next_normals_pair(z1, z2);
                        proc.step(log_s_pos, v_pos, z1, z2);
                        payoff_.update(st, log_s_pos);
                    }
                    return disc * payoff_.evaluate(st, log_s_pos, inv_steps);
                } else {
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z1, z2;
                        rng.next_normals_pair(z1, z2);
                        proc.step(log_s_pos, v_pos, z1, z2);
                    }
                    return disc * payoff_.evaluate(exp_avx2(log_s_pos));
                }
            }
        };

        std::size_t rem = chunk_obs;
        while (rem >= 4) {
            accum.add_vec(run_block4());
            rem -= 4;
        }
        if (rem > 0) {
            accum.add_partial(run_block4(), rem);
        }
        return accum.finalize();
    }

    // Output slots: [0]=price, [1]=delta, [2]=gamma, [3]=vega, [4]=theta, [5]=rho.
    [[nodiscard]] std::array<WorkerAccum, 6>
    run_gbm_euro_1step_greeks_chunk(const Process& proc, const Process& proc_tm,
                                    const Process& proc_tp, double h_theta,
                                    std::uint64_t seed, std::size_t chunk_id,
                                    std::size_t chunk_obs) const noexcept
        requires (supports_gbm_5_greeks_v<Process, Payoff> && !Payoff::is_path_dependent) {
        Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(seed, chunk_id);
        std::array<ChunkAccum, 6> acc{};

        const Jet<Vec4d, 3> disc_jet = proc.discount_factor_jet();
        const Vec4d disc_tm = Vec4d::splat(proc_tm.discount_factor());
        const Vec4d disc_tp = Vec4d::splat(proc_tp.discount_factor());
        const Vec4d inv_2h = Vec4d::splat(0.5 / h_theta);
        const Vec4d half = Vec4d::splat(0.5);

        const double vol_sqrt_t = proc.vol_sqrt_T();
        const bool has_vol = (vol_sqrt_t > 0.0);
        const Vec4d v_vol_sqrt_t = Vec4d::splat(vol_sqrt_t);
        const Vec4d inv_s0_vol_sqrt_t = has_vol
            ? Vec4d::splat(1.0 / (proc.initial_spot() * vol_sqrt_t))
            : Vec4d::zero();

        auto eval_vec = [&](Vec4d z, std::array<Vec4d, 6>& out) noexcept {
            if constexpr (Sampling::is_antithetic) {
                Jet<Vec4d, 3> st_pos, st_neg;
                proc.terminal_spot_jet_antithetic(z, st_pos, st_neg);
                const Jet<Vec4d, 3> pv_pos = disc_jet * payoff_.evaluate(st_pos);
                const Jet<Vec4d, 3> pv_neg = disc_jet * payoff_.evaluate(st_neg);

                // Mixed Pathwise-LR Gamma (Broadie-Glasserman): Delta_PW * (Z - sigma*sqrt(T)) / (S0 * sigma * sqrt(T)).
                const Vec4d g_pos = pv_pos.d[0] * (z - v_vol_sqrt_t) * inv_s0_vol_sqrt_t;
                const Vec4d g_neg = pv_neg.d[0] * ((-z) - v_vol_sqrt_t) * inv_s0_vol_sqrt_t;

                Vec4d st_tm_pos, st_tm_neg, st_tp_pos, st_tp_neg;
                proc_tm.terminal_spot_antithetic(z, st_tm_pos, st_tm_neg);
                proc_tp.terminal_spot_antithetic(z, st_tp_pos, st_tp_neg);
                const Vec4d v_tm = half * disc_tm * (payoff_.evaluate(st_tm_pos) + payoff_.evaluate(st_tm_neg));
                const Vec4d v_tp = half * disc_tp * (payoff_.evaluate(st_tp_pos) + payoff_.evaluate(st_tp_neg));

                out[0] = half * (pv_pos.val + pv_neg.val);
                out[1] = half * (pv_pos.d[0] + pv_neg.d[0]);
                out[2] = half * (g_pos + g_neg);
                out[3] = half * (pv_pos.d[1] + pv_neg.d[1]);
                out[4] = (v_tm - v_tp) * inv_2h;
                out[5] = half * (pv_pos.d[2] + pv_neg.d[2]);
            } else {
                const Jet<Vec4d, 3> st = proc.terminal_spot_jet(z);
                const Jet<Vec4d, 3> pv = disc_jet * payoff_.evaluate(st);
                const Vec4d gamma = pv.d[0] * (z - v_vol_sqrt_t) * inv_s0_vol_sqrt_t;

                const Vec4d v_tm = disc_tm * payoff_.evaluate(proc_tm.terminal_spot(z));
                const Vec4d v_tp = disc_tp * payoff_.evaluate(proc_tp.terminal_spot(z));

                out[0] = pv.val;
                out[1] = pv.d[0];
                out[2] = gamma;
                out[3] = pv.d[1];
                out[4] = (v_tm - v_tp) * inv_2h;
                out[5] = pv.d[2];
            }
        };

        std::size_t rem = chunk_obs;
        while (rem >= 8) {
            Vec4d z0, z1;
            rng.next_normals_pair(z0, z1);
            std::array<Vec4d, 6> o0, o1;
            eval_vec(z0, o0);
            eval_vec(z1, o1);
            for (std::size_t k = 0; k < 6; ++k) {
                acc[k].add_vec(o0[k]);
                acc[k].add_vec(o1[k]);
            }
            rem -= 8;
        }
        if (rem > 0) {
            Vec4d z0, z1;
            rng.next_normals_pair(z0, z1);
            std::array<Vec4d, 6> o0, o1;
            eval_vec(z0, o0);
            if (rem >= 4) {
                for (std::size_t k = 0; k < 6; ++k) acc[k].add_vec(o0[k]);
                rem -= 4;
                if (rem > 0) {
                    eval_vec(z1, o1);
                    for (std::size_t k = 0; k < 6; ++k) acc[k].add_partial(o1[k], rem);
                }
            } else {
                for (std::size_t k = 0; k < 6; ++k) acc[k].add_partial(o0[k], rem);
            }
        }

        std::array<WorkerAccum, 6> res{};
        for (std::size_t k = 0; k < 6; ++k) {
            res[k] = acc[k].finalize();
        }
        return res;
    }

    [[nodiscard]] std::array<WorkerAccum, 6>
    run_gbm_multistep_greeks_chunk(const Process& proc, const Process& proc_tm,
                                   const Process& proc_tp, double h_theta,
                                   double eps_s0_rel, std::uint64_t seed,
                                   std::size_t chunk_id, std::size_t chunk_obs) const noexcept
        requires (supports_gbm_5_greeks_v<Process, Payoff>) {
        Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(seed, chunk_id);
        std::array<ChunkAccum, 6> acc{};

        const std::size_t m = proc.steps();
        const double dm = static_cast<double>(m);
        [[maybe_unused]] const double inv_steps = 1.0 / dm;
        const Jet<Vec4d, 3> log_s0_jet = proc.initial_log_spot_jet();
        const Vec4d log_s0 = Vec4d::splat(proc.log_initial_spot());
        const Jet<Vec4d, 3> disc_jet = proc.discount_factor_jet();
        const Vec4d disc_v = Vec4d::splat(proc.discount_factor());
        const Vec4d disc_tm = Vec4d::splat(proc_tm.discount_factor());
        const Vec4d disc_tp = Vec4d::splat(proc_tp.discount_factor());
        const Vec4d inv_2h = Vec4d::splat(0.5 / h_theta);
        const Vec4d half = Vec4d::splat(0.5);

        const MarketParams mp = proc.params();
        const double s0 = mp.s0;
        const double inv_s0 = 1.0 / s0;
        const double eps_s0 = eps_s0_rel * s0;
        const Vec4d inv_2eps_s0 = Vec4d::splat(0.5 / eps_s0);
        const Vec4d scale_plus = Vec4d::splat(1.0 + eps_s0_rel);
        const Vec4d scale_minus = Vec4d::splat(1.0 - eps_s0_rel);

        // Effective log-normal moments for single-Gaussian LR Gamma (European & GeometricAsian).
        const double sigma_eff_sqrt_T = [&]() noexcept -> double {
            if constexpr (detail::is_geometric_asian_payoff_v<Payoff>) {
                const double factor = std::sqrt(((dm + 1.0) * (2.0 * dm + 1.0)) / (6.0 * dm * dm));
                return mp.sigma * std::sqrt(mp.T) * factor;
            } else {
                return proc.vol_sqrt_T();
            }
        }();
        const double mean_log_shift = [&]() noexcept -> double {
            if constexpr (detail::is_geometric_asian_payoff_v<Payoff>) {
                return proc.log_initial_spot()
                     + (mp.r - mp.q - 0.5 * mp.sigma * mp.sigma) * mp.T * ((dm + 1.0) / (2.0 * dm));
            } else {
                return proc.log_initial_spot() + proc.drift_T();
            }
        }();
        const bool has_vol = (sigma_eff_sqrt_T > 0.0);
        const Vec4d v_mean_log_shift = Vec4d::splat(mean_log_shift);
        const Vec4d v_inv_sig_eff = has_vol ? Vec4d::splat(1.0 / sigma_eff_sqrt_T) : Vec4d::zero();
        const Vec4d v_sig_eff = Vec4d::splat(sigma_eff_sqrt_T);
        const Vec4d v_inv_s0_sig_eff = has_vol
            ? Vec4d::splat(1.0 / (s0 * sigma_eff_sqrt_T))
            : Vec4d::zero();

        [[maybe_unused]] auto compute_gamma_terminal = [&](const Jet<Vec4d, 3>& log_s_jet,
                                                           Vec4d delta_pw) noexcept -> Vec4d {
            const Vec4d z_t = (log_s_jet.val - v_mean_log_shift) * v_inv_sig_eff;
            return delta_pw * (z_t - v_sig_eff) * v_inv_s0_sig_eff;
        };

        [[maybe_unused]] auto compute_gamma_path = [&](const auto& state_jet,
                                                       Vec4d delta_pw) noexcept -> Vec4d {
            if constexpr (detail::is_floating_lookback_payoff_v<Payoff>) {
                (void)state_jet;
                (void)delta_pw;
                return Vec4d::zero();
            } else if constexpr (detail::is_geometric_asian_payoff_v<Payoff>) {
                const Vec4d log_g = state_jet.sum_log_s.val * inv_steps;
                const Vec4d z_g = (log_g - v_mean_log_shift) * v_inv_sig_eff;
                return delta_pw * (z_g - v_sig_eff) * v_inv_s0_sig_eff;
            } else if constexpr (detail::is_arithmetic_asian_payoff_v<Payoff>) {
                (void)delta_pw;
                // CRN central diff of pathwise Delta via homogeneity: A(S) = (S/S0) * A(S0).
                const Vec4d avg = state_jet.sum_s.val * inv_steps;
                const Vec4d k = Vec4d::splat(payoff_.strike());
                const Vec4d d_scale = disc_v * (avg * Vec4d::splat(inv_s0));
                if constexpr (Payoff::option_type == OptionType::Call) {
                    const Vec4d itm_p = cmp_gt(avg * scale_plus, k);
                    const Vec4d itm_m = cmp_gt(avg * scale_minus, k);
                    const Vec4d dp = bit_and(itm_p, d_scale);
                    const Vec4d dm_v = bit_and(itm_m, d_scale);
                    return (dp - dm_v) * inv_2eps_s0;
                } else {
                    const Vec4d itm_p = cmp_lt(avg * scale_plus, k);
                    const Vec4d itm_m = cmp_lt(avg * scale_minus, k);
                    const Vec4d dp = bit_and(itm_p, -d_scale);
                    const Vec4d dm_v = bit_and(itm_m, -d_scale);
                    return (dp - dm_v) * inv_2eps_s0;
                }
            } else if constexpr (detail::is_fixed_lookback_payoff_v<Payoff>) {
                (void)delta_pw;
                // CRN central diff of pathwise Delta via homogeneity: M(S) = (S/S0) * M(S0).
                const Vec4d ext_spot = exp_avx2(state_jet.extremum_log_s.val);
                const Vec4d k = Vec4d::splat(payoff_.strike());
                const Vec4d d_scale = disc_v * (ext_spot * Vec4d::splat(inv_s0));
                if constexpr (Payoff::option_type == OptionType::Call) {
                    const Vec4d itm_p = cmp_gt(ext_spot * scale_plus, k);
                    const Vec4d itm_m = cmp_gt(ext_spot * scale_minus, k);
                    const Vec4d dp = bit_and(itm_p, d_scale);
                    const Vec4d dm_v = bit_and(itm_m, d_scale);
                    return (dp - dm_v) * inv_2eps_s0;
                } else {
                    const Vec4d itm_p = cmp_lt(ext_spot * scale_plus, k);
                    const Vec4d itm_m = cmp_lt(ext_spot * scale_minus, k);
                    const Vec4d dp = bit_and(itm_p, -d_scale);
                    const Vec4d dm_v = bit_and(itm_m, -d_scale);
                    return (dp - dm_v) * inv_2eps_s0;
                }
            } else {
                static_assert(!sizeof(Payoff*),
                              "PayoffPolicy has no supported GBM Gamma estimator");
                return Vec4d::zero();
            }
        };

        auto run_pair8 = [&](std::array<Vec4d, 6>& o0, std::array<Vec4d, 6>& o1) noexcept {
            Jet<Vec4d, 3> ls0_p = log_s0_jet;
            Jet<Vec4d, 3> ls1_p = log_s0_jet;
            Vec4d tm0_p = log_s0, tm1_p = log_s0;
            Vec4d tp0_p = log_s0, tp1_p = log_s0;

            if constexpr (Sampling::is_antithetic) {
                Jet<Vec4d, 3> ls0_n = log_s0_jet;
                Jet<Vec4d, 3> ls1_n = log_s0_jet;
                Vec4d tm0_n = log_s0, tm1_n = log_s0;
                Vec4d tp0_n = log_s0, tp1_n = log_s0;

                if constexpr (Payoff::is_path_dependent) {
                    auto st0_p = payoff_.init_state(log_s0_jet);
                    auto st1_p = payoff_.init_state(log_s0_jet);
                    auto st0_n = payoff_.init_state(log_s0_jet);
                    auto st1_n = payoff_.init_state(log_s0_jet);
                    auto st_tm0_p = payoff_.init_state(log_s0);
                    auto st_tm1_p = payoff_.init_state(log_s0);
                    auto st_tm0_n = payoff_.init_state(log_s0);
                    auto st_tm1_n = payoff_.init_state(log_s0);
                    auto st_tp0_p = payoff_.init_state(log_s0);
                    auto st_tp1_p = payoff_.init_state(log_s0);
                    auto st_tp0_n = payoff_.init_state(log_s0);
                    auto st_tp1_n = payoff_.init_state(log_s0);

                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z0, z1;
                        rng.next_normals_pair(z0, z1);
                        const Vec4d nz0 = -z0;
                        const Vec4d nz1 = -z1;

                        ls0_p = proc.step_log_jet(ls0_p, z0);
                        ls1_p = proc.step_log_jet(ls1_p, z1);
                        ls0_n = proc.step_log_jet(ls0_n, nz0);
                        ls1_n = proc.step_log_jet(ls1_n, nz1);
                        payoff_.update(st0_p, ls0_p);
                        payoff_.update(st1_p, ls1_p);
                        payoff_.update(st0_n, ls0_n);
                        payoff_.update(st1_n, ls1_n);

                        tm0_p = proc_tm.step_log(tm0_p, z0);
                        tm1_p = proc_tm.step_log(tm1_p, z1);
                        tm0_n = proc_tm.step_log(tm0_n, nz0);
                        tm1_n = proc_tm.step_log(tm1_n, nz1);
                        payoff_.update(st_tm0_p, tm0_p);
                        payoff_.update(st_tm1_p, tm1_p);
                        payoff_.update(st_tm0_n, tm0_n);
                        payoff_.update(st_tm1_n, tm1_n);

                        tp0_p = proc_tp.step_log(tp0_p, z0);
                        tp1_p = proc_tp.step_log(tp1_p, z1);
                        tp0_n = proc_tp.step_log(tp0_n, nz0);
                        tp1_n = proc_tp.step_log(tp1_n, nz1);
                        payoff_.update(st_tp0_p, tp0_p);
                        payoff_.update(st_tp1_p, tp1_p);
                        payoff_.update(st_tp0_n, tp0_n);
                        payoff_.update(st_tp1_n, tp1_n);
                    }

                    const auto pv0_p = disc_jet * payoff_.evaluate(st0_p, ls0_p, inv_steps);
                    const auto pv0_n = disc_jet * payoff_.evaluate(st0_n, ls0_n, inv_steps);
                    const auto pv1_p = disc_jet * payoff_.evaluate(st1_p, ls1_p, inv_steps);
                    const auto pv1_n = disc_jet * payoff_.evaluate(st1_n, ls1_n, inv_steps);

                    const Vec4d g0 = half * (compute_gamma_path(st0_p, pv0_p.d[0]) +
                                             compute_gamma_path(st0_n, pv0_n.d[0]));
                    const Vec4d g1 = half * (compute_gamma_path(st1_p, pv1_p.d[0]) +
                                             compute_gamma_path(st1_n, pv1_n.d[0]));

                    const Vec4d v_tm0 = half * disc_tm * (payoff_.evaluate(st_tm0_p, tm0_p, inv_steps) +
                                                          payoff_.evaluate(st_tm0_n, tm0_n, inv_steps));
                    const Vec4d v_tp0 = half * disc_tp * (payoff_.evaluate(st_tp0_p, tp0_p, inv_steps) +
                                                          payoff_.evaluate(st_tp0_n, tp0_n, inv_steps));
                    const Vec4d v_tm1 = half * disc_tm * (payoff_.evaluate(st_tm1_p, tm1_p, inv_steps) +
                                                          payoff_.evaluate(st_tm1_n, tm1_n, inv_steps));
                    const Vec4d v_tp1 = half * disc_tp * (payoff_.evaluate(st_tp1_p, tp1_p, inv_steps) +
                                                          payoff_.evaluate(st_tp1_n, tp1_n, inv_steps));

                    o0 = {half * (pv0_p.val + pv0_n.val), half * (pv0_p.d[0] + pv0_n.d[0]),
                          g0, half * (pv0_p.d[1] + pv0_n.d[1]),
                          (v_tm0 - v_tp0) * inv_2h, half * (pv0_p.d[2] + pv0_n.d[2])};
                    o1 = {half * (pv1_p.val + pv1_n.val), half * (pv1_p.d[0] + pv1_n.d[0]),
                          g1, half * (pv1_p.d[1] + pv1_n.d[1]),
                          (v_tm1 - v_tp1) * inv_2h, half * (pv1_p.d[2] + pv1_n.d[2])};
                } else {
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z0, z1;
                        rng.next_normals_pair(z0, z1);
                        const Vec4d nz0 = -z0;
                        const Vec4d nz1 = -z1;
                        ls0_p = proc.step_log_jet(ls0_p, z0);
                        ls1_p = proc.step_log_jet(ls1_p, z1);
                        ls0_n = proc.step_log_jet(ls0_n, nz0);
                        ls1_n = proc.step_log_jet(ls1_n, nz1);
                        tm0_p = proc_tm.step_log(tm0_p, z0);
                        tm1_p = proc_tm.step_log(tm1_p, z1);
                        tm0_n = proc_tm.step_log(tm0_n, nz0);
                        tm1_n = proc_tm.step_log(tm1_n, nz1);
                        tp0_p = proc_tp.step_log(tp0_p, z0);
                        tp1_p = proc_tp.step_log(tp1_p, z1);
                        tp0_n = proc_tp.step_log(tp0_n, nz0);
                        tp1_n = proc_tp.step_log(tp1_n, nz1);
                    }
                    const auto pv0_p = disc_jet * payoff_.evaluate(exp(ls0_p));
                    const auto pv0_n = disc_jet * payoff_.evaluate(exp(ls0_n));
                    const auto pv1_p = disc_jet * payoff_.evaluate(exp(ls1_p));
                    const auto pv1_n = disc_jet * payoff_.evaluate(exp(ls1_n));

                    const Vec4d g0 = half * (compute_gamma_terminal(ls0_p, pv0_p.d[0]) +
                                             compute_gamma_terminal(ls0_n, pv0_n.d[0]));
                    const Vec4d g1 = half * (compute_gamma_terminal(ls1_p, pv1_p.d[0]) +
                                             compute_gamma_terminal(ls1_n, pv1_n.d[0]));

                    const Vec4d v_tm0 = half * disc_tm * (payoff_.evaluate(exp_avx2(tm0_p)) +
                                                          payoff_.evaluate(exp_avx2(tm0_n)));
                    const Vec4d v_tp0 = half * disc_tp * (payoff_.evaluate(exp_avx2(tp0_p)) +
                                                          payoff_.evaluate(exp_avx2(tp0_n)));
                    const Vec4d v_tm1 = half * disc_tm * (payoff_.evaluate(exp_avx2(tm1_p)) +
                                                          payoff_.evaluate(exp_avx2(tm1_n)));
                    const Vec4d v_tp1 = half * disc_tp * (payoff_.evaluate(exp_avx2(tp1_p)) +
                                                          payoff_.evaluate(exp_avx2(tp1_n)));

                    o0 = {half * (pv0_p.val + pv0_n.val), half * (pv0_p.d[0] + pv0_n.d[0]),
                          g0, half * (pv0_p.d[1] + pv0_n.d[1]),
                          (v_tm0 - v_tp0) * inv_2h, half * (pv0_p.d[2] + pv0_n.d[2])};
                    o1 = {half * (pv1_p.val + pv1_n.val), half * (pv1_p.d[0] + pv1_n.d[0]),
                          g1, half * (pv1_p.d[1] + pv1_n.d[1]),
                          (v_tm1 - v_tp1) * inv_2h, half * (pv1_p.d[2] + pv1_n.d[2])};
                }
            } else {
                if constexpr (Payoff::is_path_dependent) {
                    auto st0 = payoff_.init_state(log_s0_jet);
                    auto st1 = payoff_.init_state(log_s0_jet);
                    auto st_tm0 = payoff_.init_state(log_s0);
                    auto st_tm1 = payoff_.init_state(log_s0);
                    auto st_tp0 = payoff_.init_state(log_s0);
                    auto st_tp1 = payoff_.init_state(log_s0);

                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z0, z1;
                        rng.next_normals_pair(z0, z1);
                        ls0_p = proc.step_log_jet(ls0_p, z0);
                        ls1_p = proc.step_log_jet(ls1_p, z1);
                        payoff_.update(st0, ls0_p);
                        payoff_.update(st1, ls1_p);

                        tm0_p = proc_tm.step_log(tm0_p, z0);
                        tm1_p = proc_tm.step_log(tm1_p, z1);
                        payoff_.update(st_tm0, tm0_p);
                        payoff_.update(st_tm1, tm1_p);

                        tp0_p = proc_tp.step_log(tp0_p, z0);
                        tp1_p = proc_tp.step_log(tp1_p, z1);
                        payoff_.update(st_tp0, tp0_p);
                        payoff_.update(st_tp1, tp1_p);
                    }

                    const auto pv0 = disc_jet * payoff_.evaluate(st0, ls0_p, inv_steps);
                    const auto pv1 = disc_jet * payoff_.evaluate(st1, ls1_p, inv_steps);
                    const Vec4d g0 = compute_gamma_path(st0, pv0.d[0]);
                    const Vec4d g1 = compute_gamma_path(st1, pv1.d[0]);
                    const Vec4d th0 = (disc_tm * payoff_.evaluate(st_tm0, tm0_p, inv_steps) -
                                       disc_tp * payoff_.evaluate(st_tp0, tp0_p, inv_steps)) * inv_2h;
                    const Vec4d th1 = (disc_tm * payoff_.evaluate(st_tm1, tm1_p, inv_steps) -
                                       disc_tp * payoff_.evaluate(st_tp1, tp1_p, inv_steps)) * inv_2h;

                    o0 = {pv0.val, pv0.d[0], g0, pv0.d[1], th0, pv0.d[2]};
                    o1 = {pv1.val, pv1.d[0], g1, pv1.d[1], th1, pv1.d[2]};
                } else {
                    for (std::size_t s = 0; s < m; ++s) {
                        Vec4d z0, z1;
                        rng.next_normals_pair(z0, z1);
                        ls0_p = proc.step_log_jet(ls0_p, z0);
                        ls1_p = proc.step_log_jet(ls1_p, z1);
                        tm0_p = proc_tm.step_log(tm0_p, z0);
                        tm1_p = proc_tm.step_log(tm1_p, z1);
                        tp0_p = proc_tp.step_log(tp0_p, z0);
                        tp1_p = proc_tp.step_log(tp1_p, z1);
                    }
                    const auto pv0 = disc_jet * payoff_.evaluate(exp(ls0_p));
                    const auto pv1 = disc_jet * payoff_.evaluate(exp(ls1_p));
                    const Vec4d g0 = compute_gamma_terminal(ls0_p, pv0.d[0]);
                    const Vec4d g1 = compute_gamma_terminal(ls1_p, pv1.d[0]);
                    const Vec4d th0 = (disc_tm * payoff_.evaluate(exp_avx2(tm0_p)) -
                                       disc_tp * payoff_.evaluate(exp_avx2(tp0_p))) * inv_2h;
                    const Vec4d th1 = (disc_tm * payoff_.evaluate(exp_avx2(tm1_p)) -
                                       disc_tp * payoff_.evaluate(exp_avx2(tp1_p))) * inv_2h;

                    o0 = {pv0.val, pv0.d[0], g0, pv0.d[1], th0, pv0.d[2]};
                    o1 = {pv1.val, pv1.d[0], g1, pv1.d[1], th1, pv1.d[2]};
                }
            }
        };

        std::size_t rem = chunk_obs;
        while (rem >= 8) {
            std::array<Vec4d, 6> o0, o1;
            run_pair8(o0, o1);
            for (std::size_t k = 0; k < 6; ++k) {
                acc[k].add_vec(o0[k]);
                acc[k].add_vec(o1[k]);
            }
            rem -= 8;
        }
        if (rem > 0) {
            std::array<Vec4d, 6> o0, o1;
            run_pair8(o0, o1);
            if (rem >= 4) {
                for (std::size_t k = 0; k < 6; ++k) acc[k].add_vec(o0[k]);
                rem -= 4;
                if (rem > 0) {
                    for (std::size_t k = 0; k < 6; ++k) acc[k].add_partial(o1[k], rem);
                }
            } else {
                for (std::size_t k = 0; k < 6; ++k) acc[k].add_partial(o0[k], rem);
            }
        }

        std::array<WorkerAccum, 6> res{};
        for (std::size_t k = 0; k < 6; ++k) {
            res[k] = acc[k].finalize();
        }
        return res;
    }

    // Output slots: [0]=price, [1]=delta, [2]=vega, [3]=theta, [4]=rho.
    [[nodiscard]] std::array<WorkerAccum, 5>
    run_barrier_greeks_chunk(const Process& proc,
                             const Process& proc_vp, const Process& proc_vm,
                             bool vega_forward, double eps_sig,
                             const Process& proc_rp, const Process& proc_rm, double eps_r,
                             const Process& proc_tm, const Process& proc_tp, double h_theta,
                             double eps_s0_rel, std::uint64_t seed,
                             std::size_t chunk_id, std::size_t chunk_obs) const noexcept
        requires (supports_barrier_greeks_v<Process, Payoff>) {
        Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(seed, chunk_id);
        std::array<ChunkAccum, 5> acc{};

        const std::size_t m = proc.steps();
        const Vec4d log_s0 = Vec4d::splat(proc.log_initial_spot());
        const Vec4d log_b = Vec4d::splat(payoff_.log_barrier());

        // Asymmetric log shifts keep S_plus - S_minus == 2 * eps_s0_rel * S0 exact.
        const double shift_sp = std::log(1.0 + eps_s0_rel);
        const double shift_sm = std::log(1.0 - eps_s0_rel);
        const Vec4d log_b_sp = Vec4d::splat(payoff_.log_barrier() - shift_sp);
        const Vec4d log_b_sm = Vec4d::splat(payoff_.log_barrier() - shift_sm);
        const Vec4d scale_sp = Vec4d::splat(1.0 + eps_s0_rel);
        const Vec4d scale_sm = Vec4d::splat(1.0 - eps_s0_rel);

        const double s0 = proc.initial_spot();
        const Vec4d inv_2eps_s0 = Vec4d::splat(0.5 / (eps_s0_rel * s0));
        const Vec4d inv_vega_denom = Vec4d::splat(vega_forward ? (1.0 / eps_sig) : (0.5 / eps_sig));
        const Vec4d inv_2eps_r = Vec4d::splat(0.5 / eps_r);
        const Vec4d inv_2h = Vec4d::splat(0.5 / h_theta);

        const Vec4d disc = Vec4d::splat(proc.discount_factor());
        const Vec4d disc_rp = Vec4d::splat(proc_rp.discount_factor());
        const Vec4d disc_rm = Vec4d::splat(proc_rm.discount_factor());
        const Vec4d disc_tm = Vec4d::splat(proc_tm.discount_factor());
        const Vec4d disc_tp = Vec4d::splat(proc_tp.discount_factor());
        const Vec4d half = Vec4d::splat(0.5);

        struct BarrierLaneState {
            Vec4d ls_0, ls_vp, ls_vm, ls_rp, ls_rm, ls_tm, ls_tp;
            Vec4d m_0, m_sp, m_sm, m_vp, m_vm, m_rp, m_rm, m_tm, m_tp;

            void init(Vec4d ls0) noexcept {
                ls_0 = ls_vp = ls_vm = ls_rp = ls_rm = ls_tm = ls_tp = ls0;
                const Vec4d z = Vec4d::zero();
                m_0 = m_sp = m_sm = m_vp = m_vm = m_rp = m_rm = m_tm = m_tp = z;
            }
        };

        auto step_lane = [&](BarrierLaneState& s, Vec4d z) noexcept {
            s.ls_0 = proc.step_log(s.ls_0, z);
            Payoff::update_threshold(s.m_0, s.ls_0, log_b);
            Payoff::update_threshold(s.m_sp, s.ls_0, log_b_sp);
            Payoff::update_threshold(s.m_sm, s.ls_0, log_b_sm);

            s.ls_vp = proc_vp.step_log(s.ls_vp, z);
            Payoff::update_threshold(s.m_vp, s.ls_vp, log_b);
            if (!vega_forward) {
                s.ls_vm = proc_vm.step_log(s.ls_vm, z);
                Payoff::update_threshold(s.m_vm, s.ls_vm, log_b);
            }

            s.ls_rp = proc_rp.step_log(s.ls_rp, z);
            Payoff::update_threshold(s.m_rp, s.ls_rp, log_b);
            s.ls_rm = proc_rm.step_log(s.ls_rm, z);
            Payoff::update_threshold(s.m_rm, s.ls_rm, log_b);

            s.ls_tm = proc_tm.step_log(s.ls_tm, z);
            Payoff::update_threshold(s.m_tm, s.ls_tm, log_b);
            s.ls_tp = proc_tp.step_log(s.ls_tp, z);
            Payoff::update_threshold(s.m_tp, s.ls_tp, log_b);
        };

        auto eval_lane = [&](const BarrierLaneState& s, std::array<Vec4d, 5>& out) noexcept {
            const Vec4d st_0 = exp_avx2(s.ls_0);
            const Vec4d p_0 = disc * payoff_.evaluate_from_spot(s.m_0, st_0);
            const Vec4d p_sp = disc * payoff_.evaluate_from_spot(s.m_sp, st_0 * scale_sp);
            const Vec4d p_sm = disc * payoff_.evaluate_from_spot(s.m_sm, st_0 * scale_sm);

            const Vec4d p_vp = disc * payoff_.evaluate_from_spot(s.m_vp, exp_avx2(s.ls_vp));
            const Vec4d p_vm = vega_forward
                ? p_0
                : (disc * payoff_.evaluate_from_spot(s.m_vm, exp_avx2(s.ls_vm)));

            const Vec4d p_rp = disc_rp * payoff_.evaluate_from_spot(s.m_rp, exp_avx2(s.ls_rp));
            const Vec4d p_rm = disc_rm * payoff_.evaluate_from_spot(s.m_rm, exp_avx2(s.ls_rm));

            const Vec4d p_tm = disc_tm * payoff_.evaluate_from_spot(s.m_tm, exp_avx2(s.ls_tm));
            const Vec4d p_tp = disc_tp * payoff_.evaluate_from_spot(s.m_tp, exp_avx2(s.ls_tp));

            out[0] = p_0;
            out[1] = (p_sp - p_sm) * inv_2eps_s0;
            out[2] = (p_vp - p_vm) * inv_vega_denom;
            out[3] = (p_tm - p_tp) * inv_2h;
            out[4] = (p_rp - p_rm) * inv_2eps_r;
        };

        auto run_pair8 = [&](std::array<Vec4d, 5>& o0, std::array<Vec4d, 5>& o1) noexcept {
            BarrierLaneState l0_p, l1_p;
            l0_p.init(log_s0);
            l1_p.init(log_s0);

            if constexpr (Sampling::is_antithetic) {
                BarrierLaneState l0_n, l1_n;
                l0_n.init(log_s0);
                l1_n.init(log_s0);
                for (std::size_t s = 0; s < m; ++s) {
                    Vec4d z0, z1;
                    rng.next_normals_pair(z0, z1);
                    step_lane(l0_p, z0);
                    step_lane(l1_p, z1);
                    step_lane(l0_n, -z0);
                    step_lane(l1_n, -z1);
                }
                std::array<Vec4d, 5> r0_p, r0_n, r1_p, r1_n;
                eval_lane(l0_p, r0_p);
                eval_lane(l0_n, r0_n);
                eval_lane(l1_p, r1_p);
                eval_lane(l1_n, r1_n);
                for (std::size_t k = 0; k < 5; ++k) {
                    o0[k] = half * (r0_p[k] + r0_n[k]);
                    o1[k] = half * (r1_p[k] + r1_n[k]);
                }
            } else {
                for (std::size_t s = 0; s < m; ++s) {
                    Vec4d z0, z1;
                    rng.next_normals_pair(z0, z1);
                    step_lane(l0_p, z0);
                    step_lane(l1_p, z1);
                }
                eval_lane(l0_p, o0);
                eval_lane(l1_p, o1);
            }
        };

        std::size_t rem = chunk_obs;
        while (rem >= 8) {
            std::array<Vec4d, 5> o0, o1;
            run_pair8(o0, o1);
            for (std::size_t k = 0; k < 5; ++k) {
                acc[k].add_vec(o0[k]);
                acc[k].add_vec(o1[k]);
            }
            rem -= 8;
        }
        if (rem > 0) {
            std::array<Vec4d, 5> o0, o1;
            run_pair8(o0, o1);
            if (rem >= 4) {
                for (std::size_t k = 0; k < 5; ++k) acc[k].add_vec(o0[k]);
                rem -= 4;
                if (rem > 0) {
                    for (std::size_t k = 0; k < 5; ++k) acc[k].add_partial(o1[k], rem);
                }
            } else {
                for (std::size_t k = 0; k < 5; ++k) acc[k].add_partial(o0[k], rem);
            }
        }

        std::array<WorkerAccum, 5> res{};
        for (std::size_t k = 0; k < 5; ++k) {
            res[k] = acc[k].finalize();
        }
        return res;
    }

    // Output slots: [0]=price, [1]=delta, [2]=d_v0, [3]=d_long_run_var, [4]=theta, [5]=rho.
    [[nodiscard]] std::array<WorkerAccum, 6>
    run_heston_greeks_chunk(const Process& proc,
                            const Process& proc_v0p, const Process& proc_v0m,
                            bool v0_forward, double eps_v0,
                            const Process& proc_lrvp, const Process& proc_lrvm, double eps_lrv,
                            const Process& proc_rp, const Process& proc_rm, double eps_r,
                            const Process& proc_tm, const Process& proc_tp, double h_theta,
                            std::uint64_t seed, std::size_t chunk_id,
                            std::size_t chunk_obs) const noexcept
        requires (supports_heston_greeks_v<Process, Payoff>) {
        Xoshiro256PlusPlusX4 rng = Xoshiro256PlusPlusX4::from_chunk(seed, chunk_id);
        std::array<ChunkAccum, 6> acc{};

        const std::size_t m = proc.steps();
        [[maybe_unused]] const double inv_steps = 1.0 / static_cast<double>(m);
        const Vec4d log_s0 = Vec4d::splat(proc.log_initial_spot());
        const Vec4d inv_s0 = Vec4d::splat(1.0 / proc.initial_spot());
        const Dual<Vec4d> log_s0_dual{log_s0, inv_s0};

        const Vec4d v0 = Vec4d::splat(proc.initial_variance());
        const Vec4d v0_p = Vec4d::splat(proc_v0p.initial_variance());
        const Vec4d v0_m = Vec4d::splat(proc_v0m.initial_variance());

        const Vec4d disc = Vec4d::splat(proc.discount_factor());
        const Vec4d disc_rp = Vec4d::splat(proc_rp.discount_factor());
        const Vec4d disc_rm = Vec4d::splat(proc_rm.discount_factor());
        const Vec4d disc_tm = Vec4d::splat(proc_tm.discount_factor());
        const Vec4d disc_tp = Vec4d::splat(proc_tp.discount_factor());

        const Vec4d inv_v0_denom = Vec4d::splat(v0_forward ? (1.0 / eps_v0) : (0.5 / eps_v0));
        const Vec4d inv_2eps_lrv = Vec4d::splat(0.5 / eps_lrv);
        const Vec4d inv_2eps_r = Vec4d::splat(0.5 / eps_r);
        const Vec4d inv_2h = Vec4d::splat(0.5 / h_theta);
        const Vec4d half = Vec4d::splat(0.5);

        // d(log_S_t)/dS0 == 1/S0 at every step, avoiding differentiation of sqrt(max(v, 0)).
        auto run_single_side = [&](auto& step_fn, std::array<Vec4d, 6>& out) noexcept {
            Vec4d ls0 = log_s0, v_0 = v0;
            Vec4d ls_v0p = log_s0, v_v0p = v0_p;
            Vec4d ls_v0m = log_s0, v_v0m = v0_m;
            Vec4d ls_lrvp = log_s0, v_lrvp = v0;
            Vec4d ls_lrvm = log_s0, v_lrvm = v0;
            Vec4d ls_rp = log_s0, v_rp = v0;
            Vec4d ls_rm = log_s0, v_rm = v0;
            Vec4d ls_tm = log_s0, v_tm = v0;
            Vec4d ls_tp = log_s0, v_tp = v0;

            if constexpr (Payoff::is_path_dependent) {
                auto st_0 = payoff_.init_state(log_s0_dual);
                auto st_v0p = payoff_.init_state(log_s0);
                auto st_v0m = payoff_.init_state(log_s0);
                auto st_lrvp = payoff_.init_state(log_s0);
                auto st_lrvm = payoff_.init_state(log_s0);
                auto st_rp = payoff_.init_state(log_s0);
                auto st_rm = payoff_.init_state(log_s0);
                auto st_tm = payoff_.init_state(log_s0);
                auto st_tp = payoff_.init_state(log_s0);

                for (std::size_t s = 0; s < m; ++s) {
                    Vec4d z1, z2;
                    step_fn(s, z1, z2);

                    proc.step(ls0, v_0, z1, z2);
                    payoff_.update(st_0, Dual<Vec4d>{ls0, inv_s0});

                    proc_v0p.step(ls_v0p, v_v0p, z1, z2);
                    payoff_.update(st_v0p, ls_v0p);
                    if (!v0_forward) {
                        proc_v0m.step(ls_v0m, v_v0m, z1, z2);
                        payoff_.update(st_v0m, ls_v0m);
                    }

                    proc_lrvp.step(ls_lrvp, v_lrvp, z1, z2);
                    payoff_.update(st_lrvp, ls_lrvp);
                    proc_lrvm.step(ls_lrvm, v_lrvm, z1, z2);
                    payoff_.update(st_lrvm, ls_lrvm);

                    proc_rp.step(ls_rp, v_rp, z1, z2);
                    payoff_.update(st_rp, ls_rp);
                    proc_rm.step(ls_rm, v_rm, z1, z2);
                    payoff_.update(st_rm, ls_rm);

                    proc_tm.step(ls_tm, v_tm, z1, z2);
                    payoff_.update(st_tm, ls_tm);
                    proc_tp.step(ls_tp, v_tp, z1, z2);
                    payoff_.update(st_tp, ls_tp);
                }

                const Dual<Vec4d> pv0 = payoff_.evaluate(st_0, Dual<Vec4d>{ls0, inv_s0}, inv_steps) * disc;
                const Vec4d p_v0p = disc * payoff_.evaluate(st_v0p, ls_v0p, inv_steps);
                const Vec4d p_v0m = v0_forward ? pv0.val : (disc * payoff_.evaluate(st_v0m, ls_v0m, inv_steps));
                const Vec4d p_lrvp = disc * payoff_.evaluate(st_lrvp, ls_lrvp, inv_steps);
                const Vec4d p_lrvm = disc * payoff_.evaluate(st_lrvm, ls_lrvm, inv_steps);
                const Vec4d p_rp = disc_rp * payoff_.evaluate(st_rp, ls_rp, inv_steps);
                const Vec4d p_rm = disc_rm * payoff_.evaluate(st_rm, ls_rm, inv_steps);
                const Vec4d p_tm = disc_tm * payoff_.evaluate(st_tm, ls_tm, inv_steps);
                const Vec4d p_tp = disc_tp * payoff_.evaluate(st_tp, ls_tp, inv_steps);

                out[0] = pv0.val;
                out[1] = pv0.d;
                out[2] = (p_v0p - p_v0m) * inv_v0_denom;
                out[3] = (p_lrvp - p_lrvm) * inv_2eps_lrv;
                out[4] = (p_tm - p_tp) * inv_2h;
                out[5] = (p_rp - p_rm) * inv_2eps_r;
            } else {
                for (std::size_t s = 0; s < m; ++s) {
                    Vec4d z1, z2;
                    step_fn(s, z1, z2);
                    proc.step(ls0, v_0, z1, z2);
                    proc_v0p.step(ls_v0p, v_v0p, z1, z2);
                    if (!v0_forward) {
                        proc_v0m.step(ls_v0m, v_v0m, z1, z2);
                    }
                    proc_lrvp.step(ls_lrvp, v_lrvp, z1, z2);
                    proc_lrvm.step(ls_lrvm, v_lrvm, z1, z2);
                    proc_rp.step(ls_rp, v_rp, z1, z2);
                    proc_rm.step(ls_rm, v_rm, z1, z2);
                    proc_tm.step(ls_tm, v_tm, z1, z2);
                    proc_tp.step(ls_tp, v_tp, z1, z2);
                }
                const Dual<Vec4d> pv0 = payoff_.evaluate(exp(Dual<Vec4d>{ls0, inv_s0})) * disc;
                const Vec4d p_v0p = disc * payoff_.evaluate(exp_avx2(ls_v0p));
                const Vec4d p_v0m = v0_forward ? pv0.val : (disc * payoff_.evaluate(exp_avx2(ls_v0m)));
                const Vec4d p_lrvp = disc * payoff_.evaluate(exp_avx2(ls_lrvp));
                const Vec4d p_lrvm = disc * payoff_.evaluate(exp_avx2(ls_lrvm));
                const Vec4d p_rp = disc_rp * payoff_.evaluate(exp_avx2(ls_rp));
                const Vec4d p_rm = disc_rm * payoff_.evaluate(exp_avx2(ls_rm));
                const Vec4d p_tm = disc_tm * payoff_.evaluate(exp_avx2(ls_tm));
                const Vec4d p_tp = disc_tp * payoff_.evaluate(exp_avx2(ls_tp));

                out[0] = pv0.val;
                out[1] = pv0.d;
                out[2] = (p_v0p - p_v0m) * inv_v0_denom;
                out[3] = (p_lrvp - p_lrvm) * inv_2eps_lrv;
                out[4] = (p_tm - p_tp) * inv_2h;
                out[5] = (p_rp - p_rm) * inv_2eps_r;
            }
        };

        auto run_block4 = [&](std::array<Vec4d, 6>& out) noexcept {
            if constexpr (Sampling::is_antithetic) {
                // Snapshot RNG so the negative leg replays exact (-z1, -z2) without buffering paths.
                Xoshiro256PlusPlusX4 rng_saved = rng;
                std::array<Vec4d, 6> r_pos, r_neg;
                auto draw_pos = [&](std::size_t, Vec4d& z1, Vec4d& z2) noexcept {
                    rng.next_normals_pair(z1, z2);
                };
                auto draw_neg = [&](std::size_t, Vec4d& z1, Vec4d& z2) noexcept {
                    rng_saved.next_normals_pair(z1, z2);
                    z1 = -z1;
                    z2 = -z2;
                };
                run_single_side(draw_pos, r_pos);
                run_single_side(draw_neg, r_neg);
                for (std::size_t k = 0; k < 6; ++k) {
                    out[k] = half * (r_pos[k] + r_neg[k]);
                }
            } else {
                auto draw_pos = [&](std::size_t, Vec4d& z1, Vec4d& z2) noexcept {
                    rng.next_normals_pair(z1, z2);
                };
                run_single_side(draw_pos, out);
            }
        };

        std::size_t rem = chunk_obs;
        while (rem >= 4) {
            std::array<Vec4d, 6> o;
            run_block4(o);
            for (std::size_t k = 0; k < 6; ++k) {
                acc[k].add_vec(o[k]);
            }
            rem -= 4;
        }
        if (rem > 0) {
            std::array<Vec4d, 6> o;
            run_block4(o);
            for (std::size_t k = 0; k < 6; ++k) {
                acc[k].add_partial(o[k], rem);
            }
        }

        std::array<WorkerAccum, 6> res{};
        for (std::size_t k = 0; k < 6; ++k) {
            res[k] = acc[k].finalize();
        }
        return res;
    }

    Process process_;
    Payoff payoff_;
    ThreadPool* pool_ = nullptr;
    std::vector<PaddedAccum> chunk_accums_;
    std::vector<PaddedGreeksAccum> greeks_accums_;
};

} // namespace pricer

#endif // PRICER_ENGINES_MONTE_CARLO_HPP
