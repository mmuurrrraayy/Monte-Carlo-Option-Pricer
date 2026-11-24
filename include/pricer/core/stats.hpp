#ifndef PRICER_CORE_STATS_HPP
#define PRICER_CORE_STATS_HPP

#include "pricer/core/types.hpp"
#include "pricer/simd/vec4d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace pricer {

// m2 = sum_i (x_i - mean)^2; n counts paths (standard) or pairs (antithetic).
struct WorkerAccum {
    std::size_t n = 0;
    double mean = 0.0;
    double m2 = 0.0;
};

// Shifts samples by the first observation's mean to prevent cancellation in
// sum_d2 - sum_d^2 / n when mean^2 >> var.
class ChunkAccum {
public:
    ChunkAccum() noexcept
        : sum_d_(Vec4d::zero()),
          sum_d2_(Vec4d::zero()),
          shift_v_(Vec4d::zero()) {}

    PRICER_FORCE_INLINE void add_vec(Vec4d x) noexcept {
        if (!has_shift_) [[unlikely]] {
            shift_ = 0.25 * hsum(x);
            shift_v_ = Vec4d::splat(shift_);
            has_shift_ = true;
        }
        const Vec4d d = x - shift_v_;
        sum_d_ += d;
        sum_d2_ = fma(d, d, sum_d2_);
        n_ += 4;
    }

    inline void add_partial(Vec4d x, std::size_t lanes) noexcept {
        if (lanes == 0) {
            return;
        }
        if (lanes >= 4) {
            add_vec(x);
            return;
        }
        alignas(32) double tmp[4];
        x.store_aligned(tmp);
        if (!has_shift_) {
            double s = 0.0;
            for (std::size_t k = 0; k < lanes; ++k) {
                s += tmp[k];
            }
            shift_ = s / static_cast<double>(lanes);
            shift_v_ = Vec4d::splat(shift_);
            has_shift_ = true;
        }
        for (std::size_t k = 0; k < lanes; ++k) {
            const double d = tmp[k] - shift_;
            tail_sum_d_ += d;
            tail_sum_d2_ += d * d;
        }
        n_ += lanes;
    }

    inline void add_scalar(double x) noexcept {
        if (!has_shift_) {
            shift_ = x;
            shift_v_ = Vec4d::splat(shift_);
            has_shift_ = true;
        }
        const double d = x - shift_;
        tail_sum_d_ += d;
        tail_sum_d2_ += d * d;
        ++n_;
    }

    [[nodiscard]] inline WorkerAccum finalize() const noexcept {
        if (n_ == 0) {
            return WorkerAccum{};
        }
        const double sd = hsum(sum_d_) + tail_sum_d_;
        const double sd2 = hsum(sum_d2_) + tail_sum_d2_;
        const double dn = static_cast<double>(n_);
        const double mean = shift_ + sd / dn;
        const double raw_m2 = sd2 - (sd * sd) / dn;
        const double m2 = (raw_m2 < 0.0) ? 0.0 : raw_m2;
        return WorkerAccum{
            .n = n_,
            .mean = mean,
            .m2 = m2,
        };
    }

private:
    Vec4d sum_d_;
    Vec4d sum_d2_;
    Vec4d shift_v_;
    double shift_ = 0.0;
    double tail_sum_d_ = 0.0;
    double tail_sum_d2_ = 0.0;
    std::size_t n_ = 0;
    bool has_shift_ = false;
};

// Chan (1979) parallel variance merge.
[[nodiscard]] inline WorkerAccum merge_chan(const WorkerAccum& a,
                                            const WorkerAccum& b) noexcept {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    const std::size_t n = a.n + b.n;
    const double delta = b.mean - a.mean;
    const double na = static_cast<double>(a.n);
    const double nb = static_cast<double>(b.n);
    const double dn = static_cast<double>(n);
    return WorkerAccum{
        .n = n,
        .mean = a.mean + delta * (nb / dn),
        .m2 = a.m2 + b.m2 + delta * delta * (na * nb / dn),
    };
}

[[nodiscard]] inline WorkerAccum reduce_accum(std::span<const WorkerAccum> chunks) noexcept {
    WorkerAccum total{};
    for (const WorkerAccum& c : chunks) {
        total = merge_chan(total, c);
    }
    return total;
}

// reported_paths lets antithetic runs report 2 * pairs while SE uses n = pairs.
[[nodiscard]] inline PricingResult to_pricing_result(
    const WorkerAccum& acc, std::size_t reported_paths = 0) noexcept {
    constexpr double kZ95 = 1.959963984540054;
    const std::size_t paths = (reported_paths != 0) ? reported_paths : acc.n;
    if (acc.n == 0) {
        return PricingResult{};
    }
    const double var = (acc.n > 1)
        ? (acc.m2 / static_cast<double>(acc.n - 1))
        : 0.0;
    const double se = (acc.n > 1)
        ? std::sqrt(var / static_cast<double>(acc.n))
        : 0.0;
    return PricingResult{
        .price = acc.mean,
        .std_error = se,
        .ci_low = acc.mean - kZ95 * se,
        .ci_high = acc.mean + kZ95 * se,
        .variance = var,
        .paths = paths,
    };
}

[[nodiscard]] inline PricingResult reduce_chunks(
    std::span<const WorkerAccum> chunks, std::size_t reported_paths = 0) noexcept {
    return to_pricing_result(reduce_accum(chunks), reported_paths);
}

[[nodiscard]] inline GreekEstimate to_greek_estimate(const WorkerAccum& acc) noexcept {
    if (acc.n == 0) {
        return GreekEstimate{};
    }
    const double var = (acc.n > 1)
        ? (acc.m2 / static_cast<double>(acc.n - 1))
        : 0.0;
    const double se = (acc.n > 1)
        ? std::sqrt(var / static_cast<double>(acc.n))
        : 0.0;
    return GreekEstimate{
        .value = acc.mean,
        .std_error = se,
    };
}

[[nodiscard]] inline GreekEstimate reduce_greek(std::span<const WorkerAccum> chunks) noexcept {
    return to_greek_estimate(reduce_accum(chunks));
}

} // namespace pricer

#endif // PRICER_CORE_STATS_HPP
