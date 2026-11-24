#ifndef PRICER_ENGINES_BLACK_SCHOLES_HPP
#define PRICER_ENGINES_BLACK_SCHOLES_HPP

#include "pricer/core/dual.hpp"
#include "pricer/core/types.hpp"
#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace pricer {

struct AlignedBuffer {
    static constexpr std::size_t kAlignment = 64;

    AlignedBuffer() noexcept = default;

    explicit AlignedBuffer(std::size_t count) noexcept : size_(count) {
        if (count == 0) {
            return;
        }
        const std::size_t raw_bytes = count * sizeof(double);
        const std::size_t padded_bytes = (raw_bytes + (kAlignment - 1)) & ~(kAlignment - 1);
#if defined(_WIN32)
        ptr_ = static_cast<double*>(_aligned_malloc(padded_bytes, kAlignment));
#else
        ptr_ = static_cast<double*>(std::aligned_alloc(kAlignment, padded_bytes));
#endif
        if (ptr_ != nullptr) {
            std::memset(ptr_, 0, padded_bytes);
        } else {
            size_ = 0;
        }
    }

    ~AlignedBuffer() noexcept { reset(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
#if defined(_WIN32)
            _aligned_free(ptr_);
#else
            std::free(ptr_);
#endif
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    [[nodiscard]] PRICER_FORCE_INLINE double* data() noexcept { return ptr_; }
    [[nodiscard]] PRICER_FORCE_INLINE const double* data() const noexcept { return ptr_; }
    [[nodiscard]] PRICER_FORCE_INLINE double& operator[](std::size_t i) noexcept { return ptr_[i]; }
    [[nodiscard]] PRICER_FORCE_INLINE const double& operator[](std::size_t i) const noexcept { return ptr_[i]; }
    [[nodiscard]] PRICER_FORCE_INLINE std::size_t size() const noexcept { return size_; }
    [[nodiscard]] PRICER_FORCE_INLINE bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] PRICER_FORCE_INLINE explicit operator bool() const noexcept { return ptr_ != nullptr; }

    void fill(double value) noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            ptr_[i] = value;
        }
    }

private:
    double* ptr_ = nullptr;
    std::size_t size_ = 0;
};

struct OptionChainSoA {
    AlignedBuffer spot;
    AlignedBuffer strike;
    AlignedBuffer rate;
    AlignedBuffer div_yield;
    AlignedBuffer vol;
    AlignedBuffer tau;
    std::size_t count = 0;
    std::size_t padded_n = 0;

    OptionChainSoA() noexcept = default;

    explicit OptionChainSoA(std::size_t n) noexcept
        : spot((n + 3ULL) & ~std::size_t{3}),
          strike((n + 3ULL) & ~std::size_t{3}),
          rate((n + 3ULL) & ~std::size_t{3}),
          div_yield((n + 3ULL) & ~std::size_t{3}),
          vol((n + 3ULL) & ~std::size_t{3}),
          tau((n + 3ULL) & ~std::size_t{3}),
          count(n),
          padded_n((n + 3ULL) & ~std::size_t{3}) {
        // Fill padding lanes [count, padded_n) so 4-wide tail loads avoid 0-div.
        if (is_allocated()) {
            for (std::size_t i = count; i < padded_n; ++i) {
                spot[i] = 100.0;
                strike[i] = 100.0;
                rate[i] = 0.0;
                div_yield[i] = 0.0;
                vol[i] = 0.2;
                tau[i] = 1.0;
            }
        }
    }

    PRICER_FORCE_INLINE void set(std::size_t i, double s, double k, double r,
                                 double q, double v, double t) noexcept {
        spot[i] = s;
        strike[i] = k;
        rate[i] = r;
        div_yield[i] = q;
        vol[i] = v;
        tau[i] = t;
    }

    [[nodiscard]] bool is_allocated() const noexcept {
        return count == 0 || (spot && strike && rate && div_yield && vol && tau);
    }
};

struct ChainGreeksSoA {
    AlignedBuffer price;
    AlignedBuffer delta;
    AlignedBuffer gamma;
    AlignedBuffer vega;
    AlignedBuffer theta;
    AlignedBuffer rho;
    std::size_t count = 0;
    std::size_t padded_n = 0;

    ChainGreeksSoA() noexcept = default;

    explicit ChainGreeksSoA(std::size_t n) noexcept
        : price((n + 3ULL) & ~std::size_t{3}),
          delta((n + 3ULL) & ~std::size_t{3}),
          gamma((n + 3ULL) & ~std::size_t{3}),
          vega((n + 3ULL) & ~std::size_t{3}),
          theta((n + 3ULL) & ~std::size_t{3}),
          rho((n + 3ULL) & ~std::size_t{3}),
          count(n),
          padded_n((n + 3ULL) & ~std::size_t{3}) {}

    [[nodiscard]] bool is_allocated() const noexcept {
        return count == 0 || (price && delta && gamma && vega && theta && rho);
    }
};

namespace detail {

template <typename T>
struct IsDual : std::false_type {};
template <typename U>
struct IsDual<Dual<U>> : std::true_type {};

template <typename T>
struct IsJet : std::false_type {};
template <typename U, std::size_t N>
struct IsJet<Jet<U, N>> : std::true_type {};

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE auto bs_primal(const T& x) noexcept {
    if constexpr (IsDual<T>::value || IsJet<T>::value) {
        return x.val;
    } else {
        return x;
    }
}

template <typename CondT, typename T>
[[nodiscard]] PRICER_FORCE_INLINE T
bs_select_le_zero(const CondT& cond, const T& if_le_zero, const T& if_gt_zero) noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return cond <= 0.0 ? if_le_zero : if_gt_zero;
    } else if constexpr (std::is_same_v<T, Vec4d>) {
        return blend(if_gt_zero, if_le_zero, cmp_le(cond, Vec4d::zero()));
    } else if constexpr (IsDual<T>::value) {
        return T{
            bs_select_le_zero(cond, if_le_zero.val, if_gt_zero.val),
            bs_select_le_zero(cond, if_le_zero.d, if_gt_zero.d),
        };
    } else if constexpr (IsJet<T>::value) {
        T out;
        out.val = bs_select_le_zero(cond, if_le_zero.val, if_gt_zero.val);
        for (std::size_t i = 0; i < out.d.size(); ++i) {
            out.d[i] = bs_select_le_zero(cond, if_le_zero.d[i], if_gt_zero.d[i]);
        }
        return out;
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T bs_const_one() noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return 1.0;
    } else if constexpr (std::is_same_v<T, Vec4d>) {
        return Vec4d::splat(1.0);
    } else {
        return T::constant(1.0);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T bs_safe_pos(const T& x) noexcept {
    return bs_select_le_zero(bs_primal(x), bs_const_one<T>(), x);
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T bs_max_zero(const T& x) noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return x < 0.0 ? 0.0 : x;
    } else if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::max(Vec4d::zero(), x);
    } else {
        return pricer::max(x, 0.0);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T bs_sqrt(const T& x) noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return std::sqrt(x);
    } else if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::sqrt(x);
    } else {
        return pricer::sqrt(x);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T bs_exp(const T& x) noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return std::exp(x);
    } else if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::exp_avx2(x);
    } else {
        return pricer::exp(x);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T bs_log(const T& x) noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return std::log(x);
    } else if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::log_avx2(x);
    } else {
        return pricer::log(x);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T bs_norm_cdf(const T& x) noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return pricer::norm_cdf(x);
    } else if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::norm_cdf_avx2(x);
    } else {
        return pricer::norm_cdf(x);
    }
}

template <OptionType Type>
[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL
bs_price_vec4(Vec4d S, Vec4d K, Vec4d r, Vec4d q, Vec4d sigma, Vec4d tau) noexcept {
    const Vec4d zero = Vec4d::zero();
    const Vec4d one = Vec4d::splat(1.0);
    const Vec4d half = Vec4d::splat(0.5);

    const Vec4d mask_exp = cmp_le(tau, zero);
    const Vec4d mask_zvol = cmp_le(sigma, zero);
    const Vec4d safe_tau = blend(tau, one, mask_exp);
    const Vec4d safe_sigma = blend(sigma, one, mask_zvol);

    const Vec4d sqrt_t = sqrt(safe_tau);
    const Vec4d vol_sqrt_t = safe_sigma * sqrt_t;
    const Vec4d drift = fma(half * safe_sigma, safe_sigma, r - q);
    const Vec4d d1 = fma(drift, safe_tau, log_avx2(S / K)) / vol_sqrt_t;
    const Vec4d d2 = d1 - vol_sqrt_t;

    const Vec4d disc_r = exp_avx2(-(r * safe_tau));
    const Vec4d disc_q = exp_avx2(-(q * safe_tau));
    const Vec4d s_dq = S * disc_q;
    const Vec4d k_dr = K * disc_r;

    Vec4d bs_val;
    Vec4d zero_vol_val;
    Vec4d intrinsic;
    if constexpr (Type == OptionType::Call) {
        bs_val = fms(s_dq, norm_cdf_avx2(d1), k_dr * norm_cdf_avx2(d2));
        zero_vol_val = max(zero, s_dq - k_dr);
        intrinsic = max(zero, S - K);
    } else {
        bs_val = fms(k_dr, norm_cdf_avx2(-d2), s_dq * norm_cdf_avx2(-d1));
        zero_vol_val = max(zero, k_dr - s_dq);
        intrinsic = max(zero, K - S);
    }

    return blend(blend(bs_val, zero_vol_val, mask_zvol), intrinsic, mask_exp);
}

template <OptionType Type>
[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL
implied_vol_vec4(Vec4d S, Vec4d K, Vec4d r, Vec4d q, Vec4d tau, Vec4d p_mkt) noexcept {
    const Vec4d zero = Vec4d::zero();
    const Vec4d one = Vec4d::splat(1.0);
    const Vec4d half = Vec4d::splat(0.5);
    const Vec4d two = Vec4d::splat(2.0);
    const Vec4d inf_v = Vec4d::splat(std::numeric_limits<double>::infinity());
    const Vec4d nan_v = Vec4d::splat(std::numeric_limits<double>::quiet_NaN());

    const Vec4d inputs_ok = bit_and(
        bit_and(bit_and(cmp_gt(S, zero), cmp_lt(S, inf_v)),
                bit_and(cmp_gt(K, zero), cmp_lt(K, inf_v))),
        bit_and(bit_and(cmp_gt(tau, zero), cmp_lt(tau, inf_v)),
                bit_and(cmp_lt(abs(r), inf_v), cmp_lt(abs(q), inf_v))));

    const Vec4d safe_S = blend(one, S, inputs_ok);
    const Vec4d safe_K = blend(one, K, inputs_ok);
    const Vec4d safe_r = blend(zero, r, inputs_ok);
    const Vec4d safe_q = blend(zero, q, inputs_ok);
    const Vec4d safe_tau = blend(one, tau, inputs_ok);

    const Vec4d disc_r = exp_avx2(-(safe_r * safe_tau));
    const Vec4d disc_q = exp_avx2(-(safe_q * safe_tau));
    const Vec4d s_dq = safe_S * disc_q;
    const Vec4d k_dr = safe_K * disc_r;

    Vec4d intrinsic;
    Vec4d upper;
    if constexpr (Type == OptionType::Call) {
        intrinsic = max(s_dq - k_dr, zero);
        upper = s_dq;
    } else {
        intrinsic = max(k_dr - s_dq, zero);
        upper = k_dr;
    }

    const Vec4d valid_mask = bit_and(
        bit_and(inputs_ok, cmp_gt(p_mkt, intrinsic)),
        cmp_lt(p_mkt, upper));
    if (movemask(valid_mask) == 0) {
        return nan_v;
    }

    // OTM reduction in discounted space, solving in s = sigma * sqrt(tau):
    //   P_otm(s) = A * Phi(-u/s + s/2) - B * Phi(-u/s - s/2),  u = |ln(F/K)|.
    const Vec4d A = min(s_dq, k_dr);
    const Vec4d B = max(s_dq, k_dr);
    const Vec4d u = abs(fma(safe_r - safe_q, safe_tau, log_avx2(safe_S / safe_K)));
    const Vec4d p_target = blend(A * Vec4d::splat(0.1), p_mkt - intrinsic, valid_mask);

    const Vec4d sqrt_tau = sqrt(safe_tau);

    // Manaster-Koehler initial guess: s_0 = sqrt(2*u), clamped to [1e-4, 5.0] * sqrt(tau).
    const Vec4d s_mk = sqrt(u + u);
    Vec4d s = min(max(s_mk, Vec4d::splat(1e-4) * sqrt_tau), Vec4d::splat(5.0) * sqrt_tau);

    Vec4d lo_s = Vec4d::splat(1e-6) * sqrt_tau;
    Vec4d hi_s = Vec4d::splat(10.0) * sqrt_tau;
    const Vec4d s_hard_min = Vec4d::splat(1e-12) * sqrt_tau;
    const Vec4d s_hard_max = Vec4d::splat(1e4) * sqrt_tau;

    Vec4d lo_confirmed = zero;
    Vec4d hi_confirmed = zero;
    Vec4d active = valid_mask;

    const Vec4d vega_floor = Vec4d::splat(1e-30);
    const Vec4d quarter = Vec4d::splat(0.25);
    const Vec4d tol_rel = Vec4d::splat(1e-12);

    auto halley_step = [&]() noexcept {
        const Vec4d inv_s = one / s;
        const Vec4d half_s = half * s;
        const Vec4d neg_u_over_s = -(u * inv_s);
        const Vec4d z1 = neg_u_over_s + half_s;
        const Vec4d z2 = neg_u_over_s - half_s;

        Vec4d cdf_z1, phi_z1;
        norm_cdf_and_pdf_avx2(z1, cdf_z1, phi_z1);
        const Vec4d cdf_z2 = norm_cdf_avx2(z2);

        const Vec4d price = fms(A, cdf_z1, B * cdf_z2);
        const Vec4d f = price - p_target;
        const Vec4d vega_s = A * phi_z1;

        const Vec4d f_neg = cmp_lt(f, zero);
        const Vec4d f_pos = cmp_gt(f, zero);
        const Vec4d f_zero = cmp_eq(f, zero);

        const Vec4d upd_lo = bit_and(active, f_neg);
        const Vec4d upd_hi = bit_and(active, f_pos);
        lo_s = blend(lo_s, max(lo_s, s), upd_lo);
        hi_s = blend(hi_s, min(hi_s, s), upd_hi);
        lo_confirmed = bit_or(lo_confirmed, upd_lo);
        hi_confirmed = bit_or(hi_confirmed, upd_hi);

        // Safeguarded Halley: fall back to Newton when denom <= 0.25 * vega_s.
        const Vec4d vega_ok = cmp_ge(vega_s, vega_floor);
        const Vec4d halley_denom = fnma(half * f, (z1 * z2) * inv_s, vega_s);
        const Vec4d use_halley = cmp_gt(halley_denom, quarter * vega_s);
        const Vec4d eff_denom = blend(vega_s, halley_denom, use_halley);
        const Vec4d safe_denom = blend(one, eff_denom, vega_ok);
        const Vec4d s_cand = s - (f / safe_denom);

        // Expand unconfirmed bounds if the root lies outside the initial [1e-6, 10.0].
        const Vec4d unconf_hi = bit_andnot(hi_confirmed, upd_lo);
        const Vec4d low_vega = bit_andnot(vega_ok, active);
        const Vec4d want_hi_expand = bit_and(
            unconf_hi,
            bit_or(low_vega,
                   bit_or(cmp_ge(s_cand, hi_s), cmp_ge(s, half * hi_s))));
        hi_s = blend(hi_s, min(hi_s * two, s_hard_max), want_hi_expand);

        const Vec4d unconf_lo = bit_andnot(lo_confirmed, upd_hi);
        const Vec4d want_lo_expand = bit_and(
            unconf_lo,
            cmp_le(s, two * lo_s));
        lo_s = blend(lo_s, max(lo_s * half, s_hard_min), want_lo_expand);

        const Vec4d in_bracket = bit_and(cmp_gt(s_cand, lo_s), cmp_lt(s_cand, hi_s));
        const Vec4d accept_halley = bit_and(vega_ok, in_bracket);

        const Vec4d s_mid = half * (lo_s + hi_s);
        const Vec4d s_next = blend(s_mid, s_cand, accept_halley);

        // |delta_sigma| <= 1e-12 * max(1, sigma); bisection must also shrink bracket to tol_s.
        const Vec4d tol_s = tol_rel * max(sqrt_tau, s_next);
        const Vec4d halley_conv = bit_and(accept_halley, cmp_le(abs(s_next - s), tol_s));
        const Vec4d bracket_conv = bit_and(
            bit_and(lo_confirmed, hi_confirmed),
            cmp_le(hi_s - lo_s, tol_s));
        const Vec4d converged = bit_or(f_zero, bit_or(halley_conv, bracket_conv));

        s = blend(s, s_next, active);
        active = bit_andnot(converged, active);
    };

    for (int iter = 0; iter < 12; ++iter) {
        halley_step();
        if (movemask(active) == 0) {
            break;
        }
    }

    if (movemask(active) != 0) {
        for (int iter = 0; iter < 64; ++iter) {
            halley_step();
            if (movemask(active) == 0) {
                break;
            }
        }
    }

    return blend(nan_v, s / sqrt_tau, valid_mask);
}

} // namespace detail

template <OptionType Type, typename T>
[[nodiscard]] PRICER_FORCE_INLINE T
black_scholes_price(T S, T K, T r, T q, T sigma, T tau) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return detail::bs_price_vec4<Type>(S, K, r, q, sigma, tau);
    } else {
        const T safe_tau = detail::bs_safe_pos(tau);
        const T safe_sigma = detail::bs_safe_pos(sigma);

        const T sqrt_t = detail::bs_sqrt(safe_tau);
        const T vol_sqrt_t = safe_sigma * sqrt_t;

        const T d1 = (detail::bs_log(S / K) + (r - q + 0.5 * safe_sigma * safe_sigma) * safe_tau)
                   / vol_sqrt_t;
        const T d2 = d1 - vol_sqrt_t;

        const T disc_r = detail::bs_exp(-r * safe_tau);
        const T disc_q = detail::bs_exp(-q * safe_tau);
        const T s_dq = S * disc_q;
        const T k_dr = K * disc_r;

        T bs_val;
        T zero_vol_val;
        T intrinsic;
        if constexpr (Type == OptionType::Call) {
            bs_val = s_dq * detail::bs_norm_cdf(d1) - k_dr * detail::bs_norm_cdf(d2);
            zero_vol_val = detail::bs_max_zero(s_dq - k_dr);
            intrinsic = detail::bs_max_zero(S - K);
        } else {
            bs_val = k_dr * detail::bs_norm_cdf(-d2) - s_dq * detail::bs_norm_cdf(-d1);
            zero_vol_val = detail::bs_max_zero(k_dr - s_dq);
            intrinsic = detail::bs_max_zero(K - S);
        }

        const T price_pos_tau = detail::bs_select_le_zero(
            detail::bs_primal(sigma), zero_vol_val, bs_val);
        return detail::bs_select_le_zero(
            detail::bs_primal(tau), intrinsic, price_pos_tau);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T
black_scholes_price(OptionType type, T S, T K, T r, T q, T sigma, T tau) noexcept {
    if (type == OptionType::Call) {
        return black_scholes_price<OptionType::Call>(S, K, r, q, sigma, tau);
    }
    return black_scholes_price<OptionType::Put>(S, K, r, q, sigma, tau);
}

// Theta = -dV / dtau.
template <OptionType Type>
[[nodiscard]] inline GreeksResult
black_scholes_greeks(double S, double K, double r, double q, double sigma, double tau) noexcept {
    constexpr double kInvSqrt2Pi = 0.398942280401432677939946059934381868;

    if (tau <= 0.0) {
        if constexpr (Type == OptionType::Call) {
            return GreeksResult{
                .price = S > K ? (S - K) : 0.0,
                .delta = S > K ? 1.0 : 0.0,
                .gamma = 0.0,
                .vega = 0.0,
                .theta = 0.0,
                .rho = 0.0,
            };
        } else {
            return GreeksResult{
                .price = K > S ? (K - S) : 0.0,
                .delta = K > S ? -1.0 : 0.0,
                .gamma = 0.0,
                .vega = 0.0,
                .theta = 0.0,
                .rho = 0.0,
            };
        }
    }

    const double disc_r = std::exp(-r * tau);
    const double disc_q = std::exp(-q * tau);
    const double s_dq = S * disc_q;
    const double k_dr = K * disc_r;

    if (sigma <= 0.0) {
        if constexpr (Type == OptionType::Call) {
            const bool itm = s_dq > k_dr;
            return GreeksResult{
                .price = itm ? (s_dq - k_dr) : 0.0,
                .delta = itm ? disc_q : 0.0,
                .gamma = 0.0,
                .vega = 0.0,
                .theta = itm ? (q * s_dq - r * k_dr) : 0.0,
                .rho = itm ? (k_dr * tau) : 0.0,
            };
        } else {
            const bool itm = k_dr > s_dq;
            return GreeksResult{
                .price = itm ? (k_dr - s_dq) : 0.0,
                .delta = itm ? -disc_q : 0.0,
                .gamma = 0.0,
                .vega = 0.0,
                .theta = itm ? (r * k_dr - q * s_dq) : 0.0,
                .rho = itm ? (-k_dr * tau) : 0.0,
            };
        }
    }

    const double sqrt_t = std::sqrt(tau);
    const double vol_sqrt_t = sigma * sqrt_t;
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / vol_sqrt_t;
    const double d2 = d1 - vol_sqrt_t;
    const double phi_d1 = kInvSqrt2Pi * std::exp(-0.5 * d1 * d1);

    const double gamma = (disc_q * phi_d1) / (S * vol_sqrt_t);
    const double vega = s_dq * phi_d1 * sqrt_t;
    const double theta_diff = -0.5 * (s_dq * phi_d1 * sigma) / sqrt_t;

    if constexpr (Type == OptionType::Call) {
        const double nd1 = norm_cdf(d1);
        const double nd2 = norm_cdf(d2);
        return GreeksResult{
            .price = s_dq * nd1 - k_dr * nd2,
            .delta = disc_q * nd1,
            .gamma = gamma,
            .vega = vega,
            .theta = theta_diff - r * k_dr * nd2 + q * s_dq * nd1,
            .rho = k_dr * tau * nd2,
        };
    } else {
        const double n_neg_d1 = norm_cdf(-d1);
        const double n_neg_d2 = norm_cdf(-d2);
        return GreeksResult{
            .price = k_dr * n_neg_d2 - s_dq * n_neg_d1,
            .delta = -disc_q * n_neg_d1,
            .gamma = gamma,
            .vega = vega,
            .theta = theta_diff + r * k_dr * n_neg_d2 - q * s_dq * n_neg_d1,
            .rho = -k_dr * tau * n_neg_d2,
        };
    }
}

[[nodiscard]] inline GreeksResult
black_scholes_greeks(OptionType type, double S, double K, double r,
                     double q, double sigma, double tau) noexcept {
    if (type == OptionType::Call) {
        return black_scholes_greeks<OptionType::Call>(S, K, r, q, sigma, tau);
    }
    return black_scholes_greeks<OptionType::Put>(S, K, r, q, sigma, tau);
}

template <OptionType Type>
inline void price_chain_avx2(const OptionChainSoA& in,
                             double* PRICER_RESTRICT out_price) noexcept {
    const double* PRICER_RESTRICT spot = in.spot.data();
    const double* PRICER_RESTRICT strike = in.strike.data();
    const double* PRICER_RESTRICT rate = in.rate.data();
    const double* PRICER_RESTRICT div_yield = in.div_yield.data();
    const double* PRICER_RESTRICT vol = in.vol.data();
    const double* PRICER_RESTRICT tau = in.tau.data();

    const std::size_t vec_end = in.count & ~std::size_t{3};
    for (std::size_t i = 0; i < vec_end; i += 4) {
        const Vec4d p = detail::bs_price_vec4<Type>(
            Vec4d::load_aligned(spot + i),
            Vec4d::load_aligned(strike + i),
            Vec4d::load_aligned(rate + i),
            Vec4d::load_aligned(div_yield + i),
            Vec4d::load_aligned(vol + i),
            Vec4d::load_aligned(tau + i));
        p.store_unaligned(out_price + i);
    }

    if (vec_end < in.count) {
        alignas(32) double tmp[4];
        const Vec4d p = detail::bs_price_vec4<Type>(
            Vec4d::load_aligned(spot + vec_end),
            Vec4d::load_aligned(strike + vec_end),
            Vec4d::load_aligned(rate + vec_end),
            Vec4d::load_aligned(div_yield + vec_end),
            Vec4d::load_aligned(vol + vec_end),
            Vec4d::load_aligned(tau + vec_end));
        p.store_aligned(tmp);
        for (std::size_t k = 0; k < (in.count - vec_end); ++k) {
            out_price[vec_end + k] = tmp[k];
        }
    }
}

template <OptionType Type>
inline void price_chain_avx2(const OptionChainSoA& in,
                             AlignedBuffer& out_price) noexcept {
    const double* PRICER_RESTRICT spot = in.spot.data();
    const double* PRICER_RESTRICT strike = in.strike.data();
    const double* PRICER_RESTRICT rate = in.rate.data();
    const double* PRICER_RESTRICT div_yield = in.div_yield.data();
    const double* PRICER_RESTRICT vol = in.vol.data();
    const double* PRICER_RESTRICT tau = in.tau.data();
    double* PRICER_RESTRICT out = out_price.data();

    for (std::size_t i = 0; i < in.padded_n; i += 4) {
        const Vec4d p = detail::bs_price_vec4<Type>(
            Vec4d::load_aligned(spot + i),
            Vec4d::load_aligned(strike + i),
            Vec4d::load_aligned(rate + i),
            Vec4d::load_aligned(div_yield + i),
            Vec4d::load_aligned(vol + i),
            Vec4d::load_aligned(tau + i));
        p.store_aligned(out + i);
    }
}

template <OptionType Type>
inline void price_and_greeks_chain_avx2(const OptionChainSoA& in,
                                        ChainGreeksSoA& out) noexcept {
    const double* PRICER_RESTRICT spot = in.spot.data();
    const double* PRICER_RESTRICT strike = in.strike.data();
    const double* PRICER_RESTRICT rate = in.rate.data();
    const double* PRICER_RESTRICT div_yield = in.div_yield.data();
    const double* PRICER_RESTRICT vol = in.vol.data();
    const double* PRICER_RESTRICT tau_arr = in.tau.data();

    double* PRICER_RESTRICT out_price = out.price.data();
    double* PRICER_RESTRICT out_delta = out.delta.data();
    double* PRICER_RESTRICT out_gamma = out.gamma.data();
    double* PRICER_RESTRICT out_vega = out.vega.data();
    double* PRICER_RESTRICT out_theta = out.theta.data();
    double* PRICER_RESTRICT out_rho = out.rho.data();

    const Vec4d zero = Vec4d::zero();
    const Vec4d one = Vec4d::splat(1.0);
    const Vec4d neg_half = Vec4d::splat(-0.5);
    const Vec4d half = Vec4d::splat(0.5);

    for (std::size_t i = 0; i < in.padded_n; i += 4) {
        const Vec4d S = Vec4d::load_aligned(spot + i);
        const Vec4d K = Vec4d::load_aligned(strike + i);
        const Vec4d r = Vec4d::load_aligned(rate + i);
        const Vec4d q = Vec4d::load_aligned(div_yield + i);
        const Vec4d sigma = Vec4d::load_aligned(vol + i);
        const Vec4d tau = Vec4d::load_aligned(tau_arr + i);

        const Vec4d mask_exp = cmp_le(tau, zero);
        const Vec4d mask_zvol = cmp_le(sigma, zero);
        const Vec4d mask_deg = bit_or(mask_exp, mask_zvol);

        const Vec4d safe_tau = blend(tau, one, mask_exp);
        const Vec4d safe_sigma = blend(sigma, one, mask_zvol);

        const Vec4d sqrt_t = sqrt(safe_tau);
        const Vec4d vol_sqrt_t = safe_sigma * sqrt_t;
        const Vec4d drift = fma(half * safe_sigma, safe_sigma, r - q);
        const Vec4d d1 = fma(drift, safe_tau, log_avx2(S / K)) / vol_sqrt_t;
        const Vec4d d2 = d1 - vol_sqrt_t;

        const Vec4d disc_r = exp_avx2(-(r * safe_tau));
        const Vec4d disc_q = exp_avx2(-(q * safe_tau));
        const Vec4d s_dq = S * disc_q;
        const Vec4d k_dr = K * disc_r;

        Vec4d n1, phi_d1;
        Vec4d n2;
        if constexpr (Type == OptionType::Call) {
            norm_cdf_and_pdf_avx2(d1, n1, phi_d1);
            n2 = norm_cdf_avx2(d2);
        } else {
            // phi(-d1) == phi(d1)
            norm_cdf_and_pdf_avx2(-d1, n1, phi_d1);
            n2 = norm_cdf_avx2(-d2);
        }

        const Vec4d dq_phi = disc_q * phi_d1;
        const Vec4d s_dq_phi = S * dq_phi;
        const Vec4d gamma = bit_andnot(mask_deg, dq_phi / (S * vol_sqrt_t));
        const Vec4d vega = bit_andnot(mask_deg, s_dq_phi * sqrt_t);
        gamma.store_aligned(out_gamma + i);
        vega.store_aligned(out_vega + i);

        const Vec4d theta_diff = (neg_half * s_dq_phi * safe_sigma) / sqrt_t;
        const Vec4d s_dq_n1 = s_dq * n1;
        const Vec4d k_dr_n2 = k_dr * n2;

        if constexpr (Type == OptionType::Call) {
            const Vec4d itm_zv = cmp_gt(s_dq, k_dr);
            const Vec4d itm_exp = cmp_gt(S, K);

            const Vec4d p_bs = s_dq_n1 - k_dr_n2;
            const Vec4d p_zv = max(zero, s_dq - k_dr);
            const Vec4d p_exp = max(zero, S - K);
            blend(blend(p_bs, p_zv, mask_zvol), p_exp, mask_exp).store_aligned(out_price + i);

            const Vec4d d_bs = disc_q * n1;
            const Vec4d d_zv = bit_and(itm_zv, disc_q);
            const Vec4d d_exp = bit_and(itm_exp, one);
            blend(blend(d_bs, d_zv, mask_zvol), d_exp, mask_exp).store_aligned(out_delta + i);

            const Vec4d th_bs = fma(q, s_dq_n1, fnma(r, k_dr_n2, theta_diff));
            const Vec4d th_zv = bit_and(itm_zv, fms(q, s_dq, r * k_dr));
            bit_andnot(mask_exp, blend(th_bs, th_zv, mask_zvol)).store_aligned(out_theta + i);

            const Vec4d rh_bs = safe_tau * k_dr_n2;
            const Vec4d rh_zv = bit_and(itm_zv, safe_tau * k_dr);
            bit_andnot(mask_exp, blend(rh_bs, rh_zv, mask_zvol)).store_aligned(out_rho + i);
        } else {
            const Vec4d itm_zv = cmp_gt(k_dr, s_dq);
            const Vec4d itm_exp = cmp_gt(K, S);

            const Vec4d p_bs = k_dr_n2 - s_dq_n1;
            const Vec4d p_zv = max(zero, k_dr - s_dq);
            const Vec4d p_exp = max(zero, K - S);
            blend(blend(p_bs, p_zv, mask_zvol), p_exp, mask_exp).store_aligned(out_price + i);

            const Vec4d d_bs = -(disc_q * n1);
            const Vec4d d_zv = bit_and(itm_zv, -disc_q);
            const Vec4d d_exp = bit_and(itm_exp, -one);
            blend(blend(d_bs, d_zv, mask_zvol), d_exp, mask_exp).store_aligned(out_delta + i);

            const Vec4d th_bs = fnma(q, s_dq_n1, fma(r, k_dr_n2, theta_diff));
            const Vec4d th_zv = bit_and(itm_zv, fms(r, k_dr, q * s_dq));
            bit_andnot(mask_exp, blend(th_bs, th_zv, mask_zvol)).store_aligned(out_theta + i);

            const Vec4d rh_bs = -(safe_tau * k_dr_n2);
            const Vec4d rh_zv = bit_and(itm_zv, -(safe_tau * k_dr));
            bit_andnot(mask_exp, blend(rh_bs, rh_zv, mask_zvol)).store_aligned(out_rho + i);
        }
    }
}

template <OptionType Type>
inline void implied_vol_chain_avx2(const OptionChainSoA& in,
                                   const double* PRICER_RESTRICT market_prices,
                                   double* PRICER_RESTRICT out_iv) noexcept {
    const double* PRICER_RESTRICT spot = in.spot.data();
    const double* PRICER_RESTRICT strike = in.strike.data();
    const double* PRICER_RESTRICT rate = in.rate.data();
    const double* PRICER_RESTRICT div_yield = in.div_yield.data();
    const double* PRICER_RESTRICT tau = in.tau.data();

    const std::size_t vec_end = in.count & ~std::size_t{3};
    for (std::size_t i = 0; i < vec_end; i += 4) {
        const Vec4d iv = detail::implied_vol_vec4<Type>(
            Vec4d::load_aligned(spot + i),
            Vec4d::load_aligned(strike + i),
            Vec4d::load_aligned(rate + i),
            Vec4d::load_aligned(div_yield + i),
            Vec4d::load_aligned(tau + i),
            Vec4d::load_unaligned(market_prices + i));
        iv.store_unaligned(out_iv + i);
    }

    if (vec_end < in.count) {
        const std::size_t rem = in.count - vec_end;
        alignas(32) double mp_tmp[4] = {0.0, 0.0, 0.0, 0.0};
        for (std::size_t k = 0; k < rem; ++k) {
            mp_tmp[k] = market_prices[vec_end + k];
        }
        alignas(32) double iv_tmp[4];
        const Vec4d iv = detail::implied_vol_vec4<Type>(
            Vec4d::load_aligned(spot + vec_end),
            Vec4d::load_aligned(strike + vec_end),
            Vec4d::load_aligned(rate + vec_end),
            Vec4d::load_aligned(div_yield + vec_end),
            Vec4d::load_aligned(tau + vec_end),
            Vec4d::load_aligned(mp_tmp));
        iv.store_aligned(iv_tmp);
        for (std::size_t k = 0; k < rem; ++k) {
            out_iv[vec_end + k] = iv_tmp[k];
        }
    }
}

template <OptionType Type>
[[nodiscard]] inline double
black_scholes_implied_vol(double S, double K, double r, double q,
                          double tau, double market_price) noexcept {
    alignas(32) double out[4];
    const Vec4d iv = detail::implied_vol_vec4<Type>(
        Vec4d::set(S, 1.0, 1.0, 1.0),
        Vec4d::set(K, 1.0, 1.0, 1.0),
        Vec4d::set(r, 0.0, 0.0, 0.0),
        Vec4d::set(q, 0.0, 0.0, 0.0),
        Vec4d::set(tau, 1.0, 1.0, 1.0),
        Vec4d::set(market_price, 0.0, 0.0, 0.0));
    iv.store_aligned(out);
    return out[0];
}

[[nodiscard]] inline double
black_scholes_implied_vol(OptionType type, double S, double K, double r,
                          double q, double tau, double market_price) noexcept {
    if (type == OptionType::Call) {
        return black_scholes_implied_vol<OptionType::Call>(S, K, r, q, tau, market_price);
    }
    return black_scholes_implied_vol<OptionType::Put>(S, K, r, q, tau, market_price);
}

} // namespace pricer

#endif // PRICER_ENGINES_BLACK_SCHOLES_HPP
