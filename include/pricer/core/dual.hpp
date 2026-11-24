#ifndef PRICER_CORE_DUAL_HPP
#define PRICER_CORE_DUAL_HPP

#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace pricer {

namespace detail {

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T splat_like(double x) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return Vec4d::splat(x);
    } else {
        return x;
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T zero_like() noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return Vec4d::zero();
    } else {
        return 0.0;
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T one_like() noexcept {
    return splat_like<T>(1.0);
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T sqrt_val(T x) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::sqrt(x);
    } else {
        return std::sqrt(x);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T exp_val(T x) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::exp_avx2(x);
    } else {
        return std::exp(x);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T log_val(T x) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::log_avx2(x);
    } else {
        return std::log(x);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T norm_cdf_val(T x) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::norm_cdf_avx2(x);
    } else {
        return pricer::norm_cdf(x);
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T norm_pdf_val(T x) noexcept {
    constexpr double kInvSqrt2Pi = 0.398942280401432677939946059934381868;
    return splat_like<T>(kInvSqrt2Pi) * exp_val(splat_like<T>(-0.5) * x * x);
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T max_val(T a, T b) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::max(a, b);
    } else {
        return a > b ? a : b;
    }
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T min_val(T a, T b) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return pricer::min(a, b);
    } else {
        return a < b ? a : b;
    }
}

// (a > k) ? deriv : 0
template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T mask_gt(T a, double k, T deriv) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return bit_and(cmp_gt(a, Vec4d::splat(k)), deriv);
    } else {
        return a > k ? deriv : 0.0;
    }
}

// (a > b) ? if_gt : if_le
template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE T select_gt(T a, T b, T if_gt, T if_le) noexcept {
    if constexpr (std::is_same_v<T, Vec4d>) {
        return blend(if_le, if_gt, cmp_gt(a, b));
    } else {
        return a > b ? if_gt : if_le;
    }
}

} // namespace detail

template <typename T>
struct Dual {
    T val{};
    T d{};

    PRICER_FORCE_INLINE Dual() noexcept = default;
    PRICER_FORCE_INLINE Dual(T v, T deriv) noexcept : val(v), d(deriv) {}
    PRICER_FORCE_INLINE explicit Dual(T v) noexcept : val(v), d(detail::zero_like<T>()) {}
    PRICER_FORCE_INLINE explicit Dual(double v) noexcept
        requires (!std::is_same_v<T, double>)
        : val(detail::splat_like<T>(v)), d(detail::zero_like<T>()) {}

    [[nodiscard]] PRICER_FORCE_INLINE static Dual constant(T v) noexcept {
        return Dual{v, detail::zero_like<T>()};
    }

    [[nodiscard]] PRICER_FORCE_INLINE static Dual constant(double v) noexcept
        requires (!std::is_same_v<T, double>) {
        return Dual{detail::splat_like<T>(v), detail::zero_like<T>()};
    }

    [[nodiscard]] PRICER_FORCE_INLINE static Dual variable(T v) noexcept {
        return Dual{v, detail::one_like<T>()};
    }

    PRICER_FORCE_INLINE Dual& operator+=(const Dual& rhs) noexcept {
        val = val + rhs.val;
        d = d + rhs.d;
        return *this;
    }

    PRICER_FORCE_INLINE Dual& operator-=(const Dual& rhs) noexcept {
        val = val - rhs.val;
        d = d - rhs.d;
        return *this;
    }

    PRICER_FORCE_INLINE Dual& operator*=(const Dual& rhs) noexcept {
        d = d * rhs.val + val * rhs.d;
        val = val * rhs.val;
        return *this;
    }

    PRICER_FORCE_INLINE Dual& operator/=(const Dual& rhs) noexcept {
        const T inv = detail::one_like<T>() / rhs.val;
        const T q = val * inv;
        d = (d - q * rhs.d) * inv;
        val = q;
        return *this;
    }

    PRICER_FORCE_INLINE Dual& operator+=(double s) noexcept {
        val = val + s;
        return *this;
    }

    PRICER_FORCE_INLINE Dual& operator-=(double s) noexcept {
        val = val - s;
        return *this;
    }

    PRICER_FORCE_INLINE Dual& operator*=(double s) noexcept {
        val = val * s;
        d = d * s;
        return *this;
    }

    PRICER_FORCE_INLINE Dual& operator/=(double s) noexcept {
        const double inv = 1.0 / s;
        val = val * inv;
        d = d * inv;
        return *this;
    }
};

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator+(const Dual<T>& x) noexcept {
    return x;
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator-(const Dual<T>& x) noexcept {
    return Dual<T>{-x.val, -x.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator+(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>{a.val + b.val, a.d + b.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator-(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>{a.val - b.val, a.d - b.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator*(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>{a.val * b.val, a.d * b.val + a.val * b.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator/(const Dual<T>& a, const Dual<T>& b) noexcept {
    const T inv = detail::one_like<T>() / b.val;
    const T q = a.val * inv;
    return Dual<T>{q, (a.d - q * b.d) * inv};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator+(const Dual<T>& a, double s) noexcept {
    return Dual<T>{a.val + s, a.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator+(double s, const Dual<T>& b) noexcept {
    return Dual<T>{s + b.val, b.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator-(const Dual<T>& a, double s) noexcept {
    return Dual<T>{a.val - s, a.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator-(double s, const Dual<T>& b) noexcept {
    return Dual<T>{s - b.val, -b.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator*(const Dual<T>& a, double s) noexcept {
    return Dual<T>{a.val * s, a.d * s};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator*(double s, const Dual<T>& b) noexcept {
    return Dual<T>{s * b.val, s * b.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator/(const Dual<T>& a, double s) noexcept {
    const double inv = 1.0 / s;
    return Dual<T>{a.val * inv, a.d * inv};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> operator/(double s, const Dual<T>& b) noexcept {
    const T inv = detail::one_like<T>() / b.val;
    const T q = s * inv;
    return Dual<T>{q, -(q * inv) * b.d};
}

[[nodiscard]] PRICER_FORCE_INLINE Dual<Vec4d> operator+(const Dual<Vec4d>& a, Vec4d b) noexcept {
    return Dual<Vec4d>{a.val + b, a.d};
}

[[nodiscard]] PRICER_FORCE_INLINE Dual<Vec4d> operator+(Vec4d a, const Dual<Vec4d>& b) noexcept {
    return Dual<Vec4d>{a + b.val, b.d};
}

[[nodiscard]] PRICER_FORCE_INLINE Dual<Vec4d> operator-(const Dual<Vec4d>& a, Vec4d b) noexcept {
    return Dual<Vec4d>{a.val - b, a.d};
}

[[nodiscard]] PRICER_FORCE_INLINE Dual<Vec4d> operator-(Vec4d a, const Dual<Vec4d>& b) noexcept {
    return Dual<Vec4d>{a - b.val, -b.d};
}

[[nodiscard]] PRICER_FORCE_INLINE Dual<Vec4d> operator*(const Dual<Vec4d>& a, Vec4d b) noexcept {
    return Dual<Vec4d>{a.val * b, a.d * b};
}

[[nodiscard]] PRICER_FORCE_INLINE Dual<Vec4d> operator*(Vec4d a, const Dual<Vec4d>& b) noexcept {
    return Dual<Vec4d>{a * b.val, a * b.d};
}

[[nodiscard]] PRICER_FORCE_INLINE Dual<Vec4d> operator/(const Dual<Vec4d>& a, Vec4d b) noexcept {
    const Vec4d inv = Vec4d::splat(1.0) / b;
    return Dual<Vec4d>{a.val * inv, a.d * inv};
}

[[nodiscard]] PRICER_FORCE_INLINE Dual<Vec4d> operator/(Vec4d a, const Dual<Vec4d>& b) noexcept {
    const Vec4d inv = Vec4d::splat(1.0) / b.val;
    const Vec4d q = a * inv;
    return Dual<Vec4d>{q, -(q * inv) * b.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> sqrt(const Dual<T>& x) noexcept {
    const T s = detail::sqrt_val(x.val);
    const T inv_2s = detail::splat_like<T>(0.5) / s;
    return Dual<T>{s, x.d * inv_2s};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> exp(const Dual<T>& x) noexcept {
    const T e = detail::exp_val(x.val);
    return Dual<T>{e, e * x.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> log(const Dual<T>& x) noexcept {
    return Dual<T>{detail::log_val(x.val), x.d / x.val};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> norm_cdf(const Dual<T>& x) noexcept {
    const T c = detail::norm_cdf_val(x.val);
    const T p = detail::norm_pdf_val(x.val);
    return Dual<T>{c, p * x.d};
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> max(const Dual<T>& x, double k) noexcept {
    return Dual<T>{
        detail::max_val(x.val, detail::splat_like<T>(k)),
        detail::mask_gt(x.val, k, x.d),
    };
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> max(double k, const Dual<T>& x) noexcept {
    return max(x, k);
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> max(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>{
        detail::max_val(a.val, b.val),
        detail::select_gt(a.val, b.val, a.d, b.d),
    };
}

template <typename T>
[[nodiscard]] PRICER_FORCE_INLINE Dual<T> min(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>{
        detail::min_val(a.val, b.val),
        detail::select_gt(b.val, a.val, a.d, b.d),
    };
}

template <typename T, std::size_t N>
struct Jet {
    T val{};
    std::array<T, N> d{};

    PRICER_FORCE_INLINE Jet() noexcept = default;

    PRICER_FORCE_INLINE Jet(T v, const std::array<T, N>& derivs) noexcept
        : val(v), d(derivs) {}

    PRICER_FORCE_INLINE explicit Jet(T v) noexcept : val(v) {
        for (std::size_t i = 0; i < N; ++i) {
            d[i] = detail::zero_like<T>();
        }
    }

    PRICER_FORCE_INLINE explicit Jet(double v) noexcept
        requires (!std::is_same_v<T, double>)
        : Jet(detail::splat_like<T>(v)) {}

    [[nodiscard]] PRICER_FORCE_INLINE static Jet constant(T v) noexcept {
        return Jet(v);
    }

    [[nodiscard]] PRICER_FORCE_INLINE static Jet constant(double v) noexcept
        requires (!std::is_same_v<T, double>) {
        return Jet(detail::splat_like<T>(v));
    }

    template <std::size_t Idx>
    [[nodiscard]] PRICER_FORCE_INLINE static Jet variable(T v) noexcept {
        static_assert(Idx < N, "Jet variable index out of bounds");
        Jet j(v);
        j.d[Idx] = detail::one_like<T>();
        return j;
    }

    PRICER_FORCE_INLINE Jet& operator+=(const Jet& rhs) noexcept {
        val = val + rhs.val;
        for (std::size_t i = 0; i < N; ++i) {
            d[i] = d[i] + rhs.d[i];
        }
        return *this;
    }

    PRICER_FORCE_INLINE Jet& operator-=(const Jet& rhs) noexcept {
        val = val - rhs.val;
        for (std::size_t i = 0; i < N; ++i) {
            d[i] = d[i] - rhs.d[i];
        }
        return *this;
    }

    PRICER_FORCE_INLINE Jet& operator*=(const Jet& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            d[i] = d[i] * rhs.val + val * rhs.d[i];
        }
        val = val * rhs.val;
        return *this;
    }

    PRICER_FORCE_INLINE Jet& operator/=(const Jet& rhs) noexcept {
        const T inv = detail::one_like<T>() / rhs.val;
        const T q = val * inv;
        for (std::size_t i = 0; i < N; ++i) {
            d[i] = (d[i] - q * rhs.d[i]) * inv;
        }
        val = q;
        return *this;
    }

    PRICER_FORCE_INLINE Jet& operator+=(double s) noexcept {
        val = val + s;
        return *this;
    }

    PRICER_FORCE_INLINE Jet& operator-=(double s) noexcept {
        val = val - s;
        return *this;
    }

    PRICER_FORCE_INLINE Jet& operator*=(double s) noexcept {
        val = val * s;
        for (std::size_t i = 0; i < N; ++i) {
            d[i] = d[i] * s;
        }
        return *this;
    }

    PRICER_FORCE_INLINE Jet& operator/=(double s) noexcept {
        const double inv = 1.0 / s;
        val = val * inv;
        for (std::size_t i = 0; i < N; ++i) {
            d[i] = d[i] * inv;
        }
        return *this;
    }
};

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator+(const Jet<T, N>& x) noexcept {
    return x;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator-(const Jet<T, N>& x) noexcept {
    Jet<T, N> r;
    r.val = -x.val;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = -x.d[i];
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator+(const Jet<T, N>& a, const Jet<T, N>& b) noexcept {
    Jet<T, N> r;
    r.val = a.val + b.val;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = a.d[i] + b.d[i];
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator-(const Jet<T, N>& a, const Jet<T, N>& b) noexcept {
    Jet<T, N> r;
    r.val = a.val - b.val;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = a.d[i] - b.d[i];
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator*(const Jet<T, N>& a, const Jet<T, N>& b) noexcept {
    Jet<T, N> r;
    r.val = a.val * b.val;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = a.d[i] * b.val + a.val * b.d[i];
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator/(const Jet<T, N>& a, const Jet<T, N>& b) noexcept {
    const T inv = detail::one_like<T>() / b.val;
    const T q = a.val * inv;
    Jet<T, N> r;
    r.val = q;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = (a.d[i] - q * b.d[i]) * inv;
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator+(const Jet<T, N>& a, double s) noexcept {
    return Jet<T, N>{a.val + s, a.d};
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator+(double s, const Jet<T, N>& b) noexcept {
    return Jet<T, N>{s + b.val, b.d};
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator-(const Jet<T, N>& a, double s) noexcept {
    return Jet<T, N>{a.val - s, a.d};
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator-(double s, const Jet<T, N>& b) noexcept {
    Jet<T, N> r;
    r.val = s - b.val;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = -b.d[i];
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator*(const Jet<T, N>& a, double s) noexcept {
    Jet<T, N> r;
    r.val = a.val * s;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = a.d[i] * s;
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator*(double s, const Jet<T, N>& b) noexcept {
    return b * s;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator/(const Jet<T, N>& a, double s) noexcept {
    const double inv = 1.0 / s;
    return a * inv;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> operator/(double s, const Jet<T, N>& b) noexcept {
    const T inv = detail::one_like<T>() / b.val;
    const T q = s * inv;
    const T scale = -(q * inv);
    Jet<T, N> r;
    r.val = q;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = scale * b.d[i];
    }
    return r;
}

template <std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, N> operator+(const Jet<Vec4d, N>& a, Vec4d b) noexcept {
    return Jet<Vec4d, N>{a.val + b, a.d};
}

template <std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, N> operator+(Vec4d a, const Jet<Vec4d, N>& b) noexcept {
    return Jet<Vec4d, N>{a + b.val, b.d};
}

template <std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, N> operator-(const Jet<Vec4d, N>& a, Vec4d b) noexcept {
    return Jet<Vec4d, N>{a.val - b, a.d};
}

template <std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, N> operator-(Vec4d a, const Jet<Vec4d, N>& b) noexcept {
    Jet<Vec4d, N> r;
    r.val = a - b.val;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = -b.d[i];
    }
    return r;
}

template <std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, N> operator*(const Jet<Vec4d, N>& a, Vec4d b) noexcept {
    Jet<Vec4d, N> r;
    r.val = a.val * b;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = a.d[i] * b;
    }
    return r;
}

template <std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, N> operator*(Vec4d a, const Jet<Vec4d, N>& b) noexcept {
    return b * a;
}

template <std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, N> operator/(const Jet<Vec4d, N>& a, Vec4d b) noexcept {
    const Vec4d inv = Vec4d::splat(1.0) / b;
    return a * inv;
}

template <std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<Vec4d, N> operator/(Vec4d a, const Jet<Vec4d, N>& b) noexcept {
    const Vec4d inv = Vec4d::splat(1.0) / b.val;
    const Vec4d q = a * inv;
    const Vec4d scale = -(q * inv);
    Jet<Vec4d, N> r;
    r.val = q;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = scale * b.d[i];
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> sqrt(const Jet<T, N>& x) noexcept {
    const T s = detail::sqrt_val(x.val);
    const T inv_2s = detail::splat_like<T>(0.5) / s;
    Jet<T, N> r;
    r.val = s;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = x.d[i] * inv_2s;
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> exp(const Jet<T, N>& x) noexcept {
    const T e = detail::exp_val(x.val);
    Jet<T, N> r;
    r.val = e;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = e * x.d[i];
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> log(const Jet<T, N>& x) noexcept {
    const T inv = detail::one_like<T>() / x.val;
    Jet<T, N> r;
    r.val = detail::log_val(x.val);
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = x.d[i] * inv;
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> norm_cdf(const Jet<T, N>& x) noexcept {
    const T c = detail::norm_cdf_val(x.val);
    const T p = detail::norm_pdf_val(x.val);
    Jet<T, N> r;
    r.val = c;
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = p * x.d[i];
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> max(const Jet<T, N>& x, double k) noexcept {
    Jet<T, N> r;
    r.val = detail::max_val(x.val, detail::splat_like<T>(k));
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = detail::mask_gt(x.val, k, x.d[i]);
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> max(double k, const Jet<T, N>& x) noexcept {
    return max(x, k);
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> max(const Jet<T, N>& a, const Jet<T, N>& b) noexcept {
    Jet<T, N> r;
    r.val = detail::max_val(a.val, b.val);
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = detail::select_gt(a.val, b.val, a.d[i], b.d[i]);
    }
    return r;
}

template <typename T, std::size_t N>
[[nodiscard]] PRICER_FORCE_INLINE Jet<T, N> min(const Jet<T, N>& a, const Jet<T, N>& b) noexcept {
    Jet<T, N> r;
    r.val = detail::min_val(a.val, b.val);
    for (std::size_t i = 0; i < N; ++i) {
        r.d[i] = detail::select_gt(b.val, a.val, a.d[i], b.d[i]);
    }
    return r;
}

} // namespace pricer

#endif // PRICER_CORE_DUAL_HPP
