#ifndef PRICER_SIMD_VEC4D_HPP
#define PRICER_SIMD_VEC4D_HPP

#include <immintrin.h>

#include <utility>

#if defined(_MSC_VER)
    #define PRICER_FORCE_INLINE __forceinline
    #define PRICER_VEC_CALL     __vectorcall
    #define PRICER_RESTRICT     __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define PRICER_FORCE_INLINE [[gnu::always_inline]] inline
    #define PRICER_VEC_CALL
    #define PRICER_RESTRICT     __restrict__
#else
    #define PRICER_FORCE_INLINE inline
    #define PRICER_VEC_CALL
    #define PRICER_RESTRICT
#endif

#if defined(__has_cpp_attribute)
    #if __has_cpp_attribute(assume) >= 202207L
        #define PRICER_ASSUME(expr) [[assume(expr)]]
    #endif
#endif
#ifndef PRICER_ASSUME
    #if defined(_MSC_VER)
        #define PRICER_ASSUME(expr) __assume(expr)
    #else
        #define PRICER_ASSUME(expr) do { if (!(expr)) std::unreachable(); } while (false)
    #endif
#endif

namespace pricer {

struct alignas(32) Vec4d {
    __m256d v;

    PRICER_FORCE_INLINE Vec4d() noexcept = default;
    PRICER_FORCE_INLINE explicit Vec4d(__m256d x) noexcept : v(x) {}
    PRICER_FORCE_INLINE explicit Vec4d(double x) noexcept : v(_mm256_set1_pd(x)) {}

    [[nodiscard]] PRICER_FORCE_INLINE static Vec4d splat(double x) noexcept {
        return Vec4d{_mm256_set1_pd(x)};
    }

    [[nodiscard]] PRICER_FORCE_INLINE static Vec4d zero() noexcept {
        return Vec4d{_mm256_setzero_pd()};
    }

    [[nodiscard]] PRICER_FORCE_INLINE static Vec4d set(double a, double b, double c, double d) noexcept {
        return Vec4d{_mm256_setr_pd(a, b, c, d)};
    }

    [[nodiscard]] PRICER_FORCE_INLINE static Vec4d load_aligned(const double* ptr) noexcept {
        return Vec4d{_mm256_load_pd(ptr)};
    }

    [[nodiscard]] PRICER_FORCE_INLINE static Vec4d load_unaligned(const double* ptr) noexcept {
        return Vec4d{_mm256_loadu_pd(ptr)};
    }

    PRICER_FORCE_INLINE void store_aligned(double* ptr) const noexcept {
        _mm256_store_pd(ptr, v);
    }

    PRICER_FORCE_INLINE void store_unaligned(double* ptr) const noexcept {
        _mm256_storeu_pd(ptr, v);
    }

    PRICER_FORCE_INLINE Vec4d& operator+=(Vec4d rhs) noexcept {
        v = _mm256_add_pd(v, rhs.v);
        return *this;
    }

    PRICER_FORCE_INLINE Vec4d& operator-=(Vec4d rhs) noexcept {
        v = _mm256_sub_pd(v, rhs.v);
        return *this;
    }

    PRICER_FORCE_INLINE Vec4d& operator*=(Vec4d rhs) noexcept {
        v = _mm256_mul_pd(v, rhs.v);
        return *this;
    }

    PRICER_FORCE_INLINE Vec4d& operator/=(Vec4d rhs) noexcept {
        v = _mm256_div_pd(v, rhs.v);
        return *this;
    }

    PRICER_FORCE_INLINE Vec4d& operator+=(double rhs) noexcept {
        v = _mm256_add_pd(v, _mm256_set1_pd(rhs));
        return *this;
    }

    PRICER_FORCE_INLINE Vec4d& operator-=(double rhs) noexcept {
        v = _mm256_sub_pd(v, _mm256_set1_pd(rhs));
        return *this;
    }

    PRICER_FORCE_INLINE Vec4d& operator*=(double rhs) noexcept {
        v = _mm256_mul_pd(v, _mm256_set1_pd(rhs));
        return *this;
    }

    PRICER_FORCE_INLINE Vec4d& operator/=(double rhs) noexcept {
        v = _mm256_div_pd(v, _mm256_set1_pd(rhs));
        return *this;
    }
};

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator+(Vec4d x) noexcept {
    return x;
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL negate(Vec4d x) noexcept {
    return Vec4d{_mm256_xor_pd(x.v, _mm256_set1_pd(-0.0))};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator-(Vec4d x) noexcept {
    return negate(x);
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator+(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_add_pd(a.v, b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator-(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_sub_pd(a.v, b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator*(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_mul_pd(a.v, b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator/(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_div_pd(a.v, b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator+(Vec4d a, double b) noexcept {
    return Vec4d{_mm256_add_pd(a.v, _mm256_set1_pd(b))};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator+(double a, Vec4d b) noexcept {
    return Vec4d{_mm256_add_pd(_mm256_set1_pd(a), b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator-(Vec4d a, double b) noexcept {
    return Vec4d{_mm256_sub_pd(a.v, _mm256_set1_pd(b))};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator-(double a, Vec4d b) noexcept {
    return Vec4d{_mm256_sub_pd(_mm256_set1_pd(a), b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator*(Vec4d a, double b) noexcept {
    return Vec4d{_mm256_mul_pd(a.v, _mm256_set1_pd(b))};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator*(double a, Vec4d b) noexcept {
    return Vec4d{_mm256_mul_pd(_mm256_set1_pd(a), b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator/(Vec4d a, double b) noexcept {
    return Vec4d{_mm256_div_pd(a.v, _mm256_set1_pd(b))};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL operator/(double a, Vec4d b) noexcept {
    return Vec4d{_mm256_div_pd(_mm256_set1_pd(a), b.v)};
}

// fma = a*b + c, fnma = c - a*b, fms = a*b - c
[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL fma(Vec4d a, Vec4d b, Vec4d c) noexcept {
    return Vec4d{_mm256_fmadd_pd(a.v, b.v, c.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL fnma(Vec4d a, Vec4d b, Vec4d c) noexcept {
    return Vec4d{_mm256_fnmadd_pd(a.v, b.v, c.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL fms(Vec4d a, Vec4d b, Vec4d c) noexcept {
    return Vec4d{_mm256_fmsub_pd(a.v, b.v, c.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL max(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_max_pd(a.v, b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL min(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_min_pd(a.v, b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL abs(Vec4d x) noexcept {
    return Vec4d{_mm256_andnot_pd(_mm256_set1_pd(-0.0), x.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL sqrt(Vec4d x) noexcept {
    return Vec4d{_mm256_sqrt_pd(x.v)};
}

// Comparisons (ordered quiet predicates; result is all-1s or all-0s per 64-bit lane)
[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL cmp_gt(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL cmp_lt(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL cmp_ge(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL cmp_le(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL cmp_eq(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL cmp_neq(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_cmp_pd(a.v, b.v, _CMP_NEQ_UQ)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL bit_and(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_and_pd(a.v, b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL bit_or(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_or_pd(a.v, b.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL bit_xor(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_xor_pd(a.v, b.v)};
}

// (~a) & b
[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL bit_andnot(Vec4d a, Vec4d b) noexcept {
    return Vec4d{_mm256_andnot_pd(a.v, b.v)};
}

// (mask MSB set) ? b : a
[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL blend(Vec4d a, Vec4d b, Vec4d mask) noexcept {
    return Vec4d{_mm256_blendv_pd(a.v, b.v, mask.v)};
}

[[nodiscard]] PRICER_FORCE_INLINE int PRICER_VEC_CALL movemask(Vec4d x) noexcept {
    return _mm256_movemask_pd(x.v);
}

[[nodiscard]] PRICER_FORCE_INLINE double PRICER_VEC_CALL hsum(Vec4d x) noexcept {
    const __m128d lo = _mm256_castpd256_pd128(x.v);
    const __m128d hi = _mm256_extractf128_pd(x.v, 1);
    const __m128d sum2 = _mm_add_pd(lo, hi);
    const __m128d shuf = _mm_unpackhi_pd(sum2, sum2);
    return _mm_cvtsd_f64(_mm_add_sd(sum2, shuf));
}

} // namespace pricer

#endif // PRICER_SIMD_VEC4D_HPP
