#ifndef PRICER_SIMD_AVX2_MATH_HPP
#define PRICER_SIMD_AVX2_MATH_HPP

#include "pricer/simd/vec4d.hpp"

#include <cmath>
#include <cstdint>

namespace pricer {

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL exp_avx2(Vec4d x) noexcept {
    // Keeps k in [-1022, 1023] so 2^k stays normal.
    const Vec4d x_min = Vec4d::splat(-708.396);
    const Vec4d x_max = Vec4d::splat(709.08);
    x = min(x_max, max(x_min, x));

    const Vec4d log2e  = Vec4d::splat(0x1.71547652b82fep+0);  // 1 / ln(2)
    const Vec4d magic  = Vec4d::splat(6755399441055744.0);    // 1.5 * 2^52
    const Vec4d ln2_hi = Vec4d::splat(0x1.62e42fefa38p-1);    // 13 trailing zero bits (Cody-Waite)
    const Vec4d ln2_lo = Vec4d::splat(0x1.ef35793c7673p-45);  // ln(2) - ln2_hi

    // Magic-number round to even: k_magic holds signed integer k in low bits.
    const Vec4d k_magic = fma(x, log2e, magic);
    const Vec4d k = k_magic - magic;
    const Vec4d r = fnma(k, ln2_lo, fnma(k, ln2_hi, x));

    // Degree-11 Remez minimax for exp(r) on [-ln(2)/2, ln(2)/2]; c0 = c1 = 1.
    const Vec4d c2  = Vec4d::splat(0x1.0000000000011p-1);
    const Vec4d c3  = Vec4d::splat(0x1.555555555556ap-3);
    const Vec4d c4  = Vec4d::splat(0x1.555555554f0cdp-5);
    const Vec4d c5  = Vec4d::splat(0x1.111111110c4c4p-7);
    const Vec4d c6  = Vec4d::splat(0x1.6c16c187fc372p-10);
    const Vec4d c7  = Vec4d::splat(0x1.a01a01bec1a41p-13);
    const Vec4d c8  = Vec4d::splat(0x1.a01991abb830fp-16);
    const Vec4d c9  = Vec4d::splat(0x1.71ddd95f28ec5p-19);
    const Vec4d c10 = Vec4d::splat(0x1.28b4062b9abaep-22);
    const Vec4d c11 = Vec4d::splat(0x1.af8d9fe87cef4p-26);

    Vec4d p = fma(c11, r, c10);
    p = fma(p, r, c9);
    p = fma(p, r, c8);
    p = fma(p, r, c7);
    p = fma(p, r, c6);
    p = fma(p, r, c5);
    p = fma(p, r, c4);
    p = fma(p, r, c3);
    p = fma(p, r, c2);
    p = fma(p, r, Vec4d::splat(1.0));
    p = fma(p, r, Vec4d::splat(1.0));

    // Place (k + 1023) into IEEE-754 exponent bits to form 2^k.
    const __m256i ki = _mm256_add_epi64(_mm256_castpd_si256(k_magic.v),
                                        _mm256_set1_epi64x(1023));
    const Vec4d pow2k{_mm256_castsi256_pd(_mm256_slli_epi64(ki, 52))};

    return p * pow2k;
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL log_avx2(Vec4d x) noexcept {
    const __m256i bits = _mm256_castpd_si256(x.v);

    // Biased exponent -> double without AVX-512 cvtepi64_pd: OR with 2^52, subtract 2^52 + 1023.
    const __m256i e_bits = _mm256_and_si256(_mm256_srli_epi64(bits, 52),
                                            _mm256_set1_epi64x(0x7FF));
    const __m256i e_magic_i = _mm256_or_si256(e_bits,
                                              _mm256_set1_epi64x(0x4330000000000000LL));
    Vec4d e{_mm256_sub_pd(_mm256_castsi256_pd(e_magic_i),
                          _mm256_set1_pd(4503599627371519.0))};

    // Mantissa m in [1, 2), then shift to [sqrt(2)/2, sqrt(2)].
    const __m256i mant_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL);
    const __m256i exp_one   = _mm256_set1_epi64x(0x3FF0000000000000LL);
    Vec4d m{_mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(bits, mant_mask), exp_one))};

    const Vec4d sqrt2 = Vec4d::splat(0x1.6a09e667f3bcdp+0);
    const Vec4d mask_ge = cmp_ge(m, sqrt2);
    m = blend(m, m * Vec4d::splat(0.5), mask_ge);
    e = e + bit_and(mask_ge, Vec4d::splat(1.0));

    const Vec4d one = Vec4d::splat(1.0);
    const Vec4d f = (m - one) / (m + one);
    const Vec4d s = f * f;

    // Odd Remez minimax: ln(m) = 2*f * (1 + s * P(s)) on |f| <= (sqrt(2)-1)/(sqrt(2)+1).
    const Vec4d c1 = Vec4d::splat(0x1.5555555555558p-2);
    const Vec4d c2 = Vec4d::splat(0x1.999999999522ep-3);
    const Vec4d c3 = Vec4d::splat(0x1.2492492e02b0fp-3);
    const Vec4d c4 = Vec4d::splat(0x1.c71c62d1c7546p-4);
    const Vec4d c5 = Vec4d::splat(0x1.7462b9b7cb0c5p-4);
    const Vec4d c6 = Vec4d::splat(0x1.39fdc7dff97bp-4);
    const Vec4d c7 = Vec4d::splat(0x1.2b5f0a3e7cb11p-4);

    Vec4d p = fma(c7, s, c6);
    p = fma(p, s, c5);
    p = fma(p, s, c4);
    p = fma(p, s, c3);
    p = fma(p, s, c2);
    p = fma(p, s, c1);

    const Vec4d two_f = f + f;
    const Vec4d ln_m = fma(two_f * s, p, two_f);

    const Vec4d ln2_hi = Vec4d::splat(0x1.62e42fefa38p-1);
    const Vec4d ln2_lo = Vec4d::splat(0x1.ef35793c7673p-45);
    const Vec4d res = fma(e, ln2_hi, fma(e, ln2_lo, ln_m));
    const __m256d invalid = _mm256_cmp_pd(x.v, _mm256_setzero_pd(), _CMP_NGT_UQ);
    return Vec4d{_mm256_or_pd(res.v, invalid)};
}

PRICER_FORCE_INLINE void PRICER_VEC_CALL
sincos_avx2(Vec4d x, Vec4d& out_sin, Vec4d& out_cos) noexcept {
    const Vec4d inv_pi_over_2 = Vec4d::splat(0x1.45f306dc9c883p-1); // 2 / pi
    const Vec4d magic         = Vec4d::splat(6755399441055744.0);   // 1.5 * 2^52
    const Vec4d pi2_hi        = Vec4d::splat(0x1.921fb5444p+0);     // pi/2 high (16 zero bits)
    const Vec4d pi2_lo        = Vec4d::splat(0x1.68c234c4c6629p-39);

    const Vec4d k_magic = fma(x, inv_pi_over_2, magic);
    const Vec4d k = k_magic - magic;
    const __m256i ki = _mm256_castpd_si256(k_magic.v);

    const Vec4d r = fnma(k, pi2_lo, fnma(k, pi2_hi, x));
    const Vec4d s = r * r;

    // 6-term Remez minimax on [-pi/4, pi/4].
    const Vec4d s1 = Vec4d::splat(-0x1.5555555555555p-3);
    const Vec4d s2 = Vec4d::splat( 0x1.1111111110bb3p-7);
    const Vec4d s3 = Vec4d::splat(-0x1.a01a019e83e3p-13);
    const Vec4d s4 = Vec4d::splat( 0x1.71de3796c9c41p-19);
    const Vec4d s5 = Vec4d::splat(-0x1.ae600b3db902bp-26);
    const Vec4d s6 = Vec4d::splat( 0x1.5e0b2d1aacbc5p-33);

    Vec4d ps = fma(s6, s, s5);
    ps = fma(ps, s, s4);
    ps = fma(ps, s, s3);
    ps = fma(ps, s, s2);
    ps = fma(ps, s, s1);
    const Vec4d sin_r = fma(r * s, ps, r);

    const Vec4d c2 = Vec4d::splat( 0x1.5555555555555p-5);
    const Vec4d c3 = Vec4d::splat(-0x1.6c16c16c16967p-10);
    const Vec4d c4 = Vec4d::splat( 0x1.a01a019f4ec9p-16);
    const Vec4d c5 = Vec4d::splat(-0x1.27e4fa17f65f2p-22);
    const Vec4d c6 = Vec4d::splat( 0x1.1eeb69037a64ap-29);
    const Vec4d c7 = Vec4d::splat(-0x1.907db46c8b55p-37);

    Vec4d pc = fma(c7, s, c6);
    pc = fma(pc, s, c5);
    pc = fma(pc, s, c4);
    pc = fma(pc, s, c3);
    pc = fma(pc, s, c2);
    const Vec4d cos_r = fma(s * s, pc, fnma(Vec4d::splat(0.5), s, Vec4d::splat(1.0)));

    // Quadrant from low 2 bits of k: bit 0 swaps sin/cos; bit 1 flips sin; (k+1)&2 flips cos.
    const Vec4d swap_mask{_mm256_castsi256_pd(_mm256_slli_epi64(ki, 63))};
    const Vec4d sin_abs = blend(sin_r, cos_r, swap_mask);
    const Vec4d cos_abs = blend(cos_r, sin_r, swap_mask);

    const __m256i two = _mm256_set1_epi64x(2);
    const __m256i one = _mm256_set1_epi64x(1);
    const __m256i sin_sign_i = _mm256_slli_epi64(_mm256_and_si256(ki, two), 62);
    const __m256i cos_sign_i = _mm256_slli_epi64(
        _mm256_and_si256(_mm256_add_epi64(ki, one), two), 62);

    out_sin = Vec4d{_mm256_xor_pd(sin_abs.v, _mm256_castsi256_pd(sin_sign_i))};
    out_cos = Vec4d{_mm256_xor_pd(cos_abs.v, _mm256_castsi256_pd(cos_sign_i))};
}

// Cody (1969) 3-region rational erfc for Phi(x) and phi(x) sharing one exp(-x^2/2).
PRICER_FORCE_INLINE void PRICER_VEC_CALL
norm_cdf_and_pdf_avx2(Vec4d x, Vec4d& out_cdf, Vec4d& out_pdf) noexcept {
    const Vec4d abs_x = abs(x);
    const Vec4d inv_sqrt2 = Vec4d::splat(0x1.6a09e667f3bcdp-1); // 1 / sqrt(2)
    const Vec4d z = abs_x * inv_sqrt2;

    const Vec4d one = Vec4d::splat(1.0);
    const Vec4d half = Vec4d::splat(0.5);

    // Clamp unselected lanes away from rational poles / overflow.
    const Vec4d z1 = min(half, z);
    const Vec4d z2 = min(Vec4d::splat(4.0), z);

    // Region 1: z <= 0.46875 -> erf(z) = z * P1(z^2) / Q1(z^2)
    const Vec4d y1 = z1 * z1;
    Vec4d num1 = Vec4d::splat(1.85777706184603153e-01);
    num1 = fma(num1, y1, Vec4d::splat(3.16112374387056560e+00));
    num1 = fma(num1, y1, Vec4d::splat(1.13864154151050156e+02));
    num1 = fma(num1, y1, Vec4d::splat(3.77485237685302021e+02));
    num1 = fma(num1, y1, Vec4d::splat(3.20937758913846947e+03));

    Vec4d den1 = y1 + Vec4d::splat(2.36012909523441209e+01);
    den1 = fma(den1, y1, Vec4d::splat(2.44024637934444173e+02));
    den1 = fma(den1, y1, Vec4d::splat(1.28261652607737228e+03));
    den1 = fma(den1, y1, Vec4d::splat(2.84423683343917062e+03));

    const Vec4d erf_z1 = z1 * (num1 / den1);
    const Vec4d half_erfc_r1 = fnma(half, erf_z1, half);

    // Split x^2 = x2_hi + x2_lo via FMA so exp(-0.5*x^2) keeps tail accuracy up to |x| = 37.5.
    const Vec4d x_tail = min(Vec4d::splat(37.5), abs_x);
    const Vec4d x2_hi = x_tail * x_tail;
    const Vec4d x2_lo = fms(x_tail, x_tail, x2_hi);
    const Vec4d neg_half = Vec4d::splat(-0.5);
    const Vec4d exp_hi = exp_avx2(neg_half * x2_hi);
    const Vec4d exp_neg_z2 = exp_hi * fma(neg_half, x2_lo, one);

    // Region 2: 0.46875 < z <= 4.0 -> exp(z^2) * erfc(z) = P2(z) / Q2(z)
    Vec4d num2 = Vec4d::splat(2.15311535474403846e-08);
    num2 = fma(num2, z2, Vec4d::splat(5.64188496988670089e-01));
    num2 = fma(num2, z2, Vec4d::splat(8.88314979438837594e+00));
    num2 = fma(num2, z2, Vec4d::splat(6.61191906371416295e+01));
    num2 = fma(num2, z2, Vec4d::splat(2.98635138197400131e+02));
    num2 = fma(num2, z2, Vec4d::splat(8.81952221241769090e+02));
    num2 = fma(num2, z2, Vec4d::splat(1.71204761263407058e+03));
    num2 = fma(num2, z2, Vec4d::splat(2.05107837782607147e+03));
    num2 = fma(num2, z2, Vec4d::splat(1.23033935479799725e+03));

    Vec4d den2 = z2 + Vec4d::splat(1.57449261107098347e+01);
    den2 = fma(den2, z2, Vec4d::splat(1.17693950891312499e+02));
    den2 = fma(den2, z2, Vec4d::splat(5.37181101862009858e+02));
    den2 = fma(den2, z2, Vec4d::splat(1.62138957456669019e+03));
    den2 = fma(den2, z2, Vec4d::splat(3.29079923573345963e+03));
    den2 = fma(den2, z2, Vec4d::splat(4.36261909014324716e+03));
    den2 = fma(den2, z2, Vec4d::splat(3.43936767414372164e+03));
    den2 = fma(den2, z2, Vec4d::splat(1.23033935480374942e+03));

    const Vec4d half_erfc_r2 = (half * exp_neg_z2) * (num2 / den2);

    const Vec4d mask_r1 = cmp_le(z, Vec4d::splat(0.46875));
    const Vec4d mask_r3 = cmp_gt(z, Vec4d::splat(4.0));
    Vec4d half_erfc = blend(half_erfc_r2, half_erfc_r1, mask_r1);

    const Vec4d inv_sqrt_2pi = Vec4d::splat(0.398942280401432677939946059934381868);
    out_pdf = inv_sqrt_2pi * exp_neg_z2;

    if (movemask(mask_r3) != 0) [[unlikely]] {
        // Region 3: z > 4.0 -> erfc(z) = (exp(-z^2) / z) * (1/sqrt(pi) - u * P3(u) / Q3(u))
        const Vec4d z3 = max(Vec4d::splat(4.0), z);
        const Vec4d u = one / (z3 * z3);
        Vec4d num3 = Vec4d::splat(1.63153871373020978e-02);
        num3 = fma(num3, u, Vec4d::splat(3.05326634961232344e-01));
        num3 = fma(num3, u, Vec4d::splat(3.60344899949804439e-01));
        num3 = fma(num3, u, Vec4d::splat(1.25781726111229246e-01));
        num3 = fma(num3, u, Vec4d::splat(1.60837851487422766e-02));
        num3 = fma(num3, u, Vec4d::splat(6.58749161529837803e-04));

        Vec4d den3 = u + Vec4d::splat(2.56852019228982242e+00);
        den3 = fma(den3, u, Vec4d::splat(1.87295284992346047e+00));
        den3 = fma(den3, u, Vec4d::splat(5.27905102951428412e-01));
        den3 = fma(den3, u, Vec4d::splat(6.05183413124413191e-02));
        den3 = fma(den3, u, Vec4d::splat(2.33520497626869185e-03));

        const Vec4d inv_sqrt_pi = Vec4d::splat(0x1.20dd750429b6dp-1); // 1 / sqrt(pi)
        const Vec4d asym = inv_sqrt_pi - u * (num3 / den3);
        const Vec4d half_erfc_r3 = (half * exp_neg_z2 / z3) * asym;
        half_erfc = blend(half_erfc, half_erfc_r3, mask_r3);

        const Vec4d mask_extreme = cmp_gt(abs_x, Vec4d::splat(37.5));
        half_erfc = bit_andnot(mask_extreme, half_erfc);
        out_pdf = bit_andnot(mask_extreme, out_pdf);
    }

    // Use half_erfc directly for x < 0 to preserve left-tail relative accuracy.
    const Vec4d mask_neg = cmp_lt(x, Vec4d::zero());
    out_cdf = blend(one - half_erfc, half_erfc, mask_neg);
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL norm_cdf_avx2(Vec4d x) noexcept {
    Vec4d cdf, pdf;
    norm_cdf_and_pdf_avx2(x, cdf, pdf);
    return cdf;
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL sqrt_avx2(Vec4d x) noexcept {
    return sqrt(x);
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL exp(Vec4d x) noexcept {
    return exp_avx2(x);
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL log(Vec4d x) noexcept {
    return log_avx2(x);
}

[[nodiscard]] PRICER_FORCE_INLINE double norm_cdf(double x) noexcept {
    constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

[[nodiscard]] PRICER_FORCE_INLINE Vec4d PRICER_VEC_CALL norm_cdf(Vec4d x) noexcept {
    return norm_cdf_avx2(x);
}

} // namespace pricer

#endif // PRICER_SIMD_AVX2_MATH_HPP
