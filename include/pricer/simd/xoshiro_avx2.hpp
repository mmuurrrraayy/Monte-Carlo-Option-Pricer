#ifndef PRICER_SIMD_XOSHIRO_AVX2_HPP
#define PRICER_SIMD_XOSHIRO_AVX2_HPP

#include "pricer/simd/avx2_math.hpp"
#include "pricer/simd/vec4d.hpp"

#include <array>
#include <bit>
#include <cstdint>

namespace pricer {

[[nodiscard]] constexpr std::uint64_t mix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct SplitMix64 {
    std::uint64_t state = 0;

    constexpr explicit SplitMix64(std::uint64_t seed) noexcept : state(seed) {}

    [[nodiscard]] constexpr std::uint64_t next() noexcept {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

class Xoshiro256PlusPlus {
public:
    std::array<std::uint64_t, 4> s{};

    constexpr Xoshiro256PlusPlus() noexcept : Xoshiro256PlusPlus(42ULL) {}

    constexpr explicit Xoshiro256PlusPlus(std::uint64_t seed) noexcept {
        SplitMix64 sm(seed);
        s[0] = sm.next();
        s[1] = sm.next();
        s[2] = sm.next();
        s[3] = sm.next();
    }

    constexpr explicit Xoshiro256PlusPlus(std::array<std::uint64_t, 4> state) noexcept
        : s(state) {}

    [[nodiscard]] static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    [[nodiscard]] constexpr std::uint64_t next() noexcept {
        const std::uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const std::uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Advances state by 2^128 steps.
    constexpr void jump() noexcept {
        static constexpr std::array<std::uint64_t, 4> kJump = {
            0x180ec6d33cfd0abaULL,
            0xd5a61266f0c9392cULL,
            0xa9582618e03fc9aaULL,
            0x39abdc4529b1661cULL,
        };
        jump_impl(kJump);
    }

    // Advances state by 2^192 steps.
    constexpr void long_jump() noexcept {
        static constexpr std::array<std::uint64_t, 4> kLongJump = {
            0x76e15d3efefdcbbfULL,
            0xc5004e441c522fb3ULL,
            0x77710069854ee241ULL,
            0x39109bb02acbe635ULL,
        };
        jump_impl(kLongJump);
    }

    // Maps 52 top bits to (k + 0.5) * 2^-52 in [2^-53, 1 - 2^-53].
    [[nodiscard]] static constexpr double bits_to_open01(std::uint64_t bits) noexcept {
        const std::uint64_t mant = (bits >> 12) | 0x3FF0000000000000ULL;
        return std::bit_cast<double>(mant) - (1.0 - 0x1.0p-53);
    }

    [[nodiscard]] constexpr double next_uniform_open01() noexcept {
        return bits_to_open01(next());
    }

private:
    constexpr void jump_impl(const std::array<std::uint64_t, 4>& poly) noexcept {
        std::uint64_t s0 = 0;
        std::uint64_t s1 = 0;
        std::uint64_t s2 = 0;
        std::uint64_t s3 = 0;
        for (std::uint64_t word : poly) {
            for (int b = 0; b < 64; ++b) {
                if ((word & (1ULL << b)) != 0ULL) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                (void)next();
            }
        }
        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }
};

// 4 independent 256-bit states transposed across four __m256i registers.
class alignas(32) Xoshiro256PlusPlusX4 {
public:
    __m256i s0;
    __m256i s1;
    __m256i s2;
    __m256i s3;

    Xoshiro256PlusPlusX4() noexcept
        : s0(_mm256_setzero_si256()),
          s1(_mm256_setzero_si256()),
          s2(_mm256_setzero_si256()),
          s3(_mm256_setzero_si256()) {}

    [[nodiscard]] static Xoshiro256PlusPlusX4 from_chunk(
        std::uint64_t seed, std::uint64_t chunk_id) noexcept {
        std::array<Xoshiro256PlusPlus, 4> lanes;
        for (std::uint64_t i = 0; i < 4; ++i) {
            const std::uint64_t lane_seed = seed ^ mix64((chunk_id << 2) | i);
            lanes[static_cast<std::size_t>(i)] = Xoshiro256PlusPlus(lane_seed);
        }
        return from_states(lanes);
    }

    [[nodiscard]] static Xoshiro256PlusPlusX4 from_states(
        const std::array<Xoshiro256PlusPlus, 4>& lanes) noexcept {
        Xoshiro256PlusPlusX4 rng;
        rng.s0 = _mm256_setr_epi64x(
            static_cast<long long>(lanes[0].s[0]),
            static_cast<long long>(lanes[1].s[0]),
            static_cast<long long>(lanes[2].s[0]),
            static_cast<long long>(lanes[3].s[0]));
        rng.s1 = _mm256_setr_epi64x(
            static_cast<long long>(lanes[0].s[1]),
            static_cast<long long>(lanes[1].s[1]),
            static_cast<long long>(lanes[2].s[1]),
            static_cast<long long>(lanes[3].s[1]));
        rng.s2 = _mm256_setr_epi64x(
            static_cast<long long>(lanes[0].s[2]),
            static_cast<long long>(lanes[1].s[2]),
            static_cast<long long>(lanes[2].s[2]),
            static_cast<long long>(lanes[3].s[2]));
        rng.s3 = _mm256_setr_epi64x(
            static_cast<long long>(lanes[0].s[3]),
            static_cast<long long>(lanes[1].s[3]),
            static_cast<long long>(lanes[2].s[3]),
            static_cast<long long>(lanes[3].s[3]));
        return rng;
    }

    [[nodiscard]] PRICER_FORCE_INLINE __m256i next_u64x4() noexcept {
        const __m256i sum03 = _mm256_add_epi64(s0, s3);
        const __m256i rot23 = _mm256_or_si256(
            _mm256_slli_epi64(sum03, 23),
            _mm256_srli_epi64(sum03, 41));
        const __m256i result = _mm256_add_epi64(rot23, s0);

        const __m256i t = _mm256_slli_epi64(s1, 17);
        s2 = _mm256_xor_si256(s2, s0);
        s3 = _mm256_xor_si256(s3, s1);
        s1 = _mm256_xor_si256(s1, s2);
        s0 = _mm256_xor_si256(s0, s3);
        s2 = _mm256_xor_si256(s2, t);
        s3 = _mm256_or_si256(
            _mm256_slli_epi64(s3, 45),
            _mm256_srli_epi64(s3, 19));
        return result;
    }

    // Maps 52 top bits to (k + 0.5) * 2^-52 in [2^-53, 1 - 2^-53] so log(u) never sees 0.
    [[nodiscard]] PRICER_FORCE_INLINE static Vec4d PRICER_VEC_CALL
    bits_to_open01(__m256i bits) noexcept {
        const __m256i exp_one_mask = _mm256_set1_epi64x(0x3FF0000000000000LL);
        const __m256i mant = _mm256_or_si256(_mm256_srli_epi64(bits, 12), exp_one_mask);
        return Vec4d{_mm256_sub_pd(
            _mm256_castsi256_pd(mant),
            _mm256_set1_pd(1.0 - 0x1.0p-53))};
    }

    [[nodiscard]] PRICER_FORCE_INLINE Vec4d next_uniform_open01() noexcept {
        return bits_to_open01(next_u64x4());
    }

    PRICER_FORCE_INLINE void next_normals_pair(Vec4d& z0, Vec4d& z1) noexcept;
};

struct BoxMullerAVX2 {
    PRICER_FORCE_INLINE static void PRICER_VEC_CALL
    from_uniforms(Vec4d u1, Vec4d u2, Vec4d& z0, Vec4d& z1) noexcept {
        const Vec4d r = sqrt(Vec4d::splat(-2.0) * log_avx2(u1));
        const Vec4d theta = Vec4d::splat(6.283185307179586476925286766559) * u2;
        Vec4d s;
        Vec4d c;
        sincos_avx2(theta, s, c);
        z0 = r * c;
        z1 = r * s;
    }

    PRICER_FORCE_INLINE static void
    next_pair(Xoshiro256PlusPlusX4& rng, Vec4d& z0, Vec4d& z1) noexcept {
        const Vec4d u1 = rng.next_uniform_open01();
        const Vec4d u2 = rng.next_uniform_open01();
        from_uniforms(u1, u2, z0, z1);
    }
};

PRICER_FORCE_INLINE void Xoshiro256PlusPlusX4::next_normals_pair(Vec4d& z0, Vec4d& z1) noexcept {
    BoxMullerAVX2::next_pair(*this, z0, z1);
}

} // namespace pricer

#endif // PRICER_SIMD_XOSHIRO_AVX2_HPP
