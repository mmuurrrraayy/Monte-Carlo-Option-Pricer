#include "cpuid_guard.hpp"

#include <cstdint>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace {

std::uint64_t read_xcr0() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return _xgetbv(0);
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0u));
    return (static_cast<std::uint64_t>(edx) << 32) | static_cast<std::uint64_t>(eax);
#else
    return 0;
#endif
}

} // namespace

bool pricer_has_avx2_fma() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) {
        return false;
    }

    __cpuid(regs, 1);
    const auto ecx1 = static_cast<std::uint32_t>(regs[2]);
    constexpr std::uint32_t kFmaBit     = 1u << 12;
    constexpr std::uint32_t kOsxsaveBit = 1u << 27;
    constexpr std::uint32_t kAvxBit     = 1u << 28;
    if ((ecx1 & (kFmaBit | kOsxsaveBit | kAvxBit)) != (kFmaBit | kOsxsaveBit | kAvxBit)) {
        return false;
    }

    // XCR0 bits 1 & 2: OS saves XMM and YMM state.
    if ((read_xcr0() & 0x6u) != 0x6u) {
        return false;
    }

    __cpuidex(regs, 7, 0);
    const auto ebx7 = static_cast<std::uint32_t>(regs[1]);
    constexpr std::uint32_t kBmi1Bit = 1u << 3;
    constexpr std::uint32_t kAvx2Bit = 1u << 5;
    constexpr std::uint32_t kBmi2Bit = 1u << 8;
    return (ebx7 & (kBmi1Bit | kAvx2Bit | kBmi2Bit)) == (kBmi1Bit | kAvx2Bit | kBmi2Bit);
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0u, &eax, &ebx, &ecx, &edx) == 0 || eax < 7u) {
        return false;
    }

    if (__get_cpuid(1u, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }
    constexpr unsigned int kFmaBit     = 1u << 12;
    constexpr unsigned int kOsxsaveBit = 1u << 27;
    constexpr unsigned int kAvxBit     = 1u << 28;
    if ((ecx & (kFmaBit | kOsxsaveBit | kAvxBit)) != (kFmaBit | kOsxsaveBit | kAvxBit)) {
        return false;
    }

    // XCR0 bits 1 & 2: OS saves XMM and YMM state.
    if ((read_xcr0() & 0x6u) != 0x6u) {
        return false;
    }

    if (__get_cpuid_count(7u, 0u, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }
    constexpr unsigned int kBmi1Bit = 1u << 3;
    constexpr unsigned int kAvx2Bit = 1u << 5;
    constexpr unsigned int kBmi2Bit = 1u << 8;
    return (ebx & (kBmi1Bit | kAvx2Bit | kBmi2Bit)) == (kBmi1Bit | kAvx2Bit | kBmi2Bit);
#else
    return false;
#endif
}
