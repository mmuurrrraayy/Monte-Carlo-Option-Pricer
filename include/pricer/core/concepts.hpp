#ifndef PRICER_CORE_CONCEPTS_HPP
#define PRICER_CORE_CONCEPTS_HPP

#include "pricer/core/types.hpp"
#include "pricer/simd/vec4d.hpp"

#include <concepts>
#include <cstddef>

namespace pricer {

template <typename T>
concept NumericType = requires(T a, T b, double s) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { -a }    -> std::convertible_to<T>;
    { a + s } -> std::convertible_to<T>;
    { s + a } -> std::convertible_to<T>;
    { a - s } -> std::convertible_to<T>;
    { s - a } -> std::convertible_to<T>;
    { a * s } -> std::convertible_to<T>;
    { s * a } -> std::convertible_to<T>;
    { a / s } -> std::convertible_to<T>;
    { s / a } -> std::convertible_to<T>;
};

struct StandardSampling {
    static constexpr bool is_antithetic = false;
};

struct AntitheticSampling {
    static constexpr bool is_antithetic = true;
};

template <typename S>
concept SamplingPolicy = requires {
    { S::is_antithetic } -> std::convertible_to<bool>;
};

template <typename P>
concept ProcessPolicy = requires(const P& p) {
    { P::supports_exact_terminal } -> std::convertible_to<bool>;
    { P::is_gbm }                  -> std::convertible_to<bool>;
    { p.initial_spot() }           -> std::convertible_to<double>;
    { p.maturity() }               -> std::convertible_to<double>;
    { p.discount_factor() }        -> std::convertible_to<double>;
} && (!P::supports_exact_terminal || requires(const P& p, Vec4d z) {
    { p.terminal_spot(z) } -> std::same_as<Vec4d>;
}) && (!P::is_gbm || requires(const P& p, Vec4d log_s, Vec4d z) {
    { p.step_log(log_s, z) } -> std::same_as<Vec4d>;
});

template <typename P>
concept PayoffPolicy = requires {
    { P::is_path_dependent }   -> std::convertible_to<bool>;
    { P::needs_spot_per_step } -> std::convertible_to<bool>;
    { P::is_smooth }           -> std::convertible_to<bool>;
} && (
    (!P::is_path_dependent && requires(const P& p, Vec4d terminal_spot) {
        { p.evaluate(terminal_spot) } -> std::same_as<Vec4d>;
    }) ||
    (P::is_path_dependent && requires(const P& p, Vec4d log_s, double inv_steps) {
        typename P::template State<Vec4d>;
        { p.init_state(log_s) } -> std::same_as<typename P::template State<Vec4d>>;
        requires requires(typename P::template State<Vec4d>& st) {
            p.update(st, log_s);
            { p.evaluate(st, log_s, inv_steps) } -> std::same_as<Vec4d>;
        };
    })
);

namespace detail {

template <typename P>
inline constexpr bool is_geometric_asian_payoff_v = false;

template <typename P>
    requires requires { P::is_geometric_asian; }
inline constexpr bool is_geometric_asian_payoff_v<P> = P::is_geometric_asian;

template <typename P>
inline constexpr bool is_arithmetic_asian_payoff_v = false;

template <typename P>
    requires requires { P::is_arithmetic_asian; }
inline constexpr bool is_arithmetic_asian_payoff_v<P> = P::is_arithmetic_asian;

template <typename P>
inline constexpr bool is_floating_lookback_payoff_v = false;

template <typename P>
    requires requires { P::is_floating_lookback; }
inline constexpr bool is_floating_lookback_payoff_v<P> = P::is_floating_lookback;

template <typename P>
inline constexpr bool is_fixed_lookback_payoff_v = false;

template <typename P>
    requires requires { P::is_fixed_lookback; }
inline constexpr bool is_fixed_lookback_payoff_v<P> = P::is_fixed_lookback;

} // namespace detail

// Rejects custom smooth path-dependent payoffs without a known Gamma strategy
// rather than silently returning Gamma = 0.
template <typename Payoff>
inline constexpr bool has_gbm_gamma_estimator_v =
    PayoffPolicy<Payoff> && (
        !Payoff::is_path_dependent ||
        detail::is_geometric_asian_payoff_v<Payoff> ||
        detail::is_arithmetic_asian_payoff_v<Payoff> ||
        detail::is_floating_lookback_payoff_v<Payoff> ||
        detail::is_fixed_lookback_payoff_v<Payoff>);

template <typename Process, typename Payoff>
inline constexpr bool supports_gbm_5_greeks_v =
    ProcessPolicy<Process> && PayoffPolicy<Payoff> &&
    Process::is_gbm && Payoff::is_smooth &&
    has_gbm_gamma_estimator_v<Payoff>;

template <typename Process, typename Payoff>
inline constexpr bool supports_barrier_greeks_v =
    ProcessPolicy<Process> && PayoffPolicy<Payoff> &&
    Process::is_gbm && Payoff::is_path_dependent && !Payoff::is_smooth;

template <typename Process, typename Payoff>
inline constexpr bool supports_heston_greeks_v =
    ProcessPolicy<Process> && PayoffPolicy<Payoff> &&
    !Process::is_gbm && Payoff::is_smooth;

} // namespace pricer

#endif // PRICER_CORE_CONCEPTS_HPP
