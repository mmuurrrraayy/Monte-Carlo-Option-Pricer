#include "mcpricer/Precision.h"

#include <cmath>

Precision::Precision(const totals& t_, double discount_, double z_):
    t(t_), discount(discount_), z(z_) {}

double Precision::price() const {
    return discount * (t.sum / (double) t.paths);
}

double Precision::std_error() const {
    const auto n = static_cast<double>(t.paths);
    const auto mean = t.sum / n;                                   // undiscounted
    const auto var = (t.sumsq / n - mean * mean) * (n / (n - 1.0)); // Bessel-corrected
    return discount * std::sqrt(var / n);
}

Precision::interval Precision::calc_CI() const {
    const auto p = price();
    const auto se = std_error();
    return interval{p - z * se, p + z * se};
}
