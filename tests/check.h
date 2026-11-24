#ifndef PRICER_TESTS_CHECK_H
#define PRICER_TESTS_CHECK_H

#include <cmath>
#include <cstdio>

namespace check {

inline int g_checks = 0;
inline int g_failures = 0;

inline bool record(bool ok) {
    ++g_checks;
    if (!ok) ++g_failures;
    return ok;
}

// Scale-free check in standard errors: (mc - ref) / se ~ N(0, 1).
inline bool within_se(const char* name, double mc, double se, double ref,
                      double max_se = 4.0) {
    const auto ok = std::isfinite(mc) && std::isfinite(se) && se > 0.0
                 && std::isfinite(ref)
                 && std::fabs((mc - ref) / se) <= max_se;
    record(ok);
    std::printf("%-28s mc %14.8f  se %12.8f  ref %14.8f  err_se %8.4f  %s\n",
                name, mc, se, ref,
                se > 0.0 ? (mc - ref) / se : 0.0, ok ? "ok" : "FAIL");
    return ok;
}

inline bool close(const char* name, double got, double want, double tol) {
    const auto ok = std::isfinite(got) && std::isfinite(want)
                 && std::fabs(got - want) <= tol;
    record(ok);
    std::printf("%-28s got %14.8f  want %14.8f  diff %12.3g  tol %8.3g  %s\n",
                name, got, want, got - want, tol, ok ? "ok" : "FAIL");
    return ok;
}

inline bool is_true(const char* name, bool cond) {
    const auto ok = record(cond);
    std::printf("%-28s %s\n", name, ok ? "ok" : "FAIL");
    return ok;
}

inline int summary() {
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures > 0 ? 1 : 0;
}

} // namespace check

#endif
