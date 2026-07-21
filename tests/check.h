// Minimal assertion harness shared by the check_* binaries.
//
// Deliberately not Catch2/GoogleTest: those would need FetchContent (a network
// dependency at configure time) or a vendored copy, and what this project
// actually needs from a framework is a failure counter and a non-zero exit
// status. That is small enough to own outright, and it keeps `cmake -B build`
// working offline.
//
// Every check funnels through record() so that a check which is never printed
// cannot also be a check which never fails.
#ifndef MCPRICER_TESTS_CHECK_H
#define MCPRICER_TESTS_CHECK_H

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

// Agreement measured in standard errors, not as an absolute tolerance.
//
// An absolute tolerance would have to be re-tuned every time N changed, and the
// usual way that goes is that someone loosens it until it passes. In SE units
// the bound is scale-free: under a correct estimator (mc - ref)/se is ~N(0,1)
// regardless of N, so 4.0 means the same thing at every sample size.
//
// NaN must be rejected explicitly. fabs(NaN) > 4.0 is FALSE, so without the
// isfinite guards a NaN -- the most likely catastrophic failure -- would be the
// one result that silently reported a pass.
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

// For exact/analytic identities (put-call parity, call-put equality of gamma
// and vega), where the only expected discrepancy is floating-point rounding.
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

// Returns the process exit status: non-zero if anything failed, so ctest sees
// a failure instead of a clean run that happened to print the word FAIL.
inline int summary() {
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures > 0 ? 1 : 0;
}

} // namespace check

#endif
