// price -- command-line front end to the Monte Carlo pricer.
//
// Prices a European call or put, reports the estimate with its standard error
// and a 95% confidence interval, and (unless --no-reference) prints the
// closed-form Black-Scholes value alongside it with the gap expressed in
// standard errors. Greeks are optional, since they cost an extra pass.
//
//   price --s0 100 --k 100 --r 0.05 --sigma 0.2 --T 1 --paths 1000000 --greeks

#include "mcpricer/DataVariables.h"
#include "mcpricer/Option.h"
#include "mcpricer/Black-Scholes.h"
#include "mcpricer/ParallelEngine.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

static void usage() {
    std::printf(
        "usage: price [options]\n"
        "\n"
        "  --s0 <f>        spot price                      (default 100)\n"
        "  --k <f>         strike                          (default 100)\n"
        "  --r <f>         risk-free rate, annualised      (default 0.05)\n"
        "  --sigma <f>     volatility, annualised          (default 0.20)\n"
        "  --T <f>         time to expiry in years         (default 1.0)\n"
        "  --type <s>      call | put                      (default call)\n"
        "  --paths <n>     simulated paths                 (default 1000000)\n"
        "  --threads <n>   worker threads                  (default hardware)\n"
        "  --seed <n>      RNG seed                        (default 42)\n"
        "  --antithetic    use antithetic variates (--paths is then rounded\n"
        "                  down to an even number and run as paths/2 pairs)\n"
        "  --greeks        also report delta/gamma/vega/theta/rho\n"
        "  --theta-h <f>   bump for the CRN theta estimate (default 0.005)\n"
        "  --no-reference  suppress the closed-form comparison\n"
        "  -h, --help      this message\n");
}

// Returns false if the value is missing or unparseable, so a typo like
// `--sigma --greeks` is rejected rather than silently read as 0.
static bool arg_double(int argc, char** argv, int& i, double& out) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", argv[i]);
        return false;
    }
    char* end = nullptr;
    const auto v = std::strtod(argv[i + 1], &end);
    if (end == argv[i + 1] || *end != '\0') {
        std::fprintf(stderr, "error: %s: '%s' is not a number\n", argv[i], argv[i + 1]);
        return false;
    }
    out = v;
    ++i;
    return true;
}

static bool arg_u64(int argc, char** argv, int& i, std::uint64_t& out) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", argv[i]);
        return false;
    }
    char* end = nullptr;
    const auto v = std::strtoull(argv[i + 1], &end, 10);
    if (end == argv[i + 1] || *end != '\0') {
        std::fprintf(stderr, "error: %s: '%s' is not an integer\n", argv[i], argv[i + 1]);
        return false;
    }
    out = v;
    ++i;
    return true;
}

int main(int argc, char** argv) {
    auto s0 = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0, theta_h = 0.005;
    auto paths = std::uint64_t{1000000};
    auto seed = std::uint64_t{42};
    auto threads = std::uint64_t{0};
    auto type = OptionType::Call;
    auto want_greeks = false, want_anti = false, want_ref = true;

    for (int i = 1; i < argc; ++i) {
        const auto a = std::string(argv[i]);
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--s0")    { if (!arg_double(argc, argv, i, s0)) return 2; }
        else if (a == "--k")     { if (!arg_double(argc, argv, i, K)) return 2; }
        else if (a == "--r")     { if (!arg_double(argc, argv, i, r)) return 2; }
        else if (a == "--sigma") { if (!arg_double(argc, argv, i, sigma)) return 2; }
        else if (a == "--T")     { if (!arg_double(argc, argv, i, T)) return 2; }
        else if (a == "--theta-h") { if (!arg_double(argc, argv, i, theta_h)) return 2; }
        else if (a == "--paths") { if (!arg_u64(argc, argv, i, paths)) return 2; }
        else if (a == "--seed")  { if (!arg_u64(argc, argv, i, seed)) return 2; }
        else if (a == "--threads") { if (!arg_u64(argc, argv, i, threads)) return 2; }
        else if (a == "--greeks") want_greeks = true;
        else if (a == "--antithetic") want_anti = true;
        else if (a == "--no-reference") want_ref = false;
        else if (a == "--type") {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: --type needs a value\n"); return 2; }
            const auto v = std::string(argv[++i]);
            if (v == "call") type = OptionType::Call;
            else if (v == "put") type = OptionType::Put;
            else { std::fprintf(stderr, "error: --type must be 'call' or 'put'\n"); return 2; }
        }
        else { std::fprintf(stderr, "error: unknown option %s\n", a.c_str()); usage(); return 2; }
    }

    // Validate up front. Every one of these would otherwise produce a NaN or a
    // silently meaningless price rather than an error.
    if (s0 <= 0.0)    { std::fprintf(stderr, "error: --s0 must be > 0\n"); return 2; }
    if (K <= 0.0)     { std::fprintf(stderr, "error: --k must be > 0\n"); return 2; }
    if (sigma <= 0.0) { std::fprintf(stderr, "error: --sigma must be > 0\n"); return 2; }
    if (T <= 0.0)     { std::fprintf(stderr, "error: --T must be > 0\n"); return 2; }
    if (paths < 2)    { std::fprintf(stderr, "error: --paths must be >= 2\n"); return 2; }

    if (threads == 0) {
        const auto hw = std::thread::hardware_concurrency();
        threads = hw > 0 ? hw : 1;
    }

    market_data md; md.s0 = s0; md.r = r; md.mu = r; md.sigma = sigma;
    option_data od; od.K = K; od.T = T; od.option_type = type;
    sim_data sd; sd.N = 0; sd.seed = static_cast<unsigned int>(seed);
    Option option(md, od, sd);
    ParallelEngine eng(md, od, sd, option);

    if (eng.steps() < 1) {
        std::fprintf(stderr, "error: --T %.6g is shorter than one %.6g-year step\n",
                     T, sd.time_step);
        return 2;
    }

    // Antithetic consumes paths in pairs, so an odd --paths would otherwise
    // quietly simulate one more path than requested.
    const auto pairs = paths / 2;
    const auto effective_paths = want_anti ? pairs * 2 : paths;

    const auto res = want_anti ? eng.price_antithetic(pairs, static_cast<int>(threads))
                               : eng.price(paths, static_cast<int>(threads));

    const auto c = 1.96;
    std::printf("%s  S0 %.6g  K %.6g  r %.6g  sigma %.6g  T %.6g\n",
                type == OptionType::Call ? "call" : "put", s0, K, r, sigma, T);
    std::printf("paths %llu  steps/path %d  threads %llu  seed %llu%s\n",
                (unsigned long long)effective_paths, eng.steps(),
                (unsigned long long)threads, (unsigned long long)seed,
                want_anti ? "  [antithetic]" : "");
    std::printf("\nprice      %.8f\n", res.price);
    std::printf("std_error  %.8f\n", res.std_error);
    std::printf("95%% CI     [%.8f, %.8f]\n",
                res.price - c * res.std_error, res.price + c * res.std_error);

    Black_Scholes bs(0.0, md, od);
    if (want_ref) {
        const auto ref = bs.price(s0);
        std::printf("\nblack_scholes %.8f   diff %+.8f   (%+.2f se)\n",
                    ref, res.price - ref,
                    res.std_error > 0.0 ? (res.price - ref) / res.std_error : 0.0);
    }

    if (want_greeks) {
        // Greeks always come from the plain estimator: the pathwise/LR
        // derivations assume one path per sample, and pairing them would need a
        // separate derivation rather than a reuse of this one.
        const auto g = eng.price_and_greeks(paths, static_cast<int>(threads));
        const auto th = eng.theta_crn(paths, static_cast<int>(threads), theta_h);

        std::printf("\n%-6s %14s %13s", "greek", "value", "std_error");
        if (want_ref) std::printf(" %14s %9s", "black_scholes", "err_se");
        std::printf("\n");

        struct Row { const char* name; GreekEstimate e; double ref; };
        const Row rows[] = {
            {"delta", g.delta, bs.delta(s0)},
            {"gamma", g.gamma, bs.gamma(s0)},
            {"vega",  g.vega,  bs.vega(s0)},
            {"theta", th,      bs.theta(s0)},
            {"rho",   g.rho,   bs.rho(s0)},
        };
        for (const auto& row : rows) {
            std::printf("%-6s %14.8f %13.8f", row.name, row.e.value, row.e.std_error);
            if (want_ref)
                std::printf(" %14.8f %9.2f", row.ref,
                            row.e.std_error > 0.0 ? (row.e.value - row.ref) / row.e.std_error : 0.0);
            std::printf("\n");
        }
    }
    return 0;
}
