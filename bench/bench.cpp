// Benchmark + RNG verification harness.
//   bench calibrate            -> throughput probe, prints paths/sec for 1 thread
//   bench sweep <N>            -> thread sweep, median of 5 after 1 warm-up
//   bench converge <R>         -> Black-Scholes convergence + 1/sqrt(N) SE fit
//   bench repro <N>            -> same seed across thread counts, bitwise compare
//   bench anti <pairs> [K]     -> antithetic vs plain at equal work, strike K

#include "mcpricer/DataVariables.h"
#include "mcpricer/Option.h"
#include "mcpricer/Black-Scholes.h"
#include "mcpricer/ParallelEngine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif

// ---------------------------------------------------------------- machine info

// One affinity mask per PHYSICAL core. hardware_concurrency() would count SMT
// siblings, and an unpinned thread may land on a sibling of an already-busy
// core, which halves that core's throughput for no visible reason.
static std::vector<ULONG_PTR> core_masks() {
    std::vector<ULONG_PTR> masks;
#ifdef _WIN32
    auto len = DWORD{0};
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return masks;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()), &len))
        return masks;
    auto off = DWORD{0};
    while (off < len) {
        auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (p->Relationship == RelationProcessorCore && p->Processor.GroupCount > 0)
            masks.push_back(p->Processor.GroupMask[0].Mask);
        off += p->Size;
    }
#endif
    return masks;
}

static int physical_cores() { return static_cast<int>(core_masks().size()); }

// Pin the calling thread to one logical processor of physical core `idx`.
bool g_pin = false;
void pin_to_core(int idx) {
#ifdef _WIN32
    if (!g_pin) return;
    const auto m = core_masks();
    if (m.empty()) return;
    auto mask = m[idx % m.size()];
    mask &= ~(mask - 1);  // lowest set bit: one logical proc, not both siblings
    SetThreadAffinityMask(GetCurrentThread(), mask);
#endif
}

static std::string cpu_brand() {
#ifdef _WIN32
    int regs[4];
    char brand[49] = {0};
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return "unknown";
    for (int i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002 + i);
        std::memcpy(brand + i * 16, regs, 16);
    }
    auto s = std::string(brand);
    while (!s.empty() && s.back() == ' ') s.pop_back();
    const auto b = s.find_first_not_of(' ');
    return b == std::string::npos ? s : s.substr(b);
#else
    return "unknown";
#endif
}

// ---------------------------------------------------------------- problem setup

struct Setup {
    market_data md;
    option_data od;
    sim_data sd;
    Option option;
};

// The strike is a parameter rather than a post-construction assignment because
// sim_data::time_step is const, which deletes Option's copy-assignment operator;
// a Setup can be built but not reassigned.
static Setup make_setup(unsigned int seed, double strike = 100.0) {
    market_data md;
    md.s0 = 100.0;
    md.r = 0.05;
    md.mu = 0.05;   // unused by the pricer now, kept so the struct stays honest
    md.sigma = 0.20;
    option_data od;
    od.K = strike;
    od.T = 1.0;
    od.option_type = OptionType::Call;
    sim_data sd;
    sd.N = 0;
    sd.seed = seed;
    return Setup{md, od, sd, Option(md, od, sd)};
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const auto n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Times ONLY the compute region. Engine construction, the accumulator vector,
// and all printing sit outside the steady_clock window. Thread spawn/join IS
// inside it, because that cost is intrinsic to running with p threads.
static double time_once(const ParallelEngine& eng, std::uint64_t N, int threads,
                        PriceResult& out) {
    const auto t0 = std::chrono::steady_clock::now();
    out = eng.price(N, threads);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ---------------------------------------------------------------------- modes

static void mode_calibrate() {
    auto s = make_setup(42);
    ParallelEngine eng(s.md, s.od, s.sd, s.option);
    auto probe = std::uint64_t{20000};
    PriceResult r;
    time_once(eng, probe, 1, r);  // warm
    const auto t = time_once(eng, probe, 1, r);
    std::printf("steps_per_path %d\n", eng.steps());
    std::printf("probe_paths %llu\n", (unsigned long long)probe);
    std::printf("probe_seconds %.6f\n", t);
    std::printf("paths_per_sec_1t %.1f\n", probe / t);
    std::printf("N_for_2s %.0f\n", 2.0 * probe / t);
}

static void mode_sweep(std::uint64_t N) {
    auto s = make_setup(42);
    ParallelEngine eng(s.md, s.od, s.sd, s.option);
    eng.on_thread_start = pin_to_core;
    const auto cores = physical_cores();

    std::vector<int> counts;
    for (int t = 1; t <= cores; t *= 2) counts.push_back(t);
    if (counts.empty()) counts.push_back(1);

    std::printf("cpu %s\n", cpu_brand().c_str());
    std::printf("physical_cores %d\n", cores);
    std::printf("logical_procs %u\n", std::thread::hardware_concurrency());
    std::printf("N %llu\n", (unsigned long long)N);
    std::printf("steps_per_path %d\n", eng.steps());

    // Interleaved: round-major, not thread-count-major. A blocked schedule runs
    // all five 1-thread timings on a cold chip and all five 4-thread timings on a
    // hot one, so thermal drift is aliased directly into the speedup. Round-robin
    // spreads any drift evenly across every thread count.
    std::vector<PriceResult> last(counts.size());
    std::vector<std::vector<double>> runs(counts.size());
    for (int round = 0; round < 6; ++round) {
        for (size_t c = 0; c < counts.size(); ++c) {
            const auto dt = time_once(eng, N, counts[c], last[c]);
            if (round > 0) runs[c].push_back(dt);  // round 0 discarded as warm-up
        }
    }

    auto t1 = 0.0;
    for (size_t c = 0; c < counts.size(); ++c) {
        const auto med = median(runs[c]);
        if (counts[c] == 1) t1 = med;
        std::printf("thread %d median %.6f speedup %.4f efficiency %.4f price %.10f",
                    counts[c], med, t1 / med, (t1 / med) / counts[c], last[c].price);
        std::printf(" runs");
        for (double x : runs[c]) std::printf(" %.6f", x);
        std::printf("\n");
    }
}

static void mode_converge(int reps) {
    // Observed standard error is measured EMPIRICALLY: for each N, run `reps`
    // independent replications (distinct seeds -> distinct streams) and take the
    // sample standard deviation of the resulting prices. That is a real test of
    // stream independence -- correlated streams would not shrink as 1/sqrt(N).
    const std::uint64_t Ns[] = {1000, 10000, 100000, 1000000};
    const auto nN = 4;
    const auto cores = physical_cores() > 0 ? physical_cores() : 1;

    auto ref = make_setup(42);
    Black_Scholes bs(0.0, ref.md, ref.od);
    const auto closed_form = bs.price(ref.md.s0);
    std::printf("black_scholes %.10f\n", closed_form);
    std::printf("reps %d\n", reps);

    std::vector<double> lx, ly;
    for (int k = 0; k < nN; ++k) {
        const auto N = Ns[k];
        std::vector<double> prices;
        for (int rep = 0; rep < reps; ++rep) {
            auto s = make_setup(1000u + 7919u * static_cast<unsigned>(rep));
            ParallelEngine eng(s.md, s.od, s.sd, s.option);
            prices.push_back(eng.price(N, cores).price);
        }
        auto mean = 0.0;
        for (double p : prices) mean += p;
        mean /= prices.size();
        auto var = 0.0;
        for (double p : prices) var += (p - mean) * (p - mean);
        var /= (prices.size() - 1);
        const auto se = std::sqrt(var);

        // Also report the internal per-run SE estimate for one replication, as a
        // cross-check that the engine's own error bar agrees with the observed
        // spread across replications.
        auto s0 = make_setup(1000u);
        ParallelEngine eng0(s0.md, s0.od, s0.sd, s0.option);
        const auto one = eng0.price(N, cores);

        std::printf("N %llu mean %.10f bias %.10f observed_se %.10f internal_se %.10f\n",
                    (unsigned long long)N, mean, mean - closed_form, se, one.std_error);
        lx.push_back(std::log(static_cast<double>(N)));
        ly.push_back(std::log(se));
    }

    // Least-squares slope of log(SE) on log(N); theory says -0.5.
    auto sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    const auto n = nN;
    for (int i = 0; i < n; ++i) {
        sx += lx[i]; sy += ly[i]; sxx += lx[i] * lx[i]; sxy += lx[i] * ly[i];
    }
    const auto slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const auto icpt = (sy - slope * sx) / n;
    auto ss_res = 0.0, ss_tot = 0.0;
    const auto ybar = sy / n;
    for (int i = 0; i < n; ++i) {
        const auto fit = icpt + slope * lx[i];
        ss_res += (ly[i] - fit) * (ly[i] - fit);
        ss_tot += (ly[i] - ybar) * (ly[i] - ybar);
    }
    std::printf("fit_exponent %.6f r_squared %.6f\n", slope, 1.0 - ss_res / ss_tot);
}

// Antithetic vs plain at EQUAL WORK.
//
// The comparison that matters is not "does pairing shrink the standard error"
// (it must, by construction) but "does it shrink it by more than simply
// spending the same paths on the plain estimator would". So `pairs` pair-
// averages (= 2*pairs simulated paths) is timed against plain at 2*pairs paths.
//
// Two figures are reported:
//   var_ratio     -- (se_plain / se_anti)^2 at equal PATH COUNT. >1 means
//                    antithetic wins on variance per path.
//   efficiency    -- (se_plain^2 * t_plain) / (se_anti^2 * t_anti), the standard
//                    work-normalised MC efficiency. This is the honest number:
//                    it charges antithetic for the extra exp() per step, so it
//                    can come out below var_ratio, or below 1 entirely.
// The strike is a parameter because the size of the antithetic gain is not a
// property of the technique alone -- it depends on how nearly monotone the
// payoff is in Z over the region carrying the variance. Sweeping K is what
// turns "antithetic helps" into a claim with a stated range of validity.
static void mode_anti(std::uint64_t pairs, double strike) {
    auto s = make_setup(42, strike);
    ParallelEngine eng(s.md, s.od, s.sd, s.option);
    eng.on_thread_start = pin_to_core;
    const auto threads = physical_cores() > 0 ? physical_cores() : 1;
    const auto N_plain = 2 * pairs;

    Black_Scholes bs(0.0, s.md, s.od);
    const auto closed_form = bs.price(s.md.s0);

    PriceResult plain{}, anti{};
    std::vector<double> tp, ta;
    // Round-robin, first round discarded -- same protocol as mode_sweep, so
    // thermal drift is spread across both estimators instead of favouring one.
    for (int round = 0; round < 6; ++round) {
        auto t0 = std::chrono::steady_clock::now();
        plain = eng.price(N_plain, threads);
        auto t1 = std::chrono::steady_clock::now();
        anti = eng.price_antithetic(pairs, threads);
        auto t2 = std::chrono::steady_clock::now();
        if (round > 0) {
            tp.push_back(std::chrono::duration<double>(t1 - t0).count());
            ta.push_back(std::chrono::duration<double>(t2 - t1).count());
        }
    }

    const auto t_plain = median(tp), t_anti = median(ta);
    const auto var_ratio = (plain.std_error * plain.std_error)
                         / (anti.std_error * anti.std_error);
    const auto efficiency = var_ratio * (t_plain / t_anti);

    std::printf("black_scholes %.10f\n", closed_form);
    std::printf("threads %d\n", threads);
    std::printf("pairs %llu paths_equivalent %llu\n",
                (unsigned long long)pairs, (unsigned long long)N_plain);
    std::printf("plain  paths %llu price %.10f se %.10f err %+.10f time %.6f\n",
                (unsigned long long)N_plain, plain.price, plain.std_error,
                plain.price - closed_form, t_plain);
    std::printf("anti   pairs %llu price %.10f se %.10f err %+.10f time %.6f\n",
                (unsigned long long)pairs, anti.price, anti.std_error,
                anti.price - closed_form, t_anti);
    std::printf("var_ratio %.4f\n", var_ratio);
    std::printf("time_ratio_anti_over_plain %.4f\n", t_anti / t_plain);
    std::printf("efficiency %.4f\n", efficiency);
}

static void mode_repro(std::uint64_t N) {
    auto s = make_setup(42);
    ParallelEngine eng(s.md, s.od, s.sd, s.option);
    eng.on_thread_start = pin_to_core;
    const auto cores = physical_cores();
    auto base = 0.0;
    auto first = true;
    for (int t = 1; t <= cores; t *= 2) {
        const auto r = eng.price(N, t);
        if (first) { base = r.price; first = false; }
        const auto d = r.price - base;
        std::printf("threads %d price %.17g delta %.3g bitwise_equal %d\n",
                    t, r.price, d, r.price == base ? 1 : 0);
    }
}

// Sustained load, so an external sampler can read the actual clock while the
// benchmark's own hot loop is what's running.
static void mode_load(int threads, double seconds) {
    auto s = make_setup(42);
    ParallelEngine eng(s.md, s.od, s.sd, s.option);
    eng.on_thread_start = pin_to_core;
    const auto t0 = std::chrono::steady_clock::now();
    auto paths = std::uint64_t{0};
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        eng.price(50000, threads);
        paths += 50000;
    }
    const auto el = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("threads %d elapsed %.3f paths %llu paths_per_sec %.1f per_core %.1f\n",
                threads, el, (unsigned long long)paths, paths / el, paths / el / threads);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "pin") == 0) g_pin = true;
    const auto mode = std::string(argc > 1 ? argv[1] : "calibrate");
    if (mode == "calibrate") mode_calibrate();
    else if (mode == "sweep") mode_sweep(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 500000);
    else if (mode == "converge") mode_converge(argc > 2 ? std::atoi(argv[2]) : 32);
    else if (mode == "load") mode_load(argc > 2 ? std::atoi(argv[2]) : 1,
                                       argc > 3 ? std::atof(argv[3]) : 20.0);
    else if (mode == "repro") mode_repro(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 200000);
    else if (mode == "anti") mode_anti(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 250000,
                                      argc > 3 ? std::atof(argv[3]) : 100.0);
    else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 1; }
    return 0;
}
