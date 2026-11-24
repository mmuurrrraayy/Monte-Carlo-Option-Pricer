#ifndef PRICER_ENGINES_THREAD_POOL_HPP
#define PRICER_ENGINES_THREAD_POOL_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <immintrin.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#endif

namespace pricer {

namespace detail {

#if defined(_WIN32)
using CorePinId = DWORD_PTR;
#else
using CorePinId = std::size_t;
#endif

// One pinning target per physical core (lowest logical CPU in each core/sibling group).
[[nodiscard]] inline std::vector<CorePinId> discover_physical_cores() {
    std::vector<CorePinId> cores;
#if defined(_WIN32)
    DWORD len = 0;
    (void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return cores;
    }
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()),
            &len)) {
        return cores;
    }
    DWORD off = 0;
    while (off < len) {
        const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buf.data() + off);
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount > 0) {
            const DWORD_PTR mask = info->Processor.GroupMask[0].Mask;
            if (mask != 0) {
                // Lowest set bit without unsigned unary minus (MSVC C4146).
                const DWORD_PTR lowest_bit = mask & (~mask + DWORD_PTR{1});
                cores.push_back(lowest_bit);
            }
        }
        if (info->Size == 0) {
            break;
        }
        off += info->Size;
    }
#elif defined(__linux__)
    DIR* dir = ::opendir("/sys/devices/system/cpu");
    if (dir == nullptr) {
        return cores;
    }
    while (const struct dirent* ent = ::readdir(dir)) {
        const char* name = ent->d_name;
        if (std::strncmp(name, "cpu", 3) != 0 || name[3] == '\0') {
            continue;
        }
        bool digits_only = true;
        for (const char* p = name + 3; *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') {
                digits_only = false;
                break;
            }
        }
        if (!digits_only) {
            continue;
        }

        char path[256];
        const int written = std::snprintf(
            path, sizeof(path),
            "/sys/devices/system/cpu/%s/topology/thread_siblings_list", name);
        if (written <= 0 || static_cast<std::size_t>(written) >= sizeof(path)) {
            continue;
        }

        std::FILE* fp = std::fopen(path, "r");
        if (fp == nullptr) {
            continue;
        }
        char line[256];
        const bool ok = (std::fgets(line, sizeof(line), fp) != nullptr);
        std::fclose(fp);
        if (!ok) {
            continue;
        }

        bool found_any = false;
        std::size_t min_cpu = 0;
        const char* cur = line;
        while (*cur != '\0') {
            while (*cur != '\0' && (*cur < '0' || *cur > '9')) {
                ++cur;
            }
            if (*cur == '\0') {
                break;
            }
            char* end = nullptr;
            const unsigned long val = std::strtoul(cur, &end, 10);
            if (end == cur) {
                break;
            }
            const auto cpu_id = static_cast<std::size_t>(val);
            if (!found_any || cpu_id < min_cpu) {
                min_cpu = cpu_id;
                found_any = true;
            }
            cur = end;
        }
        if (found_any) {
            cores.push_back(min_cpu);
        }
    }
    ::closedir(dir);

    std::sort(cores.begin(), cores.end());
    cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
#endif
    return cores;
}

inline void pin_thread_to_target(CorePinId target) noexcept {
#if defined(_WIN32)
    if (target != 0) {
        (void)SetThreadAffinityMask(GetCurrentThread(), target);
    }
#elif defined(__linux__)
    if (target < static_cast<std::size_t>(CPU_SETSIZE)) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(static_cast<int>(target), &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
    }
#else
    (void)target;
#endif
}

} // namespace detail

[[nodiscard]] inline std::size_t physical_core_count() {
    const std::size_t phys = detail::discover_physical_cores().size();
    if (phys > 0) {
        return phys;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<std::size_t>(hw) : std::size_t{1};
}

// Persistent pool owning (pool_size - 1) workers; the caller runs as worker 0
// with its original CPU affinity restored via RAII on return.
// Not thread-safe or reentrant: at most one parallel_for call may be active at a time.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads = physical_core_count(),
                        bool pin_to_physical_cores = true)
        : pool_size_(num_threads > 0 ? num_threads : std::size_t{1}),
          pin_to_physical_cores_(pin_to_physical_cores),
          core_ids_(pin_to_physical_cores ? detail::discover_physical_cores()
                                          : std::vector<detail::CorePinId>{}) {
        const std::size_t num_workers = pool_size_ - 1;
        if (num_workers > 0) {
            slots_ = std::make_unique<WorkerSlot[]>(num_workers);
            workers_.reserve(num_workers);

            // If spawning worker k throws, ~ThreadPool() won't run before ~vector<jthread>()
            // joins workers 0..k-1; signal stop on unwind to prevent deadlock.
            struct SpawnRollbackGuard {
                ThreadPool* self;
                ~SpawnRollbackGuard() noexcept {
                    if (self != nullptr) {
                        self->shutdown_workers();
                    }
                }
            } rollback{this};

            for (std::size_t i = 0; i < num_workers; ++i) {
                workers_.emplace_back([this, i]() noexcept { worker_loop(i); });
            }
            rollback.self = nullptr;
        }
    }

    ~ThreadPool() noexcept {
        shutdown_workers();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return pool_size_; }
    [[nodiscard]] bool pin_to_physical_cores() const noexcept { return pin_to_physical_cores_; }

    // active_threads == 0 uses pool_size_. Single caller only; fn must not re-enter.
    template <typename Fn>
    void parallel_for(std::size_t num_chunks, std::size_t active_threads, Fn&& fn) noexcept {
        if (num_chunks == 0) {
            return;
        }
        const std::size_t requested = (active_threads == 0) ? pool_size_ : active_threads;
        const std::size_t active = (std::min)({requested, pool_size_, num_chunks});

        if (active <= 1) {
            for (std::size_t c = 0; c < num_chunks; ++c) {
                fn(c);
            }
            return;
        }

        const CallerAffinityGuard affinity_guard(*this);

        using FnType = std::remove_reference_t<Fn>;
        dispatch_.thunk = +[](void* ctx, std::size_t c) noexcept {
            (*static_cast<FnType*>(ctx))(c);
        };
        dispatch_.ctx = static_cast<void*>(&fn);
        dispatch_.num_chunks = num_chunks;

        const std::size_t wake_count = active - 1;
        work_.next_chunk.store(0, std::memory_order_relaxed);
        done_.remaining.store(static_cast<std::uint32_t>(wake_count), std::memory_order_relaxed);

        for (std::size_t i = 0; i < wake_count; ++i) {
            slots_[i].epoch.fetch_add(1, std::memory_order_release);
            slots_[i].epoch.notify_one();
        }

        for (;;) {
            const std::size_t c = work_.next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (c >= num_chunks) {
                break;
            }
            fn(c);
        }

        int spins = 0;
        for (;;) {
            const std::uint32_t rem = done_.remaining.load(std::memory_order_acquire);
            if (rem == 0) {
                break;
            }
            if (spins < kSpinIters) {
                _mm_pause();
                ++spins;
            } else {
                done_.remaining.wait(rem, std::memory_order_acquire);
            }
        }
    }

private:
    static constexpr int kSpinIters = 1000;

    using ThunkFn = void (*)(void*, std::size_t) noexcept;

    struct alignas(64) DispatchState {
        ThunkFn thunk = nullptr;
        void* ctx = nullptr;
        std::size_t num_chunks = 0;
        bool stop = false;
    };

    struct alignas(64) WorkCounter {
        std::atomic<std::size_t> next_chunk{0};
    };

    struct alignas(64) DoneCounter {
        std::atomic<std::uint32_t> remaining{0};
    };

    struct alignas(64) WorkerSlot {
        std::atomic<std::uint64_t> epoch{0};
    };

    class CallerAffinityGuard {
    public:
        explicit CallerAffinityGuard(const ThreadPool& pool) noexcept {
            if (!pool.pin_to_physical_cores_ || pool.core_ids_.empty()) {
                return;
            }
#if defined(_WIN32)
            const DWORD_PTR target = pool.core_ids_[0];
            if (target != 0) {
                prev_mask_ = SetThreadAffinityMask(GetCurrentThread(), target);
            }
#elif defined(__linux__)
            const std::size_t target_cpu = pool.core_ids_[0];
            if (target_cpu < static_cast<std::size_t>(CPU_SETSIZE) &&
                pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &prev_set_) == 0) {
                cpu_set_t target;
                CPU_ZERO(&target);
                CPU_SET(static_cast<int>(target_cpu), &target);
                if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &target) == 0) {
                    has_prev_ = true;
                }
            }
#endif
        }

        ~CallerAffinityGuard() noexcept {
#if defined(_WIN32)
            if (prev_mask_ != 0) {
                (void)SetThreadAffinityMask(GetCurrentThread(), prev_mask_);
            }
#elif defined(__linux__)
            if (has_prev_) {
                (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &prev_set_);
            }
#endif
        }

        CallerAffinityGuard(const CallerAffinityGuard&) = delete;
        CallerAffinityGuard& operator=(const CallerAffinityGuard&) = delete;

    private:
#if defined(_WIN32)
        DWORD_PTR prev_mask_ = 0;
#elif defined(__linux__)
        cpu_set_t prev_set_{};
        bool has_prev_ = false;
#endif
    };

    void shutdown_workers() noexcept {
        const std::size_t num_workers = workers_.size();
        if (num_workers > 0) {
            dispatch_.stop = true;
            for (std::size_t i = 0; i < num_workers; ++i) {
                slots_[i].epoch.fetch_add(1, std::memory_order_release);
                slots_[i].epoch.notify_one();
            }
            workers_.clear(); // joins all std::jthreads before member teardown
        }
    }

    void worker_loop(std::size_t slot_idx) noexcept {
        if (pin_to_physical_cores_ && !core_ids_.empty()) {
            const detail::CorePinId target = core_ids_[(slot_idx + 1) % core_ids_.size()];
            detail::pin_thread_to_target(target);
        }

        std::uint64_t seen_epoch = 0;
        for (;;) {
            std::uint64_t cur = slots_[slot_idx].epoch.load(std::memory_order_acquire);
            if (cur == seen_epoch) {
                for (int spin = 0; spin < kSpinIters; ++spin) {
                    _mm_pause();
                    cur = slots_[slot_idx].epoch.load(std::memory_order_acquire);
                    if (cur != seen_epoch) {
                        break;
                    }
                }
                while (cur == seen_epoch) {
                    slots_[slot_idx].epoch.wait(seen_epoch, std::memory_order_acquire);
                    cur = slots_[slot_idx].epoch.load(std::memory_order_acquire);
                }
            }
            seen_epoch = cur;

            if (dispatch_.stop) {
                return;
            }

            const ThunkFn thunk = dispatch_.thunk;
            void* const ctx = dispatch_.ctx;
            const std::size_t n = dispatch_.num_chunks;

            for (;;) {
                const std::size_t c = work_.next_chunk.fetch_add(1, std::memory_order_relaxed);
                if (c >= n) {
                    break;
                }
                thunk(ctx, c);
            }

            if (done_.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                done_.remaining.notify_one();
            }
        }
    }

    std::size_t pool_size_ = 1;
    bool pin_to_physical_cores_ = true;
    std::vector<detail::CorePinId> core_ids_;

    DispatchState dispatch_{};
    WorkCounter work_{};
    DoneCounter done_{};
    std::unique_ptr<WorkerSlot[]> slots_;
    std::vector<std::jthread> workers_; // declared last so workers join before teardown
};

} // namespace pricer

#endif // PRICER_ENGINES_THREAD_POOL_HPP
