cmake_minimum_required(VERSION 3.15)
project(MonteCarloOptionPricer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Compile as -std=c++17, not -std=gnu++17. Without this CMake defaults to the
# GNU dialect, which would let a compiler extension slip in unnoticed and
# silently break the "C++17" claim on a non-GNU toolchain.
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release when the user has not chosen a build type. CACHE ... FORCE
# is required: a plain set() creates a normal variable that the generator honours
# but never writes to CMakeCache.txt, so `cmake -L` and IDEs would report the
# build type as empty. Guarded on CMAKE_CONFIGURATION_TYPES because the variable
# is meaningless for multi-config generators (Visual Studio, Ninja Multi-Config),
# which pick the configuration at build time instead.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

find_package(Threads REQUIRED)

add_library(mcpricer STATIC
    src/Option.cpp
    src/GBM.cpp
    src/MonteCarloEngine.cpp
    src/AntitheticEngine.cpp
    src/Precision.cpp
    src/Black-Scholes.cpp
    src/ParallelEngine.cpp
)
target_include_directories(mcpricer PUBLIC include)
target_link_libraries(mcpricer PUBLIC Threads::Threads)

# The user-facing pricer.
add_executable(price apps/price.cpp)
target_link_libraries(price PRIVATE mcpricer)

# Performance benchmark harness (bench/) -- not run as part of `ctest`, since
# throughput numbers aren't pass/fail; run manually via the built executable.
add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE mcpricer)

# Correctness checks (tests/) -- gated assertions against closed-form
# Black-Scholes values and analytic identities. Every check runs through
# tests/check.h, which counts failures and returns a non-zero exit status, so a
# regression turns ctest red rather than merely printing a bad number.
enable_testing()

add_executable(check_greeks tests/check_greeks.cpp)
target_link_libraries(check_greeks PRIVATE mcpricer)
add_test(NAME check_greeks COMMAND check_greeks)

add_executable(check_serial tests/check_serial.cpp)
target_link_libraries(check_serial PRIVATE mcpricer)
add_test(NAME check_serial COMMAND check_serial)

# CLI smoke test -- runs through ctest rather than an OS-specific shell step,
# so the Debug/Release output subdirectory that MSVC's multi-config generator
# adds (build/Release/price.exe vs. build/price on single-config generators)
# is resolved by ctest itself instead of being hard-coded here.
add_test(NAME price_smoke COMMAND price --paths 20000 --threads 1)
