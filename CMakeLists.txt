cmake_minimum_required(VERSION 3.25)
project(MonteCarloOptionPricer LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

option(PRICER_USE_CXX26 "Build pricer targets with C++26 instead of C++23" OFF)
option(PRICER_MARCH_NATIVE "Tune with -march=native instead of portable -march=x86-64-v3" OFF)
option(PRICER_ENABLE_SANITIZERS "Enable ASan + UBSan on GCC/Clang" OFF)
option(PRICER_FAST_TESTS "Use reduced multi-step MC path counts for fast Debug/Sanitizer runs" OFF)
option(PRICER_USE_GOOGLE_BENCHMARK "Fetch Google Benchmark v1.9.1" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    if(PRICER_ENABLE_SANITIZERS)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING
            "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    else()
        set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    endif()
endif()

if(PRICER_ENABLE_SANITIZERS AND MSVC)
    message(FATAL_ERROR "PRICER_ENABLE_SANITIZERS is only supported on GCC and Clang")
endif()

find_package(Threads REQUIRED)
include(CheckCXXSourceCompiles)

add_library(pricer INTERFACE)
target_include_directories(pricer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pricer INTERFACE Threads::Threads)
if(PRICER_USE_CXX26)
    target_compile_features(pricer INTERFACE cxx_std_26)
else()
    target_compile_features(pricer INTERFACE cxx_std_23)
endif()

add_library(pricer_flags INTERFACE)
add_library(pricer_no_exceptions_flags INTERFACE)

if(MSVC)
    target_compile_options(pricer_flags INTERFACE
        /W4
        /wd4324
        /fp:precise
        /arch:AVX2
        /Zc:__cplusplus
        $<$<CONFIG:Release>:/O2;/GL>
    )
    target_link_options(pricer_flags INTERFACE
        $<$<CONFIG:Release>:/LTCG>
    )
    target_compile_options(pricer_no_exceptions_flags INTERFACE
        /EHs-c-
        /D_HAS_EXCEPTIONS=0
        /GR-
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(PRICER_MARCH_NATIVE)
        set(_pricer_arch_flag -march=native)
    else()
        set(_pricer_arch_flag -march=x86-64-v3)
    endif()

    # Probe linker LTO support (e.g. MSYS2 Clang may lack lld); disabled under sanitisers.
    set(_pricer_lto_compile_flags "")
    set(_pricer_lto_link_flags "")
    if(NOT PRICER_ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            string(REGEX MATCH "^[0-9]+" _pricer_clang_major "${CMAKE_CXX_COMPILER_VERSION}")
            set(CMAKE_REQUIRED_FLAGS "-flto")
            set(CMAKE_REQUIRED_LINK_OPTIONS "-flto;-fuse-ld=lld-${_pricer_clang_major}")
            check_cxx_source_compiles("int main() { return 0; }" PRICER_CLANG_LTO_LLD_VERSIONED)
            if(PRICER_CLANG_LTO_LLD_VERSIONED)
                set(_pricer_lto_compile_flags -flto)
                set(_pricer_lto_link_flags -flto "-fuse-ld=lld-${_pricer_clang_major}")
            else()
                set(CMAKE_REQUIRED_FLAGS "-flto")
                set(CMAKE_REQUIRED_LINK_OPTIONS "-flto;-fuse-ld=lld")
                check_cxx_source_compiles("int main() { return 0; }" PRICER_CLANG_LTO_LLD)
                if(PRICER_CLANG_LTO_LLD)
                    set(_pricer_lto_compile_flags -flto)
                    set(_pricer_lto_link_flags -flto -fuse-ld=lld)
                else()
                    set(CMAKE_REQUIRED_FLAGS "-flto")
                    set(CMAKE_REQUIRED_LINK_OPTIONS "-flto")
                    check_cxx_source_compiles("int main() { return 0; }" PRICER_CLANG_LTO_DEFAULT)
                    if(PRICER_CLANG_LTO_DEFAULT)
                        set(_pricer_lto_compile_flags -flto)
                        set(_pricer_lto_link_flags -flto)
                    endif()
                endif()
            endif()
        else()
            set(CMAKE_REQUIRED_FLAGS "-flto=auto")
            set(CMAKE_REQUIRED_LINK_OPTIONS "-flto=auto")
            check_cxx_source_compiles("int main() { return 0; }" PRICER_GCC_LTO_AUTO)
            if(PRICER_GCC_LTO_AUTO)
                set(_pricer_lto_compile_flags -flto=auto)
                set(_pricer_lto_link_flags -flto=auto)
            else()
                set(CMAKE_REQUIRED_FLAGS "-flto")
                set(CMAKE_REQUIRED_LINK_OPTIONS "-flto")
                check_cxx_source_compiles("int main() { return 0; }" PRICER_GCC_LTO)
                if(PRICER_GCC_LTO)
                    set(_pricer_lto_compile_flags -flto)
                    set(_pricer_lto_link_flags -flto)
                endif()
            endif()
        endif()
        unset(CMAKE_REQUIRED_FLAGS)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
    endif()

    target_compile_options(pricer_flags INTERFACE
        ${_pricer_arch_flag}
        -ffp-contract=off
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        $<$<CONFIG:Release>:-O3;-fno-math-errno;-fno-trapping-math;${_pricer_lto_compile_flags}>
    )
    target_link_options(pricer_flags INTERFACE
        $<$<CONFIG:Release>:${_pricer_lto_link_flags}>
    )

    if(PRICER_ENABLE_SANITIZERS)
        target_compile_options(pricer_flags INTERFACE
            $<$<CONFIG:Debug>:-O1>
            -g
            -fno-omit-frame-pointer
            -fsanitize=address,undefined
        )
        target_link_options(pricer_flags INTERFACE
            -fsanitize=address,undefined
        )
    else()
        target_compile_options(pricer_flags INTERFACE
            $<$<CONFIG:Debug>:-O0;-g>
        )
    endif()

    target_compile_options(pricer_no_exceptions_flags INTERFACE
        -fno-exceptions
        -fno-rtti
    )
endif()

# Compiled without AVX2/FMA/LTO so it can run safely on pre-AVX2 CPUs.
add_library(pricer_cpuid_guard STATIC apps/cpuid_guard.cpp)
target_include_directories(pricer_cpuid_guard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps)
if(PRICER_USE_CXX26)
    target_compile_features(pricer_cpuid_guard PRIVATE cxx_std_26)
else()
    target_compile_features(pricer_cpuid_guard PRIVATE cxx_std_23)
endif()
if(MSVC)
    target_compile_options(pricer_cpuid_guard PRIVATE
        /W4
        /Zc:__cplusplus
        $<$<CONFIG:Release>:/O2>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pricer_cpuid_guard PRIVATE
        -mno-avx
        -mno-avx2
        -mno-fma
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        $<$<CONFIG:Release>:-O2>
    )
    if(PRICER_ENABLE_SANITIZERS)
        target_compile_options(pricer_cpuid_guard PRIVATE
            $<$<CONFIG:Debug>:-O1>
            -g
            -fno-omit-frame-pointer
            -fsanitize=address,undefined
        )
    else()
        target_compile_options(pricer_cpuid_guard PRIVATE
            $<$<CONFIG:Debug>:-O0;-g>
        )
    endif()
endif()

if(PRICER_USE_GOOGLE_BENCHMARK)
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
    )
    FetchContent_MakeAvailable(googlebenchmark)
endif()

add_executable(price apps/price.cpp)
target_link_libraries(price PRIVATE pricer pricer_flags pricer_cpuid_guard)

add_executable(bench_pricer bench/bench_pricer.cpp)
target_link_libraries(bench_pricer PRIVATE pricer pricer_flags pricer_cpuid_guard)
if(PRICER_USE_GOOGLE_BENCHMARK)
    target_link_libraries(bench_pricer PRIVATE benchmark::benchmark)
    target_compile_definitions(bench_pricer PRIVATE PRICER_USE_GOOGLE_BENCHMARK=1)
endif()

enable_testing()

add_executable(pricer_tests tests/test_pricer.cpp)
target_link_libraries(pricer_tests PRIVATE pricer pricer_flags pricer_cpuid_guard)
if(PRICER_FAST_TESTS OR PRICER_ENABLE_SANITIZERS)
    target_compile_definitions(pricer_tests PRIVATE PRICER_FAST_TESTS=1)
endif()
add_test(NAME pricer_tests COMMAND pricer_tests)

add_executable(test_no_exceptions tests/test_no_exceptions.cpp)
target_link_libraries(test_no_exceptions PRIVATE
    pricer
    pricer_flags
    pricer_no_exceptions_flags
    pricer_cpuid_guard
)
add_test(NAME test_no_exceptions COMMAND test_no_exceptions)

add_test(NAME price_smoke COMMAND price --paths 20000 --threads 1)
