#pragma once

// Compiled without AVX2/FMA so startup checks stay safe on pre-AVX2 CPUs.
bool pricer_has_avx2_fma() noexcept;
