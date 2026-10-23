// newton-vision Dynamic Runtime CPUID Dispatch & SIMD Engine.
// Güvenli dinamik CPU Tespiti (Dynamic CPUID Dispatching) — Eski CPU ve ARM'larda çökme riski sıfırlanmıştır.
#pragma once
#include <cstdint>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#include <immintrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#endif

namespace nv {

inline bool has_avx2_hardware() {
#if defined(__GNUC__) || defined(__clang__)
    #if __has_builtin(__builtin_cpu_supports)
        return __builtin_cpu_supports("avx2");
    #else
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            return (ebx & (1 << 5)) != 0;
        }
        return false;
    #endif
#elif defined(_MSC_VER)
    int cpuInfo[4] = {0};
    __cpuidex(cpuInfo, 7, 0);
    return (cpuInfo[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}

#if (defined(__GNUC__) || defined(__clang__)) && !defined(__AVX2__)
__attribute__((target("avx2,fma")))
#endif
inline void gemm_avx2_kernel(int M, int N, int K, const float* A, const float* B, float* C, const float* bias) {
#if defined(__x86_64__) || defined(_M_X64)
    #pragma omp parallel for if(!omp_in_parallel())
    for (int m = 0; m < M; ++m) {
        float b_val = bias ? bias[m] : 0.0f;
        int n = 0;

        for (; n <= N - 8; n += 8) {
            __m256 c_vec = _mm256_set1_ps(b_val);
            for (int k = 0; k < K; ++k) {
                __m256 a_vec = _mm256_set1_ps(A[m * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + n]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            }
            _mm256_storeu_ps(&C[m * N + n], c_vec);
        }

        for (; n < N; ++n) {
            float sum = b_val;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
#else
    #pragma omp parallel for if(!omp_in_parallel())
    for (int m = 0; m < M; ++m) {
        float b_val = bias ? bias[m] : 0.0f;
        for (int n = 0; n < N; ++n) {
            C[m * N + n] = b_val;
        }
        for (int k = 0; k < K; ++k) {
            float a_val = A[m * K + k];
            for (int n = 0; n < N; ++n) {
                C[m * N + n] += a_val * B[k * N + n];
            }
        }
    }
#endif
}

inline void gemm_avx2(int M, int N, int K, const float* A, const float* B, float* C, const float* bias = nullptr) {
    static bool supports_avx2 = has_avx2_hardware();
    if (supports_avx2) {
        gemm_avx2_kernel(M, N, K, A, B, C, bias);
    } else {
        #pragma omp parallel for if(!omp_in_parallel())
        for (int m = 0; m < M; ++m) {
            float b_val = bias ? bias[m] : 0.0f;
            for (int n = 0; n < N; ++n) {
                C[m * N + n] = b_val;
            }
            for (int k = 0; k < K; ++k) {
                float a_val = A[m * K + k];
                for (int n = 0; n < N; ++n) {
                    C[m * N + n] += a_val * B[k * N + n];
                }
            }
        }
    }
}

inline void gemm_int8_cpu(
    int M, int N, int K,
    const int8_t* A, const int8_t* B, float* C,
    float scale_a, float scale_b, const float* bias = nullptr
) {
    float combined_scale = scale_a * scale_b;
    #pragma omp parallel for if(!omp_in_parallel())
    for (int m = 0; m < M; ++m) {
        float b_val = bias ? bias[m] : 0.0f;
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<float>(acc) * combined_scale + b_val;
        }
    }
}

} // namespace nv
