#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx512f,avx512cd,avx512bw,avx512dq,avx512vl,bmi,bmi2,lzcnt,popcnt")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <omp.h>

namespace solution {
    // Block sizes tuned for L1/L2 cache
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_M = 64;  // multiple of 8 for AVX2
    constexpr int BLOCK_K = 128;
    constexpr int VEC_SIZE = 16;

    std::string compute(const std::string &m1_path,
                        const std::string &m2_path,
                        int n, int k, int m) {
        // Prepare file I/O
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        // Aligned allocations for A, B, and result
        size_t sizeA = static_cast<size_t>(n) * k;
        size_t sizeB = static_cast<size_t>(k) * m;
        size_t sizeC = static_cast<size_t>(n) * m;

        float *m1 = static_cast<float*>(aligned_alloc(64, sizeof(float) * sizeA));
        float *m2 = static_cast<float*>(aligned_alloc(64, sizeof(float) * sizeB));
        float *result = static_cast<float*>(aligned_alloc(64, sizeof(float) * sizeC));
        m1_fs.read(reinterpret_cast<char*>(m1), sizeof(float) * sizeA);
        m2_fs.read(reinterpret_cast<char*>(m2), sizeof(float) * sizeB);
        m1_fs.close(); m2_fs.close();

        // Set number of threads based on available cores
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);

        // Zero initialize result
        std::fill_n(result, sizeC, 0.0f);

        // Parallelize the outermost loop with OpenMP
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < n; i += BLOCK_N) {
            // Thread-local packed buffers
            float packA[BLOCK_N * BLOCK_K] __attribute__((aligned(64))); // Use 64 for AVX-512 alignment
            float packB[BLOCK_K * BLOCK_M] __attribute__((aligned(64)));

            int i_max = std::min(i + BLOCK_N, n);
            for (int j = 0; j < m; j += BLOCK_M) {
                int j_max = std::min(j + BLOCK_M, m);
                float temp_C_block[BLOCK_N * BLOCK_M] __attribute__((aligned(64)));
                std::fill_n(temp_C_block, BLOCK_N * BLOCK_M, 0.0f);

                for (int kk = 0; kk < k; kk += BLOCK_K) {
                    int k_max = std::min(kk + BLOCK_K, k);
                    for (int ii = i; ii < i_max; ++ii) {
                        std::memcpy(&packA[(ii - i) * BLOCK_K], &m1[static_cast<size_t>(ii) * k + kk], sizeof(float) * (k_max - kk));
                    }
                    for (int ll = kk; ll < k_max; ++ll) {
                        std::memcpy(&packB[(ll - kk) * BLOCK_M], &m2[static_cast<size_t>(ll) * m + j], sizeof(float) * (j_max - j));
                    }

                    // Micro-kernel
                    for (int ii = i; ii < i_max; ++ii) {
                        for (int jj = j; jj < j_max; jj += VEC_SIZE) {
                            // Load current value from temp_C_block
                            __m512 c_vec = _mm512_load_ps(&temp_C_block[(ii - i) * BLOCK_M + (jj - j)]);

                            // Accumulate contributions from A and B blocks
                            for (int ll = kk; ll < k_max; ++ll) {
                                __m512 a_vec = _mm512_set1_ps(packA[(ii - i) * BLOCK_K + (ll - kk)]);
                                __m512 b_vec = _mm512_load_ps(&packB[(ll - kk) * BLOCK_M + (jj - j)]);
                                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                            }

                            // Store accumulated value back to temp_C_block
                            _mm512_store_ps(&temp_C_block[(ii - i) * BLOCK_M + (jj - j)], c_vec); // Use aligned store
                        }
                    }
                }

                // Add the completed temporary C block to the main result matrix
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; jj += VEC_SIZE) {
                         __m512 res_vec = _mm512_loadu_ps(&result[static_cast<size_t>(ii) * m + jj]);
                         __m512 tmp_vec = _mm512_load_ps(&temp_C_block[(ii - i) * BLOCK_M + (jj - j)]);
                         res_vec = _mm512_add_ps(res_vec, tmp_vec);
                         _mm512_storeu_ps(&result[static_cast<size_t>(ii) * m + jj], res_vec);
                    }
                }

            }
        }

        // Write output matrix
        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * sizeC);
        sol_fs.close();

        // Free aligned buffers
        free(m1);
        free(m2);
        free(result);

        return sol_path;
    }
}