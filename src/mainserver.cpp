#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

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
    constexpr int BLOCK_N = 64;  // Multiple of 8
    constexpr int BLOCK_M = 64;  // Multiple of 16
    constexpr int BLOCK_K = 128; // Larger K block for better reuse
    constexpr int VEC_SIZE = 16; // AVX-512 width

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
        #pragma omp parallel for
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
                    // Process 8 rows at a time
                    for (int ii = i; ii < i_max; ii += 8) {
                        // Make sure we don't go out of bounds
                        if (ii + 8 <= i_max) {
                            for (int jj = j; jj < j_max; jj += VEC_SIZE) {
                                // Load 8 rows of accumulator vectors
                                __m512 c_vec0 = _mm512_load_ps(&temp_C_block[(ii - i) * BLOCK_M + (jj - j)]);
                                __m512 c_vec1 = _mm512_load_ps(&temp_C_block[(ii + 1 - i) * BLOCK_M + (jj - j)]);
                                __m512 c_vec2 = _mm512_load_ps(&temp_C_block[(ii + 2 - i) * BLOCK_M + (jj - j)]);
                                __m512 c_vec3 = _mm512_load_ps(&temp_C_block[(ii + 3 - i) * BLOCK_M + (jj - j)]);
                                __m512 c_vec4 = _mm512_load_ps(&temp_C_block[(ii + 4 - i) * BLOCK_M + (jj - j)]);
                                __m512 c_vec5 = _mm512_load_ps(&temp_C_block[(ii + 5 - i) * BLOCK_M + (jj - j)]);
                                __m512 c_vec6 = _mm512_load_ps(&temp_C_block[(ii + 6 - i) * BLOCK_M + (jj - j)]);
                                __m512 c_vec7 = _mm512_load_ps(&temp_C_block[(ii + 7 - i) * BLOCK_M + (jj - j)]);
                                
                                // Unroll the inner loop for better pipelining
                                int ll = kk;
                                for (; ll + 1 < k_max; ll += 2) {
                                    // First K iteration
                                    __m512 b_vec0 = _mm512_load_ps(&packB[(ll - kk) * BLOCK_M + (jj - j)]);
                                    
                                    __m512 a_vec0 = _mm512_set1_ps(packA[(ii - i) * BLOCK_K + (ll - kk)]);
                                    c_vec0 = _mm512_fmadd_ps(a_vec0, b_vec0, c_vec0);
                                    
                                    __m512 a_vec1 = _mm512_set1_ps(packA[(ii + 1 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec1 = _mm512_fmadd_ps(a_vec1, b_vec0, c_vec1);
                                    
                                    __m512 a_vec2 = _mm512_set1_ps(packA[(ii + 2 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec2 = _mm512_fmadd_ps(a_vec2, b_vec0, c_vec2);
                                    
                                    __m512 a_vec3 = _mm512_set1_ps(packA[(ii + 3 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec3 = _mm512_fmadd_ps(a_vec3, b_vec0, c_vec3);
                                    
                                    __m512 a_vec4 = _mm512_set1_ps(packA[(ii + 4 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec4 = _mm512_fmadd_ps(a_vec4, b_vec0, c_vec4);
                                    
                                    __m512 a_vec5 = _mm512_set1_ps(packA[(ii + 5 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec5 = _mm512_fmadd_ps(a_vec5, b_vec0, c_vec5);
                                    
                                    __m512 a_vec6 = _mm512_set1_ps(packA[(ii + 6 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec6 = _mm512_fmadd_ps(a_vec6, b_vec0, c_vec6);
                                    
                                    __m512 a_vec7 = _mm512_set1_ps(packA[(ii + 7 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec7 = _mm512_fmadd_ps(a_vec7, b_vec0, c_vec7);
                                    
                                    // Second K iteration
                                    __m512 b_vec1 = _mm512_load_ps(&packB[(ll + 1 - kk) * BLOCK_M + (jj - j)]);
                                    
                                    __m512 a_vec0_1 = _mm512_set1_ps(packA[(ii - i) * BLOCK_K + (ll + 1 - kk)]);
                                    c_vec0 = _mm512_fmadd_ps(a_vec0_1, b_vec1, c_vec0);
                                    
                                    __m512 a_vec1_1 = _mm512_set1_ps(packA[(ii + 1 - i) * BLOCK_K + (ll + 1 - kk)]);
                                    c_vec1 = _mm512_fmadd_ps(a_vec1_1, b_vec1, c_vec1);
                                    
                                    __m512 a_vec2_1 = _mm512_set1_ps(packA[(ii + 2 - i) * BLOCK_K + (ll + 1 - kk)]);
                                    c_vec2 = _mm512_fmadd_ps(a_vec2_1, b_vec1, c_vec2);
                                    
                                    __m512 a_vec3_1 = _mm512_set1_ps(packA[(ii + 3 - i) * BLOCK_K + (ll + 1 - kk)]);
                                    c_vec3 = _mm512_fmadd_ps(a_vec3_1, b_vec1, c_vec3);
                                    
                                    __m512 a_vec4_1 = _mm512_set1_ps(packA[(ii + 4 - i) * BLOCK_K + (ll + 1 - kk)]);
                                    c_vec4 = _mm512_fmadd_ps(a_vec4_1, b_vec1, c_vec4);
                                    
                                    __m512 a_vec5_1 = _mm512_set1_ps(packA[(ii + 5 - i) * BLOCK_K + (ll + 1 - kk)]);
                                    c_vec5 = _mm512_fmadd_ps(a_vec5_1, b_vec1, c_vec5);
                                    
                                    __m512 a_vec6_1 = _mm512_set1_ps(packA[(ii + 6 - i) * BLOCK_K + (ll + 1 - kk)]);
                                    c_vec6 = _mm512_fmadd_ps(a_vec6_1, b_vec1, c_vec6);
                                    
                                    __m512 a_vec7_1 = _mm512_set1_ps(packA[(ii + 7 - i) * BLOCK_K + (ll + 1 - kk)]);
                                    c_vec7 = _mm512_fmadd_ps(a_vec7_1, b_vec1, c_vec7);
                                }
                                
                                // Handle remaining K iterations
                                for (; ll < k_max; ++ll) {
                                    __m512 b_vec = _mm512_load_ps(&packB[(ll - kk) * BLOCK_M + (jj - j)]);
                                    
                                    __m512 a_vec0 = _mm512_set1_ps(packA[(ii - i) * BLOCK_K + (ll - kk)]);
                                    c_vec0 = _mm512_fmadd_ps(a_vec0, b_vec, c_vec0);
                                    
                                    __m512 a_vec1 = _mm512_set1_ps(packA[(ii + 1 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec1 = _mm512_fmadd_ps(a_vec1, b_vec, c_vec1);
                                    
                                    __m512 a_vec2 = _mm512_set1_ps(packA[(ii + 2 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec2 = _mm512_fmadd_ps(a_vec2, b_vec, c_vec2);
                                    
                                    __m512 a_vec3 = _mm512_set1_ps(packA[(ii + 3 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec3 = _mm512_fmadd_ps(a_vec3, b_vec, c_vec3);
                                    
                                    __m512 a_vec4 = _mm512_set1_ps(packA[(ii + 4 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec4 = _mm512_fmadd_ps(a_vec4, b_vec, c_vec4);
                                    
                                    __m512 a_vec5 = _mm512_set1_ps(packA[(ii + 5 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec5 = _mm512_fmadd_ps(a_vec5, b_vec, c_vec5);
                                    
                                    __m512 a_vec6 = _mm512_set1_ps(packA[(ii + 6 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec6 = _mm512_fmadd_ps(a_vec6, b_vec, c_vec6);
                                    
                                    __m512 a_vec7 = _mm512_set1_ps(packA[(ii + 7 - i) * BLOCK_K + (ll - kk)]);
                                    c_vec7 = _mm512_fmadd_ps(a_vec7, b_vec, c_vec7);
                                }
                                
                                // Store results back
                                _mm512_store_ps(&temp_C_block[(ii - i) * BLOCK_M + (jj - j)], c_vec0);
                                _mm512_store_ps(&temp_C_block[(ii + 1 - i) * BLOCK_M + (jj - j)], c_vec1);
                                _mm512_store_ps(&temp_C_block[(ii + 2 - i) * BLOCK_M + (jj - j)], c_vec2);
                                _mm512_store_ps(&temp_C_block[(ii + 3 - i) * BLOCK_M + (jj - j)], c_vec3);
                                _mm512_store_ps(&temp_C_block[(ii + 4 - i) * BLOCK_M + (jj - j)], c_vec4);
                                _mm512_store_ps(&temp_C_block[(ii + 5 - i) * BLOCK_M + (jj - j)], c_vec5);
                                _mm512_store_ps(&temp_C_block[(ii + 6 - i) * BLOCK_M + (jj - j)], c_vec6);
                                _mm512_store_ps(&temp_C_block[(ii + 7 - i) * BLOCK_M + (jj - j)], c_vec7);
                            }
                        } else {
                            // Handle remaining rows (fewer than 8)
                            for (int ii_rem = ii; ii_rem < i_max; ++ii_rem) {
                                for (int jj = j; jj < j_max; jj += VEC_SIZE) {
                                    __m512 c_vec = _mm512_load_ps(&temp_C_block[(ii_rem - i) * BLOCK_M + (jj - j)]);
                                    
                                    for (int ll = kk; ll < k_max; ++ll) {
                                        __m512 a_vec = _mm512_set1_ps(packA[(ii_rem - i) * BLOCK_K + (ll - kk)]);
                                        __m512 b_vec = _mm512_load_ps(&packB[(ll - kk) * BLOCK_M + (jj - j)]);
                                        c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                    }
                                    
                                    _mm512_store_ps(&temp_C_block[(ii_rem - i) * BLOCK_M + (jj - j)], c_vec);
                                }
                            }
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

        // collapse
        // avx512 tune to 8 rows 16 columns
        // 

        return sol_path;
    }
}