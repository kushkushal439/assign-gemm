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
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_M = 32;  // multiple of 8 for AVX2
    constexpr int BLOCK_K = 32;
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

        float *m1 = static_cast<float*>(aligned_alloc(32, sizeof(float) * sizeA));
        float *m2 = static_cast<float*>(aligned_alloc(32, sizeof(float) * sizeB));
        float *result = static_cast<float*>(aligned_alloc(32, sizeof(float) * sizeC));
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
                        // Prefetch next line of A if helpful
                        // _mm_prefetch((const char*)(&m1[static_cast<size_t>(ii) * k + kk + 16]), _MM_HINT_T0);
                        std::memcpy(&packA[(ii - i) * BLOCK_K], &m1[static_cast<size_t>(ii) * k + kk], sizeof(float) * (k_max - kk));
                    }
                    for (int ll_pack = kk; ll_pack < k_max; ++ll_pack) {
                        for (int jj_pack = j; jj_pack < j_max; ++jj_pack) {
                            // packB layout: [k_block][m_block] -> packB[inner_k * BLOCK_M + inner_j]
                            // We want layout: [m_block / VEC_SIZE][k_block][VEC_SIZE]
                            // Or simpler: packB[inner_k * BLOCK_M + inner_j] -> packB[inner_j * BLOCK_K + inner_k] (transposed)
                            // Let's try the simple transpose first: packB[inner_j * BLOCK_K + inner_k]
                            packB[(jj_pack - j) * BLOCK_K + (ll_pack - kk)] = m2[static_cast<size_t>(ll_pack) * m + jj_pack];
                        }
                        // Prefetch next row of B if helpful
                        // _mm_prefetch((const char*)(&m2[static_cast<size_t>(ll_pack + 1) * m + j]), _MM_HINT_T0);
                    }

                    // Micro-kernel: Accumulate into temp_C_block
                    for (int ii = i; ii < i_max; ++ii) {
                        for (int jj = j; jj < j_max; jj += VEC_SIZE) {
                            __m512 c_vec = _mm512_load_ps(&temp_C_block[(ii - i) * BLOCK_M + (jj - j)]);

                            int ll = kk;
                            // Unroll inner loop (example: unroll by 2)
                            for (; ll + 1 < k_max; ll += 2) {
                                __m512 a_vec0 = _mm512_set1_ps(packA[(ii - i) * BLOCK_K + (ll - kk)]);
                                // Load from OPTIMIZED packB - now contiguous access pattern for b_vec!
                                // Address calculation depends on the chosen packing layout. For simple transpose:
                                __m512 b_vec0 = _mm512_load_ps(&packB[(jj - j) * BLOCK_K + (ll - kk)]); // Load 16 floats for ll=0
                                c_vec = _mm512_fmadd_ps(a_vec0, b_vec0, c_vec);

                                __m512 a_vec1 = _mm512_set1_ps(packA[(ii - i) * BLOCK_K + (ll + 1 - kk)]);
                                __m512 b_vec1 = _mm512_load_ps(&packB[(jj - j) * BLOCK_K + (ll + 1 - kk)]); // Load 16 floats for ll=1
                                c_vec = _mm512_fmadd_ps(a_vec1, b_vec1, c_vec);
                            }
                            // Handle remaining iterations
                            for (; ll < k_max; ++ll) {
                                __m512 a_vec = _mm512_set1_ps(packA[(ii - i) * BLOCK_K + (ll - kk)]);
                                __m512 b_vec = _mm512_load_ps(&packB[(jj - j) * BLOCK_K + (ll - kk)]);
                                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                            }

                            _mm512_store_ps(&temp_C_block[(ii - i) * BLOCK_M + (jj - j)], c_vec);
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