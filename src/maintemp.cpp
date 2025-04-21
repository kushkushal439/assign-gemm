#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx512f,avx512cd,avx512bw,avx512dq,avx512vl") // Target relevant AVX-512 instruction sets for float

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm> // For std::min
#include <omp.h> // For OpenMP
#include <stdexcept> // For std::runtime_error

namespace solution{
	std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m){
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
		std::ofstream sol_fs(sol_path, std::ios::binary);
		std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

		// Use aligned allocation for better performance with AVX-512 loads/stores
		// Allocate with 64-byte alignment for AVX-512
		float* m1_data = (float*)_mm_malloc(n * k * sizeof(float), 64);
		float* m2_data = (float*)_mm_malloc(k * m * sizeof(float), 64);
		float* result_data = (float*)_mm_malloc(n * m * sizeof(float), 64);

		auto m1 = std::unique_ptr<float, void(*)(void*)>(m1_data, _mm_free);
		auto m2 = std::unique_ptr<float, void(*)(void*)>(m2_data, _mm_free);
		auto result = std::unique_ptr<float, void(*)(void*)>(result_data, _mm_free);


		if (!m1 || !m2 || !result) {
            // Clean up any successfully allocated memory before throwing
            if(m1) _mm_free(m1.get());
            if(m2) _mm_free(m2.get());
            if(result) _mm_free(result.get());
            throw std::runtime_error("Failed to allocate aligned memory for matrices.");
        }

		m1_fs.read(reinterpret_cast<char*>(m1.get()), sizeof(float) * n * k);
		m2_fs.read(reinterpret_cast<char*>(m2.get()), sizeof(float) * k * m);
		m1_fs.close(); m2_fs.close();

		// Initialize result matrix to zero (important for accumulation)
		// Can be partially vectorized, but a simple loop is clear here.
		for(int i = 0; i < n * m; ++i) result.get()[i] = 0.0f;

		// Define blocking sizes
		const int Mr = 6; // Register block size for rows of A and C
		const int Nr = 16; // Register block size for columns of B and C (AVX-512 float vector width)

		// Cache block sizes (example values, requires extensive tuning for target CPU)
		// These should be chosen to fit blocks of A and B into L2/L3 cache
		// and should be multiples of Mr and Nr respectively for full efficiency
		const int Mc = 96; // Example: Must be a multiple of Mr
		const int Nc = 256; // Example: Must be a multiple of Nr
		const int Kc = 128; // Example: Can be tuned


		// Optimized GEMM with multi-level blocking, multi-threading, and AVX-512
		// Parallelize the outer loops (cache blocks of C) using OpenMP
		#pragma omp parallel for collapse(2) schedule(static) // Static schedule is often good for predictable workloads
		for (int i_c = 0; i_c < n; i_c += Mc) { // Iterate through cache blocks of rows of A and C
			for (int j_c = 0; j_c < m; j_c += Nc) { // Iterate through cache blocks of columns of B and C

				// Iterate over the inner dimension in cache blocks
				for (int l_c = 0; l_c < k; l_c += Kc) {

					// Register blocking loops within the current cache block
					// These loops iterate over the smaller blocks that fit into registers
					for (int i = i_c; i < std::min(i_c + Mc, n); i += Mr) {
						for (int j = j_c; j < std::min(j_c + Nc, m); j += Nr) {

							// Accumulate Mr x Nr block of C in registers
							__m512 c_acc[Mr];

							// Determine number of columns to process in this j-block for masking
                            int current_Nr = std::min(j + Nr, std::min(j_c + Nc, m)) - j;
                            __mmask16 mask_j = (current_Nr == Nr) ? (__mmask16)-1 : (1 << current_Nr) - 1;

							// Load initial values of C block from memory into registers (with masking for j remainder)
							for(int row = 0; row < Mr; ++row) {
								if (i + row < std::min(i_c + Mc, n)) { // Check if the current row is within the bounds of the cache block and matrix
									// Use masked load to load existing values of C, handling the j-dimension remainder
									c_acc[row] = _mm512_maskz_loadu_ps(mask_j, &result.get()[(i+row)*m + j]);
								} else {
									// If the row is out of bounds, initialize the corresponding accumulator register to zero
                                     c_acc[row] = _mm512_setzero_ps();
                                }
                            }


							// Inner loop over the Kc dimension (k-dimension within the cache block)
							for (int l = l_c; l < std::min(l_c + Kc, k); ++l) {
								// Load and broadcast elements from A (Mr elements)
								__m512 a_scalar[Mr];
								for(int row = 0; row < Mr; ++row) {
									if (i + row < std::min(i_c + Mc, n)) { // Check if the current row is within bounds
										// Broadcast the scalar element A[i+row, l] to a vector
										a_scalar[row] = _mm512_set1_ps(m1.get()[(i+row)*k + l]);
									} else {
										// If the row is out of bounds, broadcast zero
										a_scalar[row] = _mm512_setzero_ps();
									}
								}

								// Load Nr elements from B (with masking for j remainder)
								// Use masked unaligned load for flexibility at block edges
								__m512 b_vec = _mm512_maskz_loadu_ps(mask_j, &m2.get()[l*m + j]);

								// Perform Mr Fused Multiply-Accumulate operations (FMAs)
								for(int row = 0; row < Mr; ++row) {
                                     if (i + row < std::min(i_c + Mc, n)) { // Only perform FMAs for rows within bounds
										// Perform: c_acc[row] = c_acc[row] + a_scalar[row] * b_vec
										// The mask_j applied to b_vec ensures that only the relevant elements
										// of c_acc[row] are updated when multiplying by b_vec.
                                        c_acc[row] = _mm512_fmadd_ps(a_scalar[row], b_vec, c_acc[row]);
                                    }
								}
							}

							// Store the accumulated results for the Mr x Nr block of C back to memory
							for(int row = 0; row < Mr; ++row) {
                                if (i + row < std::min(i_c + Mc, n)) { // Only store for rows within bounds
                                     // Use masked unaligned store to write the updated C block, handling the j-dimension remainder
                                     _mm512_mask_storeu_ps(&result.get()[(i+row)*m + j], mask_j, c_acc[row]);
                                }
							}
						}
					}
				}
			}
		}

		sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
		sol_fs.close();

		// Aligned memory will be deallocated automatically by unique_ptr with custom deleter
		return sol_path;
	}
};