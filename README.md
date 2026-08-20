# GEMM - General Matrix Multiplication

High-performance dense **GEMM** kernel (course assignment, SPP - Systems Programming for Performance).

## Implementation

Single-threaded blocked GEMM tuned for the CPU:

- **Blocked algorithm** with tile sizes `BLOCK_M=128`, `BLOCK_N=128`, `BLOCK_K=192` for better cache reuse
- **AVX-512 SIMD** intrinsics (`_mm512_*`) with 16-wide vector loads/stores and FMA
- **Register blocking** - 8 accumulator rows kept in registers per iteration
- **Loop unrolling** for better instruction pipelining
- **64-byte aligned allocations** for AVX-512 alignment

## Build & Run

```bash
bash runner_script.sh   # or use the provided CMake build
```
