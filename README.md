# Accelerated Graph Analytics: Blocked Floyd-Warshall in CUDA

![Language](https://img.shields.io/badge/Language-C++%20%7C%20CUDA-green) ![Platform](https://img.shields.io/badge/Platform-NVIDIA%20GPU-76b900) ![Status](https://img.shields.io/badge/Status-Complete-success)

**Author:** Alessandro Secci  
**Project:** Blocked Floyd-Warshall on CUDA

## 🚀 Project Overview
The goal of this project was to implement a high-performance **All-Pairs Shortest Path (APSP)** solver using the Floyd-Warshall algorithm. The challenge lies in accelerating the computational complexity of $O(N^3)$ by leveraging the massive parallelism of NVIDIA GPUs using CUDA.

By treating the graph adjacency matrix similarly to dense matrix multiplication, this engine optimizes for **memory bandwidth efficiency**, a critical skill in modern high-performance computing infrastructure.

## 🧠 Technical Implementation
To achieve high performance, the **Blocked (Tiled)** version of the Floyd-Warshall algorithm was implemented. This approach minimizes global memory accesses and maximizes data reuse. The matrix is divided into $32 \times 32$ tiles.

### The 3-Phase Kernel Dispatch Strategy
In each iteration of the outer loop $k$ (corresponding to the $k$-th block row/column), the following phases are executed to avoid race conditions:

1.  **Phase 1 (Pivot Update):** 
    *   **Goal:** Update the diagonal block $(k, k)$, which is self-dependent.
    *   **Configuration:** Launched with 1 block of threads, block size `dim3(8, 32)`. Using `float4` vectorization, 8 threads in the x-dimension cover 32 elements ($8 \times 4 = 32$), and 32 threads in the y-dimension cover the 32 rows.
    *   **Operation:** Loads the $32 \times 32$ diagonal block into Shared Memory, computes APSP for this block using a local iterative update, and writes it back to Global Memory.

2.  **Phase 2 (Cross Update):**
    *   **Goal:** Update the $k$-th row of blocks and the $k$-th column of blocks (excluding the pivot itself), using the updated pivot from Phase 1.
    *   **Configuration:** Launched with `dim3(numBlocks, 2)`. The y size of 2 separates the workload into "Row Blocks" and "Column Blocks". Block size `dim3(8, 32)`.
    *   **Operation:** Each block loads its own data and the updated Pivot block into Shared Memory, then updates its own values.

3.  **Phase 3 (Independent Update):**
    *   **Goal:** Update all remaining blocks $(i, j)$ where $i \neq k$ and $j \neq k$.
    *   **Configuration:** Launched with `dim3(numBlocks, numBlocks)` to cover the entire grid. Block size `dim3(8, 32)`.
    *   **Operation:** Skips dependent blocks (Pivot and Cross). Each block loads the corresponding updated block from the $k$-th row and $k$-th column (computed in Phase 2) into Shared Memory to compute the final update.

### Optimization Techniques
*   **Shared Memory Tiling:** Blocks of the graph are loaded into fast Shared Memory to reduce slow Global Memory accesses. `__shared__ float tile[32][33]` was used (padding of 1) to avoid shared memory bank conflicts.
*   **Memory Coalescing:** Memory accesses were aligned to ensure that threads in a warp access contiguous memory addresses, maximizing memory bandwidth.
*   **Vectorization:** The implementation uses `float4` types, processing 4 elements per thread per instruction. This reduces the number of memory instructions by a factor of 4 and increases arithmetic intensity.

## 📊 Performance Benchmarks

### Test System
*   **CPU:** AMD Ryzen 6800H
*   **GPU:** NVIDIA RTX 3070 Ti Laptop GPU (Ampere, Compute Capability 8.6)
*   **OS:** Windows 11 Pro 25H2 (build 26200.7623)
*   **Build System:** CMake 4.2.3
*   **Optimization flags:** `-O3`

### Test Results
Measured GPU Time includes memory transfer device-to-host and host-to-device.

| Graph Size (N) | CPU Time (ms) | GPU Time (ms) | GPU Throughput | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| **32** | 0.1218 | 89.8191 | 0.73 MFLOPS | 737x Slower |
| **64** | 0.5253 | 92.8062 | 5.65 MFLOPS | 176x Slower |
| **128** | 2.3199 | 91.9685 | 45.61 MFLOPS | 39.5x Slower |
| **256** | 12.8719 | 85.3998 | 392.91 MFLOPS | 6.5x Slower |
| **512** | 84.8751 | 84.7777 | 3.17 GFLOPS | **1x** |
| **1024** | 589.709 | 106.711 | 20.12 GFLOPS | **5.5x** |
| **2048** | 4570.83 | 103.946 | 165.28 GFLOPS | **44x** |
| **4096** | 35214.9 | 306.632 | 448.22 GFLOPS | **115x** |
| **8192** | 273884 | 716.955 | 1.53 TFLOPS | **382x** |
| **16384** | N/A | 4808.43 | 1.83 TFLOPS | N/A |
| **32768** | N/A | 37122.7 | 1.90 TFLOPS | N/A |

### Profiling (NVIDIA Nsight Compute)
Key metrics demonstrating the efficiency of the implementation:
*   **Compute (SM) Throughput:** 88.93%
*   **Memory Throughput:** 89.07%
*   **L1/TEX Cache Throughput:** 89.51%
*   **DRAM Throughput:** 46.80%
*   **Achieved Occupancy:** 96.52%

## 🔬 Testing & Correctness Verification

### Data Generation
A custom graph generation utility generates adjacency matrices where the graph size $N$ is enforced to be a multiple of 32 (aligning with the GPU warp size and tiling strategy without padding) and the density is configurable.
```bash
.\DataGenerator.exe <Density [0, 100]> <Size>
```

### Verification Methodology
Correctness was verified by comparing the GPU output against a standard $O(N^3)$ sequential CPU implementation.
```bash
.\FloydWarshall.exe <mode> <input graph path> <output folder>
```
*(Modes: `0` = cpu-only, `1` = gpu-only, `2` = both + benchmark)*

Floating-point arithmetic and GPU parallelization can introduce minor precision differences exceeding machine epsilon, leading to strict equality (`!=`) check warnings, which are normal.

## 🏆 Conclusion
The implemented Blocked Floyd-Warshall algorithm achieves high-throughput efficiency. While execution at small graph sizes ($N < 512$) is dominated by PCIe transfer and driver overhead, the solver achieves massive scalability on larger datasets.

At $N=8192$, it delivers a **382x speedup** over the CPU baseline, culminating in a peak throughput of **1.90 TFLOPS** at $N=32768$.

The kernel achieves an exceptional **96.5% Occupancy** and **~89% SM Throughput**, indicating near-total saturation of the GPU’s compute resources. The high L1 cache hit rate (~89.5%) versus lower DRAM usage (~46.8%) proves the Tiling Strategy successfully shifted the bottleneck from memory bandwidth to compute.

## 🛠️ Build & Run
### Prerequisites
* CMake (3.10+)
* NVIDIA CUDA Toolkit (11.0+)
* C++ Compiler (MSVC for Windows / GCC for Linux)

### Compilation
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```