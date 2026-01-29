# Accelerated Graph Analytics: Blocked Floyd-Warshall in CUDA

![Language](https://img.shields.io/badge/Language-C++%20%7C%20CUDA-green) ![Platform](https://img.shields.io/badge/Platform-NVIDIA%20GPU-76b900) ![Status](https://img.shields.io/badge/Status-Early%20Development-7e0101)

## 🚀 Project Overview
This project implements a high-performance **All-Pairs Shortest Path (APSP)** solver using the Floyd-Warshall algorithm.

While the naive algorithm operates in $O(N^3)$ time, this implementation utilizes a **Blocked (Tiled)** approach to leverage the massive parallelism of modern GPUs. By treating the graph adjacency matrix similarly to dense matrix multiplication (GEMM), this engine optimizes for **memory bandwidth efficiency**, a critical skill in modern Deep Learning infrastructure (e.g., optimizing Transformer attention mechanisms).

## 🧠 Technical Implementation
The core challenge of parallelizing Floyd-Warshall is the strict data dependency: the $k$-th iteration depends on the results of the $(k-1)$-th iteration. To solve this in parallel without race conditions, I implemented a **3-Phase Kernel Dispatch** strategy:

### The "Blocked" Strategy
Instead of processing single nodes, the graph is divided into $B \times B$ tiles (blocks). Each iteration of the outer loop ($k$) requires three synchronized kernel launches:

1.  **Phase 1 (Dependent Block):** Compute the diagonal "Pivot" block $(k, k)$.
2.  **Phase 2 (Cross Blocks):** Compute blocks in the same row ($k$) and column ($k$) using the Pivot.
3.  **Phase 3 (Independent Blocks):** Compute the remaining blocks in parallel using the Cross blocks.

### Optimization Techniques
This implementation moves beyond naive parallelization to maximize GPU occupancy:
* **Shared Memory Tiling:** Loads $32 \times 32$ sub-matrices into on-chip Shared Memory to reduce high-latency Global Memory (VRAM) accesses by approximately **32x**.
* **Memory Coalescing:** Aligns memory reads/writes to 128-byte cache lines to maximize memory controller throughput.
* **Vectorized Loads:** Utilizes `float4` (or `int4`) instructions to load 128 bits per instruction, reducing instruction overhead.
* **Bank Conflict Avoidance:** Implements memory padding in Shared Memory to prevent serialization of thread access.

## 📊 Performance Benchmarks
*Note: Benchmarks run on [Insert Your GPU Model Here]*

| Graph Size (Nodes) | CPU Time (ms) | GPU Time (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| 1024 | TBD | TBD | **TBD** |
| 2048 | TBD | TBD | **TBD** |
| 4096 | TBD | TBD | **TBD** |
| 8192 | TBD | TBD | **TBD** |

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