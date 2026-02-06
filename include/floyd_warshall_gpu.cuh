#ifndef FLOYD_WARSHALL_GPU_CUH
#define FLOYD_WARSHALL_GPU_CUH

#include <cuda_runtime.h>
#include <vector>

void floydWarshallGPU(std::vector<float>& graph, int n);

void floydWarshallGPUNaive(std::vector<float>& graph, int n);

#endif // FLOYD_WARSHALL_GPU_CUH
