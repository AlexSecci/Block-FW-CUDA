#ifndef FLOYD_WARSHALL_GPU_CUH
#define FLOYD_WARSHALL_GPU_CUH

#include <cuda_runtime.h>

void floydWarshallGPU(float* h_matrix, int n);

#endif // FLOYD_WARSHALL_GPU_CUH
