#include "floyd_warshall_gpu.cuh"
#include <cstdio>

__global__ void floyd_warshall_kernel(float* d_matrix, int n, int k) {
    // TODO: Implement CUDA kernel
    // int i = blockIdx.y * blockDim.y + threadIdx.y;
    // int j = blockIdx.x * blockDim.x + threadIdx.x;
}

void floydWarshallGPU(float* h_matrix, int n) {
    // TODO: Allocate device memory
    // TODO: Copy data to device
    // TODO: Run kernel loop
    // TODO: Copy result back
}
