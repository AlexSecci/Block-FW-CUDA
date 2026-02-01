#include "floyd_warshall_gpu.cuh"
#include <cstdio>
#include <iostream>

using namespace std;

__global__ void floyd_warshall_kernel(float* d_matrix, int n, int k) {
    // TODO: Implement CUDA kernel
    // int i = blockIdx.y * blockDim.y + threadIdx.y;
    // int j = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void floyd_warshall_naive_kernel(float* d_matrix, int n, int k) {
    // TODO: Implement CUDA kernel
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n){
        if (d_matrix[i * n + j] > d_matrix[i * n + k] + d_matrix[k * n + j])
        {
            d_matrix[i * n + j] = d_matrix[i * n + k] + d_matrix[k * n + j];
        }
    }
}

void floydWarshallGPU(vector<float>& graph, int n) {
    cout << "Running Floyd-Warshall on GPU" << endl;
    float *d_array;
    size_t size = n * n * sizeof(float);
    // Allocate device memory
    cudaMalloc((void**)&d_array, size);
    // Copy data to device
    cudaMemcpy(d_array, graph.data(), size, cudaMemcpyHostToDevice);
    // Run kernel loop
    dim3 threadsPerBlock(32, 32); 
    dim3 numBlocks(n / threadsPerBlock.x, n / threadsPerBlock.y);
    for (int k = 0; k < n; k++) {
        floyd_warshall_naive_kernel<<<numBlocks, threadsPerBlock>>>(d_array, n, k);
    }
    // Copy result back
    cudaMemcpy(graph.data(), d_array, size, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}
