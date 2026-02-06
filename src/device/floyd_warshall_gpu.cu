#include "floyd_warshall_gpu.cuh"
#include <cstdio>
#include <iostream>

using namespace std;

__global__ void floyd_warshall_phase1_kernel(float4* d_matrix, int n, int k) {
    // Shared memory for the single diagonal block padded to avoid warps accessing the same memory block
    __shared__ float pivot[32][33];

    int threadX = threadIdx.x; // 0..7
    int threadY = threadIdx.y; // 0..31

    // Load the Diagonal Block (k, k) into Shared Memory
    int matrixRow = k * 32 + threadY;
    int matrixColumn = k * 8 + threadX;
    
    float4 vector = d_matrix[(matrixRow) * 8 + (matrixColumn)];

    pivot[threadY][threadX * 4 + 0] = vector.x;
    pivot[threadY][threadX * 4 + 1] = vector.y;
    pivot[threadY][threadX * 4 + 2] = vector.z;
    pivot[threadY][threadX * 4 + 3] = vector.w;

    __syncthreads();

    // Compute FW locally on the 32x32 block inside Shared Memory
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        float p_row = pivot[threadY][i];
        // Instead of the if that we used in the naive kernel, we prefer fminf to avoid divergence since it maps to a single instruction
        pivot[threadY][threadX * 4 + 0] = fminf(pivot[threadY][threadX * 4 + 0], p_row + pivot[i][threadX * 4 + 0]);
        pivot[threadY][threadX * 4 + 1] = fminf(pivot[threadY][threadX * 4 + 1], p_row + pivot[i][threadX * 4 + 1]);
        pivot[threadY][threadX * 4 + 2] = fminf(pivot[threadY][threadX * 4 + 2], p_row + pivot[i][threadX * 4 + 2]);
        pivot[threadY][threadX * 4 + 3] = fminf(pivot[threadY][threadX * 4 + 3], p_row + pivot[i][threadX * 4 + 3]);
        __syncthreads();
    }

    // Write back to Global Memory
    vector.x = pivot[threadY][threadX * 4 + 0];
    vector.y = pivot[threadY][threadX * 4 + 1];
    vector.z = pivot[threadY][threadX * 4 + 2];
    vector.w = pivot[threadY][threadX * 4 + 3];

    d_matrix[(matrixRow) * 8 + (matrixColumn)] = vector;
}

__global__ void floyd_warshall_phase2_kernel(float4* d_matrix, int n, int k) {
    // TODO: Implement CUDA kernel
    // int i = blockIdx.y * blockDim.y + threadIdx.y;
    // int j = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void floyd_warshall_phase3_kernel(float4* d_matrix, int n, int k) {
    // TODO: Implement CUDA kernel
    // int i = blockIdx.y * blockDim.y + threadIdx.y;
    // int j = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void floyd_warshall_naive_kernel(float* d_matrix, int n, int k) {
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
    // Run three phases kernel loop
    dim3 threadsPerBlock(8, 32); 
    int numBlocks(n / 32);
    int numVectors(n / 4);
    for (int k = 0; k < numBlocks; ++k) {
        // Loop over diagonal blocks
        floyd_warshall_phase1_kernel<<<1, threadsPerBlock>>>( (float4*)d_array, numVectors, k );
        // Loop over the cross
        dim3 gridPhase2(numBlocks, 2);
        floyd_warshall_phase2_kernel<<<gridPhase2, threadsPerBlock>>>( (float4*)d_array, numVectors, k );
        // Loop over the whole matrix
        dim3 gridPhase3(numBlocks, numBlocks);
        floyd_warshall_phase3_kernel<<<gridPhase3, threadsPerBlock>>>( (float4*)d_array, numVectors, k );
    }
    // Copy result back
    cudaMemcpy(graph.data(), d_array, size, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

void floydWarshallGPUNaive(vector<float>& graph, int n) {
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
    for (int k = 0; k < n; ++k) {
        floyd_warshall_naive_kernel<<<numBlocks, threadsPerBlock>>>(d_array, n, k);
    }
    // Copy result back
    cudaMemcpy(graph.data(), d_array, size, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}