#include "floyd_warshall_gpu.cuh"
#include <cstdio>
#include <iostream>

using namespace std;

__global__ void floyd_warshall_phase1_kernel(float4* d_matrix, int numVectors, int k) {
    // Shared memory for the single diagonal block padded to avoid warps accessing the same memory block
    __shared__ float pivot[32][33];

    int threadX = threadIdx.x; // 0..7
    int threadY = threadIdx.y; // 0..31

    // Load the Diagonal Block (k, k) into Shared Memory
    int matrixRow = k * 32 + threadY;
    int matrixColumn = k * 8 + threadX;
    
    float4 pivotVector = d_matrix[(matrixRow) * numVectors + (matrixColumn)];

    pivot[threadY][threadX * 4 + 0] = pivotVector.x;
    pivot[threadY][threadX * 4 + 1] = pivotVector.y;
    pivot[threadY][threadX * 4 + 2] = pivotVector.z;
    pivot[threadY][threadX * 4 + 3] = pivotVector.w;

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
    pivotVector.x = pivot[threadY][threadX * 4 + 0];
    pivotVector.y = pivot[threadY][threadX * 4 + 1];
    pivotVector.z = pivot[threadY][threadX * 4 + 2];
    pivotVector.w = pivot[threadY][threadX * 4 + 3];

    d_matrix[matrixRow * numVectors + matrixColumn] = pivotVector;
}

__global__ void floyd_warshall_phase2_kernel(float4* d_matrix, int numVectors, int k) {
    // blockIdx.y == 0 -> Row Block (k, blockId)
    // blockIdx.y == 1 -> Col Block (blockId, k)
    int blockId = blockIdx.x;
    
    // Skip diagonal
    if (blockId == k) return;

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // Shared Memory for Pivot and Self Block
    __shared__ float pivot[32][33];
    __shared__ float self[32][33];

    // Determine Coordinates
    int selfRow, selfColumn;
    
    if (blockIdx.y == 0) {
        // Processing Row Block (k, blockId)
        selfRow = k * 32 + threadY;
        selfColumn = blockId * 8 + threadX;
        
        // Load Pivot (k, k)
        float4 pivotVector = d_matrix[(k * 32 + threadY) * numVectors + (k * 8 + threadX)];
        pivot[threadY][threadX * 4 + 0] = pivotVector.x;
        pivot[threadY][threadX * 4 + 1] = pivotVector.y;
        pivot[threadY][threadX * 4 + 2] = pivotVector.z;
        pivot[threadY][threadX * 4 + 3] = pivotVector.w;
    } else {
        // Processing Col Block: Row blockId, Col k
        selfRow = blockId * 32 + threadY;
        selfColumn = k * 8 + threadX;

        // Load Pivot (k, k)
        float4 pivotVector = d_matrix[(k * 32 + threadY) * numVectors + (k * 8 + threadX)];
        pivot[threadY][threadX * 4 + 0] = pivotVector.x;
        pivot[threadY][threadX * 4 + 1] = pivotVector.y;
        pivot[threadY][threadX * 4 + 2] = pivotVector.z;
        pivot[threadY][threadX * 4 + 3] = pivotVector.w;
    }

    // Load Self
    float4 selfVector = d_matrix[selfRow * numVectors + selfColumn];
    self[threadY][threadX * 4 + 0] = selfVector.x;
    self[threadY][threadX * 4 + 1] = selfVector.y;
    self[threadY][threadX * 4 + 2] = selfVector.z;
    self[threadY][threadX * 4 + 3] = selfVector.w;

    __syncthreads();

    // Compute
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        // Row Block (k, i): self[threadY][threadX] = min(self, pivot[threadY][i] + self[i][threadX])
        // Col Block (i, k): self[threadY][threadX] = min(self, self[threadY][i] + pivot[i][threadX])
        float lelfValue, rightValue0, rightValue1, rightValue2, rightValue3;

        if (blockIdx.y == 0) { // Row Strip
            // For Row strip, pivot is the left operand, self is right
            lelfValue = pivot[threadY][i];
            rightValue0 = self[i][threadX * 4 + 0];
            rightValue1 = self[i][threadX * 4 + 1];
            rightValue2 = self[i][threadX * 4 + 2];
            rightValue3 = self[i][threadX * 4 + 3];
        } else { // Col Strip
             // For Col strip, it's the opposite: self is the left operand, pivot is right
            lelfValue = self[threadY][i];
            rightValue0 = pivot[i][threadX * 4 + 0];
            rightValue1 = pivot[i][threadX * 4 + 1];
            rightValue2 = pivot[i][threadX * 4 + 2];
            rightValue3 = pivot[i][threadX * 4 + 3];
        }

        self[threadY][threadX * 4 + 0] = fminf(self[threadY][threadX * 4 + 0], lelfValue + rightValue0);
        self[threadY][threadX * 4 + 1] = fminf(self[threadY][threadX * 4 + 1], lelfValue + rightValue1);
        self[threadY][threadX * 4 + 2] = fminf(self[threadY][threadX * 4 + 2], lelfValue + rightValue2);
        self[threadY][threadX * 4 + 3] = fminf(self[threadY][threadX * 4 + 3], lelfValue + rightValue3);
        
        __syncthreads(); 
    }

    // Write back Self
    selfVector.x = self[threadY][threadX * 4 + 0];
    selfVector.y = self[threadY][threadX * 4 + 1];
    selfVector.z = self[threadY][threadX * 4 + 2];
    selfVector.w = self[threadY][threadX * 4 + 3];
    d_matrix[selfRow * numVectors + selfColumn] = selfVector;
}

__global__ void floyd_warshall_phase3_kernel(float4* d_matrix, int numVectors, int k) {
    int blockX = blockIdx.x; 
    int blockY = blockIdx.y; 

    // Skip dependent blocks
    if (blockX == k || blockY == k) return;

    int threadX = threadIdx.x; 
    int threadY = threadIdx.y;

    // Shared Memory for Pivot and Self Block
    __shared__ float row[32][33]; // Row Block (k, blockY)
    __shared__ float column[32][33]; // Col Block (blockX, k)

    // Row Block: Row k, Col blockY
    float4 rowVector = d_matrix[(k * 32 + threadY) * numVectors + (blockY * 8 + threadX)];
    
    // Col Block: Row blockX, Col k
    float4 colVector = d_matrix[(blockX * 32 + threadY) * numVectors + (k * 8 + threadX)];

    // Target Block: Row blockX, Col blockY
    float4 targetVector = d_matrix[(blockX * 32 + threadY) * numVectors + (blockY * 8 + threadX)];

    // Populate Shared Memory
    row[threadY][threadX*4+0] = rowVector.x;
    row[threadY][threadX*4+1] = rowVector.y;
    row[threadY][threadX*4+2] = rowVector.z;
    row[threadY][threadX*4+3] = rowVector.w;

    column[threadY][threadX*4+0] = colVector.x;
    column[threadY][threadX*4+1] = colVector.y;
    column[threadY][threadX*4+2] = colVector.z;
    column[threadY][threadX*4+3] = colVector.w;

    __syncthreads();

    // Compute loop
    #pragma unroll
    for (int w = 0; w < 32; ++w) {
        // Read from Shared
        float columnValue = column[threadY][w];
        float rowValue0 = row[w][threadX*4+0];
        float rowValue1 = row[w][threadX*4+1];
        float rowValue2 = row[w][threadX*4+2];
        float rowValue3 = row[w][threadX*4+3];

        // Update Values
        targetVector.x = fminf(targetVector.x, columnValue + rowValue0);
        targetVector.y = fminf(targetVector.y, columnValue + rowValue1);
        targetVector.z = fminf(targetVector.z, columnValue + rowValue2);
        targetVector.w = fminf(targetVector.w, columnValue + rowValue3);
    }

    // Write back to Global
    d_matrix[(blockX * 32 + threadY) * numVectors + (blockY * 8 + threadX)] = targetVector;
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