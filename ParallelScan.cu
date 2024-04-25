#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// CUDA runtime
#include <cuda_runtime.h>

typedef unsigned int uint;

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 65535

// Allocate device memory
cudaError_t allocateDeviceMemory(uint** devicePtr, size_t size) {
    cudaError_t err = cudaMalloc((void**)devicePtr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}

// Copy host memory to device
cudaError_t copyToDevice(uint* devicePtr, const uint* hostPtr, size_t size) {
    cudaError_t err = cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory to the device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}

// Copy results from device to host
cudaError_t copyFromDevice(uint* hostPtr, const uint* devicePtr, size_t size) {
    cudaError_t err = cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory from the device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}

// Deallocate device memory
cudaError_t deallocateDeviceMemory(uint* devicePtr) {
    cudaError_t err = cudaFree(devicePtr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}

// Work-inefficient inclusive scan kernel
__global__ void workInefficient_inclusiveScan(uint* input, uint* output, uint n) {
    extern __shared__ uint shared[];
    uint tid = threadIdx.x;
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (i < n) {
        shared[tid] = input[i];
    }
    else {
        shared[tid] = 0; // Pad with 0 for remaining threads
    }
    __syncthreads();

    // Scan in place
    for (uint stride = 1; stride < blockDim.x; stride *= 2) {
        //uint index = tid + stride;
        uint local_index;
        if (tid >= stride) {
            local_index = shared[tid - stride];
        }
        else {
            local_index = 0;
        }
        __syncthreads();
        shared[tid] = shared[tid] + local_index;
    }
    __syncthreads();

    // Write output
    if (i < n) {
        output[i] = shared[tid];
    }
}

// Work-efficient inclusive scan kernel
__global__ void workEfficient_inclusiveScan(uint* input, uint* output, uint n) {
    extern __shared__ uint shared[];
    uint tid = threadIdx.x;
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (i < n) {
        shared[tid] = input[i];
    }
    else {
        shared[tid] = 0; // Pad with 0 for remaining threads
    }
    if (i + blockDim.x < n) {
        shared[tid + blockDim.x] = input[i + blockDim.x];
    }
    else {
        shared[tid + blockDim.x] = 0; // Pad with 0 for remaining threads
    }
    __syncthreads();

    // Reduction phase
    for (uint stride = 1; stride <= blockDim.x; stride *= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x) {
            shared[index] += shared[index - stride];
        }
        __syncthreads();
    }

    // Post-reduction reverse phase
    for (uint stride = blockDim.x / 2; stride > 0; stride /= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index + stride < 2 * blockDim.x) {
            shared[index + stride] += shared[index];
        }
        __syncthreads();
    }

    // Write output
    if (i < n) {
        output[i] = shared[tid];
    }
    if (i + blockDim.x < n) {
        output[i + blockDim.x] = shared[tid + blockDim.x];
    }
}

// CPU implementation of normal inclusive scan
void cpu_normal_inclusiveScan(uint* input, uint* output, uint n) {
    output[0] = input[0];
    for (uint i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// Compare CPU and GPU results
bool compareResults(uint* cpuOutput, uint* gpuOutput, uint n) {
    for (uint i = 0; i < n; i++) {
        if (cpuOutput[i] != gpuOutput[i]) {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, cpuOutput[i], gpuOutput[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4 || (strcmp(argv[2], "-i") != 0)) {
        fprintf(stderr, "Usage: %s [scan-work-efficient | scan-work-inefficient] -i <vector_size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    uint n = atoi(argv[3]);
    if (n > MAX_THREADS_PER_BLOCK * MAX_BLOCKS) {
        fprintf(stderr, "Input size should be at most %d\n", MAX_THREADS_PER_BLOCK * MAX_BLOCKS);
        exit(EXIT_FAILURE);
    }

    uint* hostInput, * hostOutput, * cpuOutput;
    uint* deviceInput, * deviceOutput;

    // Allocate host memory
    hostInput = (uint*)malloc(n * sizeof(uint));
    hostOutput = (uint*)malloc(n * sizeof(uint));
    cpuOutput = (uint*)malloc(n * sizeof(uint));

    // Initialize input data
    srand(time(NULL));
    for (uint i = 0; i < n; i++) {
        hostInput[i] = rand() % 100;
    }

    // Allocate device memory
    allocateDeviceMemory(&deviceInput, n * sizeof(uint));
    allocateDeviceMemory(&deviceOutput, n * sizeof(uint));

    // Copy input data to device
    copyToDevice(deviceInput, hostInput, n * sizeof(uint));

    // Determine grid and block dimensions
    int blockSize = MAX_THREADS_PER_BLOCK;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch GPU kernel
    auto start_gpu = std::chrono::high_resolution_clock::now();
    if (strcmp(argv[1], "scan-work-inefficient") == 0) {
        workInefficient_inclusiveScan << <gridSize, blockSize, blockSize * sizeof(uint) >> > (deviceInput, deviceOutput, n);
    }
    else if (strcmp(argv[1], "scan-work-efficient") == 0) {
        workEfficient_inclusiveScan << <gridSize, blockSize, 2 * blockSize * sizeof(uint) >> > (deviceInput, deviceOutput, n);
    }
    else {
        fprintf(stderr, "Invalid kernel type. Choose either 'scan-work-efficient' or 'scan-work-inefficient'.\n");
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;
    printf("GPU execution time: %.3f ms\n", duration_gpu.count());

    // Copy output data from device
    copyFromDevice(hostOutput, deviceOutput, n * sizeof(uint));

    // Run CPU implementation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_normal_inclusiveScan(hostInput, cpuOutput, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    printf("CPU execution time: %.3f ms\n", duration_cpu.count());

    // Compare results
    bool matched = compareResults(cpuOutput, hostOutput, n);
    if (matched) {
        printf("CPU and GPU results match.\n");
    }
    else {
        printf("CPU and GPU results do not match.\n");
    }

    // Deallocate memory
    free(hostInput);
    free(hostOutput);
    free(cpuOutput);
    deallocateDeviceMemory(deviceInput);
    deallocateDeviceMemory(deviceOutput);

    return 0;
}
