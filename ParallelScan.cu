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

using namespace std;

// Utility functions
cudaError_t allocateDeviceMemory(uint** devicePtr, size_t size) {
    cudaError_t err = cudaMalloc((void**)devicePtr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}

cudaError_t copyToDevice(uint* devicePtr, const uint* hostPtr, size_t size) {
    cudaError_t err = cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory to the device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}

cudaError_t copyFromDevice(uint* hostPtr, const uint* devicePtr, size_t size) {
    cudaError_t err = cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory from the device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}

cudaError_t deallocateDeviceMemory(uint* devicePtr) {
    cudaError_t err = cudaFree(devicePtr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}

__global__ void workEfficient_inclusiveScan(uint* input, uint* output, uint n, uint* blockSums) {
    __shared__ uint shared[2 * 1024 * sizeof(uint)];
    int tid = threadIdx.x;
    int bIdx = blockIdx.x;
    int idx = bIdx * blockDim.x + tid;

    // Load input into shared memory
    shared[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // Up-sweep (Reduction) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            shared[index] += shared[index - stride];
        }
        __syncthreads();
    }

    // Down-sweep phase
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index + stride < blockDim.x) {
            shared[index + stride] += shared[index];
        }
        __syncthreads();
    }

    // Write the processed data back to the output
    if (idx < n) {
        output[idx] = shared[tid];
    }

    // Last thread writes the last element to blockSums
    if (tid == blockDim.x - 1) {
        blockSums[bIdx] = shared[blockDim.x - 1];
    }
}

//////////////////////////////////////////////////////////////////////////////////////
__global__ void scanBlockSumsVanila(uint* blockSums, int numBlocks) {
    extern __shared__ uint temp[];
    int tid = threadIdx.x;
    // Load block sums into shared memory
    if (tid < numBlocks) {
        temp[tid] = blockSums[tid];
    }
    else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Inclusive scan using up-sweep
    for (unsigned int stride = 1; stride < numBlocks; stride *= 2) {
        //__syncthreads();
        int index = (tid + 1) * 2 * stride - 1;
        if (index < numBlocks) {//blockDim.x
            temp[index] += temp[index - stride];
        }
        __syncthreads();
        /*
        if (tid == 0) {
            printf("After up sweep stride %d: ", stride);
            for (int i = 0; i < numBlocks; i++) {
                printf("%d ", temp[i]);
            }
            printf("\n");
        }
        */
    }

    // Set the last element to zero to start the down-sweep
    if (tid == numBlocks - 1) {
        temp[numBlocks - 1] = 0;
    }
    __syncthreads();

    // Down-Sweep
    for (int stride = 1 << (31 - __clz(numBlocks - 1)); stride > 0; stride >>= 1) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index + stride < numBlocks) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
        // Debug print after each stride
        /*
        if (tid == 0) {
            printf("After down sweep stride %d: ", stride);
            for (int i = 0; i < numBlocks; i++) {
                printf("%d ", temp[i]);
            }
            printf("\n");
        }
        */
    }
    // Write back to the block sums
    __syncthreads();

    if (tid < numBlocks) {
        blockSums[tid] = temp[tid];
    }
    //__syncthreads();
    /*
    // Correct the last element if numBlocks is not a power of two
    if (tid == numBlocks - 1 && numBlocks & (numBlocks - 1)) {
        blockSums[tid] = temp[tid - 1] + blockSums[tid - 1];
    }
    */
}
////////////////////////////////////////////////////////////////////////////////////////////
__global__ void scanBlockSums(uint* blockSums, uint* blockSumsNext, int numBlocks) {
    extern __shared__ uint temp[];
    int tid = threadIdx.x;

    // Load block sums into shared memory
    if (tid < numBlocks) {
        temp[tid] = blockSums[tid];
    }
    else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Inclusive scan using up-sweep
    for (unsigned int stride = 1; stride < numBlocks; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < numBlocks) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Set the last element to zero to start the down-sweep
    if (tid == numBlocks - 1) {
        temp[numBlocks - 1] = 0;
    }
    __syncthreads();

    // Down-Sweep
    for (int stride = 1 << (31 - __clz(numBlocks - 1)); stride > 0; stride >>= 1) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index + stride < numBlocks) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    // Write back to the block sums
    if (tid < numBlocks) {
        blockSumsNext[tid] = temp[tid];
    }
}

__global__ void addBlockSumsToOutput(uint* output, uint* blockSums, uint n, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (blockIdx.x > 0) {
            output[idx] += blockSums[blockIdx.x - 1];
        }
    }
}
__global__ void scanBlockSumsVanilla(uint* blockSums, int numBlocks) {
    extern __shared__ uint temp[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < numBlocks) {
        temp[tid] = blockSums[index];
    }
    else {
        temp[tid] = 0;  // Important to avoid out-of-bounds access in shared memory
    }
    __syncthreads();

    // Perform up-sweep
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int accessIndex = (tid + 1) * 2 * stride - 1;
        if (accessIndex < blockDim.x) {
            temp[accessIndex] += temp[accessIndex - stride];
        }
        __syncthreads();
    }

    // Down-Sweep
    for (int stride = 1 << (31 - __clz(blockDim.x - 1)); stride > 0; stride >>= 1) {
        int accessIndex = (tid + 1) * 2 * stride - 1;
        if (accessIndex + stride < blockDim.x) {
            temp[accessIndex + stride] += temp[accessIndex];
        }
        __syncthreads();
    }

    if (index < numBlocks) {
        blockSums[index] = temp[tid];
    }
}


// Improved recursive function with enhanced error handling
cudaError_t recursiveScanBlockSums(uint* blockSums, int numBlocks, uint* tempBuffer = nullptr) {
    bool isRootCall = (tempBuffer == nullptr);
    cudaError_t cudaStatus;

    if (isRootCall) {
        size_t totalBufferSize = numBlocks * sizeof(uint);
        cudaStatus = cudaMalloc((void**)&tempBuffer, totalBufferSize);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to allocate temporary buffer: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
    }

    if (numBlocks > 1) {
        int threads = min(numBlocks, MAX_THREADS_PER_BLOCK);
        int blocks = (numBlocks + threads - 1) / threads;
        uint* blockSumsNext = tempBuffer;

        scanBlockSumsVanilla << <blocks, threads, threads * sizeof(uint) >> > (blockSums, numBlocks);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CUDA Kernel error: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        int nextNumBlocks = (numBlocks + threads - 1) / threads;
        cudaStatus = recursiveScanBlockSums(blockSumsNext, nextNumBlocks, tempBuffer + nextNumBlocks * sizeof(uint));
        if (cudaStatus != cudaSuccess) {
            return cudaStatus;
        }
    }

    if (isRootCall) {
        cudaFree(tempBuffer);
    }

    return cudaSuccess;
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
    if (n > 2 * MAX_THREADS_PER_BLOCK * MAX_BLOCKS) {
        fprintf(stderr, "Input size should be at most %d\n", 2 * MAX_THREADS_PER_BLOCK * MAX_BLOCKS);
        exit(EXIT_FAILURE);
    }

    uint* hostInput = (uint*)malloc(n * sizeof(uint));
    uint* hostOutput = (uint*)malloc(n * sizeof(uint));
    uint* cpuOutput = (uint*)malloc(n * sizeof(uint));
    uint* deviceInput;
    uint* deviceOutput;
    uint* deviceBlockSums;

    // Initialize input data
    srand(time(NULL));
    for (uint i = 0; i < n; i++) {
        hostInput[i] = 1; // Simple input to demonstrate inclusive scan
    }

    // Allocate device memory
    allocateDeviceMemory(&deviceInput, n * sizeof(uint));
    allocateDeviceMemory(&deviceOutput, n * sizeof(uint));

    // Copy input data to device
    copyToDevice(deviceInput, hostInput, n * sizeof(uint));

    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    allocateDeviceMemory(&deviceBlockSums, numBlocks * sizeof(uint));

    // Launch work-efficient scan
    workEfficient_inclusiveScan << <numBlocks, threadsPerBlock >> > (deviceInput, deviceOutput, n, deviceBlockSums);
    cudaDeviceSynchronize();

    // Handle block sums recursively
    recursiveScanBlockSums(deviceBlockSums, numBlocks, nullptr);
    //scanBlockSumsVanila << < 1, numBlocks, numBlocks * sizeof(uint) >> > (deviceBlockSums, numBlocks);
    cudaDeviceSynchronize();
    // Add block sums back to the output
    addBlockSumsToOutput << <numBlocks, threadsPerBlock >> > (deviceOutput, deviceBlockSums, n, numBlocks);
    cudaDeviceSynchronize();

    // Timing GPU execution
    auto start_gpu = chrono::high_resolution_clock::now();
    // Copy output data from device
    copyFromDevice(hostOutput, deviceOutput, n * sizeof(uint));
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_gpu = end_gpu - start_gpu;
    printf("GPU execution time: %.3f ms\n", duration_gpu.count());

    // CPU inclusive scan for comparison
    auto start_cpu = chrono::high_resolution_clock::now();
    cpu_normal_inclusiveScan(hostInput, cpuOutput, n);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_cpu = end_cpu - start_cpu;
    printf("CPU execution time: %.3f ms\n", duration_cpu.count());

    // Compare CPU and GPU results
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
    deallocateDeviceMemory(deviceBlockSums);

    return 0;
}

