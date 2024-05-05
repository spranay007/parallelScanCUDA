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

__global__ void workEfficient_inclusiveScan(uint* input, uint* output, uint n, uint* blockSums) {
    __shared__ uint shared[2*1024*sizeof(uint)];
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
        // Debug: Print shared memory state after each up-sweep step
        /*
        if (tid == 0) {
            printf("Block %d - After up-sweep stride %d: ", blockIdx.x, stride);
            for (int i = 0; i < blockDim.x; i++) {
                printf("%u ", shared[i]);
            }
            printf("\n");
        }
        */
        
    }

    // Down-sweep phase
    for (int stride = blockDim.x / 4; stride > 0;  stride /= 2) {
        int index = 2 * stride * (tid + 1) - 1;
        if (index + stride < blockDim.x) {
            shared[index + stride] += shared[index];
        }
        __syncthreads();
        // Debug: Print shared memory state after each down-sweep step
        /*
        if (tid == 0) {
            printf("Block %d - After down-sweep stride %d: ", blockIdx.x, stride);
            for (int i = 0; i < blockDim.x; i++) {
                printf("%u ", shared[i]);
            }
            printf("\n");
        }
        */
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
__global__ void scanBlockSums(uint* blockSums, int numBlocks) {
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

__global__ void addBlockSumsToOutput(uint* output, uint* blockSums, uint n, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        uint before = output[idx];  // Store the current value before addition
        if (blockIdx.x > 0) {
            output[idx] += blockSums[blockIdx.x - 1];
            /*
            if(blockIdx.x > 5)
            printf("Thread %d (Block %d, Index %d): Before = %u, BlockSum[%d] = %u, After = %u\n",
                threadIdx.x, blockIdx.x, idx, before, blockIdx.x - 1, blockSums[blockIdx.x - 1], output[idx]);
                */
        }
        else {
            //printf("Thread %d (Block %d, Index %d): Initial Value = %u\n", threadIdx.x, blockIdx.x, idx, before);
        }
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
    if (n > 2* MAX_THREADS_PER_BLOCK * MAX_BLOCKS) {
        fprintf(stderr, "Input size should be at most %d\n", 2*MAX_THREADS_PER_BLOCK * MAX_BLOCKS);
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
        //hostInput[i] = rand() % 100;
        hostInput[i] = 1;
    }

    // Allocate device memory
    allocateDeviceMemory(&deviceInput, n * sizeof(uint));
    allocateDeviceMemory(&deviceOutput, n * sizeof(uint));

    // Copy input data to device
    copyToDevice(deviceInput, hostInput, n * sizeof(uint));

    // Calculate the total number of elements in the input list
    size_t totalElements = static_cast<size_t>(n);

    // Determine the maximum number of blocks and threads per block that can be used
    int maxThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int maxBlocks = MAX_BLOCKS;

    // Calculate the number of blocks and threads per block needed to process the entire input list
    int threadsPerBlock = maxThreadsPerBlock;
    
  
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // Ensures all elements are covered
    cout <<"Total Blocks : " << numBlocks << endl;

    uint* deviceBlockSums;
    allocateDeviceMemory(&deviceBlockSums, numBlocks * sizeof(uint));
    size_t sharedMemorySize = 2 * threadsPerBlock * sizeof(uint);
    size_t sharedMemSize = numBlocks * sizeof(uint);
    
    // Declare a host array to store block sums from the device
    uint* hostBlockSums = (uint*)malloc(numBlocks * sizeof(uint));
    uint* hostBlockSums2 = (uint*)malloc(numBlocks * sizeof(uint));
    // Launch GPU kernel
    auto start_gpu = std::chrono::high_resolution_clock::now();
    if (strcmp(argv[1], "-work-inefficient") == 0) {
        //to be implemented later 
        //workInefficient_inclusiveScan << <numBlocks, threadsPerBlock, threadsPerBlock * sizeof(uint) >> > (deviceInput, deviceOutput, static_cast<uint>(totalElements));
    }
    else if (strcmp(argv[1], "-work-efficient") == 0) {
        workEfficient_inclusiveScan <<< numBlocks, threadsPerBlock, sharedMemorySize >>> (deviceInput, deviceOutput, static_cast<uint>(totalElements), deviceBlockSums);
        cudaDeviceSynchronize();
        ///////////////////////////////////
        // Copy block sums from device to host
        //copyFromDevice(hostBlockSums, deviceBlockSums, numBlocks * sizeof(uint));
        // Print block sums
        //cout << "Block sums Work Efficient Kernel:" << endl;
        //for (int i = 0; i < numBlocks; i++) {
        //    cout << "Block " << i << ": " << hostBlockSums[i] << endl;
        //}
        //copyFromDevice(hostOutput, deviceOutput, n * sizeof(uint));
        //cout << "Block sums Work Efficient Kernel:" << endl;
        //for (int i = 0; i < totalElements; i++) {
        //    cout << "Block " << i << ": " << hostOutput[i] << "  ";
       // }
        //cout << endl;

        ///////////////////////////////////
        scanBlockSums <<< 1, numBlocks, sharedMemSize >>> (deviceBlockSums, numBlocks); //numBlocks *sizeof(uint)
        cudaDeviceSynchronize();
        
        addBlockSumsToOutput <<< numBlocks, MAX_THREADS_PER_BLOCK >>> (deviceOutput, deviceBlockSums, static_cast<uint>(totalElements), numBlocks);
        cudaDeviceSynchronize();
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
    cout << "kernel run successful" << endl;
    copyFromDevice(hostOutput, deviceOutput, n * sizeof(uint));
    cout << "Copy run successful" << endl;
    
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
    deallocateDeviceMemory(deviceBlockSums);
    return 0;
}
