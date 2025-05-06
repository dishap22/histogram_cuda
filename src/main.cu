#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>
#include <chrono>

void computeHistogramCPU(const int* input, int* histogram, int N, int B) {
    for (int i = 0; i < N; i++) {
        int bin = input[i];
        if (bin >= 0 && bin < B) {
            histogram[bin]++;
        }
    }
}

__global__ void computeHistogramKernelNaive(const int* input, int* histogram, int N, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int bin = input[idx];
        if (bin >= 0 && bin < B) {
            atomicAdd(&histogram[bin], 1);
        }
    }
}

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void computeHistogramKernel(const int* __restrict__ input, int* __restrict__ global_histogram, int N, int B) {
    const int padding = 1;
    const int padded_B = B + padding;

    extern __shared__ int shared_array[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    int* shared_hist = shared_array;

    for (int i = tid; i < padded_B; i += blockDim.x) {
        if (i < B) {
            shared_hist[i] = 0;
        }
    }
    __syncthreads();

    for (int i = global_id; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < B) {
            atomicAdd(&shared_hist[bin], 1);
        }
    }
    __syncthreads();

    for (int bin = tid; bin < B; bin += blockDim.x) {
        int count = shared_hist[bin];
        if (count > 0) {
            atomicAdd(&global_histogram[bin], count);
        }
    }
}

int getOptimalGridSize(int blockSize, int N) {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, computeHistogramKernel, blockSize, 0);

    int maxBlocks = maxActiveBlocks * props.multiProcessorCount;
    int blocksNeeded = (N + blockSize - 1) / blockSize;

    return min(blocksNeeded, maxBlocks);
}

namespace solution {
    std::string compute(const std::string &input_path, int N, int B) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_histogram.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream input_fs(input_path, std::ios::binary);

        const auto input_data = std::make_unique<int[]>(N);
        input_fs.read(reinterpret_cast<char*>(input_data.get()), sizeof(int) * N);
        input_fs.close();

        // Prepare histogram on the host
        auto histogram = std::make_unique<int[]>(B);
        for (int i = 0; i < B; i++) histogram[i] = 0;

        // CPU Histogram Benchmarking
        auto cpu_histogram = std::make_unique<int[]>(B);
        std::copy(histogram.get(), histogram.get() + B, cpu_histogram.get());

        auto start = std::chrono::high_resolution_clock::now();
        computeHistogramCPU(input_data.get(), cpu_histogram.get(), N, B);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> cpu_duration = end - start;
        float cpu_time_seconds = cpu_duration.count();
        std::cout << "CPU Histogram Execution Time: " << cpu_time_seconds << " seconds\n";

        // Allocate device memory
        int *d_input = nullptr;
        int *d_histogram = nullptr;
        cudaMalloc(&d_input, sizeof(int) * N);
        cudaMalloc(&d_histogram, sizeof(int) * B);

        // Copy input data to device
        cudaMemcpy(d_input, input_data.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
        cudaMemset(d_histogram, 0, sizeof(int) * B);

        // Naive GPU Histogram Benchmarking
        dim3 Db(256);  // 256 threads per block
        dim3 Dg((N + Db.x - 1) / Db.x);  // enough blocks to cover N elements

        cudaEvent_t start_gpu_naive, stop_gpu_naive;
        cudaEventCreate(&start_gpu_naive);
        cudaEventCreate(&stop_gpu_naive);

        cudaEventRecord(start_gpu_naive);
        computeHistogramKernelNaive<<<Dg, Db>>>(d_input, d_histogram, N, B);
        cudaEventRecord(stop_gpu_naive);

        cudaDeviceSynchronize();
        float gpu_naive_ms = 0;
        cudaEventElapsedTime(&gpu_naive_ms, start_gpu_naive, stop_gpu_naive);
        float gpu_naive_seconds = gpu_naive_ms / 1000.0f;  // Convert milliseconds to seconds
        std::cout << "Naive GPU Histogram Execution Time: " << gpu_naive_seconds << " seconds\n";

        // Shared Memory GPU Histogram Benchmarking
        int *d_shared_histogram = nullptr;
        cudaMalloc(&d_shared_histogram, sizeof(int) * B);

        cudaMemset(d_shared_histogram, 0, sizeof(int) * B);

        int opt_threads = 256;
        int opt_blocks = getOptimalGridSize(opt_threads, N);
        size_t shared_mem_size = (B + 1) * sizeof(int);

        cudaEvent_t start_gpu_shared, stop_gpu_shared;
        cudaEventCreate(&start_gpu_shared);
        cudaEventCreate(&stop_gpu_shared);

        cudaEventRecord(start_gpu_shared);
        computeHistogramKernel<<<opt_blocks, opt_threads, shared_mem_size>>>(d_input, d_shared_histogram, N, B);
        cudaEventRecord(stop_gpu_shared);

        cudaDeviceSynchronize();
        float gpu_shared_ms = 0;
        cudaEventElapsedTime(&gpu_shared_ms, start_gpu_shared, stop_gpu_shared);
        float gpu_shared_seconds = gpu_shared_ms / 1000.0f;  // Convert milliseconds to seconds
        std::cout << "Shared Memory GPU Histogram Execution Time: " << gpu_shared_seconds << " seconds\n";

        // Copy result back to host for Shared Memory GPU
        cudaMemcpy(histogram.get(), d_shared_histogram, sizeof(int) * B, cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_histogram);
        cudaFree(d_shared_histogram);

        // Write the output histogram for any of the results
        sol_fs.write(reinterpret_cast<const char*>(histogram.get()), sizeof(int) * B);
        sol_fs.close();

        return sol_path;
    }
}
