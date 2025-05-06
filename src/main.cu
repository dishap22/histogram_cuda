#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>

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

        // Read input data on host
        const auto input_data = std::make_unique<int[]>(N);
        input_fs.read(reinterpret_cast<char*>(input_data.get()), sizeof(int) * N);
        input_fs.close();

        // Allocate and initialize histogram on host
        auto histogram = std::make_unique<int[]>(B);
        for (int i = 0; i < B; i++) histogram[i] = 0;

        // Allocate device memory
        int *d_input = nullptr;
        int *d_histogram = nullptr;
        cudaMalloc(&d_input, sizeof(int) * N);
        cudaMalloc(&d_histogram, sizeof(int) * B);

        // Copy input data to device
        cudaMemcpy(d_input, input_data.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
        cudaMemset(d_histogram, 0, sizeof(int) * B);

        // Kernel launch parameters
        int threads_per_block = BLOCK_SIZE;
        int blocks = getOptimalGridSize(threads_per_block, N);
        const int padding = 1;
        const int padded_B = B + padding;
        size_t shared_mem_size = padded_B * sizeof(int);

        computeHistogramKernel<<<blocks, threads_per_block, shared_mem_size>>>(d_input, d_histogram, N, B);
        cudaDeviceSynchronize();  // Ensure kernel is done

        // Copy result back to host
        cudaMemcpy(histogram.get(), d_histogram, sizeof(int) * B, cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_histogram);

        // Write output
        sol_fs.write(reinterpret_cast<const char*>(histogram.get()), sizeof(int) * B);
        sol_fs.close();

        return sol_path;
    }
}