#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>

#define PADDED(i) (i + (i / 32))
#define WARP_SIZE 32

__global__ void computeHistogramKernel(const int* input, int* global_histogram, int N, int B) {
    extern __shared__ int shared_array[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    int* shared_hist = &shared_array[warp_id * B];

    for (int i = lane_id; i < B; i += WARP_SIZE) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    if (global_id < N) {
        int bin = input[global_id];
        if (bin >= 0 && bin < B) {
            atomicAdd(&shared_hist[bin], 1);
        }
    }
    __syncthreads();

    if (tid < B) {
        int total = 0;
        for (int w = 0; w < warps_per_block; ++w) {
            total += shared_hist[w * B + tid];
        }
        atomicAdd(&global_histogram[tid], total);
    }
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
        int threads_per_block = 256;
        int blocks = (N + threads_per_block - 1) / threads_per_block;
        int warps_per_block = threads_per_block / WARP_SIZE;
        size_t shared_mem_size = warps_per_block * PADDED(B) * sizeof(int);

        // Launch naive kernel
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
