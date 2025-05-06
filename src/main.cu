#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>

#define PADDED(i) (i + (i / 32))

__global__ void computeHistogramKernel(const int* input, int* global_histogram, int N, int B) {
    __shared__ int shared_hist[1024 + 32];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < B; i += blockDim.x) {
        shared_hist[PADDED(i)] = 0;
    }
    __syncthreads();

    if (global_id < N) {
        int bin = input[global_id];
        if (bin >= 0 && bin < B) {
            atomicAdd(&shared_hist[PADDED(bin)], 1);
        }
    }
    __syncthreads();

    for (int i = tid; i < B; i += blockDim.x) {
        atomicAdd(&global_histogram[i], shared_hist[PADDED(i)]);
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
        dim3 Db(256);  // 256 threads per block
        dim3 Dg((N + Db.x - 1) / Db.x);  // enough blocks to cover N elements

        // Launch naive kernel
        computeHistogramKernel<<<Dg, Db, sizeof(int) * B>>>(d_input, d_histogram, N, B);
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
