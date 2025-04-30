#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>


__global__ void computeHistogramKernel(const int* input, int* histogram, int N, int B) {
    // implementation of histogram computation
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
        int *d_input, *d_histogram;
        
        // Copy data to device
        
        // Launch kernel
        
        computeHistogramKernel<<< Dg, Db, Ns, S >>>(d_input, d_histogram, N, B);
        
        // Copy result back to host
        
        // Cleanup
        
        // Write output
        sol_fs.write(reinterpret_cast<const char*>(histogram.get()), sizeof(int) * B);
        sol_fs.close();
        
        return sol_path;
    }
}