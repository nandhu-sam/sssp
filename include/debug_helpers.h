#pragma once

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>


static inline auto& putstrln(std::string s) { return (std::cout << s << std::endl); }

#define CUDA_SAFE_CALL(stmt) checkCuda((stmt), __FILE__, __LINE__)


static inline cudaError_t checkCuda(cudaError_t e, std::string file, int line) {
    if(e != cudaSuccess) {
        std::cerr << file << ':' << line << ": "
                  << cudaGetErrorName(e) << ": "
                  << cudaGetErrorString(e) << std::endl;
    }
    return e;
}
