#pragma once

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>


static inline auto& putstrln(std::string s) { return (std::cout << s << std::endl); }
static inline auto& putstr(std::string s) { return (std::cout << s); }

#define CUDA_SAFE_CALL(stmt) checkCuda((stmt), __FILE__, __LINE__)
#define DEBUG_PRINT \
  putstr(std::string(__FILE__) + ":" + std::to_string(__LINE__) \
           + " [" + std::string(__func__) + "] ")

#define DEBUG_PRINTLN \
  putstrln(std::string(__FILE__) + ":" + std::to_string(__LINE__) \
           + " [" + std::string(__func__) + "] ")

#define DEBUG_COUT \
  (std::cout << std::string(__FILE__) + ":" + std::to_string(__LINE__) \
   + " [" + std::string(__func__) + "]: ")


static inline cudaError_t checkCuda(cudaError_t e, std::string file, int line) {
    if(e != cudaSuccess) {
        std::cerr << file << ':' << line << ": "
                  << cudaGetErrorName(e) << ": "
                  << cudaGetErrorString(e) << std::endl;
    }
    return e;
}
