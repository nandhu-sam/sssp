#include <assert.h>
#include <stdio.h>

#define CUDA_DEBUG_REACHED_PRINT   printf("%s:%d [%s] [%u,%u,%u] in [%u,%u,%u] ", \
                                          __FILE__, __LINE__, __func__, \
                                          threadIdx.x, threadIdx.y, threadIdx.z,\
                                          blockIdx.x, blockIdx.y, blockIdx.z)

#define CUDA_DEBUG_REACHED_PRINTLN printf("%s:%d [%s] [%u,%u,%u] in [%u,%u,%u]\n",\
                                          __FILE__, __LINE__, __func__, \
                                          threadIdx.x, threadIdx.y, threadIdx.z,\
                                          blockIdx.x, blockIdx.y, blockIdx.z)


#define CUDA_DEBUG_PRINTF(s) printf("%s:%d [%s] [%u,%u,%u] in [%u,%u,%u] %s\n", \
                                          __FILE__, __LINE__, __func__, \
                                          threadIdx.x, threadIdx.y, threadIdx.z,\
                                    blockIdx.x, blockIdx.y, blockIdx.z, (s))

#define CUDA_DEBUG_PRINTF_INT(s) printf("%s:%d [%s] [%u,%u,%u] in [%u,%u,%u] %d\n", \
                                          __FILE__, __LINE__, __func__, \
                                          threadIdx.x, threadIdx.y, threadIdx.z,\
                                        blockIdx.x, blockIdx.y, blockIdx.z, (s))

#define ASK_THREAD_N(n) if(threadIdx.x + \
                           threadIdx.y * blockDim.x + \
                           threadIdx.z * blockDim.y * blockDim.z == (n)) \
        CUDA_DEBUG_REACHED_PRINTLN


#define ASK_THREAD_ZERO ASK_THREAD_N(0)
