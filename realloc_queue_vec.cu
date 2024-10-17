#include "cuda_debug_helpers.h"

__global__ void realloc_queue_vec(size_t* queue, unsigned int* queue_size,
                              int* bits, size_t n_vertx) {
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    size_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    size_t n_threads = blockDim.x * blockDim.y * blockDim.z;
    if(tid == 0) *queue_size = 0;

    __syncthreads();
    
    for(size_t i=tid; i < n_vertx; i += n_threads) {
        if(bits[i]) {
            size_t idx = atomicInc(queue_size, n_vertx);
            queue[idx] = (size_t)i;}
    }
}
