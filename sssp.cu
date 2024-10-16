
#include <cmath>
#include <cassert>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>


#include "graph_types.h"
#include "debug_helpers.h"



struct queue_elem {
    size_t idx;
    queue_elem* next;
};


__global__ void realloc_queue_vec(size_t* queue, unsigned int* queue_size,
                                  int* bits, size_t n_vertx);


__global__ void relax_edges(size_t* queue,
                            unsigned int queue_size,
                            int* future_queue_bits,
                            size_t n_vertx,
                            adj_vert_t** adj_list,
                            size_t* adj_lens,
                            unsigned int* dist);


void single_source_shortest_paths(
    size_t n_vertx,
    size_t* adj_lens,
    adj_vert_t** adj_list,
    unsigned int* dist,
    size_t start) {
    
    CUDA_SAFE_CALL(cudaMemset(dist, 0xFF, sizeof(unsigned int)*n_vertx));

    size_t* queue_vec;
//    size_t* tmp_queue_vec;
    unsigned int* queue_size;
    int* future_queue_bits;
    CUDA_SAFE_CALL(cudaMallocManaged(&queue_size, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMallocManaged(&queue_vec, sizeof(size_t)*n_vertx));
    CUDA_SAFE_CALL(cudaMallocManaged(&future_queue_bits, sizeof(int)*n_vertx));
//    CUDA_SAFE_CALL(cudaMallocManaged(&tmp_queue_vec, sizeof(size_t)*queue_capacity));
    *queue_size = 0;
    queue_vec[*queue_size] = start;
    (*queue_size)++;
    dist[start] = 0.0;
    
    while(*queue_size) {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        relax_edges<<<1, dim3(32, 32, 1)>>>
            (queue_vec, *queue_size, future_queue_bits, n_vertx, adj_list, adj_lens, dist);

        realloc_queue_vec<<<1, 1>>>(queue_vec, queue_size,
                                    future_queue_bits,
                                    n_vertx);
        
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaFree(queue_vec));
}


