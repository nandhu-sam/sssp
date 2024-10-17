#include <assert.h>
#include <stdio.h>

#include "graph_types.h"
#include "cuda_debug_helpers.h"


__global__ void relax_edges(size_t* queue,
                            unsigned int queue_size,
                            int* future_queue_bits,
                            size_t n_vertx,
                            adj_vert_t** adj_list,
                            size_t* adj_lens,
                            unsigned int* dist)
{
    size_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    size_t n_threads = blockDim.x * blockDim.y * blockDim.z;


    for(size_t i=tid; i < n_vertx; i += n_threads)
        future_queue_bits[i] = 0;

    __syncthreads();
    for(unsigned int i=threadIdx.x; i < queue_size; i += blockDim.x) {
        size_t u_idx = queue[i];
        adj_vert_t* adj_of_u = adj_list[u_idx];
        size_t adj_of_u_size = adj_lens[u_idx];
        for(unsigned int k=threadIdx.y; k < adj_of_u_size; k += blockDim.y) {
            size_t v_idx = adj_of_u[k].idx;
            float uv_edge_weight = adj_of_u[k].weight;
            unsigned int new_dist = atomicAdd(&dist[u_idx], 0) + uv_edge_weight;
            unsigned int old_dist = atomicMin(&dist[v_idx], new_dist);
            if(old_dist > new_dist) future_queue_bits[v_idx] = 1;
        }
    }
}
