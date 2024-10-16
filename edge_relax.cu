#include <assert.h>
#include <stdio.h>

#include "graph_types.h"

#define MIN(a, b) ((a)<(b))?((a)):((b))

#define CUDA_DEBUG_REACHED_PRINT   printf("%s:%d [%s] [%u,%u,%u] in [%u,%u,%u] ",\
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

#undef ASK_THREAD_N

#define ASK_THREAD_N(n) if(tid == (n)) CUDA_DEBUG_REACHED_PRINT

#define ASK_THREAD_ZERO ASK_THREAD_N(0)
    

__global__ void relax_edges(size_t* queue,
                            size_t* tmp_queue,
                            size_t queue_capacity,
                            size_t* queue_size,
                            adj_vert_t** adj_list,
                            size_t* adj_lens,
                            unsigned int* dist)
{
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    size_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    extern __shared__ char* shmem[];
    unsigned int* queue_size_local = (unsigned int*)&(shmem[0]);
    if(tid == 0) (*queue_size_local) = 0;
    __syncthreads();

    for(unsigned int i=threadIdx.x; i < (*queue_size); i += blockDim.x) {
        size_t u_idx = queue[i];
        adj_vert_t* adj_of_u = adj_list[u_idx];
        size_t adj_of_u_size = adj_lens[u_idx];
        for(unsigned int k=threadIdx.y; k < adj_of_u_size; k += blockDim.y) {
            size_t v_idx = adj_of_u[k].idx;
            
            float uv_edge_weight = adj_of_u[k].weight;
            unsigned int new_dist = atomicAdd(&dist[u_idx], 0) + uv_edge_weight;
            unsigned int old_dist = atomicMin(&dist[v_idx], new_dist);
            if(old_dist > new_dist) {
                // v gets added to queue
                unsigned int idx = atomicInc(queue_size_local, queue_capacity);
                tmp_queue[idx] = v_idx;
            }
        }
    }

    __syncthreads();

    if(tid == 0) {
        *queue_size = *queue_size_local;
    }
    return;
    
}
