
#include <cmath>
#include <cassert>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>


#include "graph_types.h"
#include "debug_helpers.h"

#include "utlist.h"

__global__ void setup_dist_arr(float* , size_t);

struct queue_elem {
    size_t idx;
    queue_elem* next;
};





__global__ void relax_edges(size_t* queue,
                            size_t* tmp_queue,
                            size_t queue_capacity,
                            size_t* queue_size,
                            adj_vert_t** adj_list,
                            size_t* adj_lens,
                            unsigned int* dist);

__global__ void calc_queue_capacity(size_t*, size_t*, size_t* adj_lens);


void single_source_shortest_paths(
    size_t n_vertx,
    size_t* adj_lens,
    adj_vert_t** adj_list,
    unsigned int* dist,
    size_t start) {

    
    // {
    //     size_t blockdim = (size_t)std::ceil(n_vertx/1024.0/1024.0);
    //     //static_cast<size_t>(std::ceil(static_cast<double>(n_vertx)/1024.0));
    //     for(size_t i=0; i<n_vertx; ++i) dist[i] = 0;
    //     setup_dist_arr<<<blockdim, 1024>>>(dist, n_vertx);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // }

    CUDA_SAFE_CALL(cudaMemset(dist, 0xFF, sizeof(unsigned int)*n_vertx));

    size_t* queue_vec;
    size_t* tmp_queue_vec;
    size_t queue_capacity = n_vertx;
    size_t* queue_size;
    CUDA_SAFE_CALL(cudaMallocManaged(&queue_size, sizeof(size_t)));
    CUDA_SAFE_CALL(cudaMallocManaged(&queue_vec, sizeof(size_t)*queue_capacity));
    CUDA_SAFE_CALL(cudaMallocManaged(&tmp_queue_vec, sizeof(size_t)*queue_capacity));
    
    // queue_elem queue;
    // queue.next = nullptr;
    // queue.idx = 0;
    {
        // queue_elem* start_elem;
        // CUDA_SAFE_CALL(cudaMallocManaged(&start_elem, sizeof(queue_elem)));
        // start_elem->idx = start;
        queue_vec[*queue_size] = start;
        (*queue_size)++;
        dist[start] = 0.0;
        // queue_insert(&queue, start_elem, dist);
    }


    
    // bool* visited;
    // {
    //     CUDA_SAFE_CALL(cudaMallocManaged(&visited, sizeof(bool)*n_vertx));
    //     CUDA_SAFE_CALL(cudaMemset(visited, static_cast<int>(false), sizeof(bool)*n_vertx)); // debug reconsider
    // }


    
    while(*queue_size) {
        
        // queue_elem* temp = queue_pop(&queue);
        // size_t current = temp->idx;
        // CUDA_SAFE_CALL(cudaFree(temp));
        // visited[current] = true;
        // for(size_t i=0; i<adj_lens[current]; ++i) {
        //     size_t elem_idx = adj_list[current][i].idx;
        //     queue_elem* elem;            
        //     if(visited[elem_idx]) continue;
        //     dist[elem_idx] = std::min(dist[elem_idx], dist[current] + adj_list[current][i].weight);
        //     if(std::isinf(dist[elem_idx])) {
        //         CUDA_SAFE_CALL(cudaMallocManaged(&elem, sizeof(queue_elem)));
        //         elem->idx = elem_idx;
        //         queue_insert(&queue, elem, dist);
        //     }
        // }

        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        relax_edges<<<1, dim3(8, 8, 1), sizeof(unsigned int), 0>>>
            (queue_vec, tmp_queue_vec, queue_capacity, queue_size,
             adj_list, adj_lens, dist);

        // assert(!(queue_size >= queue_capacity));
        // calc_queue_capacity<<<1, 1024, sizeof(size_t)*32>>>
        //     (&queue_capacity, queue_size, tmp_queue_vec, adj_lens);
        
     
        std::swap(queue_vec, tmp_queue_vec);

        // CUDA_SAFE_CALL(cudaFree(tmp_queue_vec));
        // CUDA_SAFE_CALL(cudaMallocManaged(&tmp_queue_vec, sizeof(size_t)*queue_capacity));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    
    CUDA_SAFE_CALL(cudaFree(tmp_queue_vec));
    CUDA_SAFE_CALL(cudaFree(queue_vec));
}


