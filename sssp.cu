
#include <cmath>
#include <cassert>

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



static inline bool queue_empty(queue_elem* head) {
    return head->next == nullptr;
}

static inline queue_elem* queue_pop(queue_elem* head) {
    assert(!queue_empty(head));
    queue_elem* t = head->next;
    head->next = t->next;
    return t;
}


static void queue_insert(queue_elem* head, queue_elem* elem, float* dist) {
    
    if(queue_empty(head)) {
        head->next = elem;
        elem->next = nullptr;
        return; 
    }

    queue_elem* t_prev = head->next;
    queue_elem* t = t_prev->next;

    while(t != nullptr) {
        if(dist[elem->idx] <= dist[t->idx]) break;
        t_prev = t;
        t = t->next;
    }

    elem->next = t;
    t_prev->next = elem;
}


// __global__ void relax_edges(size_t idx, adj_vert_t* adj_list, size_t adj_len, float* dist)

void single_source_shortest_paths(
    size_t n_vertx,
    size_t* adj_lens,
    adj_vert_t** adj_list,
    float* dist,
    size_t start) {

    
    {
        size_t blockdim = (size_t)std::ceil(n_vertx/1024.0/1024.0);
        //static_cast<size_t>(std::ceil(static_cast<double>(n_vertx)/1024.0));
        for(size_t i=0; i<n_vertx; ++i) dist[i] = 0;
        setup_dist_arr<<<blockdim, 1024>>>(dist, n_vertx);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
        
    queue_elem queue;
    queue.next = nullptr;
    {
        queue_elem* start_elem;
        CUDA_SAFE_CALL(cudaMallocManaged(&start_elem, sizeof(queue_elem)));
        start_elem->idx = start;
        dist[start] = 0.0;
        queue_insert(&queue, start_elem, dist);

    }


    
    bool* visited;
    {
        CUDA_SAFE_CALL(cudaMallocManaged(&visited, sizeof(bool)*n_vertx));
        CUDA_SAFE_CALL(cudaMemset(visited, static_cast<int>(false), sizeof(bool)*n_vertx));
    }



    while(!queue_empty(&queue)) {
        queue_elem* temp = queue_pop(&queue);

        size_t current = temp->idx;

        CUDA_SAFE_CALL(cudaFree(temp));
        visited[current] = true;

        // relax_edges<<<1, 1024>>>(current, adj_list[current], adj_lens[current], dist);

        for(size_t i=0; i<adj_lens[current]; ++i) {
            size_t idx = adj_list[current][i].idx;
            if(visited[idx]) continue;
            
            queue_elem* elem;
            size_t elem_idx = adj_list[current][i].idx;
            
            if(std::isinf(dist[idx])) {
                CUDA_SAFE_CALL(cudaMallocManaged(&elem, sizeof(queue_elem)));
                elem->idx = elem_idx;
                queue_insert(&queue, elem, dist);

            }

            dist[elem_idx] = std::min(dist[elem_idx],
                                       dist[current] + adj_list[current][i].weight);
        }
    }

    while(!queue_empty(&queue)) {
        auto p = queue_pop(&queue);
        CUDA_SAFE_CALL(cudaFree(p));
    }
}


