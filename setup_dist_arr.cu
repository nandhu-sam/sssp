#include <assert.h>
#include <math.h>

#define MIN(a, b) ((a)<(b))?((a)):((b))

__global__ void setup_dist_arr(float* dist, size_t n_vertx) {
    
    size_t start_idx = (1024*threadIdx.x) + (1024*1024*blockIdx.x);
    size_t end_idx = MIN(start_idx + 1024, n_vertx);
    
    for(size_t idx=start_idx; idx < end_idx; ++idx) 
        dist[idx] = INFINITY;
}
