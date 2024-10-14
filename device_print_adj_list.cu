#include "graph_types.h"

extern
__global__
void
device_print_adj_list(size_t* adj_list_lens, size_t n_vertx, adj_vert_t** adj_list) {
    printf("vertex count: %llu\n", n_vertx);
}
