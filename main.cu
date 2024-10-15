
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "graph_loaders.h"
// #include "delta_stepping_fns.h"

#include "debug_helpers.h" // debug

#include "uthash.h"


extern __global__ void device_print_adj_list(size_t* adj_list_lens, size_t n_vertx, adj_vert_t** adj_list);


// struct frontier_list {
//     size_t idx;
//     struct frontier_list* next, prev;
// };

void single_source_shortest_paths(size_t n_vertx,
                                  size_t* adj_lens,
                                  adj_vert_t** adj_list,
                                  float* dist,
                                  size_t s);

int main(int argc, char** argv) {

    std::tuple<label_list_t, size_t*, adj_vert_t**> graph;
    std::string s;
    size_t s_idx;


    if(argc == 2) {

        std::string name(argv[1]);
        if(name == "sample") {
            graph = load_sample_graph();
            s = "a";
        }
        else if(name == "bitcoin") {
            graph = load_soc_bitcoin_graph();
            s = "3";
        }
        else {
            std::cout << "wrong graph: " << name << "\n";
            return 1;
        }
    } else {
        std::cout << "no graph specified\n";
        return 1;
    }


    auto& [label_list, adj_list_lens, adj_list] = graph;
    auto it_loc = std::find(label_list.begin(), label_list.end(), s);
    s_idx = std::distance(label_list.begin(), it_loc);
            

    float* dist;
    CUDA_SAFE_CALL(cudaMallocManaged(&dist, sizeof(float)*label_list.size()));



    // while(false) {
    //     std::cout << "source vertex?: ";
    //     std::cin >> s;
    //     auto it_loc = std::find(label_list.begin(), label_list.end(), s);
    //     if(it_loc == label_list.end()) {
    //         std::cout << "vertex '" << s << "' not found" << std::endl;
    //         continue;
    //     }
    //     s_idx = std::distance(label_list.begin(), it_loc);
    //     break;
    // }

    single_source_shortest_paths(label_list.size(), adj_list_lens, adj_list, dist, s_idx);
    cudaDeviceSynchronize();

    // for(size_t i=0; i<label_list.size(); ++i) {
    //     std::cout << dist[i] << " = " << label_list[i] <<  '\n';
    // }

}
