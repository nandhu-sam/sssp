
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "graph_loaders.h"


#include "debug_helpers.h" // debug



void single_source_shortest_paths(size_t n_vertx,
                                  size_t* adj_lens,
                                  adj_vert_t** adj_list,
                                  unsigned int* dist,
                                  size_t s);

int main(int argc, char** argv) {


    std::tuple<label_list_t, size_t*, adj_vert_t**> graph;
    std::string s;
    size_t s_idx;

    auto setup_start = std::chrono::steady_clock::now();
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
        else if(name == "wikitalk") {
            DEBUG_PRINTLN;
            graph = load_wiki_talk_graph();
            DEBUG_PRINTLN;
            s = "3000";
        }
        else {
            std::cout << "wrong graph: " << name << "\n";
            return 1;
        }
    } else {
        std::cout << "no graph specified\n";
        return 1;
    }

    DEBUG_PRINTLN;
    auto& [label_list, adj_list_lens, adj_list] = graph;
    auto it_loc = std::find(label_list.begin(), label_list.end(), s);
    s_idx = std::distance(label_list.begin(), it_loc);

    
    unsigned int* dist;
    CUDA_SAFE_CALL(cudaMallocManaged(&dist, sizeof(unsigned int)*label_list.size()));
    
    auto start = std::chrono::steady_clock::now();
    DEBUG_PRINTLN;
    single_source_shortest_paths(label_list.size(), adj_list_lens, adj_list, dist, s_idx);
    DEBUG_PRINTLN;
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    auto setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(start-setup_start);

    DEBUG_COUT << "label list size: " << label_list.size() << std::endl;
    std::cout << "DIST\tVERT\n";
    for(size_t i=0; i<label_list.size(); ++i) {
        std::cout << dist[i] << '\t' << label_list[i] <<  '\n';
    }

    std::cout << "load time: " << setup_duration << '\n'
              << "exec time: " << duration << std::endl;

}
