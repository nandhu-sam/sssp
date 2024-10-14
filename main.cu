
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "graph_loaders.h"
// #include "delta_stepping_fns.h"

#include "debug_helpers.h" // debug

#include "uthash.h"


extern __global__ void device_print_adj_list(size_t* adj_list_lens, size_t n_vertx, adj_vert_t** adj_list);


struct frontier_hash {
    size_t idx;
    UT_hash_handle hh;
};


int main() {
    
    auto [label_list, adj_list_lens, adj_list] = load_soc_bitcon_graph();
    

    std::unordered_set<std::string> frontier;

    for(size_t i =0; i<label_list.size(); ++i) {
        dist[i] = std::numeric_limits<float>::infinity();
        visited[i] = false;
    }

    // memory to be initialized at kernel
    
    float* dist;
    bool* visited;

    CUDA_SAFE_CALL(cudaMallocManaged(&dist, sizeof(float)*label_list.size()));
    CUDA_SAFE_CALL(cudaMallocManaged(&visited, sizeof(bool)*label_list.size()));

    
    std::string s;

    while(true) {
        std::cout << "source vertex?: ";
        std::cin >> s;
        auto it_loc = std::find(label_list.begin(), label_list.end(), s);
        if(it_loc == label_list.end()) {
            std::cout << "vertex '" << s << "' not found" << std::endl;
            continue;
        }

        break;
    }

    
    

}

#if 0
static int old_main() {

//    auto g = load_soc_bitcon_graph();
///   auto g = load_wiki_talk_graph();
    auto [label_list, adj_list_lens, adj_list] = sample_graph();
//    print_labels_of_graph(g);
    
    std::vector<float> dist(label_list.size());
    std::vector<bool> visited(label_list.size());
    std::unordered_set<std::string> frontier;

    for(size_t i =0; i<dist.size(); ++i) {
        dist[i] = std::numeric_limits<float>::infinity();
        visited[i] = false;
    }

    std::string s;

    while(true) {
        std::cout << "source vertex?: ";
        std::cin >> s;
        if(!(g.contains(s))) {
            std::cout << "vertex '" << s << "' not found" << std::endl;
            continue;
        }

        break;
    }

    frontier.insert(s);
    dist[s] = 0.0;
    visited[s] = true;

    auto count = g.size();

    while(!frontier.empty()) {

        std::cout.flush();
        float min_cost = std::numeric_limits<float>::infinity();
        std::string  min_label;

        std::set<std::string> frontier_delete_list;

        for(const auto& v: frontier) {
            bool visited_all = true;
            for(const auto& [w, weight]: g[v]) {

                float path_cost = dist[v] + weight;

                if(visited[w]) continue;
                else visited_all = false;

                if(path_cost < min_cost) {
                    min_cost = path_cost;
                    min_label = w;
                }
            }
            if(visited_all) {
                frontier_delete_list.insert(v);
            }
        }


        for(const auto& x: frontier_delete_list) {
            frontier.erase(x);
        }

        if(min_label.empty()) break;
        dist[min_label] = min_cost;
        visited[min_label] = true;
        frontier.insert(min_label);
    }


    std::cout << "DIST\tVERT\n";
    for(auto& [v, x]: dist) {
        std::cout << x << "\t" << v << '\n';
    }
    std::cout << std::endl;
}
#endif
