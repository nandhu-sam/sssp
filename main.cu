
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
    using graph_t = std::tuple<label_list_t, size_t*, adj_vert_t**>;
    using graph_fn = graph_t (*)(void);


    std::map<std::string, graph_fn> graph_loaders;


    graph_loaders["sample"] = load_sample_graph;
    graph_loaders["bitcoin"] = load_soc_bitcoin_graph;
    graph_loaders["wikitalk"] = load_wiki_talk_graph;
    graph_loaders["ca-road"] = load_road_CA_graph;
    graph_loaders["pa-road"] = load_road_PA_graph;
    graph_loaders["tx-road"] = load_road_TX_graph;
    graph_loaders["skitter"] = load_skitter_graph;
    graph_loaders["patent"] = load_cit_patent_graph;


    graph_t graph;
    std::string s;
    size_t s_idx;

    auto setup_start = std::chrono::steady_clock::now();
    if(argc >= 3) {
        std::string name(argv[1]);
        s = argv[2];
        try {
            graph = (graph_loaders.at(name))();
        }
        catch(...) {
            std::cout << "wrong graph: " << name << '\n';
            return 1;
        }

        {
            auto vec = std::get<0>(graph);
            auto start = vec.begin();
            auto end = vec.end();
            auto found = std::find(start, end, s);
            if(found == end) {
                std::cout << "vertex '" << s << "' not found in graph '"
                          << name << "'\n";
                return 1;
            }
        }



    }
    else {
        std::cout << "usage: <prog> <graph> <start>\n";
        return 1;
    }


    bool print_dist = false;
    if(argc == 4) {
        std::string arg3(argv[3]);
        print_dist = (arg3 == "print");
    }



    auto& [label_list, adj_list_lens, adj_list] = graph;
    auto it_loc = std::find(label_list.begin(), label_list.end(), s);
    s_idx = std::distance(label_list.begin(), it_loc);



    unsigned int* dist;
    CUDA_SAFE_CALL(cudaMallocManaged(&dist, sizeof(unsigned int)*label_list.size()));


    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    using std::chrono::milliseconds;
    using std::chrono::seconds;

    auto start = steady_clock::now();
    single_source_shortest_paths(label_list.size(), adj_list_lens, adj_list, dist, s_idx);
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end-start);
    auto setup_duration = duration_cast<microseconds>(start-setup_start);

    if(print_dist) {
        std::cout << "DIST\tVERT\n";
        for(size_t i=0; i<label_list.size(); ++i) {
            std::cout << dist[i] << '\t' << label_list[i] <<  '\n';
        }
    }

    std::cout << "load time: "
              << setup_duration << '\t'
              << duration_cast<milliseconds>(setup_duration) << '\t'
              << duration_cast<seconds>(setup_duration)
              << '\n'
              << "exec time: " << duration << '\t'
              << duration_cast<milliseconds>(duration) << '\t'
              << duration_cast<seconds>(duration)
              << std::endl;

}
