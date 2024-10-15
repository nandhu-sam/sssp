#include <cuda.h>
#include <cuda_runtime.h>

#include "graph_loaders.h"


#include "debug_helpers.h" // debug




typedef std::vector<std::string> label_list_t;
typedef std::tuple<std::string, std::string, float> edge_t;



static edge_t edge_from_csv_string(std::string s) {
    std::string start, end;
    std::stringstream str(s);
    std::string weight;
    
    std::getline(str, start, ',');
    std::getline(str, end, ',');
    std::getline(str, weight, ',');

    return make_tuple(start, end, std::stof(weight));
}


static
std::tuple<label_list_t, size_t*, adj_vert_t**>
cuda_graph_from_stl_graph(std::map<std::string, std::map<std::string, float>>& g) {

    std::vector<std::string> label_list;
    
    size_t count = 0;
    for(const auto& [vert, adjs]: g) {
        label_list.push_back(vert);
        count++;    
    }


    size_t* adj_list_lengths;
    adj_vert_t** adj_list;
    CUDA_SAFE_CALL(cudaMallocManaged(&adj_list_lengths, sizeof(size_t)*label_list.size()));
    CUDA_SAFE_CALL(cudaMallocManaged(&adj_list, sizeof(adj_vert_t*)*label_list.size()));

    for(size_t i=0; i<label_list.size(); ++i) {
        adj_list_lengths[i] = (g[label_list[i]]).size();
        CUDA_SAFE_CALL(cudaMallocManaged(&adj_list[i], sizeof(adj_vert_t)*adj_list_lengths[i]));
        const auto& adjs_of_i = g[label_list[i]];
        size_t k=0;
        for(const auto& [v, x]: adjs_of_i) {
            auto it_loc = std::find(label_list.begin(), label_list.end(), v);
            size_t idx = std::distance(label_list.begin(), it_loc);
            assert(idx != label_list.size());
            // if(idx == label_list.size()); 
            adj_list[i][k].idx =  idx;
            adj_list[i][k].weight = x;
            ++k;
        }
    }

    return std::make_tuple(label_list, adj_list_lengths, adj_list);
}


std::tuple<label_list_t, size_t*, adj_vert_t**>
load_soc_bitcoin_graph() {
    
    std::string fname = "soc-sign-bitcoinalpha.csv";
    auto file = std::ifstream(fname);

    std::string line;

    std::map<std::string, std::map<std::string, float>> g;
    
    while(std::getline(file, line, '\n')) {
        auto edge = edge_from_csv_string(line);
     
        std::string start = std::get<0>(edge);
        std::string end = std::get<1>(edge);
        float weight = std::get<2>(edge);

        (g[start])[end] = weight + 11;
        g[end];
    }

    return cuda_graph_from_stl_graph(g);
}

/*
graph_t load_wiki_talk_graph() {
    std::string fname = "wiki-Talk.txt";
    return graph_from_tsv(fname);
}
*/
std::tuple<label_list_t, size_t*, adj_vert_t**>
load_sample_graph() {
    std::map<std::string, std::map<std::string, float>> g;

    (g["a"])["b"] = 1;
    (g["a"])["c"] = 4;
    (g["b"])["a"] = 1;
    (g["b"])["c"] = 2;
    (g["b"])["g"] = 4;
    (g["b"])["h"] = 2;
    (g["c"])["a"] = 4;
    (g["c"])["b"] = 1;
    (g["c"])["d"] = 1;
    (g["c"])["e"] = 3;
    (g["d"])["c"] = 1;
    (g["d"])["e"] = 1;
    (g["d"])["f"] = 3;
    (g["d"])["g"] = 1;
    (g["e"])["c"] = 3;
    (g["e"])["d"] = 1;
    (g["e"])["f"] = 1;
    (g["f"])["d"] = 3;
    (g["f"])["e"] = 1;
    (g["f"])["g"] = 6;
    (g["g"])["b"] = 4;
    (g["g"])["d"] = 1;
    (g["g"])["f"] = 6;
    (g["g"])["h"] = 14;
    (g["h"])["b"] = 2;
    (g["h"])["g"] = 14;

    return cuda_graph_from_stl_graph(g);
}




