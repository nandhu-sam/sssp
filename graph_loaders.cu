#include "graph_loaders.h"


static edge_t edge_from_tsv_string(std::string s) {
    std::string first, second;
    std::stringstream str(s);
    str >> first >> second;
    return std::make_tuple(first, second, 5.0f);
}


static label_t edge_start(edge_t t) { return std::get<0>(t); }
static label_t edge_end(edge_t t) { return std::get<1>(t); }
static float edge_weight(edge_t t) { return std::get<2>(t); }


static edge_t edge_from_csv_string(std::string s) {


    std::string first, second, weight;
    std::stringstream str(s);
    std::getline(str, first, ',');
    std::getline(str, second, ',');
    std::getline(str, weight, ',');
    float f = std::stof(weight);
    return make_tuple(first, second, f+11.0);

}


static graph_t graph_from_tsv(std::string fname) {
    auto file = std::ifstream(fname);
    graph_t g;
    std::string line;


    while(std::getline(file, line, '\n')) {
        if(line.starts_with("#")) continue;
        auto edge = edge_from_tsv_string(line);
        g[edge_start(edge)].insert(make_pair(edge_end(edge), edge_weight(edge)));
        g[edge_end(edge)]; // in case of leaves
    }

    return g;
}




graph_t load_soc_bitcon_graph() {
    std::string fname = "soc-sign-bitcoinalpha.csv";

    auto file = std::ifstream(fname);
    graph_t g;
    std::string line;


    while(std::getline(file, line, '\n')) {
        auto edge = edge_from_csv_string(line);
        g[edge_start(edge)].insert(make_pair(edge_end(edge), edge_weight(edge)));
        g[edge_end(edge)]; // in case of leaves
    }

    return g;
}

graph_t load_wiki_talk_graph() {
    std::string fname = "wiki-Talk.txt";
    return graph_from_tsv(fname);
}


graph_t sample_graph() {


    graph_t g;
    g["a"]["b"] = 1;
    g["a"]["c"] = 4;

    g["b"]["a"] = 1;
    g["b"]["c"] = 2;
    g["b"]["g"] = 4;
    g["b"]["h"] = 2;


    g["c"]["a"] = 4;
    g["c"]["b"] = 1;
    g["c"]["d"] = 1;
    g["c"]["e"] = 3;

    g["d"]["c"] = 1;
    g["d"]["e"] = 1;
    g["d"]["f"] = 3;
    g["d"]["g"] = 1;

    g["e"]["c"] = 3;
    g["e"]["d"] = 1;
    g["e"]["f"] = 1;


    g["f"]["d"] = 3;
    g["f"]["e"] = 1;
    g["f"]["g"] = 6;

    g["g"]["b"] = 4;
    g["g"]["d"] = 1;
    g["g"]["f"] = 6;
    g["g"]["h"] = 14;

    g["h"]["b"] = 2;
    g["h"]["g"] = 14;

    return g;
}


void print_labels_of_graph(graph_t g) { // debug
    for(const auto& [v, _]: g)
        std::cout << "'" << v << "' ";
    std::cout << std::endl;
}
