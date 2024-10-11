#include <iostream>
#include <fstream>
#include <sstream>

#include <limits>

#include <tuple>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>


using weight_t = float;
using label_t = std::string;
using edge_t = std::tuple<label_t, label_t, weight_t>;


using edge_list_t = std::map<std::pair<label_t, label_t>, weight_t>;


using adj_list_t = std::unordered_set<label_t>;
// using adj_list_t = std::unordered_set<std::pair<label_t, float>>
using graph_t = std::unordered_map<label_t, adj_list_t>;

static label_t edge_start(const edge_t& e) { return std::get<0>(e); }
static label_t edge_end(const edge_t& e) { return std::get<1>(e); }
static weight_t edge_weight(const edge_t& e) { return std::get<2>(e); }


static edge_t edge_from_string(std::string s) {
    std::string first, second;
    std::stringstream str(s);
    str >> first >> second;
    return std::make_tuple(first, second, 1.0f);
}


static edge_t edge_from_csv_string(std::string s) {
  std::string first, second, weight;
  std::stringstream str(s);
  std::getline(str, first, ',');
  std::getline(str, second, ',');
  std::getline(str, weight, ',');
  return make_tuple(first, second, std::stof(weight));
}



static graph_t graph_from_tsv(std::string fname) {
    auto file = std::ifstream(fname);
    graph_t g;
    std::string line;
    
    while(std::getline(file, line, '\n')) {
        if(line.starts_with("#")) continue;
        auto edge = edge_from_string(line);
        g[edge_start(edge)].insert(edge_end(edge));
    }

    return g;
}


static std::pair<graph_t, edge_list_t> weighted_graph_from_csv(std::string fname) {
    auto file = std::ifstream(fname);
    graph_t g;
    edge_list_t edge_list;
    std::string line;
    
    while(std::getline(file, line, '\n')) {
        auto edge = edge_from_csv_string(line);
        g[edge_start(edge)].insert(edge_end(edge));
        edge_list[std::make_pair(edge_start(edge), edge_end(edge))] = edge_weight(edge);
    }

    return make_pair(g, edge_list);
    
}


int main() {

    
    std::cout << "reading graph...\t";
    std::cout.flush();
    auto [graph, what] = weighted_graph_from_csv("soc-sign-bitcoinalpha.csv");
    std::cout << "done" << std::endl;

    std::cout << "enter a vertex: ";
    std::string v_0;
    std::cin >> v_0;

    size_t n_vertices = graph.size();
    std::unordered_set<label_t> frontier;
    std::unordered_map<label_t, bool> visited(n_vertices);
    std::unordered_map<label_t, float> distance(n_vertices);
    

    if(!graph.contains(v_0)) {
        std::cerr << "bad vertex" << std::endl;
        return 1;
    }
    else {
        std:: cout << "vertex " << v_0 << " exists\n"
                   << "adjacents are: ";
        for(auto x: graph[v_0]) {
            std::cout << x << ' ';
        }
        std::cout << std::endl;
    }
         
    
    frontier.insert(v_0);
    visited[v_0] = true;
    distance[v_0] = 0.0f;


    
}
