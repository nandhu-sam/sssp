
#include <bits/stdc++.h>

using label_t = std::string;
using adj_list_t = std::unordered_map<label_t, float>;
//using adj_list_t = std::set<std::pair<label_t, float>>;
using label_set_t = std::unordered_set<label_t>;
using graph_t = std::unordered_map<label_t, adj_list_t>;
using edge_t = std::tuple<label_t, label_t, float>;

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



adj_list_t find_light_edges(const graph_t& g,
                            float thresh,
                            const label_set_t& frontier,
                            const std::map<label_t, float> dist) {
    adj_list_t edges;
    for(const auto& v: frontier) {
        for(const auto& [w, weight]: g.at(v)) {
            if(weight <= thresh) {
                if(edges.contains(w))
                    edges[w] = std::min(dist.at(v) + weight, edges[w]);
                else
                    edges[w] = dist.at(v) + weight;
            }
        }
    }
    return edges;
}


adj_list_t find_heavy_edges(const graph_t& g,
                            float thresh,
                            const label_set_t& frontier,
                            const std::map<label_t, float>& dist) {
    adj_list_t edges;
    for(const auto& v: frontier) {
        for(const auto& [w, weight]: g.at(v)) {
            if(weight > thresh) {
                if(edges.contains(w))
                    edges[w] = std::min(dist.at(v) + weight, edges[w]);
                else
                    edges[w] = dist.at(v) + weight;
            }
        }
    }
    return edges;
}


static size_t bucket_pos(float cost, float thresh, size_t n_buckets) {
    if(cost == std::numeric_limits<float>::infinity())
        return 0;
    return static_cast<int>(std::floor(cost/thresh)) % n_buckets;
}


void relax_edges(const adj_list_t reqs,
                 std::map<label_t, float>& dist,
                 std::vector<std::unordered_set<label_t>>& buckets)
{
    for(auto& [w, x]: reqs) {

        if(x < dist[w]) {
            auto old_bucket_pos = bucket_pos(dist[w], 1.0f, buckets.size());
            auto new_bucket_pos = bucket_pos(x, 1.0f, buckets.size());
            buckets[old_bucket_pos].erase(w);
            buckets[new_bucket_pos].insert(w);
            dist[w] = x;
        }
    }
}




static graph_t load_soc_bitcon_graph() {
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

static graph_t load_wiki_talk_graph() {
    std::string fname = "wiki-Talk.txt";
    return graph_from_tsv(fname);


}

static graph_t sample_graph() {
    
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



static void print_labels_of_graph(graph_t g) { // debug
    for(const auto& [v, _]: g)
        std::cout << "'" << v << "' ";
    std::cout << std::endl;
}

int main() {

//    auto g = load_soc_bitcon_graph();
//    auto g = load_wiki_talk_graph();
    auto g = sample_graph();
//    print_labels_of_graph(g);
    

    std::unordered_map<label_t, float> dist;
    std::unordered_map<label_t, bool> visited;
    std::unordered_set<label_t> frontier;


    for(const auto& [label, _]: g) {
        dist[label] = std::numeric_limits<float>::infinity();
        visited[label] = false;
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
        label_t min_label;

        std::set<label_t> frontier_delete_list;
        
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
