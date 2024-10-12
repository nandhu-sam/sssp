
#include <bits/stdc++.h>


using label_t = std::string;
using adj_list_t = std::unordered_map<label_t, float>;
//using adj_list_t = std::set<std::pair<label_t, float>>;
using label_set_t = std::unordered_set<label_t>;
using graph_t = std::unordered_map<label_t, adj_list_t>;
using edge_t = std::tuple<label_t, label_t, float>;

static edge_t edge_from_string(std::string s) {
    std::string first, second;
    std::stringstream str(s);
    str >> first >> second;
    return std::make_tuple(first, second, 1.0f);
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
    return make_tuple(first, second, f+10.0);
    
}



static graph_t graph_from_tsv(std::string fname) {
    auto file = std::ifstream(fname);
    graph_t g;
    std::string line;
    
    while(std::getline(file, line, '\n')) {
        if(line.starts_with("#")) continue;
        auto edge = edge_from_string(line);
        g[edge_start(edge)].insert(make_pair(edge_end(edge), edge_weight(edge)));
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
                edges[w] = dist.at(v) + weight;
            }
        }
    }
    return edges;
}


static size_t bucket_pos(float cost, float thresh, size_t n_buckets) {
    if(cost == std::numeric_limits<float>::infinity()) return 0;
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



int main() {

    auto g = load_soc_bitcon_graph();

    std::map<label_t, float> dist;
    std::vector<std::unordered_set<label_t>> buckets(20);
    
    for(auto& v: g) {
        dist[v.first] = std::numeric_limits<float>::infinity();
        buckets[bucket_pos(dist[v.first], 1.0f, buckets.size())].insert(v.first);
    }

    std::string s;

    while(true) {
        std::cout << "source vertex?: ";
        std::cin >> s;
        if(!(g.contains(s))) {
            std::cout << "vertex '" << s << "' not found" << std::endl;
        }
        break;
    };


    dist[s] = 0.0f;

    std::optional<size_t> current_bucket;
    for(size_t i=0; i<buckets.size(); i++) {
        if(!buckets[i].empty()) {
            current_bucket = i;
            break;
        }
    }

    
    while(current_bucket) {

        label_set_t tmp_bucket;
        adj_list_t reqs;
        
        while(!buckets[current_bucket.value()].empty()) {
            reqs = find_light_edges(g, 1.0f, buckets[current_bucket.value()], dist);
            tmp_bucket.merge(buckets[current_bucket.value()]);
            buckets[current_bucket.value()].clear();
            if(tmp_bucket.empty()) std::cout << "tmp_bucket is now empty" << std::endl;
            relax_edges(reqs, dist, buckets);
        }

        reqs = find_heavy_edges(g, 1.0f, tmp_bucket, dist);
        relax_edges(reqs, dist, buckets);

        current_bucket.reset();
        for(size_t i=0; i<buckets.size(); i++) {
            if(!buckets[i].empty()) {
                current_bucket = i;
                break;
            }
        }
    }

    for(auto [v, x]: dist) {
        std::cout << v << " - " << x << '\n';
    }
    std::cout << std::endl;


}
