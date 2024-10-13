
#include <bits/stdc++.h>


#include "graph_types.h"
#include "graph_loaders.h"


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



int main() {

//    auto g = load_soc_bitcon_graph();
///   auto g = load_wiki_talk_graph();
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
