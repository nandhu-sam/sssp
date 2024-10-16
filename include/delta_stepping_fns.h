#pragma once


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

