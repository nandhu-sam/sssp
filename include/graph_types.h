#pragma once
#include <vector>
#include <string>
#include <tuple>

struct adj_vert_t {
    size_t idx;
    float weight;
};

using label_list_t = std::vector<std::string>;
using graph_t = std::tuple<label_list_t, size_t*, adj_vert_t**>;
