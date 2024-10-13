#pragma once

#include <bits/stdc++.h>

using label_t = std::string;
using adj_list_t = std::unordered_map<label_t, float>;
using label_set_t = std::unordered_set<label_t>;
using graph_t = std::unordered_map<label_t, adj_list_t>;
using edge_t = std::tuple<label_t, label_t, float>;
