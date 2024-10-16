#pragma once
#include <bits/stdc++.h>
#include "graph_types.h"

graph_t load_soc_bitcoin_graph();
graph_t load_wiki_talk_graph();
graph_t load_sample_graph();

void print_labels_of_graph(graph_t g);
