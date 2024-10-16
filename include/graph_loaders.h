#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>
#include <tuple>

#include "graph_types.h"


graph_t load_sample_graph();

graph_t load_soc_bitcoin_graph();

graph_t load_wiki_talk_graph();

graph_t load_road_CA_graph();
graph_t load_road_PA_graph();
graph_t load_road_TX_graph();

graph_t load_skitter_graph();

graph_t load_cit_patent_graph();
