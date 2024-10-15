
INCLUDES = -I./include
NVCCFLAGS := --std c++20 --debug $(INCLUDES)

objects = main.o graph_loaders.o device_print_adj_list.o sssp.o setup_dist_arr.o edge_relax.o
.PHONY: all

all: main

main: $(objects)
	nvcc $^ -o main $(NVCCFLAGS)

main.o: main.cu include/graph_loaders.h 
	nvcc $< --compile $(NVCCFLAGS)

graph_loaders.o: graph_loaders.cu include/graph_loaders.h
	nvcc $< --compile $(NVCCFLAGS)

device_print_adj_list.o: device_print_adj_list.cu
	nvcc $< --compile $(NVCCFLAGS)	

sssp.o: sssp.cu setup_dist_arr.cu edge_relax.cu include/graph_types.h include/debug_helpers.h 
	nvcc $< --compile $(NVCCFLAGS)

setup_dist_arr.o: setup_dist_arr.cu
	nvcc $< --compile $(NVCCFLAGS)

edge_relax.o: edge_relax.cu
	nvcc $< --compile $(NVCCFLAGS)	

clean:
	rm --force $(objects)
