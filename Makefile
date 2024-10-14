
INCLUDES = -I./include
NVCCFLAGS := --std c++20 --debug $(INCLUDES)

.PHONY: all

all: main

main: main.o graph_loaders.o device_print_adj_list.o
	nvcc $^ -o main $(NVCCFLAGS)

main.o: main.cu include/graph_loaders.h 
	nvcc main.cu --compile $(NVCCFLAGS)

graph_loaders.o: graph_loaders.cu include/graph_loaders.h
	nvcc graph_loaders.cu --compile $(NVCCFLAGS)

device_print_adj_list.o: device_print_adj_list.cu
	nvcc device_print_adj_list.cu --compile $(NVCCFLAGS)


clean:
	rm --force main.o graph_loaders.o device_print_adj_list.o main
