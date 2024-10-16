
INCLUDES = -I./include
NVCCFLAGS := --std c++20 $(INCLUDES)

objects := main.o
objects += graph_loaders.o
objects += sssp.o edge_relax.o
objects += realloc_queue_vec.o

.PHONY: all

all: release

release: NVCCFLAGS += -O3
release: main

debug: NVCCFLAGS += --debug --device-debug
debug: main

main: $(objects)
	nvcc $^ -o main $(NVCCFLAGS)

main.o: main.cu include/graph_loaders.h 
	nvcc $< --compile $(NVCCFLAGS)

graph_loaders.o: graph_loaders.cu include/graph_loaders.h
	nvcc $< --compile $(NVCCFLAGS)

sssp.o: sssp.cu edge_relax.cu include/graph_types.h include/debug_helpers.h
	nvcc $< --compile $(NVCCFLAGS)


edge_relax.o: edge_relax.cu include/cuda_debug_helpers.h
	nvcc $< --compile $(NVCCFLAGS)

realloc_queue_vec.o: realloc_queue_vec.cu include/cuda_debug_helpers.h
	nvcc $< --compile $(NVCCFLAGS)

clean:
	rm --force $(objects) main
