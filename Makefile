
NVCCFLAGS := --std c++20 --debug

.PHONY: all

all: main

main: main.o graph_loaders.o
	nvcc main.o graph_loaders.o -o main $(NVCCFLAGS)

main.o: main.cu
	nvcc main.cu --compile $(NVCCFLAGS)

graph_loaders.o: graph_loaders.cu graph_loaders.h
	nvcc graph_loaders.cu --compile $(NVCCFLAGS)

clean:
	rm --force main.o graph_loaders.o main
