
all: main.cu
	nvcc main.cu -o main --std c++20 --debug
