CXX=g++
CXX_FLAGS= -g -std=c++11 

all: gpu_hashtable

gpu_hashtable: gpu_hashtable.cu test_map.cpp gpu_hashtable.hpp
	nvcc $(CXX_FLAGS) -lineinfo gpu_hashtable.cu -o gpu_hashtable

clean:
	rm -f gpu_hashtable
