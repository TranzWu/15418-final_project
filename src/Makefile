# Makefile

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++11
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart

# Source files and object files
SRCS := particle.cpp seq_sim.cpp omp_sim.cpp main.cpp
OBJS := $(SRCS:.cpp=.o) cuda_sim.o

# Target executable
TARGET := water

# Default target
all: $(TARGET)

# cpp: Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -fopenmp

# cpp: Compile source files to object files
cuda_sim.o: cuda_sim.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) -fopenmp

# Clean up intermediate and executable files
clean:
	rm -f $(OBJS) $(TARGET) cuda_output.xyz
