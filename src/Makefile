# Makefile

# Compiler and flags
CXX := clang++
CXXFLAGS := -std=c++11
<<<<<<< HEAD
LDFLAGS := -L/opt/homebrew/opt/libomp/lib
CPPFLAGS := -I/opt/homebrew/opt/libomp/include
=======
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
>>>>>>> 47e16f51bf6e95600627f6426714491a3d474391

# Source files and object files
SRCS := particle.cpp md.cpp main.cpp
OBJS := $(SRCS:.cpp=.o) md_cuda.o

# Target executable
TARGET := water

# Default target
all: $(TARGET)

# cpp: Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -fopenmp

# cpp: Compile source files to object files
md_cuda.o: md_cuda.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

# Link object files to create the executable
$(TARGET): $(OBJS)
<<<<<<< HEAD
	$(CXX) $(OBJS) -o $(TARGET) -Xpreprocessor -fopenmp $(LDFLAGS) $(CPPFLAGS) -lomp
# -Xpreprocessor -fopenmp -03
# -Xpreprocessor -fopenmp $(LDFLAGS) $(CPPFLAGS) -lomp
=======
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)
>>>>>>> 47e16f51bf6e95600627f6426714491a3d474391

# Clean up intermediate and executable files
clean:
	rm -f $(OBJS) $(TARGET) cuda_output.xyz output.xyz
