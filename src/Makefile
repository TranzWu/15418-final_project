# Makefile

# Compiler and flags
CXX := clang++
CXXFLAGS := -std=c++11
LDFLAGS := -L/opt/homebrew/opt/libomp/lib
CPPFLAGS := -I/opt/homebrew/opt/libomp/include

# Source files and object files
SRCS := particle.cpp md.cpp main.cpp
OBJS := $(SRCS:.cpp=.o)

# Target executable
TARGET := water

# Default target
all: $(TARGET)

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS) $(CPPFLAGS) 
# -Xpreprocessor -fopenmp

# Clean up intermediate and executable files
clean:
	rm -f $(OBJS) $(TARGET)
