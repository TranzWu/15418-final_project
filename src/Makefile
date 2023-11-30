# Makefile

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++11

# Source files and object files
SRCS := particle.cpp main.cpp md.cpp
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
	$(CXX) $(OBJS) -o $(TARGET)

# Clean up intermediate and executable files
clean:
	rm -f $(OBJS) $(TARGET)
