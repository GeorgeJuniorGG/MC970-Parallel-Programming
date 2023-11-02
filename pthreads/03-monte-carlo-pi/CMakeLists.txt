cmake_minimum_required(VERSION 3.5)
project(monte_carlo_pi)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

# Find Threads library
find_package(Threads REQUIRED)

# Add executable target for serial version
add_executable(monte_carlo_serial src/monte_carlo_serial.cpp)
target_compile_definitions(monte_carlo_serial PRIVATE SERIAL_VERSION)

# Add executable target for parallel version
add_executable(monte_carlo_parallel src/monte_carlo_parallel.cpp)
target_compile_definitions(monte_carlo_parallel PRIVATE PARALLEL_VERSION)

# Link with Threads library
target_link_libraries(monte_carlo_parallel Threads::Threads)
