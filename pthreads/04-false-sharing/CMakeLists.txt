cmake_minimum_required(VERSION 3.5)

project(sum-scalar)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pthread")

# Create executable
add_executable(sum_scalar src/sum_scalar.c)
add_executable(sum_scalar_solution src/sum_scalar_solution.c)

# Link pthreads library
target_link_libraries(sum_scalar pthread)
target_link_libraries(sum_scalar_solution pthread)