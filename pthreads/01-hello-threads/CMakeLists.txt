cmake_minimum_required(VERSION 3.5)

project(hello_threads)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pthread")

# Create executable
add_executable(hello_threads src/hello_threads.c)
add_executable(hello_threads_solution src/hello_threads_solution.c)

# Link pthreads library
target_link_libraries(hello_threads pthread )
target_link_libraries(hello_threads_solution pthread )