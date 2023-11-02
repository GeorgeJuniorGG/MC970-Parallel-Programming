cmake_minimum_required(VERSION 3.9)

project(huffman-parallel)

find_package(OpenMP REQUIRED)

set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)

include_directories(include)

add_executable(serial_compression src/serial_compression.c src/minheap.c src/huffman.c)
add_executable(parallel_compression src/parallel_compression.c src/minheap.c src/huffman.c)

target_link_libraries(serial_compression
    PUBLIC OpenMP::OpenMP_C)

target_link_libraries(parallel_compression
    PUBLIC OpenMP::OpenMP_C)
