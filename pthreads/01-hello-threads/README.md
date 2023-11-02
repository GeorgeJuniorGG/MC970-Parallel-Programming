# Hello Worlds in C

This repository contains a simple C exercise to print "Hello World" using multiple threads. The ideia is to write your first multithread program.

## Overview

The objective of this assignment is to enhance your understanding of threads by introducing you to a simple "hello world" program that utilizes the pthread library. It is crucial to research and comprehend the similarities and differences between threads and processes. For instance, threads have their own stack and registers, but they share the data section.

## Using CMake

CMake is a tool that automates the build processes. Check the `CMakeLists.txt` file to see how it finds and links your program against the `pthreads` lib.

## How to Run the Code

After cloning the repository, run:

```sh
 mkdir build
 cd build
 cmake ..
 make
```

This will create the executables in the build folder. To compile again, simply run `make`.

## Code Overview

The `hello_threads.c` file contains the main C code for this exercise. The program uses the POSIX Threads (pthreads) library to create multiple threads. Each thread simply sleeps for 3 seconds and then prints the message "Hello Worlds" along with its thread ID.

The `main()` function should initialize five threads using the `pthread_create()` function. Each thread runs the `printHello()` function, which sleeps for 3 seconds and then prints the message "Hello World" along with its thread ID using the `printf()` function.

## What you need to do

This is an introductory assignment. The only task is to understand the code and modify it to print it using threads. Compare the time it takes to run this with the sequential code, than compare again creating a lot more threads. **Write your solution in the hello_threads_solution.c file**.

Try to answer the following questions:

1. Can your program create more threads than cores in the CPU?
2. What can the function `pthread_create(...)` return? Why?
3. What is the `pthread_join(...)` function for?
4. In what order are the threads executed?
