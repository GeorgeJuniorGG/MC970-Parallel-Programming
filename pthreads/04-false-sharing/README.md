# Fixing False-Sharing in Parallel Code

## Introduction

This lab focuses on understanding false-sharing in caches and optimizing a parallelized program that computes the sum of a list of scalars. False-sharing is a performance issue that arises when two or more threads write to different variables that share the same cache line. This leads to frequent cache invalidation and cache misses, resulting in suboptimal performance.

## Overview

The code for this lab is already parallelized with pthreads. However, it suffers from false-sharing in caches, which needs to be fixed to improve performance. In this lab, you will use a tool like Perf to identify the regions of the code that are affected by false-sharing. Once identified, you can optimize the program to reduce the number of cache misses and improve its performance.

## How to run the code

After cloning the repository, run:

```sh
 mkdir build
 cd build
 cmake ..
 make
```

This will create the executables in the build folder. To compile again, simply run `make`.

The code is to be executed with the following arguments:

```sh
./sum_scalar <n> <num_threads>
./sum_scalar 100000000 4
./sum_scalar 3000000 4
```

## What you need to do

Your task is to identify the regions of the code that suffer from false-sharing in caches and optimize the program to reduce the number of cache misses. To do this, you can use a tool like Perf to identify the specific regions of the code that are affected by false-sharing. Once you have identified these regions, you can apply the necessary optimizations to reduce the number of cache misses and improve the program's performance. **Write your solution in the sum_scalar_solution.c file**.

## Acknowledgement

This lab is a modified version of a lab made by Luís Felipe Mattos and Guido Araújo.