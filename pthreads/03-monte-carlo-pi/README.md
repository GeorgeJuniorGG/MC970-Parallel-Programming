# PI calculation with Monte Carlo

This assignment is about using the Monte Carlo method to calculate the value of PI.

## Overview

The Monte Carlo method is a statistical approach used to approximate the value of a complex function. In this exercise, we will be using the Monte Carlo method to approximate the value of PI.

The idea is to generate a large number of random points inside a unit square and count how many of them fall inside a quarter of a unit circle. We can then use this ratio to estimate the value of PI using the formula:

PI = 4 * points_inside_circle / total_points

We can increase the accuracy of this approximation by generating more random points.

If you are curious, you can check how this method works in this [Wikipedia article](https://en.wikipedia.org/wiki/Monte_Carlo_method). A visual intuition can be seen in this Wikimedia GIF:

![Monte Carlo animation](https://upload.wikimedia.org/wikipedia/commons/8/84/Pi_30K.gif)

## How to Run the Code

After cloning the repository, run:

```sh
 mkdir build
 cd build
 cmake ..
 make
```

This will create the executables in the build folder. To compile again, simply run `make`.

To run the code simply execute:

```sh
./monte_carlo_serial
./monte_carlo_parallel
```

## What you need to do

Your task is to modify the monte_carlo_pi_parallel.cpp file to use threads, with the `std::threads` library, to parallelize the code and improve its execution time. **Write your solution in the monte_carlo_parallel.cpp file**.

Try to answer the following questions:

 1. What is the optimal number of threads to use on your computer?
 2. How does the accuracy of the approximation change with the number of threads?
 3. Is there any race condition on the resulting data?
