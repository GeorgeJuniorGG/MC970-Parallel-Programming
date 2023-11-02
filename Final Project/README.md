[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/SvGT8lI6)

# Parallelizing Gzip

This repository contains the code for our final project, regarding parallelizing the Gzip algorithm. As explained in our report, we decided to parallelize one specific step of the Gzip algorithm, due to the high complexity of the Gzip algorithm as a whole. This specific step is the Huffman encoding process, which translates the original message to a Huffman code representation of it.

All the code in this repository was made by ourselves and is documented with comments, to make it easier for other people to understand. You can find the algorithm explanation, as well as the description of our implementation, in our report pdf file.

## Repository Structure

This repository contains two folders:

* data: which contains the input samples;
* src: which contains the code, for both the parallel and serial implementations.
  * Inside this folder, you'll also find some auxiliary code we used to implement the encoding process.
* include: contains the headers for our data structures

We also have a CMake file on the root of our repository, to make it easier to build everything.

## Getting Started

To get started, clone this repository to your local machine using the following command:

```sh
git clone https://github.com/mc970-1s23/final-project-parallelizing-gzip
```

Once the repository is cloned, you can compile the code with the following commands:

```bash
mkdir build && cd build
cmake ..
```
