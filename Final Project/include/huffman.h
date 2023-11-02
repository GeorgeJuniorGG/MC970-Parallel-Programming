#ifndef HUFFMAN_H
#define HUFFMAN_H

#include "minheap.h" // minheap, node
#include <stdint.h>

// struct for the Huffman tree
struct hftree {
    struct node *root;          // pointer to the tree root
    size_t size;                // tree size

    struct minheap *minheap;    // auxiliary minheap
};

// struct for the Huffman code
struct hfcode {
    uint8_t code;       // code for a given symbol
    uint8_t bit_length; // length of the code
};

// function that creates a Huffman tree, given a frequency array
struct hftree* hftree_new(uint64_t frequencies[256]);
// function that frees the Huffman tree
void hftree_destroy(struct hftree *p);

// function that generates the dict that represents the code table
void hftree_generate_dict(struct hftree *p, struct hfcode dict[256]);
// auxiliary function to print the Huffman tree
void hftree_print(struct hftree *p);

#endif
