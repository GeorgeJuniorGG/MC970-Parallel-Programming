#ifndef MINHEAP_H 
#define MINHEAP_H

#include <stdint.h>
#include <stddef.h>

// enum used for describing the node type
enum node_type {
    NODE_EMPTY = 0,
    NODE_INTERNAL = 1,
    NODE_LEAF = 2,
};

// struct for the tree nodes
struct node {
    enum node_type type;    // node type
    uint64_t frequency;     // node frequency (as per our final report tree example)

    // the symbol field is zero for internal/empty nodes
    uint8_t symbol;         // node symbol, if any

    // pointers for its children
    struct node *left;
    struct node *right;
};

// struct for the minheap itself
struct minheap {
    struct node* data[256];
    size_t size;
};

// function that creates a new node, given its type
struct node* node_new(enum node_type type);
// function that destroys a node
void node_destroy(struct node *t);

// function that instantiates a new minheap
struct minheap* minheap_new();
// function that pops the next element of the heap
struct node* minheap_pop(struct minheap *p);
// function that inserts a new node on the heap
void minheap_insert(struct minheap *p, struct node *t);
// function that frees a minheap
void minheap_destroy(struct minheap *p);
// auxiliary function that prints a minheap
void minheap_print(struct minheap *p);

#endif
