#include "minheap.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define left(i) (2*i + 1)
#define right(i) (2*i + 2)
#define parent(i) ((i-1) / 2)

struct node* node_new(enum node_type type) {
    struct node *t = calloc(1, sizeof(*t));
    t->type = type;
    return t;
}

void node_destroy(struct node *t) {
    free(t);
}

void minheap_print(struct minheap *p) {
    printf("[");
    for (size_t i = 0; i < p->size; i++) {
        struct node *t = p->data[i];
        if (t->type == NODE_LEAF) {
            printf("(LEAF: freq=%lu, symbol=0x%02x), ", t->frequency, t->symbol);
        } else {
            printf("(INTERNAL: freq=%lu, ptr=0x%02x), ", t->frequency, t);
        }
    }
    printf("]\n");
}

void minheap_insert(struct minheap *p, struct node* t) {
    struct node **data = p->data;

    size_t i = p->size;
    p->size++;
    data[i] = t;

    // printf("inserting (freq=%lu, symbol=0x%02x)\n", t.frequency, t.symbol);

    while (i != 0 && data[parent(i)]->frequency > t->frequency) {
        // swap node with its parent
        struct node *tmp = data[parent(i)];
        data[parent(i)] = data[i];
        data[i] = tmp;

        i = parent(i);
    }
}

void minheap_heapify(struct minheap *p, size_t i) {
    struct node **data = p->data;

    // printf("heapyfing at %lu (freq=%lu, symbol=0x%02x)\n", i, data[i]->frequency, data[i]->symbol);
    //
    // printf("current minheap:\n");
    // minheap_print(p);
    //

    size_t smallest = i;
    size_t l = left(i);
    size_t r = right(i);

    if (l < p->size && data[l]->frequency < data[smallest]->frequency) {
        smallest = l;
    }
    if (r < p->size && data[r]->frequency < data[smallest]->frequency) {
        smallest = r;
    }

    if (smallest != i) {
        // printf("switching (freq=%lu, symbol=0x%02x) and (freq=%lu, symbol=0x%02x)\n",
        //         data[i].frequency, data[i].symbol, data[smallest].frequency, data[smallest].symbol);
        struct node *tmp = data[smallest];
        data[smallest] = data[i];
        data[i] = tmp;
        minheap_heapify(p, smallest);
    }
}

struct node* minheap_pop(struct minheap *p) {
    if (p->size == 0) {
        return NULL;
    }

    struct node **data = p->data;
    struct node *root = data[0];
    data[0] = NULL;
    // only the root left
    if (p->size == 1) {
        // printf("MINHEAP (%p): current size is %lu...\n", p, p->size);
        p->size--;
        // printf("MINHEAP (%p): size after is %lu\n", p, p->size);
        return root;
    }

    // replace the root with the last node of the last level and re-heapify
    data[0] = data[p->size - 1];
    data[p->size - 1] = NULL;
    // printf("size=%lu, just popped 0x%02x\n", p->size, root->symbol);
    p->size--;
    minheap_heapify(p, 0);
    // printf("done heapifying\n");

    return root;
}

struct minheap* minheap_new() {
    struct minheap *p = calloc(1, sizeof(*p));
    p->size = 0;

    return p;
}

void minheap_destroy(struct minheap *p) {
    free(p);
}

void test_minheap() {
    struct minheap *p = minheap_new();

    struct node nodes[5] = {
        { .type = NODE_LEAF, .frequency = 5, .symbol = 0x00, .left = NULL, .right = NULL, },
        { .type = NODE_LEAF, .frequency = 7, .symbol = 0x01, .left = NULL, .right = NULL, },
        { .type = NODE_LEAF, .frequency = 1, .symbol = 0x02, .left = NULL, .right = NULL, },
        { .type = NODE_LEAF, .frequency = 2, .symbol = 0x03, .left = NULL, .right = NULL, },
        { .type = NODE_LEAF, .frequency = 3, .symbol = 0x04, .left = NULL, .right = NULL, },
    };

    for (size_t i = 0; i < 5; i++) {
        minheap_insert(p, &nodes[i]);
    }

    uint8_t expected_output_symbols[5] = {2, 3, 4, 0, 1};

    for (size_t i = 0; i < 5; i++) {
        struct node *t = minheap_pop(p);
        assert(t->symbol == expected_output_symbols[i]);
    }

    struct node *empty = minheap_pop(p);
    assert(empty == NULL);
    
    // no need to destroy the nodes because they were stack allocated
    minheap_destroy(p);

}

// int main(int argc, char **argv) {
//     test_minheap();
// }
