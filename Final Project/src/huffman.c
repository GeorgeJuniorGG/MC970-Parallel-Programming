#include "huffman.h"
#include "minheap.h"

#include <stdio.h>
#include <stdlib.h>

struct hftree* hftree_new(uint64_t *frequencies) {
    struct hftree *p = calloc(1, sizeof(*p));
    p->minheap = minheap_new();
    p->size = 0;

    for (size_t i = 0; i < 256; i++) {
        if (frequencies[i] == 0) {
            continue;
        }

        struct node *t = node_new(NODE_LEAF);
        t->symbol = (uint8_t) i;
        t->frequency = frequencies[i];

        minheap_insert(p->minheap, t);
    }

    return p;
}

void hftree_build(struct hftree *p) {
    while (p->minheap->size > 1) {
        // printf("heap size: %lu\n", p->minheap->size);
        // minheap_print(p->minheap);

        struct node *a = minheap_pop(p->minheap);
        struct node *b = minheap_pop(p->minheap);

        struct node *internal = node_new(NODE_INTERNAL);
        internal->frequency = (a ? a->frequency : 0) + (b ? b->frequency : 0);
        
        struct node *left, *right;

        if (!a) {
            left = b;
            right = NULL;
        } else if (!b) {
            left = a;
            right = NULL;
        } else {
            left = a->frequency <= b->frequency ? a : b;
            right = a->frequency > b->frequency ? a : b;
        }

        internal->left = left;
        internal->right = right;

        minheap_insert(p->minheap, internal);
    }
    
    p->root = minheap_pop(p->minheap);
}

void hftree_destroy_node_rec(struct node *t) {
    if (!t) {
        return;
    }

    hftree_destroy_node_rec(t->left);
    hftree_destroy_node_rec(t->right);
    node_destroy(t);
}

void hftree_destroy(struct hftree *p) {
    minheap_destroy(p->minheap);
    hftree_destroy_node_rec(p->root);
    free(p);
}

void hftree_print_rec(struct node *t, uint8_t path, uint8_t bit_count) {
    if (!t) {
        return;
    }
    hftree_print_rec(t->left, (path << 1) | 0, bit_count + 1);
    hftree_print_rec(t->right, (path << 1) | 1, bit_count + 1);

    if (t->type == NODE_INTERNAL) {
        printf("[%p] INTERNAL: (freq=%lu, left=%p, right=%p), ", t, t->frequency, t->left, t->right);
    printf("path (%u bits): 0b%0*b\n", bit_count, bit_count, path);
        return;
    }

    printf("[%p] LEAF: (freq=%lu, symbol: 0x%02x), ",
            t, t->frequency, t->symbol);
    printf("path (%u bits): 0b%0*b\n", bit_count, bit_count, path);
}

void hftree_print(struct hftree *p) {
    hftree_print_rec(p->root, 0, 0);
}

void hftree_collect_codes(struct node *t, struct hfcode *dict,
                          uint8_t path, uint8_t length) {
    // we walk the tree in-order and collect the path values
    // from the leaf nodes
    if (!t) {
        return;
    }

    if (t->type == NODE_LEAF) {
        // printf("writing entry: (0x%02x, 0b%b:%u)\n", t->symbol, path, length);
        // printf("writing entry: ('0x%02x=%c', 0b%b:%u)\n", t->symbol, t->symbol, path, length);
        dict[t->symbol].code = path;
        dict[t->symbol].bit_length = length;
    }

    hftree_collect_codes(t->left, dict, (path << 1) | 0, length + 1);
    hftree_collect_codes(t->right, dict, (path << 1) | 1, length + 1);
}

void hftree_generate_dict(struct hftree *p, struct hfcode *dict) {
    hftree_build(p);
    hftree_collect_codes(p->root, dict, 0, 0);
}

// int main(int argc, char **argv) {
//     uint64_t frequencies[256] = {0};
//     frequencies[1] = 40;
//     frequencies[2] = 35;
//     frequencies[3] = 20;
//     frequencies[4] = 5;
//
//     struct hfcode dict[256] = {0};
//
//     struct hftree *p = hftree_new(frequencies);
//     hftree_generate_dict(p, dict);
//     hftree_print(p);
//
//     hftree_destroy(p);
// }
