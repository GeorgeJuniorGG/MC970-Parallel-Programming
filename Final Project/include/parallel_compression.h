#ifndef PARALLEL_COMPRESSION_H
#define PARALLEL_COMPRESSION_H

#include "huffman.h"

#include <stdio.h>

struct bitstream {
    uint8_t *buf;
    size_t capacity;    // buffer size in bytes

    uint64_t length;    // in bits
    uint64_t offset;    // current offset (in bits) into the stream
};

struct parallel_compressor {
    struct hfcode *dict;

    uint8_t *in;
    size_t in_size;

    struct bitstream *ostream;
};

struct parallel_compressor* parallel_compressor_new(uint8_t *input, size_t len);
void parallel_compressor_destroy(struct parallel_compressor *p);
void parallel_compressor_digest(struct parallel_compressor *p);

#endif
