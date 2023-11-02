#ifndef SERIAL_COMPRESSION_H
#define SERIAL_COMPRESSION_H

#include "huffman.h"

#include <stdio.h>

// struct for the bitstream that will hold the final Huffman code
struct bitstream {
    uint8_t *buf;
    size_t capacity;    // buffer size in bytes

    uint64_t length;    // in bits
    uint64_t offset;    // current offset (in bits) into the stream
};

// struct for the serial compressor
struct serial_compressor {
    struct hfcode *dict;        // pointer to the dict that represents the code table

    uint8_t *in;                // pointer to the input file
    size_t in_size;             // input file size

    struct bitstream *ostream;  // bitstream that will hold the final Huffman code
};

// function that creates a new serial compressor, given its input and input length
struct serial_compressor* serial_compressor_new(uint8_t *input, size_t len);
// function that frees a serial compressor
void serial_compressor_destroy(struct serial_compressor *p);
// function that digest the given input to produce the Huffman code
void serial_compressor_digest(struct serial_compressor *p);

#endif
