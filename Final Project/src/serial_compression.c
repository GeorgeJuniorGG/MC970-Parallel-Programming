#include "serial_compression.h"
#include "huffman.h"

#include <errno.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct bitstream* bitstream_new(size_t capacity) {
    struct bitstream *p = malloc(sizeof(struct bitstream));
    p->buf = calloc(capacity, 1);
    p->capacity = capacity;
    p->length = 0;
    p->offset = 0;

    return p;
}

void bitstream_destroy(struct bitstream *p) {
    free(p->buf);
    free(p);
}

// 1 byte chunks
// TODO: should probably inline this
void bitstream_push_chunk(struct bitstream *p, uint8_t chunk, uint8_t bit_length) {
    // how many bits between the stream's current offset and the next byte boundary
    uint64_t byte_offset = p->offset / 8;
    uint8_t offset_within_byte = p->offset % 8;
    uint8_t last_byte = p->buf[byte_offset];

    /**
     *
     *      offset_within_byte
     *        |----------|                               chunk we'd like to push
     *         1   1   0   _   _   _   _   _          1   1   1   0   1   1   1   _
     *        |-----------------------------|        |-----------------------------|
     *                  8 bits
     *
     *        how to merge these (considering the push might cross the byte boundary)?
     *        we expand both to 16 bits, right shift our chunk by offset_within_bytes bits,
     *        OR both parts together and then convert back to two 8-bit chunks:
     *
     *                   original byte chunk expanded to 16 bits
     *       |-------------------------------------------------------------|
     *        1   1   0   _   _   _   _   _ | _   _   _   _   _   _   _   _
     *        0   0   0   1   1   1   0   1 | 1   1   _   _   _   _   _   _
     *                   |-------------------------|
     *                  out chunk expanded to 16 bits
     *           and shifted offset_within_byte = 3 units to the right
     *
     */
    
    uint16_t expanded_original_byte = last_byte << 8;
    uint16_t expanded_and_shifted_chunk = (chunk << (16 - bit_length)) >> offset_within_byte;
    uint16_t result = expanded_original_byte | expanded_and_shifted_chunk;

    // we now push the two resulting bytes to the end of our buffer
    // the mask might be redundant because of the cast but better safe than sorry
    p->buf[byte_offset] = (uint8_t) (result >> 8);
    p->buf[byte_offset + 1] = (uint8_t) (result & 0xFF);
    p->offset += bit_length;
    // printf("bitstream currently at offset=%lu (grew %u bits)\n", p->offset, bit_length);
}

void test_bitstream_push_chunk() {
    struct bitstream *p = bitstream_new(16);
    bitstream_push_chunk(p, 0b1101, 4);
    assert(p->buf[0] == 0b11010000);
    assert(p->offset == 4);
    bitstream_push_chunk(p, 0b111111, 6);
    assert(p->buf[0] == 0b11011111 && p->buf[1] == 0b11000000);
    assert(p->offset == 10);
    bitstream_destroy(p);
}

void bitstream_print(struct bitstream *p) {
    printf("offset=%lu\n", p->offset);
    printf("[ ");
    size_t size = (p->offset % 8 == 0) ? p->offset / 8 : p->offset / 8 + 1;
    for (uint64_t i = 0; i < size; i++) {
        printf("%08b ", p->buf[i]);
    }
    printf("]\n");
}

struct serial_compressor* serial_compressor_new(uint8_t *input, size_t size) {
    struct serial_compressor *p = calloc(1, sizeof(*p));
    p->dict = calloc(256, sizeof(struct hfcode));
    p->in = input;
    p->in_size = size;
    p->ostream = bitstream_new(size);
    return p;
}

void serial_compressor_destroy(struct serial_compressor *p) {
    bitstream_destroy(p->ostream);
    free(p);
}

void serial_compressor_generate_frequency_table(struct serial_compressor *p, uint64_t *frequencies) {
    memset(frequencies, 0, 256 * sizeof(uint64_t));
    for (size_t i = 0; i < p->in_size; i++) {
        // printf("omp_in_parallel: %d\n", omp_in_parallel());
        uint8_t byte = p->in[i];
        frequencies[byte]++;
    }
}

void serial_compressor_digest(struct serial_compressor *p) {
    uint64_t frequencies[256];

    serial_compressor_generate_frequency_table(p, frequencies);

    struct hftree *tree = hftree_new(frequencies);
    hftree_generate_dict(tree, p->dict);
    // hftree_print(tree);
    //
    
    double start = omp_get_wtime();

    for (size_t i = 0; i < p->in_size; i++) {
        // printf("currently at %lu bytes\n", i);
        uint8_t byte = p->in[i];
        struct hfcode t = p->dict[byte];
        // printf("(%lu) pushing 0x%02x -> 0b%b:%u to the bitstream\n", i, byte, t.code, t.bit_length);
        bitstream_push_chunk(p->ostream, t.code, t.bit_length);
    }
    
    double duration = omp_get_wtime() - start;
    printf("%.6f\n", duration);
    // bitstream_print(p->ostream);
    // printf("total offset: %lu\n", p->ostream->offset);
    // printf("compression: %2fx\n", p->in_size / (p->ostream->offset/8.0));
}

void test_serial_compression(char *filename) {
    FILE *file = fopen(filename, "r+b");
    if (!file) {
        fprintf(stderr, "failed to read file %s: %s\n", filename, strerror(errno));
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    size_t filesize = ftell(file);
    fseek(file, 0, SEEK_SET);

    const size_t buf_size = filesize;
    uint8_t *buf = malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "unable to allocate memory for reading the file\n");
        exit(1);
    }
    size_t read = fread(buf, 1, buf_size, file);
    // printf("read %lu bytes from %s\n", read, filename);
    fclose(file);

    struct serial_compressor *p = serial_compressor_new(buf, read);
    serial_compressor_digest(p);

    file = fopen("out/serial.out", "wb");
    if (!file) {
        fprintf(stderr, "failed to open file %s: %s\n", "out/serial.out", strerror(errno));
        exit(1);
    }

    size_t size = (p->ostream->offset % 8 == 0) ? p->ostream->offset / 8 : p->ostream->offset / 8 + 1;
    // printf("writing %lu bytes to disk\n", size);
    fwrite(p->ostream->buf, 1, size, file);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: ./serial_compression <filename>\n");
        exit(1);
    }

    // test_bitstream_push_chunk();
    test_serial_compression(argv[1]);
}
