#include "parallel_compression.h"
#include "huffman.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <omp.h>

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

void bitstream_print(struct bitstream *p) {
    printf("offset=%lu\n", p->offset);
    printf("[ ");
    size_t size = (p->offset % 8 == 0) ? p->offset / 8 : p->offset / 8 + 1;
    for (uint64_t i = 0; i < size; i++) {
        printf("%08b ", p->buf[i]);
    }
    printf("]\n");
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

void bitstream_append(struct bitstream *p, struct bitstream *q) {
    size_t idx = p->offset / 8;
    size_t q_size = (q->offset % 8 == 0) ? q->offset / 8 : q->offset/8 + 1;
    uint8_t shift = p->offset  % 8;

    p->buf[idx] |= q->buf[0] >> shift;
    idx++;

    for (size_t i = 0; i < q_size - 1; i++) {
        p->buf[idx + i] = q->buf[i] << (8 - shift) | q->buf[i+1] >> shift;
    }

    p->buf[idx + q_size - 1] = q->buf[q_size - 1] << (8 - shift);
    p->offset += q->offset;
}

void test_bitstream_append() {
    struct bitstream *a = bitstream_new(16);
    struct bitstream *b = bitstream_new(16);
    bitstream_push_chunk(a, 0b11110111, 8);
    bitstream_push_chunk(a, 0b000, 3);
    bitstream_push_chunk(b, 0b11111, 5);
    bitstream_push_chunk(b, 0b11111111, 8);
    bitstream_append(a, b);
    assert(a->buf[0] == 0b11110111);
    assert(a->buf[1] == 0b00011111);
    assert(a->buf[2] == 0b11111111);
    bitstream_print(a);
}

struct parallel_compressor* parallel_compressor_new(uint8_t *input, size_t size) {
    struct parallel_compressor *p = calloc(1, sizeof(*p));
    p->dict = calloc(256, sizeof(struct hfcode));
    p->in = input;
    p->in_size = size;
    p->ostream = bitstream_new(size);
    return p;
}

void parallel_compressor_destroy(struct parallel_compressor *p) {
    bitstream_destroy(p->ostream);
    free(p);
}

void parallel_compressor_generate_frequency_table(struct parallel_compressor *p, uint64_t *frequencies) {
    memset(frequencies, 0, 256 * sizeof(uint64_t));
    int threads = omp_get_max_threads();
    size_t i;
    #pragma omp parallel num_threads(threads) private(i) shared(p, frequencies)
    {
        uint64_t local_frequencies[256] = {0};
        #pragma omp for schedule(static)
        for (i = 0; i < p->in_size; i++) {
            uint8_t byte = p->in[i];
            local_frequencies[byte]++;
        }

        // printf("collecting results from thread %d\n", tid);
        #pragma omp critical
        {
            for (i = 0; i < 256; i++) {
                frequencies[i] += local_frequencies[i];
            }
        }
    }
}

void parallel_compressor_digest(struct parallel_compressor *p) {
    uint64_t frequencies[256];
    parallel_compressor_generate_frequency_table(p, frequencies);

    struct hftree *tree = hftree_new(frequencies);
    hftree_generate_dict(tree, p->dict);
    
    size_t threads = omp_get_max_threads();
    // printf("available threads: %lu\n", threads);
    struct bitstream* ostreams[threads];

    // allocate buffers for each thread
    for (size_t i = 0; i < threads; i++) {
        ostreams[i] = bitstream_new(2 * p->in_size / threads);
    }

    // the compression itself starts here
    double start = omp_get_wtime();
    
    // each thread compresses a chunk of the input into
    // a separate buffer (bitstream)
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < p->in_size; i++) {
            int tid = omp_get_thread_num();
            // printf("thread %d: %lu\n", tid, i);
            // printf("currently at %lu bytes\n", i);
            uint8_t byte = p->in[i];
            struct hfcode t = p->dict[byte];
            // printf("(%lu) pushing 0x%02x -> 0b%b:%u to bitstream %d\n", i, byte, t.code, t.bit_length, tid);
            bitstream_push_chunk(ostreams[tid], t.code, t.bit_length);
        }
    }

    // compression is over
    double duration = omp_get_wtime() - start;
    // output duration to stdout
    printf("%.6f\n", duration);

    for (size_t i = 0; i < threads; i++) {
        bitstream_append(p->ostream, ostreams[i]);
        // printf("bitstream %lu has offset %lu\n", i, ostreams[i]->offset);
        // bitstream_print(ostreams[i]);
    }

    // bitstream_print(p->ostream);
    // printf("total offset: %lu\n", p->ostream->offset);
    // printf("compression: %2fx\n", p->in_size / (p->ostream->offset/8.0));
}

void test_parallel_compression(char *filename) {
    FILE *file = fopen(filename, "r+b");
    if (!file) {
        fprintf(stderr, "failed to open file %s: %s\n", filename, strerror(errno));
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

    struct parallel_compressor *p = parallel_compressor_new(buf, read);
    parallel_compressor_digest(p);

    file = fopen("out/parallel.out", "wb");
    if (!file) {
        fprintf(stderr, "failed to open file out/parallel.out: %s\n", strerror(errno));
        exit(1);
    }

    size_t size = (p->ostream->offset % 8 == 0) ? p->ostream->offset / 8 : p->ostream->offset / 8 + 1;
    // printf("writing %lu bytes to disk\n", size);
    fwrite(p->ostream->buf, 1, size, file);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: ./parallel_compression <filename>\n");
        exit(1);
    }

    // test_bitstream_push_chunk();
    // test_bitstream_append();

    test_parallel_compression(argv[1]);
}
