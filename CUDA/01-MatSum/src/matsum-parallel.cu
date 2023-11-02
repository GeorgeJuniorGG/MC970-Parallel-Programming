#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrix_sum(int *C, int *A, int *B, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
  }
}

int main(int argc, char **argv) {
  int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  int i, j;
  double t;

  // Input
  int rows, cols;
  FILE *input;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return EXIT_FAILURE;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    return EXIT_FAILURE;
  }

  fscanf(input, "%d", &rows);
  fscanf(input, "%d", &cols);

  // Allocate memory on the host
  int bytes = rows * cols * sizeof(int);
  cudaMallocHost(&h_A, bytes);
  cudaMallocHost(&h_B, bytes);
  cudaMallocHost(&h_C, bytes);

  // Initialize memory
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      h_A[i * cols + j] = h_B[i * cols + j] = i + j;
    }
  }

  // Allocate memory on the device
  cudaMalloc((void **)&d_A, bytes);
  cudaMalloc((void **)&d_B, bytes);
  cudaMalloc((void **)&d_C, bytes);

  // Copy data to device
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

  // Compute matrix sum on device
  int T = 32;
  dim3 dim_grid((cols + T - 1) / T, (rows + T - 1) / T);
  dim3 dim_block(T, T);

  // Leave only the kernel and synchronize inside the timing region!
  t = omp_get_wtime();
  matrix_sum<<<dim_grid, dim_block>>>(d_C, d_A, d_B, rows, cols);
  cudaDeviceSynchronize();
  t = omp_get_wtime() - t;

  // Copy data back to host
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  long long int sum = 0;

  // Keep this computation on the CPU
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      sum += h_C[i * cols + j];
    }
  }

  fprintf(stdout, "%lli\n", sum);
  fprintf(stderr, "%lf\n", t);

  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);

  // Free device allocated memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
