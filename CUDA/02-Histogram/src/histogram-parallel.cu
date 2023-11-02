#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void check_cuda(cudaError_t error, const char *filename, const int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line,
            cudaGetErrorName(error), cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

__global__ void histogram_kernel(float *d_h, PPMImage *d_img, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float private_h[64];

  if (threadIdx.x < 64) {
    private_h[threadIdx.x] = 0;
  }
  __syncthreads();

  if (idx < size) {
    d_img->data[idx].red = floorf((d_img->data[idx].red * 4) / 256);
    d_img->data[idx].blue = floorf((d_img->data[idx].blue * 4) / 256);
    d_img->data[idx].green = floorf((d_img->data[idx].green * 4) / 256);

    atomicAdd(&private_h[16 * d_img->data[idx].red +
                         4 * d_img->data[idx].green + d_img->data[idx].blue],
              1);
  }

  __syncthreads();
  if (threadIdx.x < 64) {
    atomicAdd(&d_h[threadIdx.x], private_h[threadIdx.x] / size);
  }
}

double Histogram(PPMImage *image, float *h_h) {
  float ms;
  cudaEvent_t start, stop;

  int cols = image->x;
  int rows = image->y;
  int size = rows * cols;

  // Create Events
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  // Allocate device pointers
  float *d_h;
  int bytes = sizeof(float) * 64;
  cudaMalloc((void **)&d_h, bytes);

  PPMImage *d_img;
  cudaMalloc((void **)&d_img, sizeof(PPMImage));

  PPMPixel *d_data;
  cudaMalloc((void **)&d_data, size * sizeof(PPMPixel));

  // Copy data to device
  cudaMemcpy(d_h, h_h, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_img, image, sizeof(PPMImage), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, image->data, size * sizeof(PPMPixel),
             cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_img->data), &d_data, sizeof(PPMPixel *),
             cudaMemcpyHostToDevice);

  int T = 224;
  // Launch kernel and compute kernel runtime.
  // Warning: make sure only the kernel is being profiled, memcpies should be
  // out of this region.
  CUDACHECK(cudaEventRecord(start));
  histogram_kernel<<<(size + T - 1) / T, T>>>(d_h, d_img, size);
  CUDACHECK(cudaEventRecord(stop));
  CUDACHECK(cudaEventSynchronize(stop));
  CUDACHECK(cudaEventElapsedTime(&ms, start, stop));

  // Copy data back to host
  cudaMemcpy(h_h, d_h, bytes, cudaMemcpyDeviceToHost);

  // Free device allocated memory
  cudaFree(d_h);
  cudaFree(d_data);
  cudaFree(d_img);

  // Destroy events
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));

  return ((double)ms) / 1000.0;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  PPMImage *image = readPPM(argv[1]);
  float *h = (float *)malloc(sizeof(float) * 64);

  // Initialize histogram
  for (int i = 0; i < 64; i++)
    h[i] = 0.0;

  // Compute histogram
  double t = Histogram(image, h);

  for (int i = 0; i < 64; i++)
    printf("%0.3f ", h[i]);
  printf("\n");

  fprintf(stderr, "%lf\n", t);
  free(h);
}
