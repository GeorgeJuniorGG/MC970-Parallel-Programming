#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MASK_WIDTH 15
#define BLOCK_SIZE 32

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

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

  cudaMallocHost(&img, sizeof(PPMImage));
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

void writePPM(PPMImage *img) {

  fprintf(stdout, "P6\n");
  fprintf(stdout, "# %s\n", COMMENT);
  fprintf(stdout, "%d %d\n", img->x, img->y);
  fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

  fwrite(img->data, 3 * img->x, img->y, stdout);
  fclose(stdout);
}

// Implement this!
__global__ void smoothing_kernel(PPMPixel *d_img, PPMPixel *d_img_copy,
                                 int cols, int rows) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ PPMPixel
      neighborhood[BLOCK_SIZE + MASK_WIDTH - 1][BLOCK_SIZE + MASK_WIDTH - 1];

  int chunk_size = (MASK_WIDTH - 1) / 2;
  int pos_x = threadIdx.x + chunk_size;
  int pos_y = threadIdx.y + chunk_size;

  // Fill the corners
  if (threadIdx.y < chunk_size && threadIdx.x < chunk_size) {
    // Top Left
    int global_pos = (row - chunk_size) * cols + col - chunk_size;

    if (row - chunk_size >= 0 && col - chunk_size >= 0 && col < cols &&
        row < rows) {
      neighborhood[threadIdx.y][threadIdx.x].red = d_img_copy[global_pos].red;
      neighborhood[threadIdx.y][threadIdx.x].green =
          d_img_copy[global_pos].green;
      neighborhood[threadIdx.y][threadIdx.x].blue = d_img_copy[global_pos].blue;
    } else {
      neighborhood[threadIdx.y][threadIdx.x].red = 0;
      neighborhood[threadIdx.y][threadIdx.x].green = 0;
      neighborhood[threadIdx.y][threadIdx.x].blue = 0;
    }

    // Top Right
    global_pos = (row - chunk_size) * cols + col + BLOCK_SIZE;

    if (row - chunk_size >= 0 && col + BLOCK_SIZE < cols && row < rows) {
      neighborhood[threadIdx.y][BLOCK_SIZE + chunk_size + threadIdx.x].red =
          d_img_copy[global_pos].red;
      neighborhood[threadIdx.y][BLOCK_SIZE + chunk_size + threadIdx.x].green =
          d_img_copy[global_pos].green;
      neighborhood[threadIdx.y][BLOCK_SIZE + chunk_size + threadIdx.x].blue =
          d_img_copy[global_pos].blue;
    } else {
      neighborhood[threadIdx.y][BLOCK_SIZE + chunk_size + threadIdx.x].red = 0;
      neighborhood[threadIdx.y][BLOCK_SIZE + chunk_size + threadIdx.x].green =
          0;
      neighborhood[threadIdx.y][BLOCK_SIZE + chunk_size + threadIdx.x].blue = 0;
    }

    // Bottom Left
    global_pos = (row + BLOCK_SIZE) * cols + col - chunk_size;

    if (row + BLOCK_SIZE < rows && col - chunk_size >= 0 && col < cols) {
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size][threadIdx.x].red =
          d_img_copy[global_pos].red;
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size][threadIdx.x].green =
          d_img_copy[global_pos].green;
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size][threadIdx.x].blue =
          d_img_copy[global_pos].blue;
    } else {
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size][threadIdx.x].red = 0;
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size][threadIdx.x].green =
          0;
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size][threadIdx.x].blue = 0;
    }

    // Bottom Right
    global_pos = (row + BLOCK_SIZE) * cols + col + BLOCK_SIZE;

    if (row + BLOCK_SIZE < rows && col + BLOCK_SIZE < cols) {
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size]
                  [BLOCK_SIZE + chunk_size + threadIdx.x]
                      .red = d_img_copy[global_pos].red;
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size]
                  [BLOCK_SIZE + chunk_size + threadIdx.x]
                      .green = d_img_copy[global_pos].green;
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size]
                  [BLOCK_SIZE + chunk_size + threadIdx.x]
                      .blue = d_img_copy[global_pos].blue;
    } else {
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size]
                  [BLOCK_SIZE + chunk_size + threadIdx.x]
                      .red = 0;
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size]
                  [BLOCK_SIZE + chunk_size + threadIdx.x]
                      .green = 0;
      neighborhood[threadIdx.y + BLOCK_SIZE + chunk_size]
                  [BLOCK_SIZE + chunk_size + threadIdx.x]
                      .blue = 0;
    }
  }

  // Fill the right and left sides
  if (threadIdx.x < chunk_size) {

    int global_pos = row * cols + col - chunk_size;
    int local_pos = pos_x - chunk_size;

    if (col - chunk_size >= 0 && col < cols && row < rows) {
      neighborhood[pos_y][local_pos].red = d_img_copy[global_pos].red;
      neighborhood[pos_y][local_pos].green = d_img_copy[global_pos].green;
      neighborhood[pos_y][local_pos].blue = d_img_copy[global_pos].blue;
    } else {
      neighborhood[pos_y][local_pos].red = 0;
      neighborhood[pos_y][local_pos].green = 0;
      neighborhood[pos_y][local_pos].blue = 0;
    }

    global_pos = row * cols + col + BLOCK_SIZE;
    local_pos = pos_x + BLOCK_SIZE;

    if (col + BLOCK_SIZE < cols && row < rows) {
      neighborhood[pos_y][local_pos].red = d_img_copy[global_pos].red;
      neighborhood[pos_y][local_pos].green = d_img_copy[global_pos].green;
      neighborhood[pos_y][local_pos].blue = d_img_copy[global_pos].blue;
    } else {
      neighborhood[pos_y][local_pos].red = 0;
      neighborhood[pos_y][local_pos].green = 0;
      neighborhood[pos_y][local_pos].blue = 0;
    }
  }

  // Fill up and down
  if (threadIdx.y < chunk_size) {
    int global_pos = (row - chunk_size) * cols + col;
    int local_pos = pos_y - chunk_size;

    if (row - chunk_size >= 0 && col < cols && row < rows) {
      neighborhood[local_pos][pos_x].red = d_img_copy[global_pos].red;
      neighborhood[local_pos][pos_x].green = d_img_copy[global_pos].green;
      neighborhood[local_pos][pos_x].blue = d_img_copy[global_pos].blue;
    } else {
      neighborhood[local_pos][pos_x].red = 0;
      neighborhood[local_pos][pos_x].green = 0;
      neighborhood[local_pos][pos_x].blue = 0;
    }

    global_pos = (row + BLOCK_SIZE) * cols + col;
    local_pos = pos_y + BLOCK_SIZE;

    if (row + BLOCK_SIZE < rows && col < cols) {
      neighborhood[local_pos][pos_x].red = d_img_copy[global_pos].red;
      neighborhood[local_pos][pos_x].green = d_img_copy[global_pos].green;
      neighborhood[local_pos][pos_x].blue = d_img_copy[global_pos].blue;
    } else {
      neighborhood[local_pos][pos_x].red = 0;
      neighborhood[local_pos][pos_x].green = 0;
      neighborhood[local_pos][pos_x].blue = 0;
    }
  }

  if (row >= rows || col >= cols) {
    // Fill its corresponding pixel
    neighborhood[pos_y][pos_x].red = 0;
    neighborhood[pos_y][pos_x].green = 0;
    neighborhood[pos_y][pos_x].blue = 0;
  }

  else if (row < rows && col < cols) {
    // Fill its corresponding pixel
    neighborhood[pos_y][pos_x].red = d_img_copy[row * cols + col].red;
    neighborhood[pos_y][pos_x].green = d_img_copy[row * cols + col].green;
    neighborhood[pos_y][pos_x].blue = d_img_copy[row * cols + col].blue;

    __syncthreads();

    int reds = 0;
    int greens = 0;
    int blues = 0;

    // Compute the sum of neighbor pixel colors
    for (int i = threadIdx.y; i < threadIdx.y + MASK_WIDTH; i++) {
      for (int j = threadIdx.x; j < threadIdx.x + MASK_WIDTH; j++) {
        reds += neighborhood[i][j].red;
        greens += neighborhood[i][j].green;
        blues += neighborhood[i][j].blue;
      }
    }

    // Compute the pixel value for all colors
    d_img[row * cols + col].red = reds / (MASK_WIDTH * MASK_WIDTH);
    d_img[row * cols + col].green = greens / (MASK_WIDTH * MASK_WIDTH);
    d_img[row * cols + col].blue = blues / (MASK_WIDTH * MASK_WIDTH);
  }
}

void Smoothing(PPMImage *image, PPMImage *image_copy) {

  int cols = image_copy->x;
  int rows = image_copy->y;
  int size = cols * rows;
  PPMPixel *d_img, *d_img_copy, *h_aux;

  // Allocate host aux
  h_aux = (PPMPixel *)malloc(size * sizeof(PPMPixel));

  // Allocate device pointers
  cudaMalloc((void **)&d_img, sizeof(PPMPixel) * size);
  cudaMalloc((void **)&d_img_copy, sizeof(PPMPixel) * size);

  // Copy data to device
  cudaMemcpy(d_img_copy, image_copy->data, sizeof(PPMPixel) * size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_img, image_copy->data, sizeof(PPMPixel) * size,
             cudaMemcpyHostToDevice);

  // Define the grid
  int T = BLOCK_SIZE;
  dim3 dim_grid((cols + T - 1) / T, (rows + T - 1) / T);
  dim3 dim_block(T, T);

  smoothing_kernel<<<dim_grid, dim_block>>>(d_img, d_img_copy, cols, rows);

  cudaDeviceSynchronize();

  // Copy data back to host
  cudaMemcpy(h_aux, d_img, sizeof(PPMPixel) * size, cudaMemcpyDeviceToHost);
  image->data = h_aux;

  // Free device allocated memory
  cudaFree(d_img);
  cudaFree(d_img_copy);
}

int main(int argc, char *argv[]) {
  FILE *input;
  char filename[255];
  double t;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return 1;
  }

  // Read input filename
  fscanf(input, "%s\n", filename);

  // Read input file
  PPMImage *image = readPPM(filename);
  PPMImage *image_output = readPPM(filename);

  // Call Smoothing Kernel
  t = omp_get_wtime();
  Smoothing(image_output, image);
  t = omp_get_wtime() - t;

  // Write result to stdout
  // printf("%u %u %u\n", image_output->data[0].red,
  // image_output->data[0].green,
  //        image_output->data[0].blue);
  // printf("%u %u %u\n", image->data[0].red, image->data[0].green,
  //        image->data[0].blue);
  writePPM(image_output);

  // Print time to stderr
  fprintf(stderr, "%lf\n", t);

  // Cleanup
  free(image_output->data);
  cudaFreeHost(image);
  cudaFreeHost(image_output);

  return 0;
}
