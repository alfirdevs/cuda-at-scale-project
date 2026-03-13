#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t status__ = (call);                                            \
    if (status__ != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__)            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

struct Options {
  int num_images = 256;
  int width = 512;
  int height = 512;
  std::string output_dir = "output";
  int save_count = 6;
};

__global__ void Blur3x3Kernel(const unsigned char* input,
                              unsigned char* output,
                              int width,
                              int height,
                              int pitch_pixels,
                              int batch_count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int image_id = blockIdx.z;

  if (x >= width || y >= height || image_id >= batch_count) {
    return;
  }

  int base = image_id * pitch_pixels * height;
  int sum = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    int yy = min(max(y + ky, 0), height - 1);
    for (int kx = -1; kx <= 1; ++kx) {
      int xx = min(max(x + kx, 0), width - 1);
      sum += input[base + yy * pitch_pixels + xx];
    }
  }
  output[base + y * pitch_pixels + x] = static_cast<unsigned char>(sum / 9);
}

__global__ void SobelKernel(const unsigned char* input,
                            unsigned char* output,
                            int width,
                            int height,
                            int pitch_pixels,
                            int batch_count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int image_id = blockIdx.z;

  if (x >= width || y >= height || image_id >= batch_count) {
    return;
  }

  int base = image_id * pitch_pixels * height;

  int xm1 = max(x - 1, 0);
  int xp1 = min(x + 1, width - 1);
  int ym1 = max(y - 1, 0);
  int yp1 = min(y + 1, height - 1);

  auto at = [&](int xx, int yy) {
    return static_cast<int>(input[base + yy * pitch_pixels + xx]);
  };

  int gx = -at(xm1, ym1) + at(xp1, ym1)
           - 2 * at(xm1, y) + 2 * at(xp1, y)
           - at(xm1, yp1) + at(xp1, yp1);

  int gy = -at(xm1, ym1) - 2 * at(x, ym1) - at(xp1, ym1)
           + at(xm1, yp1) + 2 * at(x, yp1) + at(xp1, yp1);

  int mag = min(255, static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy))));
  output[base + y * pitch_pixels + x] = static_cast<unsigned char>(mag);
}

bool ParseInt(const std::string& value, int* out) {
  try {
    *out = std::stoi(value);
    return true;
  } catch (...) {
    return false;
  }
}

Options ParseArgs(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--num_images") {
      if (!ParseInt(require_value(arg), &options.num_images)) {
        std::cerr << "Invalid value for --num_images" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--width") {
      if (!ParseInt(require_value(arg), &options.width)) {
        std::cerr << "Invalid value for --width" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--height") {
      if (!ParseInt(require_value(arg), &options.height)) {
        std::cerr << "Invalid value for --height" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--save_count") {
      if (!ParseInt(require_value(arg), &options.save_count)) {
        std::cerr << "Invalid value for --save_count" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--output_dir") {
      options.output_dir = require_value(arg);
    } else if (arg == "--help") {
      std::cout << "Usage: ./cuda_batch_image_processing.exe "
                << "[--num_images N] [--width W] [--height H] "
                << "[--save_count K] [--output_dir DIR]" << std::endl;
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (options.num_images <= 0 || options.width <= 0 || options.height <= 0 ||
      options.save_count < 0) {
    std::cerr << "All numeric arguments must be positive, except save_count may be zero."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return options;
}

void WritePgm(const fs::path& path,
              const std::vector<unsigned char>& image,
              int width,
              int height) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to write file: " << path << std::endl;
    std::exit(EXIT_FAILURE);
  }
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.data()), image.size());
}

std::vector<unsigned char> GenerateSyntheticImage(int image_id, int width, int height) {
  std::vector<unsigned char> image(width * height);
  std::mt19937 rng(image_id + 12345);
  std::uniform_int_distribution<int> noise_dist(0, 25);

  float cx = static_cast<float>((image_id * 37) % width);
  float cy = static_cast<float>((image_id * 53) % height);
  float radius = static_cast<float>(40 + (image_id % 80));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float dx = x - cx;
      float dy = y - cy;
      float dist = std::sqrt(dx * dx + dy * dy);

      int gradient = (255 * x) / std::max(1, width - 1);
      int checker = (((x / 16) + (y / 16) + image_id) % 2) ? 30 : 0;
      int circle = (std::fabs(dist - radius) < 2.5f) ? 160 : 0;
      int diagonal = ((x + y + image_id * 3) % 97 == 0) ? 100 : 0;
      int value = gradient / 2 + checker + circle + diagonal + noise_dist(rng);
      image[y * width + x] = static_cast<unsigned char>(std::min(255, value));
    }
  }
  return image;
}

int main(int argc, char** argv) {
  Options options = ParseArgs(argc, argv);
  fs::create_directories(options.output_dir);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  CUDA_CHECK(cudaSetDevice(0));

  const size_t pixels_per_image =
      static_cast<size_t>(options.width) * static_cast<size_t>(options.height);
  const size_t total_pixels = pixels_per_image * static_cast<size_t>(options.num_images);

  std::vector<unsigned char> h_input(total_pixels);
  std::vector<unsigned char> h_blur(total_pixels);
  std::vector<unsigned char> h_sobel(total_pixels);

  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < options.num_images; ++i) {
    std::vector<unsigned char> image = GenerateSyntheticImage(i, options.width, options.height);
    std::copy(image.begin(), image.end(), h_input.begin() + i * pixels_per_image);
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  double generation_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  unsigned char* d_input = nullptr;
  unsigned char* d_blur = nullptr;
  unsigned char* d_sobel = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, total_pixels * sizeof(unsigned char)));
  CUDA_CHECK(cudaMalloc(&d_blur, total_pixels * sizeof(unsigned char)));
  CUDA_CHECK(cudaMalloc(&d_sobel, total_pixels * sizeof(unsigned char)));

  cudaEvent_t start_h2d;
  cudaEvent_t end_h2d;
  cudaEvent_t start_blur;
  cudaEvent_t end_blur;
  cudaEvent_t start_sobel;
  cudaEvent_t end_sobel;
  cudaEvent_t start_d2h;
  cudaEvent_t end_d2h;

  CUDA_CHECK(cudaEventCreate(&start_h2d));
  CUDA_CHECK(cudaEventCreate(&end_h2d));
  CUDA_CHECK(cudaEventCreate(&start_blur));
  CUDA_CHECK(cudaEventCreate(&end_blur));
  CUDA_CHECK(cudaEventCreate(&start_sobel));
  CUDA_CHECK(cudaEventCreate(&end_sobel));
  CUDA_CHECK(cudaEventCreate(&start_d2h));
  CUDA_CHECK(cudaEventCreate(&end_d2h));

  CUDA_CHECK(cudaEventRecord(start_h2d));
  CUDA_CHECK(cudaMemcpy(d_input,
                        h_input.data(),
                        total_pixels * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(end_h2d));

  dim3 block(16, 16, 1);
  dim3 grid((options.width + block.x - 1) / block.x,
            (options.height + block.y - 1) / block.y,
            options.num_images);

  CUDA_CHECK(cudaEventRecord(start_blur));
  Blur3x3Kernel<<<grid, block>>>(d_input,
                                 d_blur,
                                 options.width,
                                 options.height,
                                 options.width,
                                 options.num_images);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end_blur));

  CUDA_CHECK(cudaEventRecord(start_sobel));
  SobelKernel<<<grid, block>>>(d_blur,
                               d_sobel,
                               options.width,
                               options.height,
                               options.width,
                               options.num_images);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end_sobel));

  CUDA_CHECK(cudaEventRecord(start_d2h));
  CUDA_CHECK(cudaMemcpy(h_blur.data(),
                        d_blur,
                        total_pixels * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sobel.data(),
                        d_sobel,
                        total_pixels * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(end_d2h));

  CUDA_CHECK(cudaDeviceSynchronize());

  float h2d_ms = 0.0f;
  float blur_ms = 0.0f;
  float sobel_ms = 0.0f;
  float d2h_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start_h2d, end_h2d));
  CUDA_CHECK(cudaEventElapsedTime(&blur_ms, start_blur, end_blur));
  CUDA_CHECK(cudaEventElapsedTime(&sobel_ms, start_sobel, end_sobel));
  CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start_d2h, end_d2h));

  int save_count = std::min(options.save_count, options.num_images);
  for (int i = 0; i < save_count; ++i) {
    std::vector<unsigned char> input_image(h_input.begin() + i * pixels_per_image,
                                           h_input.begin() + (i + 1) * pixels_per_image);
    std::vector<unsigned char> blur_image(h_blur.begin() + i * pixels_per_image,
                                          h_blur.begin() + (i + 1) * pixels_per_image);
    std::vector<unsigned char> sobel_image(h_sobel.begin() + i * pixels_per_image,
                                           h_sobel.begin() + (i + 1) * pixels_per_image);

    std::ostringstream suffix;
    suffix << std::setw(3) << std::setfill('0') << i;
    WritePgm(fs::path(options.output_dir) / ("input_" + suffix.str() + ".pgm"),
             input_image,
             options.width,
             options.height);
    WritePgm(fs::path(options.output_dir) / ("blur_" + suffix.str() + ".pgm"),
             blur_image,
             options.width,
             options.height);
    WritePgm(fs::path(options.output_dir) / ("sobel_" + suffix.str() + ".pgm"),
             sobel_image,
             options.width,
             options.height);
  }

  std::ofstream log(fs::path(options.output_dir) / "execution_log.txt");
  log << "CUDA at Scale Independent Project - Batch Image Processing\n";
  log << "GPU Name: " << prop.name << "\n";
  log << "Images processed: " << options.num_images << "\n";
  log << "Image size: " << options.width << " x " << options.height << "\n";
  log << "Synthetic data generation (CPU): " << generation_ms << " ms\n";
  log << "Host to Device copy: " << h2d_ms << " ms\n";
  log << "GPU blur kernel time: " << blur_ms << " ms\n";
  log << "GPU sobel kernel time: " << sobel_ms << " ms\n";
  log << "Device to Host copy: " << d2h_ms << " ms\n";
  log << "Total pixels processed: " << total_pixels << "\n";
  log << "Saved sample outputs: " << save_count << " images for each stage\n";
  log.close();

  std::cout << "Processed " << options.num_images << " images of size "
            << options.width << "x" << options.height << std::endl;
  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Blur kernel time (ms): " << blur_ms << std::endl;
  std::cout << "Sobel kernel time (ms): " << sobel_ms << std::endl;
  std::cout << "Execution log written to: "
            << (fs::path(options.output_dir) / "execution_log.txt") << std::endl;

  CUDA_CHECK(cudaEventDestroy(start_h2d));
  CUDA_CHECK(cudaEventDestroy(end_h2d));
  CUDA_CHECK(cudaEventDestroy(start_blur));
  CUDA_CHECK(cudaEventDestroy(end_blur));
  CUDA_CHECK(cudaEventDestroy(start_sobel));
  CUDA_CHECK(cudaEventDestroy(end_sobel));
  CUDA_CHECK(cudaEventDestroy(start_d2h));
  CUDA_CHECK(cudaEventDestroy(end_d2h));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_blur));
  CUDA_CHECK(cudaFree(d_sobel));

  return 0;
}
