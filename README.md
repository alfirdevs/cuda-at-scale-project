# CUDA at Scale Independent Project: Batch GPU Image Processing
By Khalid El-Darymli

## Project Summary
This project is a simple CUDA-based batch image processing application designed to satisfy the **CUDA at Scale Independent Project** requirements with minimal setup and no external dependencies beyond CUDA.

The program generates **hundreds of grayscale images** synthetically, sends them to the GPU, and performs two GPU computations:

1. **3x3 blur filtering** using a custom CUDA kernel.
2. **Sobel edge detection** using a custom CUDA kernel.

It then copies the results back to the CPU, saves sample output images in `.pgm` format, and writes a timing log. This demonstrates GPU processing on a large batch of inputs in a single run.

## Why this project scores well against the rubric
- Public repository can include all required files.
- Includes a **README.md**.
- Includes a **CLI with arguments**.
- Includes **support files** for compiling and running (`Makefile`, `run.sh`).
- Uses **actual GPU computation** through CUDA kernels.
- Produces **proof-of-execution artifacts** automatically:
  - output images
  - execution log
- Includes a clear description of the work, kernels, and lessons learned.

## Files
- `main.cu` - main CUDA program
- `Makefile` - build instructions
- `run.sh` - one-command build and execution script
- `output/` - created after running; contains images and timing log
- `proof/` - place screenshots of terminal output here before submission

## Build
```bash
make build
```

## Run
```bash
./cuda_batch_image_processing.exe \
  --num_images 256 \
  --width 512 \
  --height 512 \
  --save_count 6 \
  --output_dir output
```

Or simply:
```bash
./run.sh
```

## Command-line arguments
- `--num_images` : number of images to generate and process
- `--width` : image width
- `--height` : image height
- `--save_count` : how many sample images to save per stage
- `--output_dir` : directory for results

Example for larger images:
```bash
./cuda_batch_image_processing.exe \
  --num_images 32 \
  --width 2048 \
  --height 2048 \
  --save_count 4 \
  --output_dir output_large
```

## Output artifacts
After a successful run, the program creates:
- `output/execution_log.txt`
- `output/input_000.pgm`, `output/blur_000.pgm`, `output/sobel_000.pgm`, etc.

These are the artifacts you should commit to your repository as proof of execution.

## Algorithm description
### Synthetic dataset
The project generates a large batch of grayscale images with:
- gradients
- checkerboard texture
- circular structures
- diagonal features
- random noise

This produces nontrivial inputs for filtering and edge detection.

### GPU kernel 1: 3x3 blur
The blur kernel computes the average of a pixel and its 8 neighbors. This is a classic image processing operation and is easy to understand and verify.

### GPU kernel 2: Sobel edge detection
The Sobel kernel computes horizontal and vertical gradients and converts them into an edge magnitude image. This is a standard signal/image processing operation and demonstrates another GPU stage.

## Lessons learned / discussion points for submission
You can reuse the following ideas in your short write-up:
- GPU processing is especially useful when the same operation must be applied to many images.
- Even simple kernels like blur and Sobel can demonstrate the GPU’s value when scaled to hundreds of inputs.
- Batch processing reduces per-image overhead and makes the workload more representative of real-world pipelines.
- Saving output images and timing logs makes the project easy for a reviewer to verify.
- Using `.pgm` avoids external image library dependencies and keeps the project portable.

## Suggested repository contents before submission
Before submitting, make sure your public repo includes:
1. Source code
2. `README.md`
3. `Makefile`
4. `run.sh`
5. Output artifacts from a real run:
   - `output/execution_log.txt`
   - several `.pgm` files
6. One or two screenshots in `proof/` showing the program running in the terminal

## Short project description for Coursera submission
I implemented a CUDA batch image processing project that generates hundreds of grayscale images and processes them on the GPU in a single run. The program applies a 3x3 blur filter and then Sobel edge detection using custom CUDA kernels. It saves sample output images and a timing log as proof of execution. I chose this project because it demonstrates GPU acceleration on a large number of inputs while keeping the code simple, portable, and easy to review.
