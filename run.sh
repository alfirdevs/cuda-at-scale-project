#!/usr/bin/env bash
set -euo pipefail

make clean
make build
./cuda_batch_image_processing.exe \
  --num_images 256 \
  --width 512 \
  --height 512 \
  --save_count 6 \
  --output_dir output
