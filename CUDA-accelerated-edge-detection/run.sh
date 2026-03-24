#!/bin/bash

# Compile CUDA + OpenCV
nvcc sobel.cu -o sobel `pkg-config --cflags --libs opencv4`

# Run the program
./sobel