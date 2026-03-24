# 🚀 GPU-Accelerated Edge Detection using CUDA (Sobel Operator)

## 📌 Overview

This project demonstrates GPU-accelerated image processing using CUDA by implementing the Sobel edge detection algorithm. The goal is to highlight how parallel computing on GPUs can significantly improve performance for image processing tasks.

The program loads a real-world image (JPEG/PNG), applies Sobel filtering using a CUDA kernel, and outputs an edge-detected image.

---

## 🧠 Key Concepts Used

* CUDA Programming (C++)
* GPU Parallelism (thread blocks & grids)
* Image Processing (Sobel Edge Detection)
* Memory Management (Host ↔ Device transfer)
* OpenCV for image I/O

---

## ⚙️ How It Works

1. The input image is loaded using OpenCV in grayscale format.

2. The image is copied from CPU memory (host) to GPU memory (device).

3. A CUDA kernel is launched:

   * Each thread processes one pixel.
   * Sobel filters are applied in X and Y directions.

4. Gradient magnitude is computed:

   G = sqrt(Gx² + Gy²)

5. The result is copied back to CPU memory.

6. Output image is saved as `output.png`.

---

## 📂 Project Structure

```
gpu-edge-detection-cuda/
│
├── sobel_opencv.cu       # CUDA kernel + main program
├── run.sh                # Compile and run script
├── README.md             # Project documentation
│
├── data/
│   └── input.jpg         # Input image
│
├── outputs/
│   └── output.png        # Edge-detected output
│
└── logs/
    └── log.txt           # Execution logs (optional)
```

---

## ▶️ How to Run (Google Colab)

### 1. Enable GPU

Runtime → Change runtime → GPU

### 2. Install dependencies

```
!apt-get update
!apt-get install -y libopencv-dev
```

### 3. Add input image

```
!mkdir -p data
!wget -O data/input.jpg https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg
```

### 4. Compile and run

```
!bash run.sh
```

---

## 🖼️ Output

* Input: `data/input.jpg`
* Output: `output.png`

To visualize:

```python
from PIL import Image
from IPython.display import display

display(Image.open("output.png"))
```

---

## ⚡ GPU Implementation Details

* Each CUDA thread processes one pixel.
* Thread configuration:

  ```
  dim3 threads(16,16);
  dim3 blocks((width+15)/16, (height+15)/16);
  ```
* Memory operations:

  * `cudaMalloc`
  * `cudaMemcpy`
  * `cudaFree`

---

## 📊 Results

The GPU implementation enables parallel computation of edge detection across all pixels simultaneously, significantly improving performance compared to sequential CPU execution.

---

## 🎯 Key Learnings

* How to implement image processing algorithms in CUDA
* Efficient memory transfer between CPU and GPU
* Mapping image pixels to CUDA threads
* Using OpenCV with CUDA for real-world applications

---

## 🚀 Future Improvements

* CPU vs GPU performance comparison
* Shared memory optimization for faster access
* Real-time video edge detection
* Integration with deep learning pipelines

---

## 📌 Conclusion

This project demonstrates how GPU acceleration using CUDA can be applied to real-world image processing tasks. The Sobel operator, when parallelized, showcases the power of GPU computing in handling large-scale data efficiently.

---