#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int idx = y * width + x;

        int gx =
            -input[(y-1)*width + (x-1)] - 2*input[y*width + (x-1)] - input[(y+1)*width + (x-1)]
            +input[(y-1)*width + (x+1)] + 2*input[y*width + (x+1)] + input[(y+1)*width + (x+1)];

        int gy =
             input[(y-1)*width + (x-1)] + 2*input[(y-1)*width + x] + input[(y-1)*width + (x+1)]
            -input[(y+1)*width + (x-1)] - 2*input[(y+1)*width + x] - input[(y+1)*width + (x+1)];

        int mag = sqrtf(gx * gx + gy * gy);
        output[idx] = min(255, mag);
    }
}

int main() {
    // Load image
    cv::Mat img = cv::imread("data/input.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error loading image\n";
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    cv::Mat output(height, width, CV_8UC1);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);

    cudaMemcpy(d_input, img.data, width * height, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((width+15)/16, (height+15)/16);

    sobelKernel<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_output, width * height, cudaMemcpyDeviceToHost);

    cv::imwrite("output.png", output);

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Done! Output saved as output.png\n";
    return 0;
}