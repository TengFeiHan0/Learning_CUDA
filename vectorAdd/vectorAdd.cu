#include <stdio.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/book.h"

__global__ void vectorAdd(const float *A, const float *B, float *C,
    int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv){

     // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);
    float *h_A, *h_B, *h_C;
    float *dev_a, *dev_b, *dev_c;
    
    // allocate the memory on the CPU
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, size ) );

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / static_cast<float>(RAND_MAX);
        h_B[i] = rand() / static_cast<float>(RAND_MAX);
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, h_A, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, h_B, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_c, h_C, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
            threadsPerBlock);
            
    dim3 cudaBlockSize(threadsPerBlock, 1, 1);
    dim3 cudaGridSize(blocksPerGrid, 1, 1);

    vectorAdd<<<cudaGridSize, cudaBlockSize>>>(dev_a, dev_b, dev_c, numElements);

    HANDLE_ERROR(cudaMemcpy(h_C, dev_c, size, cudaMemcpyDeviceToHost));

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
        fprintf(stderr, "Result verification failed at element %d!\n", i);
        exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;   
}