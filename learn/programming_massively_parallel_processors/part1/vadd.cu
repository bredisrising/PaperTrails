#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


double vecAdd(float* A_h, float* B_h, float* C_h, long n, int block_dim) {
    size_t size = (size_t) n * sizeof(float);

    float *A_d, *B_d, *C_d;


    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);
    
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);


    // first allocate device memory for A, B, C
    // copy A and B to device memory

    // then call kernel - to launch grid of threads
    // perform actual vector addition

    // Copy C from device memory
    // Free device vectors
   
    int blocks = n / block_dim;

    struct timespec s, e; 
    clock_gettime(CLOCK_MONOTONIC, &s);

    vecAddKernel<<<blocks + 1, block_dim>>>(A_d, B_d, C_d, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("sync: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &e);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return (e.tv_sec - s.tv_sec) * 1000 + (e.tv_nsec - s.tv_nsec) / 1e6;

}

int main(int argc, char** argv) {
    int thread_per_block = atoi(argv[2]); 
    long n = atol(argv[1]); 

    // FILE* output = fopen("./out", "w");
    
    size_t size = (size_t) n * sizeof(float);
    
    double mb = (double) sizeof(float) * n / 1e6;

    float* A = (float*) malloc(size);
    float* B =  (float*) malloc(size);
    float* C = (float*)  malloc(size);
    
    for (int i = 0; i < n; i++) {
        A[i] = 1;
        B[i] = 2;
    }

    double time_taken = vecAdd(A,B,C,n, thread_per_block);

    printf("%ld,%f\n", n, time_taken);

    free(A);
    free(B);
    free(C);
}
