/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda and cublas*/
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* Main */
int main(int argc, char* argv[]) {

  if (argc < 3 ){
    printf("Arguement Error: Usage ./prog M K N\n");
    return -1;
  }

  cublasStatus_t status;
  float *h_A;
  float *h_B;
  float *h_C;
  float *h_C_ref;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  
  int M = atoi(argv[1]);
  int K = atoi(argv[2]);
  int N = atoi(argv[3]);

  int n2 = M * N;

  int sizeA = M * K;
  int sizeB = K * N;
  int sizeC = M * N;
  
  int i;
  float error_norm;
  float ref_norm;
  float diff;
  cublasHandle_t handle;

  status = cublasCreate(&handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  /* Allocate host memory for the matrices */
  h_A = reinterpret_cast<float *>(malloc(sizeA * sizeof(h_A[0])));

  if (h_A == 0) {
    fprintf(stderr, "!!!! host memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  h_B = reinterpret_cast<float *>(malloc(sizeB * sizeof(h_B[0])));

  if (h_B == 0) {
    fprintf(stderr, "!!!! host memory allocation error (B)\n");
    return EXIT_FAILURE;
  }

  h_C = reinterpret_cast<float *>(malloc(sizeC * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Fill the matrices with test data */
  // for (i = 0; i < n2; i++) {
  //   h_A[i] = rand() / static_cast<float>(RAND_MAX);
  //   h_B[i] = rand() / static_cast<float>(RAND_MAX);
  //   h_C[i] = rand() / static_cast<float>(RAND_MAX);
  // }

  /* Allocate device memory for the matrices */
  if (cudaMalloc(reinterpret_cast<void **>(&d_A), sizeA * sizeof(d_A[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B), sizeB * sizeof(d_B[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C), sizeC * sizeof(d_C[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  /* Initialize the device matrices with the host matrices */
  status = cublasSetVector(sizeA, sizeof(h_A[0]), h_A, 1, d_A, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write A)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(sizeB, sizeof(h_B[0]), h_B, 1, d_B, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write B)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(sizeC, sizeof(h_C[0]), h_C, 1, d_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write C)\n");
    return EXIT_FAILURE;
  }


  #define ITER 200
  // START: Added for measuring speed.
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // END: Added for measuring speed.
  for(int trial = 0; trial < ITER; trial ++) {
    /* Performs operation using cublaGemmEX */
    cublasGemmEx(handle, 
                  CUBLAS_OP_N, 
                  CUBLAS_OP_N,
                  M, 
                  N, 
                  K, 
                  &alpha, 
                  d_A, 
                  CUDA_R_8I, 
                  M, 
                  d_B, 
                  CUDA_R_8I, 
                  K, 
                  &beta, 
                  d_C, 
                  CUDA_R_32F, 
                  M,
                  CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLASG kernel execution error.\n");
    return EXIT_FAILURE;
  }

  // START: MEASURE time and flops.
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);

  // printf("Time: %f ms\n", milliseconds);
  printf("M: %d, K: %d, N: %d, TFLOPS: %.2f\n", M, K, N, static_cast<double>(ITER*(static_cast<double>(M) *
                                                K * N * 2) /
                                               (milliseconds / 1000.)) / 1e12);
  // printf("Time (ms): %.5f\n", milliseconds/ITER);
  // END: MEASURE time and flops.


  /* Allocate host memory for reading back the result from device memory */
  h_C = reinterpret_cast<float *>(malloc(sizeC * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Read the result back */
  status = cublasGetVector(sizeC, sizeof(h_C[0]), d_C, 1, h_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (read C)\n");
    return EXIT_FAILURE;
  }

  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  // free(h_C_ref);

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

}


