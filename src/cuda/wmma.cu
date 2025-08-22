// #include <stdlib.h>
// #include <stdio.h>
// #include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
// #include <driver_types.h>
// #include <helper_cuda.h>

// extern "C" __host__ int main() {return 0;}

using namespace nvcuda;
extern "C" __global__ void wmma_half(half *a, half *b, half *c, unsigned char *b_a, unsigned char *b_b, int *b_c, const size_t stride) {
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, half> dest_frag;
  (void)(b_a);
  (void)(b_b);
  (void)(b_c);

   wmma::fill_fragment(dest_frag, 0.0f);
   wmma::load_matrix_sync(a_frag, a, stride);
   wmma::load_matrix_sync(b_frag, b, stride);

   wmma::mma_sync(dest_frag, a_frag, b_frag, dest_frag);

   wmma::store_matrix_sync(c, dest_frag, stride, wmma::mem_col_major);
}

extern __global__ void dummy_quant(half *a, unsigned char *b){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  // b[x] = 1;
  b[x] = __half2uchar_rz(a[x]);
}

extern __global__ void dummy_dequant(int *a, half *b){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  // b[x] = 1.0;
  b[x] = __int2half_rz(a[x]);
  //b[x] = __uchar2float(a[x]);
}

const int byteMatrixSize = 256;
const int blockSize = 32; //read developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

extern "C" __global__ void wmma_byte(half *a, half *b, half *c, unsigned char *b_a, unsigned char *b_b, int *b_c, const size_t stride) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> dest_frag;

   // unsigned char *b_a, *b_b; 
   // int *b_c;
   // checkCudaErrors(cudaMalloc((void **)(&b_a), byteMatrixSize));
   // checkCudaErrors(cudaMalloc((void **)(&b_b), byteMatrixSize));
   // checkCudaErrors(cudaMalloc((void **)(&b_c), byteMatrixSize));

   dummy_quant<<<byteMatrixSize/blockSize, blockSize>>>(a,b_a);
   dummy_quant<<<byteMatrixSize/blockSize, blockSize>>>(b,b_b);

   wmma::fill_fragment(dest_frag, 0);
   wmma::load_matrix_sync(a_frag, b_a, stride);
   wmma::load_matrix_sync(b_frag, b_b, stride);

   wmma::mma_sync(dest_frag, a_frag, b_frag, dest_frag);

   wmma::store_matrix_sync(b_c, dest_frag, stride, wmma::mem_col_major);
  
   dummy_dequant<<<byteMatrixSize/blockSize, blockSize>>>(b_c, c);
}
