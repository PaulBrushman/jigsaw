#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
extern "C" __global__ void wmma_half(half *a, half *b, half *c, const size_t stride) {
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, half> dest_frag;

   wmma::fill_fragment(dest_frag, 0.0f);
   wmma::load_matrix_sync(a_frag, a, stride);
   wmma::load_matrix_sync(b_frag, b, stride);

   wmma::mma_sync(dest_frag, a_frag, b_frag, dest_frag);

   wmma::store_matrix_sync(c, dest_frag, stride, wmma::mem_col_major);
}

__device__ void dummy_quant(half *a, unsigned char *b){
  //__half2uchar_rz
}
 
extern "C" __global__ void wmma_byte(half *a, half *b, half *c, const size_t stride) {
   wmma::fragment<wmma::matrix_a, 16, 16, 16, byte, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, byte, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, byte> dest_frag;

   wmma::fill_fragment(dest_frag, 0.0f);
   wmma::load_matrix_sync(a_frag, a, stride);
   wmma::load_matrix_sync(b_frag, b, stride);

   wmma::mma_sync(dest_frag, a_frag, b_frag, dest_frag);

   wmma::store_matrix_sync(c, dest_frag, stride, wmma::mem_col_major);
}
