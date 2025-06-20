#include <mma.h>

using namespace nvcuda;
extern "C" __global__ void test_wmma(half *a, half *b, half *c, unsigned stride) {
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

   wmma::load_matrix_sync(a_frag, a, 16, stride);
   wmma::load_matrix_sync(b_frag, b, 16, stride);
   wmma::load_matrix_sync(c_frag, c, 16, stride);

   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
