#include <mma.h>

using namespace nvcuda;
extern "C" __global__ void test_wmma(half *a, half *b, half *c, const size_t stride) {
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, half> dest_frag;

   wmma::fill_fragment(dest_frag, 0.0f);
   wmma::load_matrix_sync(a_frag, a, stride);
   wmma::load_matrix_sync(b_frag, b, stride);

   wmma::mma_sync(dest_frag, a_frag, b_frag, dest_frag);

   wmma::store_matrix_sync(c, dest_frag, stride, wmma::mem_col_major);
}
