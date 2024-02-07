#include <iostream>
#include<ctime>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "rant.hpp"


int main() {

	xt::xarray<int> inp = xt::random::randint<int>({1, TOKENS, 3, 12, HEAD_DIM})%10;

	ap_uint<AXI_WIDTH> *in_buffer = reinterpret_cast<ap_uint<AXI_WIDTH>*>(inp.data());
	ap_uint<AXI_WIDTH> qout_buffer[TOKENS*HEAD_DIM], kout_buffer[TOKENS*HEAD_DIM], vout_buffer[TOKENS*HEAD_DIM];

	for(int head_idx=0;head_idx<12;head_idx++) {
		rant(in_buffer, in_buffer, in_buffer, qout_buffer, kout_buffer, vout_buffer, head_idx);
		xt::xarray<float> q = xt::view(inp, xt::all(), xt::all(), 0, head_idx, xt::all());
	    xt::xarray<float> k = xt::view(inp, xt::all(), xt::all(), 1, head_idx, xt::all());
	    k = xt::transpose(k, {0,2,1});
		xt::xarray<float> v = xt::view(inp, xt::all(), xt::all(), 2, head_idx, xt::all());
		xt::xarray<int> queries_out = xt::adapt((int*)qout_buffer, TOKENS*HEAD_DIM, xt::no_ownership(), std::vector<std::size_t> {1, TOKENS, HEAD_DIM});
		xt::xarray<int> keys_out = xt::adapt((int*)kout_buffer, TOKENS*HEAD_DIM, xt::no_ownership(), std::vector<std::size_t> {1, HEAD_DIM, TOKENS});
		xt::xarray<int> values_out = xt::adapt((int*)vout_buffer, TOKENS*HEAD_DIM, xt::no_ownership(), std::vector<std::size_t> {1, TOKENS, HEAD_DIM});

		if(!xt::allclose(q, queries_out) || !xt::allclose(v, values_out) || !xt::allclose(k, keys_out)) {
			std::cout << "Failed..." << '\n';
		} else {
			std::cout << "Passed..." << '\n';
		}
	}
	return 0;
}

