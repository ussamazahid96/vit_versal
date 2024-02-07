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
#include "softmax.hpp"

template<typename T>
xt::xarray<T> Softmax(xt::xarray<T> &input) {
    auto shape = xt::adapt(input.shape());
    const long unsigned int batch_size = shape[0];
    const long unsigned int tokens = shape[1];
    xt::xarray<T> maxes = xt::amax(input, -1);
    input -= maxes.reshape({batch_size,tokens,1});
    input = xt::exp(input);
    xt::xarray<T> sum = xt::sum(input, -1);
    sum = sum.reshape({batch_size,tokens,1});
    input = input/(sum+1e-5);
    return input;
}

int main()
{
	const int batch_size = 5;
    xt::random::seed(time(NULL));
	xt::xarray<float> inp = xt::random::rand<float>({batch_size, TOKENS, TOKENS})*255-128;
	xt::xarray<float> inp_copy = inp/(float) SQRT_DIM;
	// golden softmax
	xt::xarray<float> golden_sm = Softmax(inp_copy);
	inp = xt::pad(inp, { {0,0}, {0,TOKENS_PADDED-TOKENS}, {0,TOKENS_PADDED-TOKENS} }, xt::pad_mode::constant, 0);

	ap_uint<AXI_WIDTH> *input_buffer = reinterpret_cast<ap_uint<AXI_WIDTH>*>(inp.data());
	ap_uint<AXI_WIDTH> output_buffer[inp.size()*DATA_TYPE/AXI_WIDTH];

	softmax(input_buffer, output_buffer, batch_size);
	xt::xarray<float> output_f = xt::adapt((float*)output_buffer, inp.size(), xt::no_ownership(), inp.shape());

	xt::dump_npy("/home/zahidu/workspace/CHARM/training/xtensor.npy", golden_sm);
	xt::dump_npy("/home/zahidu/workspace/CHARM/training/hls_out.npy", output_f);
	return 0;

}
