#include <iostream>
#include<ctime>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "gelu.hpp"

template<typename T>
xt::xarray<T> GeLU(xt::xarray<T> &input) {
    xt::xarray<T> input_erf = 1 + xt::erf(input/1.4142135623730951); // sqrt(2)
    input = input * input_erf/2;
    return input;
}


int main() {

	const int BATCH_SIZE = 1;
    xt::xarray<float> inp = xt::random::rand<float>({BATCH_SIZE, TOKENS, EMB_DIM})*10-5;
    xt::xarray<float> inpp = xt::pad(inp, { {0,0}, {0, TOKENS_PADDED-TOKENS}, {0,0} }, xt::pad_mode::constant, 0);

    xt::xarray<float> golden = GeLU(inp);
    xt::dump_npy("/home/zahidu/workspace/CHARM/training/xtensor.npy", golden);

    ap_uint<AXI_WIDTH> *in = reinterpret_cast<ap_uint<AXI_WIDTH>*>(inpp.data());
    ap_uint<AXI_WIDTH> out[inpp.size()*DATA_TYPE/AXI_WIDTH];

    gelu(in, out, BATCH_SIZE);
    xt::xarray<float> output_f = xt::adapt((float*) out,  inpp.size(), xt::no_ownership(), inpp.shape());
    xt::dump_npy("/home/zahidu/workspace/CHARM/training/hls_out.npy", output_f);

    return 0;
}
