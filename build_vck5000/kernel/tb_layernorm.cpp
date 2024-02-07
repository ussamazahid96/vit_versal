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

#include "layernorm.hpp"

template<typename T>
xt::xarray<T> LayerNorm(xt::xarray<T> &input, xt::xarray<T> &weight, xt::xarray<T> &bias) {
    auto shape = xt::adapt(input.shape());
    const long unsigned int batch_size = shape[0];
    const long unsigned int tokens = shape[1];
    xt::xarray<T> means = xt::mean(input,2);
    means = means.reshape({batch_size, tokens, 1});
    xt::xarray<T> vars = xt::pow(input-means,2);
    xt::xarray<T> var_mean = xt::mean(vars, 2);
    xt::xarray<T> std = xt::sqrt(var_mean+1e-5);
    std = std.reshape({batch_size, tokens, 1});
    xt::xarray<T> input_norm = (input-means)/std;
    input_norm = (input_norm*weight)+bias;
    return input_norm;
}

int main()
{
	const int BATCH_SIZE = 1;
    xt::xarray<float> inp1 = xt::random::rand<float>({BATCH_SIZE, 197, EMB_DIM})*10-5;
    xt::xarray<float> inp2 = xt::random::rand<float>({BATCH_SIZE, 197, EMB_DIM})*10-5;
    xt::xarray<float> inp1p = xt::pad(inp1, { {0,0}, {0, TOKENS_PADDED-TOKENS}, {0,0} }, xt::pad_mode::constant, 0);
    xt::xarray<float> inp2p = xt::pad(inp2, { {0,0}, {0, TOKENS_PADDED-TOKENS}, {0,0} }, xt::pad_mode::constant, 0);
    auto shape = xt::adapt(inp1.shape());
    const int images = shape[0];

    xt::xarray<float> nw = xt::random::rand<float>({EMB_DIM})*2-1;
    xt::xarray<float> nb = xt::random::rand<float>({EMB_DIM})*2-1;
    xt::xarray<float> skip = inp1+inp2;
    xt::xarray<float> golden = LayerNorm(skip, nw, nb);
    std::cout << skip << '\n';
    xt::dump_npy("/home/zahidu/workspace/CHARM/training/xtensor.npy", skip);

    ap_uint<AXI_WIDTH> *in1 = reinterpret_cast<ap_uint<AXI_WIDTH>*>(inp1p.data());
    ap_uint<AXI_WIDTH> *in2 = reinterpret_cast<ap_uint<AXI_WIDTH>*>(inp2p.data());
    ap_uint<AXI_WIDTH> *norm_w = reinterpret_cast<ap_uint<AXI_WIDTH>*>(nw.data());
    ap_uint<AXI_WIDTH> *norm_b = reinterpret_cast<ap_uint<AXI_WIDTH>*>(nb.data());
    ap_uint<AXI_WIDTH> out1[inp1p.size()*DATA_TYPE/AXI_WIDTH], out2[inp2p.size()*DATA_TYPE/AXI_WIDTH];
    
    layernorm(in1, in2, norm_w, norm_b, out1, out2, images);

    xt::xarray<float> output1_f = xt::adapt((float*) out1,  inp1p.size(), xt::no_ownership(), inp1p.shape());
    xt::xarray<float> output2_f = xt::adapt((float*) out2,  inp2p.size(), xt::no_ownership(), inp2p.shape());
    std::cout << output2_f << '\n';
    output1_f = xt::view(output1_f, xt::all(), xt::range(_, TOKENS), xt::all());
    output2_f = xt::view(output2_f, xt::all(), xt::range(_, TOKENS), xt::all());
    xt::dump_npy("/home/zahidu/workspace/CHARM/training/hls_out.npy", output2_f);

    return 0;
}











