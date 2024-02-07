#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor-blas/xlinalg.hpp>

template<
const unsigned long stridex=1,
const unsigned long stridey=1,
const unsigned long paddingx=0,
const unsigned long paddingy=0,
typename T>
xt::xarray<T> Embedding(xt::xarray<T> image, xt::xarray<T> filter, xt::xarray<T> bias, xt::xarray<T> pe, xt::xarray<T> cls) {
    auto fshape = xt::adapt(filter.shape());
    auto ishape = xt::adapt(image.shape());
    if(ishape.size() != fshape.size()){
        throw std::runtime_error("Shape Mismatch for Conv2d");
    }
    if(ishape[1] != fshape[1]){
        throw std::runtime_error("No. if input channels not same");
    }
    if(ishape[0] != 1){
        throw std::runtime_error("Embedding layer works only for 1 image for the moment.");
    }
    const unsigned long out_ch = fshape[0]; 
    const unsigned long in_ch = fshape[1];
    const unsigned long kx = fshape[2];
    const unsigned long ky = fshape[3];
    const unsigned long in_x = ishape[2];
    const unsigned long in_y = ishape[3];
    const unsigned long out_x = ((in_x+2*paddingx) - kx)/stridex + 1;  
	const unsigned long out_y = ((in_y+2*paddingy) - ky)/stridey + 1;
	image = xt::view(image, 0, xt::all());
	image = xt::transpose(image, {1,2,0});
	filter = xt::transpose(filter, {2, 3, 1, 0});
	filter = filter.reshape({in_ch*kx*ky, out_ch});
	bias = bias.reshape({1, -1});
	cls = cls.reshape({1, -1});
	cls = cls + xt::view(pe, 0, 0, xt::all());
	bias = xt::view(pe, 0, xt::range(1,out_x*out_y+1), xt::all()) + bias;
	// padding
	if(paddingx != 0 || paddingy != 0){
		image = xt::pad(image, { {paddingx, paddingx}, {paddingy, paddingy}, {0,0} }, xt::pad_mode::constant, 0);
	}
	// im2col
	xt::xarray<T> inp_im2col = xt::zeros<T>({out_x*out_y, kx*ky*in_ch});
	int patch_processed = 0;
	for(unsigned long itr_x = 0; itr_x < in_x; itr_x += stridex) {
		for(unsigned long itr_y = 0; itr_y < in_y; itr_y += stridey) {
			xt::xarray<T> temp = xt::view(image, xt::range(itr_x, itr_x+kx), xt::range(itr_y, itr_y+ky), xt::all());
			temp = temp.reshape({kx*ky*in_ch});
			xt::view(inp_im2col, patch_processed, xt::all()) = temp;
			patch_processed++;
		}
	}
	// conv result
	xt::xarray<T> res = xt::linalg::dot(inp_im2col, filter)+bias;
	res  = xt::concatenate(xt::xtuple(cls, res), 0);
    res = res.reshape({1, out_x*out_y+1, out_ch});
    return res;
}

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

template<typename T>
xt::xarray<T> Softmax(xt::xarray<T> &input, xt::xarray<T> &mask) {
    auto shape = xt::adapt(input.shape());
    const long unsigned int batch_size = shape[0];
    const long unsigned int tokens = shape[1];
    input = xt::exp(input);
    input -= mask;
    xt::xarray<T> sum = xt::sum(input, -1);
    sum = sum.reshape({batch_size,tokens,1});
    input = input/(sum+1e-5);    
    return input;
}

template<typename T>
xt::xarray<T> GeLU(xt::xarray<T> &input) {
    xt::xarray<T> input_erf = 1 + xt::erf(input/1.4142135623730951); // sqrt(2)
    input = input * input_erf/2;
    return input;
}

#endif // UTILS_H
