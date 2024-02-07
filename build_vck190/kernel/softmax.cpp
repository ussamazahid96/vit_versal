#include "softmax.hpp"

datatype myexponential(datatype input) {
	datatype lower_limit = -15.9375;
	datatype output;
	if(input < lower_limit) {
		output = exp_table[255];
	}
	else {
		ap_uint<8> index = input*-16;
		output = exp_table[index];
	}
	return output;
}

void Mem2Stream(ap_uint<AXI_WIDTH> *in, hls::stream<ap_uint<AXI_WIDTH>> &out, const int batch){
#pragma HLS INLINE off
	for(int i=0;i<batch*PADDED_SIZE;i++){
#pragma HLS pipeline style=flp II=1
		ap_uint<AXI_WIDTH> el = in[i];
		out.write(el);
	}
}

void Stream2Mem(ap_uint<AXI_WIDTH> *out, hls::stream<ap_uint<AXI_WIDTH>> &in, const int batch){
#pragma HLS INLINE off
	for(int i=0;i<batch*PADDED_SIZE;i++){
#pragma HLS pipeline style=flp II=1
		ap_uint<AXI_WIDTH> el = in.read();
		out[i] = el;
	}
}

void softmax_core(hls::stream<ap_uint<AXI_WIDTH>> &in, hls::stream<ap_uint<AXI_WIDTH>> &out, const int batch) {
	datatype bias[2] = {-128.0, -128.0}, bias_local = -128.0;
	datatype sum[2]  = {0.0, 0.0}, sum_local = 0.0;
	datatype buffer[2][TOKENS_PADDED];
	ap_uint<16> dim_read = 0, dim_write = 0;
	ap_uint<1> write_counter = 0, read_counter = 0;
	const int iter = batch*PADDED_SIZE+TOKENS_PADDED;

	for(int i=0; i<iter; i++) {
#pragma HLS pipeline style=flp II=1

		// fill the buffer and calculate the sum/bias
		if(i < batch*PADDED_SIZE) {
			auto inelem = in.read();
			float elem_f = *reinterpret_cast<float*>(&inelem);
			ap_int<DATA_TYPE> elemi = elem_f*256/SQRT_DIM;
			datatype elem = (dim_read < TOKENS) ? *reinterpret_cast<datatype*>(&elemi) : (datatype) -128.0;
			buffer[write_counter][dim_read] = elem;

			datatype diff = elem-bias_local;
			datatype scale = 1.0, exp_value = 1.0;
			if(elem > bias_local){
				scale = myexponential(-diff);
				bias_local = elem;
			}
			else {
				exp_value = myexponential(diff);
			}
			datatype sum_scaled = sum_local*scale;
			sum_local = sum_scaled + exp_value;

			if(++dim_read == TOKENS_PADDED){
				dim_read = 0;
				bias[write_counter] = bias_local;
				sum[write_counter] = sum_local;
				bias_local = -128.0;
				sum_local = 0.0;
				write_counter++;
			}
		} // end write block

		// read the buffer and output the softmax
		if(i>TOKENS_PADDED-1){
			ap_uint<AXI_WIDTH> out_data;
			datatype difference = buffer[read_counter][dim_write] - bias[read_counter];
			datatype exp_value = myexponential(difference);
			float softmax = exp_value / sum[read_counter];
			out_data = *reinterpret_cast<ap_uint<AXI_WIDTH>*>(&softmax);
			out.write(out_data);
			if(++dim_write == TOKENS_PADDED) {
				dim_write = 0;
				read_counter++;
			}
		} //end read block

	}
}

void softmax(ap_uint<AXI_WIDTH> *in, ap_uint<AXI_WIDTH> *out, const int batch)
{
#pragma HLS interface m_axi offset=slave bundle=in0 port=in max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=in
#pragma HLS interface m_axi offset=slave bundle=out0 port=out max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=out
#pragma HLS interface s_axilite bundle=control port=batch
#pragma HLS interface s_axilite bundle=control port=return

#pragma HLS DATAFLOW

	hls::stream<ap_uint<AXI_WIDTH>> inter0("inter0");
#pragma HLS STREAM depth=8 type=fifo variable=inter0
	hls::stream<ap_uint<AXI_WIDTH>> inter1("inter1");
#pragma HLS STREAM depth=8 type=fifo variable=inter1

	Mem2Stream(in, inter0, batch);
	softmax_core(inter0, inter1, batch);
	Stream2Mem(out, inter1, batch);
}


