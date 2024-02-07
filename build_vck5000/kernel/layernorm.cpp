#include "layernorm.hpp"

void Mem2Stream(ap_uint<AXI_WIDTH> *in, ap_uint<AXI_WIDTH> *skip_in,
				ap_uint<AXI_WIDTH> *wei, ap_uint<AXI_WIDTH> *bias,
				hls::stream<ap_uint<AXI_WIDTH>> &out, hls::stream<ap_uint<AXI_WIDTH>> &skip_out,
				hls::stream<ap_uint<AXI_WIDTH>> &w_out, hls::stream<ap_uint<AXI_WIDTH>> &b_out,
				const int batch){
#pragma HLS INLINE off
	const int bound = batch*TOKENS_PADDED*EMB_DIM*DATA_TYPE/AXI_WIDTH;
	const int weights = EMB_DIM*DATA_TYPE/AXI_WIDTH;
	for(int i=0;i<bound;i++){
#pragma HLS pipeline style=flp II=1
		ap_uint<AXI_WIDTH> el = in[i];
		ap_uint<AXI_WIDTH> sel = skip_in[i];
		out.write(el);
		skip_out.write(sel);
		if(i<weights){
			ap_uint<AXI_WIDTH> welem = wei[i];
			ap_uint<AXI_WIDTH> belem = bias[i];
			w_out.write(welem);
			b_out.write(belem);
		}
	}
}

void Stream2Mem(ap_uint<AXI_WIDTH> *out, ap_uint<AXI_WIDTH> *skip_out,
				hls::stream<ap_uint<AXI_WIDTH>> &in,
				hls::stream<ap_uint<AXI_WIDTH>> &skip_in,
				const int batch){
#pragma HLS INLINE off
	const int bound = batch*TOKENS_PADDED*EMB_DIM*DATA_TYPE/AXI_WIDTH;
	for(int i=0;i<bound;i++){
#pragma HLS pipeline style=flp II=1
		ap_uint<AXI_WIDTH> el = in.read();
		ap_uint<AXI_WIDTH> sel = skip_in.read();
		out[i] = el;
		skip_out[i] = sel;
	}
}

#define II_CYCLES 11

void LayerNorm_Core(hls::stream<ap_uint<AXI_WIDTH>>& in_stream, hls::stream<ap_uint<AXI_WIDTH>>& skip_in_stream,
					hls::stream<ap_uint<AXI_WIDTH>>& wei_stream, hls::stream<ap_uint<AXI_WIDTH>>& bias_stream,
					hls::stream<ap_uint<AXI_WIDTH>>& out_stream, hls::stream<ap_uint<AXI_WIDTH>>& skip_out_stream,
					const int batch) {
#pragma HLS inline off

	const int iter = batch*(TOKENS_PADDED*EMB_DIM)+EMB_DIM;
	ap_uint<1> write_counter = 0, read_counter = 0;
	ap_uint<16> dim_read = 0, dim_write = 0, token_no = 0;
	datatype weight_buffer[EMB_DIM], bias_buffer[EMB_DIM], input_buffer[2][EMB_DIM];
	stattype running_mean_local = 0.0, running_mean_sqr_local = 0.0, running_mean[2], running_mean_sqr[2];

	stattype running_mean_copies[II_CYCLES], running_mean_sqr_copies[II_CYCLES];
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=running_mean_copies
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=running_mean_sqr_copies
	for(int c=0;c<II_CYCLES;c++){
	#pragma HLS UNROLL
		running_mean_copies[c] = 0.0;
		running_mean_sqr_copies[c] = 0.0;
	}

	for(int i=0; i<iter; i++) {

		if(i<batch*(TOKENS_PADDED*EMB_DIM)) { // fill the buffer and compute mean/std
			auto inelem = in_stream.read();
			auto sinelem = skip_in_stream.read();
			datatype elem1 = *reinterpret_cast<datatype*>(&inelem);
			datatype elem2 = *reinterpret_cast<datatype*>(&sinelem);
			datatype elem = elem1+elem2;
			input_buffer[write_counter][dim_read] = elem;
			if(i<EMB_DIM) { // read  weights and bias
				auto welem = wei_stream.read();
				auto belem = bias_stream.read();
				weight_buffer[i] = *reinterpret_cast<datatype*>(&welem);
				bias_buffer[i] = *reinterpret_cast<datatype*>(&belem);

			}
			stattype cur = running_mean_copies[II_CYCLES-1] + (elem/EMB_DIM);
			stattype cur_sqr = running_mean_sqr_copies[II_CYCLES-1] + (elem*elem/EMB_DIM);
			for(int c=II_CYCLES-1;c>0;c--){
#pragma HLS UNROLL
				running_mean_copies[c] = running_mean_copies[c-1];
				running_mean_sqr_copies[c] = running_mean_sqr_copies[c-1];
			}
			running_mean_copies[0] = cur;
			running_mean_sqr_copies[0] = cur_sqr;

			if(++dim_read == EMB_DIM){
				dim_read = 0;
				stattype means_sum = 0.0, means_sqr_sum = 0.0;
				for(int c=0;c<II_CYCLES;c++){
#pragma HLS UNROLL
					means_sum += running_mean_copies[c];
					means_sqr_sum += running_mean_sqr_copies[c];
				}
				running_mean[write_counter] = means_sum;
				running_mean_sqr[write_counter] = means_sqr_sum;
				for(int c=0;c<II_CYCLES;c++){
#pragma HLS UNROLL
					running_mean_copies[c] = 0.0;
					running_mean_sqr_copies[c] = 0.0;
				}
				write_counter++;
			}

		} // end read block

		if(i>EMB_DIM-1) { // read, normalize and scale the buffer

			datatype elem = input_buffer[read_counter][dim_write];
			datatype wei = weight_buffer[dim_write];
			datatype bias = bias_buffer[dim_write];
			stattype mean = running_mean[read_counter];
			stattype mean_sqr = running_mean_sqr[read_counter];

			stattype var = mean_sqr - (mean*mean);
			stattype std = hls::sqrt(var);

			stattype in_normalized = (elem-mean)/std;
			stattype in_scaled = in_normalized*wei + bias;
			ap_int<AXI_WIDTH> output = *reinterpret_cast<ap_int<AXI_WIDTH>*>(&in_scaled);
			ap_int<AXI_WIDTH> soutput = *reinterpret_cast<ap_int<AXI_WIDTH>*>(&elem);
			if(token_no > TOKENS-1){
				output = 0.0;
			}
			out_stream.write(output);
			skip_out_stream.write(soutput);
			if(++dim_write == EMB_DIM) {
				dim_write = 0;
				read_counter++;
				if(++token_no == TOKENS_PADDED){
					token_no = 0;
				}
			}

		} // end write block
	}

}


void layernorm(ap_uint<AXI_WIDTH> *inp, ap_uint<AXI_WIDTH> *skip_in,
               ap_uint<AXI_WIDTH> *norm_w, ap_uint<AXI_WIDTH> *norm_b,
               ap_uint<AXI_WIDTH> *out, ap_uint<AXI_WIDTH> *skip_out, int batch)
{
#pragma HLS interface m_axi offset=slave bundle=gmem0 port=inp max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=inp

#pragma HLS interface m_axi offset=slave bundle=gmem1 port=skip_in max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=skip_in

#pragma HLS interface m_axi offset=slave bundle=gmem2 port=norm_w max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=norm_w
#pragma HLS interface m_axi offset=slave bundle=gmem3 port=norm_b max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=norm_b

#pragma HLS interface m_axi offset=slave bundle=gmem4 port=out max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=out
#pragma HLS interface m_axi offset=slave bundle=gmem5 port=skip_out max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=skip_out

#pragma HLS interface s_axilite bundle=control port=batch
#pragma HLS interface s_axilite bundle=control port=return

#pragma HLS DATAFLOW

    hls::stream<ap_uint<AXI_WIDTH>> input_stream("input_stream");
#pragma HLS STREAM depth=8 type=fifo variable=input_stream
    hls::stream<ap_uint<AXI_WIDTH>> skip_in_stream("input_stream");
#pragma HLS STREAM depth=8 type=fifo variable=skip_in_stream
    hls::stream<ap_uint<AXI_WIDTH>> weight_stream("weight_stream");
#pragma HLS STREAM depth=8 type=fifo variable=weight_stream
    hls::stream<ap_uint<AXI_WIDTH>> bias_stream("bias_stream");
#pragma HLS STREAM depth=8 type=fifo variable=bias_stream
    hls::stream<ap_uint<AXI_WIDTH>> output_stream("output_stream");
#pragma HLS STREAM depth=8 type=fifo variable=output_stream
    hls::stream<ap_uint<AXI_WIDTH>> skip_out_stream("output_stream");
#pragma HLS STREAM depth=8 type=fifo variable=skip_out_stream

    Mem2Stream(inp, skip_in, norm_w, norm_b, input_stream, skip_in_stream, weight_stream, bias_stream, batch);
    LayerNorm_Core(input_stream, skip_in_stream, weight_stream, bias_stream, output_stream, skip_out_stream, batch);
    Stream2Mem(out, skip_out, output_stream, skip_out_stream, batch);
}

