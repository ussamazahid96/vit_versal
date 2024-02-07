#include "gelu.hpp"


void Mem2Stream(ap_uint<AXI_WIDTH> *in,
				hls::stream<ap_uint<AXI_WIDTH>> &out,
				const int batch){
#pragma HLS INLINE off
	const int bound = batch*TOKENS_PADDED*EMB_DIM*DATA_TYPE/AXI_WIDTH;
	for(int i=0;i<bound;i++){
#pragma HLS pipeline style=flp II=1
		ap_uint<AXI_WIDTH> el = in[i];
		out.write(el);
	}
}

void Stream2Mem(ap_uint<AXI_WIDTH> *out,
				hls::stream<ap_uint<AXI_WIDTH>> &in,
				const int batch){
#pragma HLS INLINE off
	const int bound = batch*TOKENS_PADDED*EMB_DIM*DATA_TYPE/AXI_WIDTH;
	for(int i=0;i<bound;i++){
#pragma HLS pipeline style=flp II=1
		ap_uint<AXI_WIDTH> el = in.read();
		out[i] = el;
	}
}


void gelu_core(hls::stream<ap_uint<AXI_WIDTH>> &input, hls::stream<ap_uint<AXI_WIDTH>> &out, const int batch) {
#pragma HLS INLINE off
	const int bound = batch*TOKENS_PADDED*EMB_DIM*DATA_TYPE/AXI_WIDTH;
	const float step = 85.3333;

	for(int i = 0; i < bound; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<AXI_WIDTH> out_pkt;
		ap_uint<AXI_WIDTH> elem = input.read();
        for(int j=0;j<AXI_WIDTH/DATA_TYPE;j++) {
#pragma HLS UNROLL
			ap_int<DATA_TYPE> temp = elem((j+1)*DATA_TYPE-1, j*DATA_TYPE);
			datatype elem_f = *reinterpret_cast<datatype*>(&temp);
			datatype relu = (elem_f < 0) ? (float) 0 : elem_f;
			if(elem_f > -3 and elem_f < 3) {
				auto index = hls::abs(elem_f);
				int index_I = index*step-1;
				relu -= delta[index_I];
			}
			ap_uint<DATA_TYPE> relu_I = *reinterpret_cast<ap_uint<DATA_TYPE>*>(&relu);
			out_pkt((j+1)*DATA_TYPE-1 ,j*DATA_TYPE) = relu_I;
        }
        out.write(out_pkt);

	}
}


void gelu(ap_uint<AXI_WIDTH> *in, ap_uint<AXI_WIDTH> *out, int batch) {
#pragma HLS interface m_axi offset=slave bundle=gmem0 port=in max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=in
#pragma HLS interface m_axi offset=slave bundle=gmem1 port=out max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=out

#pragma HLS interface s_axilite bundle=control port=batch
#pragma HLS interface s_axilite bundle=control port=return

#pragma HLS DATAFLOW

    hls::stream<ap_uint<AXI_WIDTH>> input_stream("input_stream");
#pragma HLS STREAM depth=8 type=fifo variable=input_stream
    hls::stream<ap_uint<AXI_WIDTH>> output_stream("output_stream");
#pragma HLS STREAM depth=8 type=fifo variable=output_stream

    Mem2Stream(in, input_stream, batch);
    gelu_core(input_stream, output_stream, batch);
    Stream2Mem(out, output_stream, batch);

}
