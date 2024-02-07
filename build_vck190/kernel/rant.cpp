#include "rant.hpp"

void address_in_generator(hls::stream<ap_uint<32>>& addr_out, const int TX, const int TY, const int head_idx) {
#pragma HLS inline off
    for(int tx=0;tx<HEAD_DIM/ELEM_PER_TRA;tx++){
		for(int ty=0;ty<TY;ty++){
			for(int j=0;j<ELEM_PER_TRA;j++){
#pragma HLS PIPELINE II = 1
				ap_uint<32> pos = 768/ELEM_PER_TRA + j*TX + ty*ELEM_PER_TRA*TX + tx + head_idx*HEAD_DIM/ELEM_PER_TRA;
				addr_out.write(pos);
			}
		}
    }
}

void address_out_generator(hls::stream<ap_uint<32>>& addr_out, const int TY, const int TZ) {
#pragma HLS inline off
    for(int tx=0;tx<1;tx++){
        for(int tz=0;tz<TZ;tz++){
            for(int ty=0;ty<TY;ty++){
                ap_uint<32> pos;
                for(int j=0;j<ELEM_PER_TRA;j++){
                    for(int i=0;i<1;i++){
                    #pragma HLS PIPELINE II = 1
                        pos=i+j*TY+ty+tz*ELEM_PER_TRA*TY;
                        addr_out.write(pos);
                    }
                }
            }
        }
    }
}

void transpose_core(hls::stream<ap_uint<AXI_WIDTH>>& data_in,
					ap_uint<AXI_WIDTH> *out, hls::stream<ap_uint<32>>& addr_out,
					const int TX, const int TY) {
#pragma HLS inline off
	
	const int bound = ELEM_PER_TRA*TY*TX;
	ap_uint<DATA_TYPE> input_buffer[2][ELEM_PER_TRA][ELEM_PER_TRA];
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=input_buffer
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=input_buffer
	ap_uint<8> write_row = 0, read_col = 0;
	ap_uint<1> write_buffer_idx = 0, read_buffer_idx = 0;

	for(int i=0; i<bound+ELEM_PER_TRA; i++) {
#pragma HLS PIPELINE II=1
		if(i<bound) { //initial buffer fill
			ap_uint<AXI_WIDTH> in_pkt = data_in.read();
			for(int j=0;j<ELEM_PER_TRA;j++){
#pragma HLS UNROLL
				ap_uint<DATA_TYPE> elem = in_pkt( (j+1)*DATA_TYPE-1, j*DATA_TYPE );
				input_buffer[write_buffer_idx][write_row][j] = elem;
			}
			if(++write_row == ELEM_PER_TRA) {
				write_row = 0;
				write_buffer_idx++;
			}
		}
		if(i>ELEM_PER_TRA-1) { // start writing the output
			ap_uint<AXI_WIDTH> out_pkt;
			for(int j=0;j<ELEM_PER_TRA;j++){
#pragma HLS UNROLL
				ap_uint<DATA_TYPE> elem = input_buffer[read_buffer_idx][j][read_col];
				out_pkt( (j+1)*DATA_TYPE-1, j*DATA_TYPE ) = elem;
			}
			ap_uint<32> addrout = addr_out.read();
			out[addrout] = out_pkt;
			if(++read_col == ELEM_PER_TRA){
				read_col = 0;
				read_buffer_idx++;
			}
		}
	}
}


void reshape_core(ap_uint<AXI_WIDTH> *in1, ap_uint<AXI_WIDTH> *in2, ap_uint<AXI_WIDTH> *in3, ap_uint<AXI_WIDTH> *qout, ap_uint<AXI_WIDTH> *vout,
				  hls::stream<ap_uint<32>>&addrin, hls::stream<ap_uint<AXI_WIDTH>>&kdata, const int head_idx){
#pragma HLS inline off
	int write_idx = 0;
	const int head_offset = HEAD_DIM*head_idx/ELEM_PER_TRA;
	for(int t=0; t<TOKENS;t++){
		for(int i=0;i<HEAD_DIM/ELEM_PER_TRA;i++){
	#pragma HLS PIPELINE II=1
			ap_uint<32> qpos = i+t*(2304/ELEM_PER_TRA)+head_offset;
			ap_uint<32> kpos = addrin.read();
			ap_uint<32> vpos = qpos + 2*768/ELEM_PER_TRA;
			ap_uint<AXI_WIDTH> qelem = in1[qpos];
			ap_uint<AXI_WIDTH> kelem = in2[kpos];
			kdata.write(kelem);
			ap_uint<AXI_WIDTH> velem = in3[vpos];
			qout[write_idx] = qelem;
			vout[write_idx] = velem;
			write_idx++;
		}
	}
}

void rant(ap_uint<AXI_WIDTH> *in1, ap_uint<AXI_WIDTH> *in2, ap_uint<AXI_WIDTH> *in3, ap_uint<AXI_WIDTH> *qout,
						 ap_uint<AXI_WIDTH> *kout, ap_uint<AXI_WIDTH> *vout,
						 int head_idx) {
#pragma HLS interface m_axi offset=slave bundle=gmem0 port=in1
#pragma HLS interface s_axilite bundle=control port=in1
#pragma HLS interface m_axi offset=slave bundle=gmem1 port=in2
#pragma HLS interface s_axilite bundle=control port=in2
#pragma HLS interface m_axi offset=slave bundle=gmem2 port=in3
#pragma HLS interface s_axilite bundle=control port=in3
#pragma HLS interface m_axi offset=slave bundle=gmem3 port=qout
#pragma HLS interface s_axilite bundle=control port=qout
#pragma HLS interface m_axi offset=slave bundle=gmem4 port=kout
#pragma HLS interface s_axilite bundle=control port=kout
#pragma HLS interface m_axi offset=slave bundle=gmem5 port=vout
#pragma HLS interface s_axilite bundle=control port=vout

#pragma HLS interface s_axilite bundle=control port=head_idx
#pragma HLS interface s_axilite bundle=control port=return

#pragma HLS DATAFLOW


	hls::stream<ap_uint<32>> addrin("addr_in");
	hls::stream<ap_uint<32>> addrout("addr_out");
	hls::stream<ap_uint<AXI_WIDTH>> kdata("kdata");

	address_in_generator(addrin, 2304/ELEM_PER_TRA, TOKENS/ELEM_PER_TRA, head_idx);
	address_out_generator(addrout, TOKENS/ELEM_PER_TRA, HEAD_DIM/ELEM_PER_TRA);
	reshape_core(in1, in2, in3, qout, vout, addrin, kdata, head_idx);
	transpose_core(kdata, kout, addrout, HEAD_DIM/ELEM_PER_TRA, TOKENS/ELEM_PER_TRA);

}

























































