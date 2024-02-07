#include <ap_int.h>
#include <hls_stream.h>

#define HEAD_DIM 64
#define TOKENS 256
#define AXI_WIDTH 512
#define DATA_TYPE 32
const int ELEM_PER_TRA = AXI_WIDTH/DATA_TYPE;

void rant(ap_uint<AXI_WIDTH> *in1, ap_uint<AXI_WIDTH> *in2, ap_uint<AXI_WIDTH> *in3, ap_uint<AXI_WIDTH> *qout,
						 ap_uint<AXI_WIDTH> *kout, ap_uint<AXI_WIDTH> *vout,
						 int head_idx);
