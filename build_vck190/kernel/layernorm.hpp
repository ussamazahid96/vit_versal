#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "hls_math.h"


const int AXI_WIDTH=32;
const int TOKENS = 197;
const int TOKENS_PADDED=256;
const int EMB_DIM=768;
const int DATA_TYPE=32;
const int C_PER_TRA=AXI_WIDTH/DATA_TYPE;
typedef float datatype;
typedef float stattype;

void layernorm(ap_uint<AXI_WIDTH> *inp, ap_uint<AXI_WIDTH> *skip_in,
               ap_uint<AXI_WIDTH> *norm_w, ap_uint<AXI_WIDTH> *norm_b,
               ap_uint<AXI_WIDTH> *out, ap_uint<AXI_WIDTH> *skip_out, int batch);

