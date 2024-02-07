#include "aie_graph_L1.h"

using namespace adf;

const int L1_A=2;
const int L1_B=2;
const int L1_C=4;
const int L1_A_BRO=4;
const int L1_C_BRO=2;

class mm_x2_x2_x4_graph1 : public adf::graph {
public:
    input_port in_lhs[(L1_A*L1_C/L1_A_BRO)][L1_B];
    input_port in_rhs[(L1_A*L1_C/L1_C_BRO)][L1_B];
	output_port out[L1_A*L1_C];

    simpleGraph_k1_B2_L1 <17, 0>  vit_0_0;
    simpleGraph_k1_B2_L1 <18, 1>  vit_0_1;
    simpleGraph_k1_B2_L1 <17, 2>  vit_0_2;
    simpleGraph_k1_B2_L1 <18, 3>  vit_0_3;
    simpleGraph_k1_B2_L1 <17, 4>  vit_1_0;
    simpleGraph_k1_B2_L1 <18, 5>  vit_1_1;
    simpleGraph_k1_B2_L1 <17, 6>  vit_1_2;
    simpleGraph_k1_B2_L1 <18, 7>  vit_1_3;
    

    mm_x2_x2_x4_graph1 () {

        connect<stream,window<L1_h1*L1_w1>>(in_lhs[0][0],vit_0_0.in0[0]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[0][1],vit_0_0.in0[1]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[0][0],vit_0_1.in0[0]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[0][1],vit_0_1.in0[1]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[0][0],vit_0_2.in0[0]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[0][1],vit_0_2.in0[1]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[0][0],vit_0_3.in0[0]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[0][1],vit_0_3.in0[1]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[1][0],vit_1_0.in0[0]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[1][1],vit_1_0.in0[1]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[1][0],vit_1_1.in0[0]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[1][1],vit_1_1.in0[1]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[1][0],vit_1_2.in0[0]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[1][1],vit_1_2.in0[1]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[1][0],vit_1_3.in0[0]);
        connect<stream,window<L1_h1*L1_w1>>(in_lhs[1][1],vit_1_3.in0[1]);
        

        connect<stream,window<L1_w1*L1_w2>>(in_rhs[0][0],vit_0_0.in1[0]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[0][1],vit_0_0.in1[1]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[0][0],vit_1_0.in1[0]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[0][1],vit_1_0.in1[1]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[1][0],vit_0_1.in1[0]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[1][1],vit_0_1.in1[1]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[1][0],vit_1_1.in1[0]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[1][1],vit_1_1.in1[1]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[2][0],vit_0_2.in1[0]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[2][1],vit_0_2.in1[1]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[2][0],vit_1_2.in1[0]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[2][1],vit_1_2.in1[1]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[3][0],vit_0_3.in1[0]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[3][1],vit_0_3.in1[1]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[3][0],vit_1_3.in1[0]);
        connect<stream,window<L1_w1*L1_w2>>(in_rhs[3][1],vit_1_3.in1[1]);
        


        connect<stream,stream>(vit_0_0.out,out[0]);
        connect<stream,stream>(vit_0_1.out,out[1]);
        connect<stream,stream>(vit_0_2.out,out[2]);
        connect<stream,stream>(vit_0_3.out,out[3]);
        connect<stream,stream>(vit_1_0.out,out[4]);
        connect<stream,stream>(vit_1_1.out,out[5]);
        connect<stream,stream>(vit_1_2.out,out[6]);
        connect<stream,stream>(vit_1_3.out,out[7]);
        
    }
};