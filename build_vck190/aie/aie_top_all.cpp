#include "layer0/aie_top_L0.h"
#include "layer1/aie_top_L1.h"
using namespace adf;

PLIO* LHS_in0_L0 = new PLIO("LHS_in0_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in1_L0 = new PLIO("LHS_in1_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in2_L0 = new PLIO("LHS_in2_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in3_L0 = new PLIO("LHS_in3_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in4_L0 = new PLIO("LHS_in4_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in5_L0 = new PLIO("LHS_in5_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in6_L0 = new PLIO("LHS_in6_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in7_L0 = new PLIO("LHS_in7_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in8_L0 = new PLIO("LHS_in8_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in9_L0 = new PLIO("LHS_in9_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in10_L0 = new PLIO("LHS_in10_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in11_L0 = new PLIO("LHS_in11_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in12_L0 = new PLIO("LHS_in12_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in13_L0 = new PLIO("LHS_in13_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in14_L0 = new PLIO("LHS_in14_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in15_L0 = new PLIO("LHS_in15_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in16_L0 = new PLIO("LHS_in16_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in17_L0 = new PLIO("LHS_in17_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in18_L0 = new PLIO("LHS_in18_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in19_L0 = new PLIO("LHS_in19_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in20_L0 = new PLIO("LHS_in20_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in21_L0 = new PLIO("LHS_in21_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in22_L0 = new PLIO("LHS_in22_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in23_L0 = new PLIO("LHS_in23_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in24_L0 = new PLIO("LHS_in24_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in25_L0 = new PLIO("LHS_in25_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in26_L0 = new PLIO("LHS_in26_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in27_L0 = new PLIO("LHS_in27_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in28_L0 = new PLIO("LHS_in28_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in29_L0 = new PLIO("LHS_in29_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in30_L0 = new PLIO("LHS_in30_L0", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in31_L0 = new PLIO("LHS_in31_L0", adf::plio_128_bits, "data/input0.txt",250);

PLIO* LHS_in0_L1 = new PLIO("LHS_in0_L1", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in1_L1 = new PLIO("LHS_in1_L1", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in2_L1 = new PLIO("LHS_in2_L1", adf::plio_128_bits, "data/input0.txt",250);
PLIO* LHS_in3_L1 = new PLIO("LHS_in3_L1", adf::plio_128_bits, "data/input0.txt",250);

PLIO* RHS_in0_L0 = new PLIO("RHS_in0_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in1_L0 = new PLIO("RHS_in1_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in2_L0 = new PLIO("RHS_in2_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in3_L0 = new PLIO("RHS_in3_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in4_L0 = new PLIO("RHS_in4_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in5_L0 = new PLIO("RHS_in5_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in6_L0 = new PLIO("RHS_in6_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in7_L0 = new PLIO("RHS_in7_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in8_L0 = new PLIO("RHS_in8_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in9_L0 = new PLIO("RHS_in9_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in10_L0 = new PLIO("RHS_in10_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in11_L0 = new PLIO("RHS_in11_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in12_L0 = new PLIO("RHS_in12_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in13_L0 = new PLIO("RHS_in13_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in14_L0 = new PLIO("RHS_in14_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in15_L0 = new PLIO("RHS_in15_L0", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in0_L1 = new PLIO("RHS_in0_L1", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in1_L1 = new PLIO("RHS_in1_L1", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in2_L1 = new PLIO("RHS_in2_L1", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in3_L1 = new PLIO("RHS_in3_L1", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in4_L1 = new PLIO("RHS_in4_L1", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in5_L1 = new PLIO("RHS_in5_L1", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in6_L1 = new PLIO("RHS_in6_L1", adf::plio_128_bits, "data/input1.txt",250);
PLIO* RHS_in7_L1 = new PLIO("RHS_in7_L1", adf::plio_128_bits, "data/input1.txt",250);
PLIO* out0_L0 = new PLIO("out0_L0", adf::plio_128_bits, "data/output0.txt",250);
PLIO* out1_L0 = new PLIO("out1_L0", adf::plio_128_bits, "data/output1.txt",250);
PLIO* out2_L0 = new PLIO("out2_L0", adf::plio_128_bits, "data/output2.txt",250);
PLIO* out3_L0 = new PLIO("out3_L0", adf::plio_128_bits, "data/output3.txt",250);
PLIO* out4_L0 = new PLIO("out4_L0", adf::plio_128_bits, "data/output4.txt",250);
PLIO* out5_L0 = new PLIO("out5_L0", adf::plio_128_bits, "data/output5.txt",250);
PLIO* out6_L0 = new PLIO("out6_L0", adf::plio_128_bits, "data/output6.txt",250);
PLIO* out7_L0 = new PLIO("out7_L0", adf::plio_128_bits, "data/output7.txt",250);
PLIO* out8_L0 = new PLIO("out8_L0", adf::plio_128_bits, "data/output8.txt",250);
PLIO* out9_L0 = new PLIO("out9_L0", adf::plio_128_bits, "data/output9.txt",250);
PLIO* out10_L0 = new PLIO("out10_L0", adf::plio_128_bits, "data/output10.txt",250);
PLIO* out11_L0 = new PLIO("out11_L0", adf::plio_128_bits, "data/output11.txt",250);
PLIO* out12_L0 = new PLIO("out12_L0", adf::plio_128_bits, "data/output12.txt",250);
PLIO* out13_L0 = new PLIO("out13_L0", adf::plio_128_bits, "data/output13.txt",250);
PLIO* out14_L0 = new PLIO("out14_L0", adf::plio_128_bits, "data/output14.txt",250);
PLIO* out15_L0 = new PLIO("out15_L0", adf::plio_128_bits, "data/output15.txt",250);
PLIO* out16_L0 = new PLIO("out16_L0", adf::plio_128_bits, "data/output16.txt",250);
PLIO* out17_L0 = new PLIO("out17_L0", adf::plio_128_bits, "data/output17.txt",250);
PLIO* out18_L0 = new PLIO("out18_L0", adf::plio_128_bits, "data/output18.txt",250);
PLIO* out19_L0 = new PLIO("out19_L0", adf::plio_128_bits, "data/output19.txt",250);
PLIO* out20_L0 = new PLIO("out20_L0", adf::plio_128_bits, "data/output20.txt",250);
PLIO* out21_L0 = new PLIO("out21_L0", adf::plio_128_bits, "data/output21.txt",250);
PLIO* out22_L0 = new PLIO("out22_L0", adf::plio_128_bits, "data/output22.txt",250);
PLIO* out23_L0 = new PLIO("out23_L0", adf::plio_128_bits, "data/output23.txt",250);
PLIO* out24_L0 = new PLIO("out24_L0", adf::plio_128_bits, "data/output24.txt",250);
PLIO* out25_L0 = new PLIO("out25_L0", adf::plio_128_bits, "data/output25.txt",250);
PLIO* out26_L0 = new PLIO("out26_L0", adf::plio_128_bits, "data/output26.txt",250);
PLIO* out27_L0 = new PLIO("out27_L0", adf::plio_128_bits, "data/output27.txt",250);
PLIO* out28_L0 = new PLIO("out28_L0", adf::plio_128_bits, "data/output28.txt",250);
PLIO* out29_L0 = new PLIO("out29_L0", adf::plio_128_bits, "data/output29.txt",250);
PLIO* out30_L0 = new PLIO("out30_L0", adf::plio_128_bits, "data/output30.txt",250);
PLIO* out31_L0 = new PLIO("out31_L0", adf::plio_128_bits, "data/output31.txt",250);

PLIO* out0_L1 = new PLIO("out0_L1", adf::plio_128_bits, "data/output32.txt",250);
PLIO* out1_L1 = new PLIO("out1_L1", adf::plio_128_bits, "data/output33.txt",250);
PLIO* out2_L1 = new PLIO("out2_L1", adf::plio_128_bits, "data/output34.txt",250);
PLIO* out3_L1 = new PLIO("out3_L1", adf::plio_128_bits, "data/output35.txt",250);
PLIO* out4_L1 = new PLIO("out4_L1", adf::plio_128_bits, "data/output36.txt",250);
PLIO* out5_L1 = new PLIO("out5_L1", adf::plio_128_bits, "data/output37.txt",250);
PLIO* out6_L1 = new PLIO("out6_L1", adf::plio_128_bits, "data/output38.txt",250);
PLIO* out7_L1 = new PLIO("out7_L1", adf::plio_128_bits, "data/output39.txt",250);

simulation::platform<60, 40> platform(
LHS_in0_L0,
LHS_in1_L0,
LHS_in2_L0,
LHS_in3_L0,
LHS_in4_L0,
LHS_in5_L0,
LHS_in6_L0,
LHS_in7_L0,
LHS_in8_L0,
LHS_in9_L0,
LHS_in10_L0,
LHS_in11_L0,
LHS_in12_L0,
LHS_in13_L0,
LHS_in14_L0,
LHS_in15_L0,
LHS_in16_L0,
LHS_in17_L0,
LHS_in18_L0,
LHS_in19_L0,
LHS_in20_L0,
LHS_in21_L0,
LHS_in22_L0,
LHS_in23_L0,
LHS_in24_L0,
LHS_in25_L0,
LHS_in26_L0,
LHS_in27_L0,
LHS_in28_L0,
LHS_in29_L0,
LHS_in30_L0,
LHS_in31_L0,
LHS_in0_L1,
LHS_in1_L1,
LHS_in2_L1,
LHS_in3_L1,
RHS_in0_L0, 
RHS_in1_L0, 
RHS_in2_L0, 
RHS_in3_L0, 
RHS_in4_L0, 
RHS_in5_L0, 
RHS_in6_L0, 
RHS_in7_L0, 
RHS_in8_L0, 
RHS_in9_L0, 
RHS_in10_L0, 
RHS_in11_L0, 
RHS_in12_L0, 
RHS_in13_L0, 
RHS_in14_L0, 
RHS_in15_L0, 
RHS_in0_L1, 
RHS_in1_L1, 
RHS_in2_L1, 
RHS_in3_L1, 
RHS_in4_L1, 
RHS_in5_L1, 
RHS_in6_L1, 
RHS_in7_L1, 
out0_L0,
out1_L0,
out2_L0,
out3_L0,
out4_L0,
out5_L0,
out6_L0,
out7_L0,
out8_L0,
out9_L0,
out10_L0,
out11_L0,
out12_L0,
out13_L0,
out14_L0,
out15_L0,
out16_L0,
out17_L0,
out18_L0,
out19_L0,
out20_L0,
out21_L0,
out22_L0,
out23_L0,
out24_L0,
out25_L0,
out26_L0,
out27_L0,
out28_L0,
out29_L0,
out30_L0,
out31_L0,
out0_L1,
out1_L1,
out2_L1,
out3_L1,
out4_L1,
out5_L1,
out6_L1,
out7_L1 
);


mm_x8_x4_x4_graph0  mm_graph0;
mm_x2_x2_x4_graph1  mm_graph1;


connect<> net_lhs_in0_L0 (platform.src[0], mm_graph0.in_lhs[0][0]);
connect<> net_lhs_in1_L0 (platform.src[1], mm_graph0.in_lhs[0][1]);
connect<> net_lhs_in2_L0 (platform.src[2], mm_graph0.in_lhs[0][2]);
connect<> net_lhs_in3_L0 (platform.src[3], mm_graph0.in_lhs[0][3]);
connect<> net_lhs_in4_L0 (platform.src[4], mm_graph0.in_lhs[1][0]);
connect<> net_lhs_in5_L0 (platform.src[5], mm_graph0.in_lhs[1][1]);
connect<> net_lhs_in6_L0 (platform.src[6], mm_graph0.in_lhs[1][2]);
connect<> net_lhs_in7_L0 (platform.src[7], mm_graph0.in_lhs[1][3]);
connect<> net_lhs_in8_L0 (platform.src[8], mm_graph0.in_lhs[2][0]);
connect<> net_lhs_in9_L0 (platform.src[9], mm_graph0.in_lhs[2][1]);
connect<> net_lhs_in10_L0 (platform.src[10], mm_graph0.in_lhs[2][2]);
connect<> net_lhs_in11_L0 (platform.src[11], mm_graph0.in_lhs[2][3]);
connect<> net_lhs_in12_L0 (platform.src[12], mm_graph0.in_lhs[3][0]);
connect<> net_lhs_in13_L0 (platform.src[13], mm_graph0.in_lhs[3][1]);
connect<> net_lhs_in14_L0 (platform.src[14], mm_graph0.in_lhs[3][2]);
connect<> net_lhs_in15_L0 (platform.src[15], mm_graph0.in_lhs[3][3]);
connect<> net_lhs_in16_L0 (platform.src[16], mm_graph0.in_lhs[4][0]);
connect<> net_lhs_in17_L0 (platform.src[17], mm_graph0.in_lhs[4][1]);
connect<> net_lhs_in18_L0 (platform.src[18], mm_graph0.in_lhs[4][2]);
connect<> net_lhs_in19_L0 (platform.src[19], mm_graph0.in_lhs[4][3]);
connect<> net_lhs_in20_L0 (platform.src[20], mm_graph0.in_lhs[5][0]);
connect<> net_lhs_in21_L0 (platform.src[21], mm_graph0.in_lhs[5][1]);
connect<> net_lhs_in22_L0 (platform.src[22], mm_graph0.in_lhs[5][2]);
connect<> net_lhs_in23_L0 (platform.src[23], mm_graph0.in_lhs[5][3]);
connect<> net_lhs_in24_L0 (platform.src[24], mm_graph0.in_lhs[6][0]);
connect<> net_lhs_in25_L0 (platform.src[25], mm_graph0.in_lhs[6][1]);
connect<> net_lhs_in26_L0 (platform.src[26], mm_graph0.in_lhs[6][2]);
connect<> net_lhs_in27_L0 (platform.src[27], mm_graph0.in_lhs[6][3]);
connect<> net_lhs_in28_L0 (platform.src[28], mm_graph0.in_lhs[7][0]);
connect<> net_lhs_in29_L0 (platform.src[29], mm_graph0.in_lhs[7][1]);
connect<> net_lhs_in30_L0 (platform.src[30], mm_graph0.in_lhs[7][2]);
connect<> net_lhs_in31_L0 (platform.src[31], mm_graph0.in_lhs[7][3]);

connect<> net_lhs_in0_L1 (platform.src[32], mm_graph1.in_lhs[0][0]);
connect<> net_lhs_in1_L1 (platform.src[33], mm_graph1.in_lhs[0][1]);
connect<> net_lhs_in2_L1 (platform.src[34], mm_graph1.in_lhs[1][0]);
connect<> net_lhs_in3_L1 (platform.src[35], mm_graph1.in_lhs[1][1]);

connect<> net_rhs_in0_L0 (platform.src[36], mm_graph0.in_rhs[0][0]);
connect<> net_rhs_in1_L0 (platform.src[37], mm_graph0.in_rhs[0][1]);
connect<> net_rhs_in2_L0 (platform.src[38], mm_graph0.in_rhs[0][2]);
connect<> net_rhs_in3_L0 (platform.src[39], mm_graph0.in_rhs[0][3]);
connect<> net_rhs_in4_L0 (platform.src[40], mm_graph0.in_rhs[1][0]);
connect<> net_rhs_in5_L0 (platform.src[41], mm_graph0.in_rhs[1][1]);
connect<> net_rhs_in6_L0 (platform.src[42], mm_graph0.in_rhs[1][2]);
connect<> net_rhs_in7_L0 (platform.src[43], mm_graph0.in_rhs[1][3]);
connect<> net_rhs_in8_L0 (platform.src[44], mm_graph0.in_rhs[2][0]);
connect<> net_rhs_in9_L0 (platform.src[45], mm_graph0.in_rhs[2][1]);
connect<> net_rhs_in10_L0 (platform.src[46], mm_graph0.in_rhs[2][2]);
connect<> net_rhs_in11_L0 (platform.src[47], mm_graph0.in_rhs[2][3]);
connect<> net_rhs_in12_L0 (platform.src[48], mm_graph0.in_rhs[3][0]);
connect<> net_rhs_in13_L0 (platform.src[49], mm_graph0.in_rhs[3][1]);
connect<> net_rhs_in14_L0 (platform.src[50], mm_graph0.in_rhs[3][2]);
connect<> net_rhs_in15_L0 (platform.src[51], mm_graph0.in_rhs[3][3]);
connect<> net_rhs_in0_L1 (platform.src[52], mm_graph1.in_rhs[0][0]);
connect<> net_rhs_in1_L1 (platform.src[53], mm_graph1.in_rhs[0][1]);
connect<> net_rhs_in2_L1 (platform.src[54], mm_graph1.in_rhs[1][0]);
connect<> net_rhs_in3_L1 (platform.src[55], mm_graph1.in_rhs[1][1]);
connect<> net_rhs_in4_L1 (platform.src[56], mm_graph1.in_rhs[2][0]);
connect<> net_rhs_in5_L1 (platform.src[57], mm_graph1.in_rhs[2][1]);
connect<> net_rhs_in6_L1 (platform.src[58], mm_graph1.in_rhs[3][0]);
connect<> net_rhs_in7_L1 (platform.src[59], mm_graph1.in_rhs[3][1]);
connect<> net_out0_L0 (mm_graph0.out[0], platform.sink[0]);
connect<> net_out1_L0 (mm_graph0.out[1], platform.sink[1]);
connect<> net_out2_L0 (mm_graph0.out[2], platform.sink[2]);
connect<> net_out3_L0 (mm_graph0.out[3], platform.sink[3]);
connect<> net_out4_L0 (mm_graph0.out[4], platform.sink[4]);
connect<> net_out5_L0 (mm_graph0.out[5], platform.sink[5]);
connect<> net_out6_L0 (mm_graph0.out[6], platform.sink[6]);
connect<> net_out7_L0 (mm_graph0.out[7], platform.sink[7]);
connect<> net_out8_L0 (mm_graph0.out[8], platform.sink[8]);
connect<> net_out9_L0 (mm_graph0.out[9], platform.sink[9]);
connect<> net_out10_L0 (mm_graph0.out[10], platform.sink[10]);
connect<> net_out11_L0 (mm_graph0.out[11], platform.sink[11]);
connect<> net_out12_L0 (mm_graph0.out[12], platform.sink[12]);
connect<> net_out13_L0 (mm_graph0.out[13], platform.sink[13]);
connect<> net_out14_L0 (mm_graph0.out[14], platform.sink[14]);
connect<> net_out15_L0 (mm_graph0.out[15], platform.sink[15]);
connect<> net_out16_L0 (mm_graph0.out[16], platform.sink[16]);
connect<> net_out17_L0 (mm_graph0.out[17], platform.sink[17]);
connect<> net_out18_L0 (mm_graph0.out[18], platform.sink[18]);
connect<> net_out19_L0 (mm_graph0.out[19], platform.sink[19]);
connect<> net_out20_L0 (mm_graph0.out[20], platform.sink[20]);
connect<> net_out21_L0 (mm_graph0.out[21], platform.sink[21]);
connect<> net_out22_L0 (mm_graph0.out[22], platform.sink[22]);
connect<> net_out23_L0 (mm_graph0.out[23], platform.sink[23]);
connect<> net_out24_L0 (mm_graph0.out[24], platform.sink[24]);
connect<> net_out25_L0 (mm_graph0.out[25], platform.sink[25]);
connect<> net_out26_L0 (mm_graph0.out[26], platform.sink[26]);
connect<> net_out27_L0 (mm_graph0.out[27], platform.sink[27]);
connect<> net_out28_L0 (mm_graph0.out[28], platform.sink[28]);
connect<> net_out29_L0 (mm_graph0.out[29], platform.sink[29]);
connect<> net_out30_L0 (mm_graph0.out[30], platform.sink[30]);
connect<> net_out31_L0 (mm_graph0.out[31], platform.sink[31]);
connect<> net_out0_L1 (mm_graph1.out[0], platform.sink[32]);
connect<> net_out1_L1 (mm_graph1.out[1], platform.sink[33]);
connect<> net_out2_L1 (mm_graph1.out[2], platform.sink[34]);
connect<> net_out3_L1 (mm_graph1.out[3], platform.sink[35]);
connect<> net_out4_L1 (mm_graph1.out[4], platform.sink[36]);
connect<> net_out5_L1 (mm_graph1.out[5], platform.sink[37]);
connect<> net_out6_L1 (mm_graph1.out[6], platform.sink[38]);
connect<> net_out7_L1 (mm_graph1.out[7], platform.sink[39]);
#ifdef __AIESIM__
int main(int argc, char** argv) {
    mm_graph0.init();
    mm_graph1.init();
    mm_graph0.run(4);
    mm_graph1.run(4);
    mm_graph0.end();
    mm_graph1.end();
    return 0;
}
#endif