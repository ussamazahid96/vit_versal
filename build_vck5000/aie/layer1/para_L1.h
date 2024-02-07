#ifndef PARA_L1_H
#define PARA_L1_H
#include <adf/stream/types.h>
const int L1_NUM_KERNEL=2;
const int L1_h1=32;
const int L1_w1=32;
const int L1_w2=32;
const int L1_boundary_i=L1_h1/8;
const int L1_boundary_j=L1_w2/2;
const int L1_boundary_k=L1_w1/8-1;
const int L1_jump_B0=L1_w1-4;
const int L1_jump_A0=L1_h1+8;
const int L1_judge_i=L1_boundary_i-1;
const int L1_judge_j=L1_boundary_j-1;

//matA: [W1][H1]  matB [W2][W1] matC [W2/2][H1/8][2][8]

void mm0_kernel0_L1(input_window_float* __restrict matA, input_window_float*  __restrict matB, output_stream_accfloat* __restrict matC);
void mm0_kernel1_L1(input_window_float* __restrict matA, input_window_float*  __restrict matB, input_stream_accfloat* __restrict acc_in, output_stream_accfloat* __restrict matC);
void mm0_kernel2_L1(input_window_float* __restrict matA, input_window_float*  __restrict matB, input_stream_accfloat* __restrict acc_in, output_window_float * __restrict matC);
void mm0_kernel3_L1(input_window_float* __restrict matA, input_window_float*  __restrict matB, output_window_float* __restrict matC);

#endif