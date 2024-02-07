#include <stdint.h>
#include "dma_small.hpp"

void address_A_ddr(axis_stream_32& addrA_out,const int TX,const int TY,const int TZ) {
#pragma HLS inline off
    for(int tx=0;tx<TX;tx++){
        for(int tz=0;tz<TZ;tz++){
            for(int ty=0;ty<TY;ty++){
                ap_uint<32> pos;
                for(int j=0;j<K;j++){
                    for(int i=0;i<M/A_PER_TRA;i++){
                    #pragma HLS PIPELINE II = 1
                        pos=i+j*(M/A_PER_TRA)*TX+ty*K*(M/A_PER_TRA)*TX+tx*(M/A_PER_TRA);
                        addrA_out.write(pos);
                    }
                }
            }
        }
    }
}

void loadA_ddr(ap_uint<AXI_WIDTH_A>* ina, axis_stream_32& addrA_in,axis_stream_A& dataA_out,const int TX,const int TY,const int TZ) {
#pragma HLS inline off
    ap_uint<AXI_WIDTH_A> temp_data;
    int bound=boundA*TX*TY*TZ;
    for(int i=0;i<bound;i++){
    #pragma HLS PIPELINE II = 1
        int addr = addrA_in.read();
        temp_data=ina[addr];
        dataA_out.write(temp_data);        
    }
}

void address_B_ddr(axis_stream_32& addrB_out,const int TX,const int TY,const int TZ) {
#pragma HLS inline off    
    for(int tx=0;tx<TX;tx++){
        for(int tz=0;tz<TZ;tz++){
            for(int ty=0;ty<TY;ty++){
                ap_uint<32> pos;
                for(int j=0;j<N;j++){
                    for(int i=0;i<K/B_PER_TRA;i++){
                    #pragma HLS PIPELINE II = 1
                        pos=i+j*(K/B_PER_TRA)*TY+ty*(K/B_PER_TRA)+tz*N*(K/B_PER_TRA)*TY;
                        addrB_out.write(pos);
                    }
                }
            }
        }
    }
}


void loadB_ddr(ap_uint<AXI_WIDTH_B>* inb, axis_stream_32& addrB_in,axis_stream_B& dataB_out,const int TX,const int TY,const int TZ) {
#pragma HLS inline off
    ap_uint<AXI_WIDTH_B> temp_data;
    int bound=boundB*TX*TY*TZ;
    for(int i=0;i<bound;i++){
    #pragma HLS PIPELINE II = 1
        ap_uint<32> addr = addrB_in.read();
        temp_data=inb[addr];
        dataB_out.write(temp_data);
    }  
}

void address_C_ddr(axis_stream_32& addrC_out,const int TX,const int TZ, const int head_idx) {
#pragma HLS inline off
    for(int tx=0;tx<TX;tx++){
        for(int tz=0;tz<TZ;tz++){
            ap_uint<32> pos;
            for(int j=0;j<N;j++){
                for(int i=0;i<M/C_PER_TRA;i++){
                #pragma HLS PIPELINE II = 1
                    if(head_idx == -1){
                        pos=i+j*(M/C_PER_TRA)*TX+tx*(M/C_PER_TRA)+tz*N*(M/C_PER_TRA)*TX;
                    } 
                    else {
                        pos=i+j*(M/C_PER_TRA)*TX*HEADS+tx*(M/C_PER_TRA)+tz*N*(M/C_PER_TRA)*TX*HEADS+head_idx*(M/C_PER_TRA)*TX;
                    }
                    addrC_out.write(pos);
                }
            }
        }
    }
}

void storeC_ddr(ap_uint<AXI_WIDTH_C>* outc, axis_stream_32& addrC_in,axis_stream_C& dataC_in,const int TX,const int TZ) { 
#pragma HLS inline off
    int bound=boundC*TX*TZ;
    for(int i=0;i<bound;i++){
    #pragma HLS PIPELINE II = 1
        ap_uint<AXI_WIDTH_C> temp_data=dataC_in.read();
        ap_uint<32> addr = addrC_in.read();
        outc[addr]=temp_data;
            
    }
    
}

void loadA(axis_stream_A& dataA_in, ap_uint<PLIO_WIDTH> a_buf[A*B][X][Y][W1][(H1/NUM_PER_TRA)],bool enable){
#pragma HLS inline off
    if(enable){
        for(int y=0;y<Y;y++){
            for(int b=0;b<B;b++){
                for(int k=0;k<W1;k++){
                    for(int x=0;x<X;x++){
                        for(int a=0;a<A;a++){
                            for(int i=0;i<(H1/A_PER_TRA);i++){
                            #pragma HLS PIPELINE II = 1
                            #pragma HLS dependence variable=a_buf type=intra false
                                int pos0=i*4;
                                int pos1=b+a*B;
                                ap_uint<AXI_WIDTH_A> temp_data=dataA_in.read();
                                a_buf[pos1][x][y][k][pos0]=temp_data(127,0);
                                a_buf[pos1][x][y][k][pos0+1]=temp_data(255,128);
                                a_buf[pos1][x][y][k][pos0+2]=temp_data(383,256);
                                a_buf[pos1][x][y][k][pos0+3]=temp_data(511,384);
                            }
                        }
                    }
                }
            }
        }
    }
}

void loadB(axis_stream_B& dataB_in, ap_uint<PLIO_WIDTH> b_buf[B*C][Z][Y][W2][(W1/NUM_PER_TRA)],bool enable){
#pragma HLS inline off
    if(enable){
        for(int z=0;z<Z;z++){
            for(int c=0;c<C;c++){
                for(int w2=0;w2<W2;w2++){
                    for(int y=0;y<Y;y++){
                        for(int b=0;b<B;b++){
                            for (int m=0;m<(W1/B_PER_TRA);m++){
                            #pragma HLS PIPELINE II = 1
                            #pragma HLS dependence variable=b_buf type=intra false
                                int pos0=m*4;
                                int pos1=b+c*B;
                                ap_uint<AXI_WIDTH_B> temp_data=dataB_in.read();
                                b_buf[pos1][z][y][w2][pos0]=temp_data(127,0);
                                b_buf[pos1][z][y][w2][pos0+1]=temp_data(255,128);
                                b_buf[pos1][z][y][w2][pos0+2]=temp_data(383,256);
                                b_buf[pos1][z][y][w2][pos0+3]=temp_data(511,384);
                            }
                        }
                    }
                }
            }
        }
    }
}

void storeC(axis_stream_C& dataC_out, ap_uint<PLIO_WIDTH> c_buf[C*A][Z][X][W2][(H1/NUM_PER_TRA)], bool enable){
#pragma HLS inline off
    if(enable){
        for(int z=0;z<Z;z++){
            for(int c=0;c<C;c++){
                for(int w2=0;w2<W2;w2++){
                    for(int x=0;x<X;x++){
                        for (int a=0;a<A;a++){
                            for (int n=0; n<H1/C_PER_TRA;n++){
                            #pragma HLS PIPELINE II = 1
                            #pragma HLS dependence variable=c_buf type=intra false
                                int pos0=n*4;
                                int pos1=c+a*C;
                                ap_uint<AXI_WIDTH_C> temp_data;
                                temp_data(127,0)  =c_buf[pos1][z][x][w2][pos0];
                                temp_data(255,128)=c_buf[pos1][z][x][w2][pos0+1];
                                temp_data(383,256)=c_buf[pos1][z][x][w2][pos0+2];
                                temp_data(511,384)=c_buf[pos1][z][x][w2][pos0+3];
                                dataC_out.write(temp_data);
                            }
                        }
                    }
                }
            }
        }
        for(int z=0;z<Z;z++){
            for(int c=0;c<C;c++){
                for(int w2=0;w2<W2;w2++){
                    for(int x=0;x<X;x++){
                        for (int a=0;a<A;a++){
                            for (int n=0; n<H1/C_PER_TRA;n++){
                            #pragma HLS PIPELINE II = 1
                            #pragma HLS dependence variable=c_buf type=intra false
                                int pos0=n*4;
                                int pos1=c+a*C;
                                c_buf[pos1][z][x][w2][pos0]=0;
                                c_buf[pos1][z][x][w2][pos0+1]=0;
                                c_buf[pos1][z][x][w2][pos0+2]=0;
                                c_buf[pos1][z][x][w2][pos0+3]=0;
                            }
                        }
                    }
                }
            }
        }

    }
}

template<int NC>
void sendA(ap_uint<PLIO_WIDTH> a_buf[X][Y][W1][(H1/NUM_PER_TRA)],
           axis_stream& txA0, 
           bool enable){

#pragma HLS inline off
    if(enable){
        axis_pkt tmp;
        data_t data;
        for (int z = 0; z < Z; z++) {
            for (int x = 0; x < X; x++) {
                for (int y = 0; y < Y; y++){
                    for (int j = 0; j < W1; j++){ 
                        for (int i = 0; i < (H1/NUM_PER_TRA); i++){ 
                        #pragma HLS PIPELINE II = 1
                            data = a_buf[x][y][j][i];
                            tmp.data   = data;
                            tmp.keep   = -1;
                            txA0.write(tmp);
                            
                        }
                    }
                }
            }
        }
    }
}

template<int NC>
void sendB(ap_uint<PLIO_WIDTH> b_buf[Z][Y][W2][(W1/NUM_PER_TRA)],
           axis_stream& txB0, 
           bool enable){

#pragma HLS inline off
    if(enable){
        axis_pkt tmp;
        data_t data;
        for (int z = 0; z < Z; z++) {
            for (int x = 0; x < X; x++) {
                for (int y = 0; y < Y; y++){
                    for (int j = 0; j < W2; j++){ 
                        for (int i = 0; i < (W1/NUM_PER_TRA); i++){ 
                        #pragma HLS PIPELINE II = 1
                            data = b_buf[z][y][j][i];
                            tmp.data   = data;
                            tmp.keep   = -1;
                            txB0.write(tmp);
                            
                        }
                    }
                }
            }
        }
    }
}

template<int NC>
void receiveC(ap_uint<PLIO_WIDTH> c_buf[Z][X][W2][(H1/NUM_PER_TRA)],axis_stream& rxC, bool enable){ 

#pragma HLS inline off
    if(enable){
        axis_pkt tmp;
        data_t data;
        fp_int data_temp0[(PLIO_WIDTH/DATA_TYPE)];
        #pragma HLS ARRAY_PARTITION variable=data_temp0 complete dim=1

        fp_int data_temp1[(PLIO_WIDTH/DATA_TYPE)];
        #pragma HLS ARRAY_PARTITION variable=data_temp1 complete dim=1

        fp_int data_temp2[(PLIO_WIDTH/DATA_TYPE)];
        #pragma HLS ARRAY_PARTITION variable=data_temp2 complete dim=1
        for (int z = 0; z < Z; z++) {
            for (int x = 0; x < X; x++) {
                for (int y = 0; y < Y; y++){
                    for (int i = 0; i < (H1/NUM_PER_TRA/2); i++){ 
                        for (int j = 0; j < W2; j++){ 
                            for (int h2=0; h2<2; h2++){    
                            #pragma HLS PIPELINE II = 1
                            #pragma HLS dependence variable=c_buf type=inter false
                                int pos=h2+i*2;
                                tmp=rxC.read();
                                data_temp0[0].data_uint=tmp.data(31,0);
                                data_temp0[1].data_uint=tmp.data(63,32);
                                data_temp0[2].data_uint=tmp.data(95,64);
                                data_temp0[3].data_uint=tmp.data(127,96);

                                data_temp1[0].data_uint=c_buf[z][x][j][pos](31,0);
                                data_temp1[1].data_uint=c_buf[z][x][j][pos](63,32);
                                data_temp1[2].data_uint=c_buf[z][x][j][pos](95,64);
                                data_temp1[3].data_uint=c_buf[z][x][j][pos](127,96);

                                data_temp2[0].data_float=data_temp0[0].data_float + data_temp1[0].data_float; 
                                data_temp2[1].data_float=data_temp0[1].data_float + data_temp1[1].data_float;
                                data_temp2[2].data_float=data_temp0[2].data_float + data_temp1[2].data_float;
                                data_temp2[3].data_float=data_temp0[3].data_float + data_temp1[3].data_float;

                                c_buf[z][x][j][pos](31,0)   =  data_temp2[0].data_uint;  
                                c_buf[z][x][j][pos](63,32)  =  data_temp2[1].data_uint;
                                c_buf[z][x][j][pos](95,64)  =  data_temp2[2].data_uint;
                                c_buf[z][x][j][pos](127,96) =  data_temp2[3].data_uint;

                            }
                        }
                    }
                }
            }
        }
    }
}
void compute(axis_stream_A& dataA_out, axis_stream_B& dataB_out, axis_stream_C& dataC_in,
             axis_stream& txA0, axis_stream& txA1, 
             axis_stream& txA2, axis_stream& txA3, 
             axis_stream& txB0, axis_stream& txB1, 
             axis_stream& txB2, axis_stream& txB3, 
             axis_stream& txB4, axis_stream& txB5, 
             axis_stream& txB6, axis_stream& txB7, 
             axis_stream& rxC0, axis_stream& rxC1, axis_stream& rxC2, axis_stream& rxC3, 
             axis_stream& rxC4, axis_stream& rxC5, axis_stream& rxC6, axis_stream& rxC7, 
             const int TX, const int TY, const int TZ){

    ap_uint<PLIO_WIDTH> buff0_A[A*B][X][Y][W1][(H1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff0_A type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=buff0_A complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff0_A cyclic factor=BUFFA_FACTOR dim=5
    
    ap_uint<PLIO_WIDTH> buff1_A[A*B][X][Y][W1][(H1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff1_A type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=buff1_A complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff1_A cyclic factor=BUFFA_FACTOR dim=5

    ap_uint<PLIO_WIDTH> buff0_B[B*C][Z][Y][W2][(W1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff0_B type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=buff0_B complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff0_B cyclic factor=BUFFB_FACTOR dim=5

    ap_uint<PLIO_WIDTH> buff1_B[B*C][Z][Y][W2][(W1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff1_B type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=buff1_B complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff1_B cyclic factor=BUFFB_FACTOR dim=5

    ap_uint<PLIO_WIDTH> buff0_C[C*A][Z][X][W2][(H1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff0_C type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=buff0_C complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff0_C cyclic factor=BUFFC_FACTOR dim=5

    ap_uint<PLIO_WIDTH> buff1_C[C*A][Z][X][W2][(H1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff1_C type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=buff1_C complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff1_C cyclic factor=BUFFC_FACTOR dim=5

    const int Total_rd=TX*TY*TZ;

    for (int rd=0; rd<Total_rd+2;rd++){
        int c_flg=0,s_flg=0;
        if(rd>0){
            c_flg=((rd-1)/TY)%2;
        }
        if(rd>1){
            s_flg=(rd-2)%TY;
        }
        if(rd%2==0&&c_flg==0){
            loadA(dataA_out,buff0_A,rd<Total_rd);
            loadB(dataB_out,buff0_B,rd<Total_rd);   

            sendA<0>(buff1_A[0],txA0,rd>0&&rd<Total_rd+1);
            sendA<1>(buff1_A[1],txA1,rd>0&&rd<Total_rd+1);
            sendA<2>(buff1_A[2],txA2,rd>0&&rd<Total_rd+1);
            sendA<3>(buff1_A[3],txA3,rd>0&&rd<Total_rd+1);
            

            sendB<0>(buff1_B[0],txB0,rd>0&&rd<Total_rd+1);
            sendB<1>(buff1_B[1],txB1,rd>0&&rd<Total_rd+1);
            sendB<2>(buff1_B[2],txB2,rd>0&&rd<Total_rd+1);
            sendB<3>(buff1_B[3],txB3,rd>0&&rd<Total_rd+1);
            sendB<4>(buff1_B[4],txB4,rd>0&&rd<Total_rd+1);
            sendB<5>(buff1_B[5],txB5,rd>0&&rd<Total_rd+1);
            sendB<6>(buff1_B[6],txB6,rd>0&&rd<Total_rd+1);
            sendB<7>(buff1_B[7],txB7,rd>0&&rd<Total_rd+1);
            

            receiveC<0>(buff0_C[0],rxC0, rd>0&&rd<Total_rd+1);
            receiveC<1>(buff0_C[1],rxC1, rd>0&&rd<Total_rd+1);
            receiveC<2>(buff0_C[2],rxC2, rd>0&&rd<Total_rd+1);
            receiveC<3>(buff0_C[3],rxC3, rd>0&&rd<Total_rd+1);
            receiveC<4>(buff0_C[4],rxC4, rd>0&&rd<Total_rd+1);
            receiveC<5>(buff0_C[5],rxC5, rd>0&&rd<Total_rd+1);
            receiveC<6>(buff0_C[6],rxC6, rd>0&&rd<Total_rd+1);
            receiveC<7>(buff0_C[7],rxC7, rd>0&&rd<Total_rd+1);
            

            storeC(dataC_in, buff1_C, rd>TY&&s_flg==(TY-1));
        }
        else if(rd%2==1&&c_flg==0){
            loadA(dataA_out,buff1_A,rd<Total_rd);
            loadB(dataB_out,buff1_B,rd<Total_rd);   

            sendA<0>(buff0_A[0],txA0,rd>0&&rd<Total_rd+1);
            sendA<1>(buff0_A[1],txA1,rd>0&&rd<Total_rd+1);
            sendA<2>(buff0_A[2],txA2,rd>0&&rd<Total_rd+1);
            sendA<3>(buff0_A[3],txA3,rd>0&&rd<Total_rd+1);
            

            sendB<0>(buff0_B[0],txB0,rd>0&&rd<Total_rd+1);
            sendB<1>(buff0_B[1],txB1,rd>0&&rd<Total_rd+1);
            sendB<2>(buff0_B[2],txB2,rd>0&&rd<Total_rd+1);
            sendB<3>(buff0_B[3],txB3,rd>0&&rd<Total_rd+1);
            sendB<4>(buff0_B[4],txB4,rd>0&&rd<Total_rd+1);
            sendB<5>(buff0_B[5],txB5,rd>0&&rd<Total_rd+1);
            sendB<6>(buff0_B[6],txB6,rd>0&&rd<Total_rd+1);
            sendB<7>(buff0_B[7],txB7,rd>0&&rd<Total_rd+1);
            

            receiveC<0>(buff0_C[0],rxC0, rd>0&&rd<Total_rd+1);
            receiveC<1>(buff0_C[1],rxC1, rd>0&&rd<Total_rd+1);
            receiveC<2>(buff0_C[2],rxC2, rd>0&&rd<Total_rd+1);
            receiveC<3>(buff0_C[3],rxC3, rd>0&&rd<Total_rd+1);
            receiveC<4>(buff0_C[4],rxC4, rd>0&&rd<Total_rd+1);
            receiveC<5>(buff0_C[5],rxC5, rd>0&&rd<Total_rd+1);
            receiveC<6>(buff0_C[6],rxC6, rd>0&&rd<Total_rd+1);
            receiveC<7>(buff0_C[7],rxC7, rd>0&&rd<Total_rd+1);
            

            storeC(dataC_in, buff1_C, rd>TY&&s_flg==(TY-1));
        }
        else if(rd%2==0&&c_flg==1){
            loadA(dataA_out,buff0_A,rd<Total_rd);
            loadB(dataB_out,buff0_B,rd<Total_rd);   

            sendA<0>(buff1_A[0],txA0,rd>0&&rd<Total_rd+1);
            sendA<1>(buff1_A[1],txA1,rd>0&&rd<Total_rd+1);
            sendA<2>(buff1_A[2],txA2,rd>0&&rd<Total_rd+1);
            sendA<3>(buff1_A[3],txA3,rd>0&&rd<Total_rd+1);
            

            sendB<0>(buff1_B[0],txB0,rd>0&&rd<Total_rd+1);
            sendB<1>(buff1_B[1],txB1,rd>0&&rd<Total_rd+1);
            sendB<2>(buff1_B[2],txB2,rd>0&&rd<Total_rd+1);
            sendB<3>(buff1_B[3],txB3,rd>0&&rd<Total_rd+1);
            sendB<4>(buff1_B[4],txB4,rd>0&&rd<Total_rd+1);
            sendB<5>(buff1_B[5],txB5,rd>0&&rd<Total_rd+1);
            sendB<6>(buff1_B[6],txB6,rd>0&&rd<Total_rd+1);
            sendB<7>(buff1_B[7],txB7,rd>0&&rd<Total_rd+1);
            

            receiveC<0>(buff1_C[0],rxC0, rd>0&&rd<Total_rd+1);
            receiveC<1>(buff1_C[1],rxC1, rd>0&&rd<Total_rd+1);
            receiveC<2>(buff1_C[2],rxC2, rd>0&&rd<Total_rd+1);
            receiveC<3>(buff1_C[3],rxC3, rd>0&&rd<Total_rd+1);
            receiveC<4>(buff1_C[4],rxC4, rd>0&&rd<Total_rd+1);
            receiveC<5>(buff1_C[5],rxC5, rd>0&&rd<Total_rd+1);
            receiveC<6>(buff1_C[6],rxC6, rd>0&&rd<Total_rd+1);
            receiveC<7>(buff1_C[7],rxC7, rd>0&&rd<Total_rd+1);
            

            storeC(dataC_in, buff0_C, rd>TY&&s_flg==(TY-1));
        }
        else{ //if(rd%2==1&&c_flg==1)
            loadA(dataA_out,buff1_A,rd<Total_rd);
            loadB(dataB_out,buff1_B,rd<Total_rd);   

            sendA<0>(buff0_A[0],txA0,rd>0&&rd<Total_rd+1);
            sendA<1>(buff0_A[1],txA1,rd>0&&rd<Total_rd+1);
            sendA<2>(buff0_A[2],txA2,rd>0&&rd<Total_rd+1);
            sendA<3>(buff0_A[3],txA3,rd>0&&rd<Total_rd+1);
            

            sendB<0>(buff0_B[0],txB0,rd>0&&rd<Total_rd+1);
            sendB<1>(buff0_B[1],txB1,rd>0&&rd<Total_rd+1);
            sendB<2>(buff0_B[2],txB2,rd>0&&rd<Total_rd+1);
            sendB<3>(buff0_B[3],txB3,rd>0&&rd<Total_rd+1);
            sendB<4>(buff0_B[4],txB4,rd>0&&rd<Total_rd+1);
            sendB<5>(buff0_B[5],txB5,rd>0&&rd<Total_rd+1);
            sendB<6>(buff0_B[6],txB6,rd>0&&rd<Total_rd+1);
            sendB<7>(buff0_B[7],txB7,rd>0&&rd<Total_rd+1);
            

            receiveC<0>(buff1_C[0],rxC0, rd>0&&rd<Total_rd+1);
            receiveC<1>(buff1_C[1],rxC1, rd>0&&rd<Total_rd+1);
            receiveC<2>(buff1_C[2],rxC2, rd>0&&rd<Total_rd+1);
            receiveC<3>(buff1_C[3],rxC3, rd>0&&rd<Total_rd+1);
            receiveC<4>(buff1_C[4],rxC4, rd>0&&rd<Total_rd+1);
            receiveC<5>(buff1_C[5],rxC5, rd>0&&rd<Total_rd+1);
            receiveC<6>(buff1_C[6],rxC6, rd>0&&rd<Total_rd+1);
            receiveC<7>(buff1_C[7],rxC7, rd>0&&rd<Total_rd+1);
            

            storeC(dataC_in, buff0_C, rd>TY&&s_flg==(TY-1));
        }
    }
}
void dma_small(ap_uint<AXI_WIDTH_A>* ina, ap_uint<AXI_WIDTH_B>* inb, ap_uint<AXI_WIDTH_C>* out0,
             axis_stream& txA0, axis_stream& txA1, 
             axis_stream& txA2, axis_stream& txA3, 
             axis_stream& txB0, axis_stream& txB1, 
             axis_stream& txB2, axis_stream& txB3, 
             axis_stream& txB4, axis_stream& txB5, 
             axis_stream& txB6, axis_stream& txB7, 
             axis_stream& rxC0, axis_stream& rxC1, axis_stream& rxC2, axis_stream& rxC3, 
             axis_stream& rxC4, axis_stream& rxC5, axis_stream& rxC6, axis_stream& rxC7, 
             const int TX, const int TY, const int TZ, const int head_idx){
    
    #pragma HLS interface m_axi offset=slave bundle=gmem0 port=ina max_read_burst_length=16 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=ina
    #pragma HLS interface m_axi offset=slave bundle=gmem1 port=inb max_read_burst_length=16 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=inb
    #pragma HLS interface m_axi offset=slave bundle=gmem2 port=out0 max_write_burst_length=16 num_write_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=out0
    #pragma HLS interface s_axilite bundle=control port=TX
    #pragma HLS interface s_axilite bundle=control port=TY
    #pragma HLS interface s_axilite bundle=control port=TZ
    #pragma HLS interface s_axilite bundle=control port=head_idx
    #pragma HLS interface axis port=txA0
    #pragma HLS interface axis port=txA1
    #pragma HLS interface axis port=txA2
    #pragma HLS interface axis port=txA3
    #pragma HLS interface axis port=txB0
    #pragma HLS interface axis port=txB1
    #pragma HLS interface axis port=txB2
    #pragma HLS interface axis port=txB3
    #pragma HLS interface axis port=txB4
    #pragma HLS interface axis port=txB5
    #pragma HLS interface axis port=txB6
    #pragma HLS interface axis port=txB7
    #pragma HLS interface axis port=rxC0
    #pragma HLS interface axis port=rxC1
    #pragma HLS interface axis port=rxC2
    #pragma HLS interface axis port=rxC3
    #pragma HLS interface axis port=rxC4
    #pragma HLS interface axis port=rxC5
    #pragma HLS interface axis port=rxC6
    #pragma HLS interface axis port=rxC7
    #pragma HLS interface s_axilite bundle=control port=return

    #pragma HLS dataflow
    axis_stream_A dataA_out;
    axis_stream_B dataB_out;
    axis_stream_C dataC_in;
    axis_stream_32 addrA_out;
    axis_stream_32 addrB_out;
    axis_stream_32 addrC_out;

    address_A_ddr(addrA_out,TX,TY,TZ);
    loadA_ddr(ina, addrA_out,dataA_out,TX,TY,TZ);

    address_B_ddr(addrB_out,TX,TY,TZ);
    loadB_ddr(inb,addrB_out,dataB_out,TX,TY,TZ);

    address_C_ddr(addrC_out,TX,TZ, head_idx);
    storeC_ddr(out0,addrC_out,dataC_in,TX,TZ);

    compute(dataA_out, dataB_out, dataC_in,
            txA0, txA1, 
            txA2, txA3, 
            txB0, txB1, 
            txB2, txB3, 
            txB4, txB5, 
            txB6, txB7, 
            rxC0, rxC1, rxC2, rxC3, 
            rxC4, rxC5, rxC6, rxC7, 
            TX, TY, TZ);

}