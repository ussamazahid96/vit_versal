#include <stdint.h>
#include "dma.hpp"

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

void address_C_ddr(axis_stream_32& addrC_out,const int TX,const int TZ) {
#pragma HLS inline off
    for(int tx=0;tx<TX;tx++){
        for(int tz=0;tz<TZ;tz++){
            ap_uint<32> pos;
            for(int j=0;j<N;j++){
                for(int i=0;i<M/C_PER_TRA;i++){
                #pragma HLS PIPELINE II = 1
                    pos=i+j*(M/C_PER_TRA)*TX+tx*(M/C_PER_TRA)+tz*N*(M/C_PER_TRA)*TX;
                    addrC_out.write(pos);
                }
            }
        }
    }
}

void address_C_ddr(axis_stream_32& addrC_out,axis_stream_32& addrbias_out,const int TX,const int TZ) {
#pragma HLS inline off
    for(int tx=0;tx<TX;tx++){
        for(int tz=0;tz<TZ;tz++){
            ap_uint<32> pos;
            for(int j=0;j<N;j++){
                for(int i=0;i<M/C_PER_TRA;i++){
                #pragma HLS PIPELINE II = 1
                    pos=i+j*(M/C_PER_TRA)*TX+tx*(M/C_PER_TRA)+tz*N*(M/C_PER_TRA)*TX;
                    ap_uint<32> posb = i+tx*(M/C_PER_TRA);
                    addrC_out.write(pos);
                    addrbias_out.write(posb);
                }
            }
        }
    }
}

void loadBias_ddr(ap_uint<AXI_WIDTH_C>* bias, axis_stream_32& addrBias_in,
                  axis_stream_C& dataBias_out,const int TX,const int TZ) {
#pragma HLS inline off
    ap_uint<AXI_WIDTH_C> temp_data;
    int bound=boundC*TX*TZ;
    for(int i=0;i<bound;i++){
    #pragma HLS PIPELINE II = 1
        ap_uint<32> addr = addrBias_in.read();
        temp_data=bias[addr];
        dataBias_out.write(temp_data);
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

void storeC_ddr(ap_uint<AXI_WIDTH_C>* outc, axis_stream_32& addrC_in,
                axis_stream_C& dataC_in, axis_stream_C& dataBias_in,
                const int TX,const int TZ) { 
#pragma HLS inline off
    int bound=boundC*TX*TZ;
    for(int i=0;i<bound;i++){
    #pragma HLS PIPELINE II = 1
        ap_uint<32> addr = addrC_in.read();
        ap_uint<AXI_WIDTH_C> in_pkt = dataC_in.read();
        ap_uint<AXI_WIDTH_C> bias_val = dataBias_in.read();
        ap_uint<AXI_WIDTH_C> out_pkt;
        for(int j=0;j<C_PER_TRA;j++) {
#pragma HLS UNROLL
            ap_int<DATA_TYPE> elem  = in_pkt( (j+1)*DATA_TYPE-1, j*DATA_TYPE );
            ap_int<DATA_TYPE> belem = bias_val( (j+1)*DATA_TYPE-1, j*DATA_TYPE );
            float elem_f = *reinterpret_cast<float*>(&elem);
            float bias_f = *reinterpret_cast<float*>(&belem);
            float sum = elem_f + bias_f;
            ap_uint<DATA_TYPE> sum_I = *reinterpret_cast<ap_uint<DATA_TYPE>*>(&sum);
            out_pkt( (j+1)*DATA_TYPE-1, j*DATA_TYPE ) = sum_I;
        }
        outc[addr]=out_pkt;             
    }
    
}

void storeC_ddr(ap_uint<AXI_WIDTH_C>* outc,ap_uint<AXI_WIDTH_C>* bias, 
                axis_stream_32& addrC_in, axis_stream_32& addrbias_in,
                axis_stream_C& dataC_in,const int TX,const int TZ) { 
#pragma HLS inline off
    int bound=boundC*TX*TZ;
    for(int i=0;i<bound;i++){
    #pragma HLS PIPELINE II = 1
        ap_uint<32> addr = addrC_in.read();
        ap_uint<AXI_WIDTH_C> in_pkt=dataC_in.read();

        ap_uint<32> addrbias = addrbias_in.read();
        ap_uint<AXI_WIDTH_C> bias_val = bias[addrbias];

        ap_uint<AXI_WIDTH_C> out_pkt;
        for(int j=0;j<C_PER_TRA;j++) {
#pragma HLS UNROLL
            ap_int<DATA_TYPE> elem  = in_pkt( (j+1)*DATA_TYPE-1, j*DATA_TYPE );
            ap_int<DATA_TYPE> belem = bias_val( (j+1)*DATA_TYPE-1, j*DATA_TYPE );
            float elem_f = *reinterpret_cast<float*>(&elem);
            float bias_f = *reinterpret_cast<float*>(&belem);
            float sum = elem_f + bias_f;
            ap_uint<DATA_TYPE> sum_I = *reinterpret_cast<ap_uint<DATA_TYPE>*>(&sum);
            out_pkt( (j+1)*DATA_TYPE-1, j*DATA_TYPE ) = sum_I;
        }
        outc[addr]=out_pkt;   
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
             axis_stream& txA0, axis_stream& txA1, axis_stream& txA2, axis_stream& txA3, 
             axis_stream& txA4, axis_stream& txA5, axis_stream& txA6, axis_stream& txA7, 
             axis_stream& txA8, axis_stream& txA9, axis_stream& txA10, axis_stream& txA11, 
             axis_stream& txA12, axis_stream& txA13, axis_stream& txA14, axis_stream& txA15, 
             axis_stream& txA16, axis_stream& txA17, axis_stream& txA18, axis_stream& txA19, 
             axis_stream& txA20, axis_stream& txA21, axis_stream& txA22, axis_stream& txA23, 
             axis_stream& txA24, axis_stream& txA25, axis_stream& txA26, axis_stream& txA27, 
             axis_stream& txA28, axis_stream& txA29, axis_stream& txA30, axis_stream& txA31, 
             axis_stream& txB0, axis_stream& txB1, axis_stream& txB2, axis_stream& txB3, 
             axis_stream& txB4, axis_stream& txB5, axis_stream& txB6, axis_stream& txB7, 
             axis_stream& txB8, axis_stream& txB9, axis_stream& txB10, axis_stream& txB11, 
             axis_stream& txB12, axis_stream& txB13, axis_stream& txB14, axis_stream& txB15, 
             axis_stream& rxC0, axis_stream& rxC1, axis_stream& rxC2, axis_stream& rxC3, 
             axis_stream& rxC4, axis_stream& rxC5, axis_stream& rxC6, axis_stream& rxC7, 
             axis_stream& rxC8, axis_stream& rxC9, axis_stream& rxC10, axis_stream& rxC11, 
             axis_stream& rxC12, axis_stream& rxC13, axis_stream& rxC14, axis_stream& rxC15, 
             axis_stream& rxC16, axis_stream& rxC17, axis_stream& rxC18, axis_stream& rxC19, 
             axis_stream& rxC20, axis_stream& rxC21, axis_stream& rxC22, axis_stream& rxC23, 
             axis_stream& rxC24, axis_stream& rxC25, axis_stream& rxC26, axis_stream& rxC27, 
             axis_stream& rxC28, axis_stream& rxC29, axis_stream& rxC30, axis_stream& rxC31, 
             const int TX, const int TY, const int TZ){

    ap_uint<PLIO_WIDTH> buff0_A[A*B][X][Y][W1][(H1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff0_A type=RAM_T2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=buff0_A complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff0_A cyclic factor=BUFFA_FACTOR dim=5
    
    ap_uint<PLIO_WIDTH> buff1_A[A*B][X][Y][W1][(H1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff1_A type=RAM_T2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=buff1_A complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff1_A cyclic factor=BUFFA_FACTOR dim=5

    ap_uint<PLIO_WIDTH> buff0_B[B*C][Z][Y][W2][(W1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff0_B type=RAM_T2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=buff0_B complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff0_B cyclic factor=BUFFB_FACTOR dim=5

    ap_uint<PLIO_WIDTH> buff1_B[B*C][Z][Y][W2][(W1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff1_B type=RAM_T2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=buff1_B complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff1_B cyclic factor=BUFFB_FACTOR dim=5

    ap_uint<PLIO_WIDTH> buff0_C[C*A][Z][X][W2][(H1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff0_C type=RAM_T2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=buff0_C complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buff0_C cyclic factor=BUFFC_FACTOR dim=5

    ap_uint<PLIO_WIDTH> buff1_C[C*A][Z][X][W2][(H1/NUM_PER_TRA)];
    #pragma HLS bind_storage variable=buff1_C type=RAM_T2P impl=URAM
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
            sendA<4>(buff1_A[4],txA4,rd>0&&rd<Total_rd+1);
            sendA<5>(buff1_A[5],txA5,rd>0&&rd<Total_rd+1);
            sendA<6>(buff1_A[6],txA6,rd>0&&rd<Total_rd+1);
            sendA<7>(buff1_A[7],txA7,rd>0&&rd<Total_rd+1);
            sendA<8>(buff1_A[8],txA8,rd>0&&rd<Total_rd+1);
            sendA<9>(buff1_A[9],txA9,rd>0&&rd<Total_rd+1);
            sendA<10>(buff1_A[10],txA10,rd>0&&rd<Total_rd+1);
            sendA<11>(buff1_A[11],txA11,rd>0&&rd<Total_rd+1);
            sendA<12>(buff1_A[12],txA12,rd>0&&rd<Total_rd+1);
            sendA<13>(buff1_A[13],txA13,rd>0&&rd<Total_rd+1);
            sendA<14>(buff1_A[14],txA14,rd>0&&rd<Total_rd+1);
            sendA<15>(buff1_A[15],txA15,rd>0&&rd<Total_rd+1);
            sendA<16>(buff1_A[16],txA16,rd>0&&rd<Total_rd+1);
            sendA<17>(buff1_A[17],txA17,rd>0&&rd<Total_rd+1);
            sendA<18>(buff1_A[18],txA18,rd>0&&rd<Total_rd+1);
            sendA<19>(buff1_A[19],txA19,rd>0&&rd<Total_rd+1);
            sendA<20>(buff1_A[20],txA20,rd>0&&rd<Total_rd+1);
            sendA<21>(buff1_A[21],txA21,rd>0&&rd<Total_rd+1);
            sendA<22>(buff1_A[22],txA22,rd>0&&rd<Total_rd+1);
            sendA<23>(buff1_A[23],txA23,rd>0&&rd<Total_rd+1);
            sendA<24>(buff1_A[24],txA24,rd>0&&rd<Total_rd+1);
            sendA<25>(buff1_A[25],txA25,rd>0&&rd<Total_rd+1);
            sendA<26>(buff1_A[26],txA26,rd>0&&rd<Total_rd+1);
            sendA<27>(buff1_A[27],txA27,rd>0&&rd<Total_rd+1);
            sendA<28>(buff1_A[28],txA28,rd>0&&rd<Total_rd+1);
            sendA<29>(buff1_A[29],txA29,rd>0&&rd<Total_rd+1);
            sendA<30>(buff1_A[30],txA30,rd>0&&rd<Total_rd+1);
            sendA<31>(buff1_A[31],txA31,rd>0&&rd<Total_rd+1);
            

            sendB<0>(buff1_B[0],txB0,rd>0&&rd<Total_rd+1);
            sendB<1>(buff1_B[1],txB1,rd>0&&rd<Total_rd+1);
            sendB<2>(buff1_B[2],txB2,rd>0&&rd<Total_rd+1);
            sendB<3>(buff1_B[3],txB3,rd>0&&rd<Total_rd+1);
            sendB<4>(buff1_B[4],txB4,rd>0&&rd<Total_rd+1);
            sendB<5>(buff1_B[5],txB5,rd>0&&rd<Total_rd+1);
            sendB<6>(buff1_B[6],txB6,rd>0&&rd<Total_rd+1);
            sendB<7>(buff1_B[7],txB7,rd>0&&rd<Total_rd+1);
            sendB<8>(buff1_B[8],txB8,rd>0&&rd<Total_rd+1);
            sendB<9>(buff1_B[9],txB9,rd>0&&rd<Total_rd+1);
            sendB<10>(buff1_B[10],txB10,rd>0&&rd<Total_rd+1);
            sendB<11>(buff1_B[11],txB11,rd>0&&rd<Total_rd+1);
            sendB<12>(buff1_B[12],txB12,rd>0&&rd<Total_rd+1);
            sendB<13>(buff1_B[13],txB13,rd>0&&rd<Total_rd+1);
            sendB<14>(buff1_B[14],txB14,rd>0&&rd<Total_rd+1);
            sendB<15>(buff1_B[15],txB15,rd>0&&rd<Total_rd+1);
            

            receiveC<0>(buff0_C[0],rxC0, rd>0&&rd<Total_rd+1);
            receiveC<1>(buff0_C[1],rxC1, rd>0&&rd<Total_rd+1);
            receiveC<2>(buff0_C[2],rxC2, rd>0&&rd<Total_rd+1);
            receiveC<3>(buff0_C[3],rxC3, rd>0&&rd<Total_rd+1);
            receiveC<4>(buff0_C[4],rxC4, rd>0&&rd<Total_rd+1);
            receiveC<5>(buff0_C[5],rxC5, rd>0&&rd<Total_rd+1);
            receiveC<6>(buff0_C[6],rxC6, rd>0&&rd<Total_rd+1);
            receiveC<7>(buff0_C[7],rxC7, rd>0&&rd<Total_rd+1);
            receiveC<8>(buff0_C[8],rxC8, rd>0&&rd<Total_rd+1);
            receiveC<9>(buff0_C[9],rxC9, rd>0&&rd<Total_rd+1);
            receiveC<10>(buff0_C[10],rxC10, rd>0&&rd<Total_rd+1);
            receiveC<11>(buff0_C[11],rxC11, rd>0&&rd<Total_rd+1);
            receiveC<12>(buff0_C[12],rxC12, rd>0&&rd<Total_rd+1);
            receiveC<13>(buff0_C[13],rxC13, rd>0&&rd<Total_rd+1);
            receiveC<14>(buff0_C[14],rxC14, rd>0&&rd<Total_rd+1);
            receiveC<15>(buff0_C[15],rxC15, rd>0&&rd<Total_rd+1);
            receiveC<16>(buff0_C[16],rxC16, rd>0&&rd<Total_rd+1);
            receiveC<17>(buff0_C[17],rxC17, rd>0&&rd<Total_rd+1);
            receiveC<18>(buff0_C[18],rxC18, rd>0&&rd<Total_rd+1);
            receiveC<19>(buff0_C[19],rxC19, rd>0&&rd<Total_rd+1);
            receiveC<20>(buff0_C[20],rxC20, rd>0&&rd<Total_rd+1);
            receiveC<21>(buff0_C[21],rxC21, rd>0&&rd<Total_rd+1);
            receiveC<22>(buff0_C[22],rxC22, rd>0&&rd<Total_rd+1);
            receiveC<23>(buff0_C[23],rxC23, rd>0&&rd<Total_rd+1);
            receiveC<24>(buff0_C[24],rxC24, rd>0&&rd<Total_rd+1);
            receiveC<25>(buff0_C[25],rxC25, rd>0&&rd<Total_rd+1);
            receiveC<26>(buff0_C[26],rxC26, rd>0&&rd<Total_rd+1);
            receiveC<27>(buff0_C[27],rxC27, rd>0&&rd<Total_rd+1);
            receiveC<28>(buff0_C[28],rxC28, rd>0&&rd<Total_rd+1);
            receiveC<29>(buff0_C[29],rxC29, rd>0&&rd<Total_rd+1);
            receiveC<30>(buff0_C[30],rxC30, rd>0&&rd<Total_rd+1);
            receiveC<31>(buff0_C[31],rxC31, rd>0&&rd<Total_rd+1);
            

            storeC(dataC_in, buff1_C, rd>TY&&s_flg==(TY-1));
        }
        else if(rd%2==1&&c_flg==0){
            loadA(dataA_out,buff1_A,rd<Total_rd);
            loadB(dataB_out,buff1_B,rd<Total_rd);   

            sendA<0>(buff0_A[0],txA0,rd>0&&rd<Total_rd+1);
            sendA<1>(buff0_A[1],txA1,rd>0&&rd<Total_rd+1);
            sendA<2>(buff0_A[2],txA2,rd>0&&rd<Total_rd+1);
            sendA<3>(buff0_A[3],txA3,rd>0&&rd<Total_rd+1);
            sendA<4>(buff0_A[4],txA4,rd>0&&rd<Total_rd+1);
            sendA<5>(buff0_A[5],txA5,rd>0&&rd<Total_rd+1);
            sendA<6>(buff0_A[6],txA6,rd>0&&rd<Total_rd+1);
            sendA<7>(buff0_A[7],txA7,rd>0&&rd<Total_rd+1);
            sendA<8>(buff0_A[8],txA8,rd>0&&rd<Total_rd+1);
            sendA<9>(buff0_A[9],txA9,rd>0&&rd<Total_rd+1);
            sendA<10>(buff0_A[10],txA10,rd>0&&rd<Total_rd+1);
            sendA<11>(buff0_A[11],txA11,rd>0&&rd<Total_rd+1);
            sendA<12>(buff0_A[12],txA12,rd>0&&rd<Total_rd+1);
            sendA<13>(buff0_A[13],txA13,rd>0&&rd<Total_rd+1);
            sendA<14>(buff0_A[14],txA14,rd>0&&rd<Total_rd+1);
            sendA<15>(buff0_A[15],txA15,rd>0&&rd<Total_rd+1);
            sendA<16>(buff0_A[16],txA16,rd>0&&rd<Total_rd+1);
            sendA<17>(buff0_A[17],txA17,rd>0&&rd<Total_rd+1);
            sendA<18>(buff0_A[18],txA18,rd>0&&rd<Total_rd+1);
            sendA<19>(buff0_A[19],txA19,rd>0&&rd<Total_rd+1);
            sendA<20>(buff0_A[20],txA20,rd>0&&rd<Total_rd+1);
            sendA<21>(buff0_A[21],txA21,rd>0&&rd<Total_rd+1);
            sendA<22>(buff0_A[22],txA22,rd>0&&rd<Total_rd+1);
            sendA<23>(buff0_A[23],txA23,rd>0&&rd<Total_rd+1);
            sendA<24>(buff0_A[24],txA24,rd>0&&rd<Total_rd+1);
            sendA<25>(buff0_A[25],txA25,rd>0&&rd<Total_rd+1);
            sendA<26>(buff0_A[26],txA26,rd>0&&rd<Total_rd+1);
            sendA<27>(buff0_A[27],txA27,rd>0&&rd<Total_rd+1);
            sendA<28>(buff0_A[28],txA28,rd>0&&rd<Total_rd+1);
            sendA<29>(buff0_A[29],txA29,rd>0&&rd<Total_rd+1);
            sendA<30>(buff0_A[30],txA30,rd>0&&rd<Total_rd+1);
            sendA<31>(buff0_A[31],txA31,rd>0&&rd<Total_rd+1);
            

            sendB<0>(buff0_B[0],txB0,rd>0&&rd<Total_rd+1);
            sendB<1>(buff0_B[1],txB1,rd>0&&rd<Total_rd+1);
            sendB<2>(buff0_B[2],txB2,rd>0&&rd<Total_rd+1);
            sendB<3>(buff0_B[3],txB3,rd>0&&rd<Total_rd+1);
            sendB<4>(buff0_B[4],txB4,rd>0&&rd<Total_rd+1);
            sendB<5>(buff0_B[5],txB5,rd>0&&rd<Total_rd+1);
            sendB<6>(buff0_B[6],txB6,rd>0&&rd<Total_rd+1);
            sendB<7>(buff0_B[7],txB7,rd>0&&rd<Total_rd+1);
            sendB<8>(buff0_B[8],txB8,rd>0&&rd<Total_rd+1);
            sendB<9>(buff0_B[9],txB9,rd>0&&rd<Total_rd+1);
            sendB<10>(buff0_B[10],txB10,rd>0&&rd<Total_rd+1);
            sendB<11>(buff0_B[11],txB11,rd>0&&rd<Total_rd+1);
            sendB<12>(buff0_B[12],txB12,rd>0&&rd<Total_rd+1);
            sendB<13>(buff0_B[13],txB13,rd>0&&rd<Total_rd+1);
            sendB<14>(buff0_B[14],txB14,rd>0&&rd<Total_rd+1);
            sendB<15>(buff0_B[15],txB15,rd>0&&rd<Total_rd+1);
            

            receiveC<0>(buff0_C[0],rxC0, rd>0&&rd<Total_rd+1);
            receiveC<1>(buff0_C[1],rxC1, rd>0&&rd<Total_rd+1);
            receiveC<2>(buff0_C[2],rxC2, rd>0&&rd<Total_rd+1);
            receiveC<3>(buff0_C[3],rxC3, rd>0&&rd<Total_rd+1);
            receiveC<4>(buff0_C[4],rxC4, rd>0&&rd<Total_rd+1);
            receiveC<5>(buff0_C[5],rxC5, rd>0&&rd<Total_rd+1);
            receiveC<6>(buff0_C[6],rxC6, rd>0&&rd<Total_rd+1);
            receiveC<7>(buff0_C[7],rxC7, rd>0&&rd<Total_rd+1);
            receiveC<8>(buff0_C[8],rxC8, rd>0&&rd<Total_rd+1);
            receiveC<9>(buff0_C[9],rxC9, rd>0&&rd<Total_rd+1);
            receiveC<10>(buff0_C[10],rxC10, rd>0&&rd<Total_rd+1);
            receiveC<11>(buff0_C[11],rxC11, rd>0&&rd<Total_rd+1);
            receiveC<12>(buff0_C[12],rxC12, rd>0&&rd<Total_rd+1);
            receiveC<13>(buff0_C[13],rxC13, rd>0&&rd<Total_rd+1);
            receiveC<14>(buff0_C[14],rxC14, rd>0&&rd<Total_rd+1);
            receiveC<15>(buff0_C[15],rxC15, rd>0&&rd<Total_rd+1);
            receiveC<16>(buff0_C[16],rxC16, rd>0&&rd<Total_rd+1);
            receiveC<17>(buff0_C[17],rxC17, rd>0&&rd<Total_rd+1);
            receiveC<18>(buff0_C[18],rxC18, rd>0&&rd<Total_rd+1);
            receiveC<19>(buff0_C[19],rxC19, rd>0&&rd<Total_rd+1);
            receiveC<20>(buff0_C[20],rxC20, rd>0&&rd<Total_rd+1);
            receiveC<21>(buff0_C[21],rxC21, rd>0&&rd<Total_rd+1);
            receiveC<22>(buff0_C[22],rxC22, rd>0&&rd<Total_rd+1);
            receiveC<23>(buff0_C[23],rxC23, rd>0&&rd<Total_rd+1);
            receiveC<24>(buff0_C[24],rxC24, rd>0&&rd<Total_rd+1);
            receiveC<25>(buff0_C[25],rxC25, rd>0&&rd<Total_rd+1);
            receiveC<26>(buff0_C[26],rxC26, rd>0&&rd<Total_rd+1);
            receiveC<27>(buff0_C[27],rxC27, rd>0&&rd<Total_rd+1);
            receiveC<28>(buff0_C[28],rxC28, rd>0&&rd<Total_rd+1);
            receiveC<29>(buff0_C[29],rxC29, rd>0&&rd<Total_rd+1);
            receiveC<30>(buff0_C[30],rxC30, rd>0&&rd<Total_rd+1);
            receiveC<31>(buff0_C[31],rxC31, rd>0&&rd<Total_rd+1);
            

            storeC(dataC_in, buff1_C, rd>TY&&s_flg==(TY-1));
        }
        else if(rd%2==0&&c_flg==1){
            loadA(dataA_out,buff0_A,rd<Total_rd);
            loadB(dataB_out,buff0_B,rd<Total_rd);   

            sendA<0>(buff1_A[0],txA0,rd>0&&rd<Total_rd+1);
            sendA<1>(buff1_A[1],txA1,rd>0&&rd<Total_rd+1);
            sendA<2>(buff1_A[2],txA2,rd>0&&rd<Total_rd+1);
            sendA<3>(buff1_A[3],txA3,rd>0&&rd<Total_rd+1);
            sendA<4>(buff1_A[4],txA4,rd>0&&rd<Total_rd+1);
            sendA<5>(buff1_A[5],txA5,rd>0&&rd<Total_rd+1);
            sendA<6>(buff1_A[6],txA6,rd>0&&rd<Total_rd+1);
            sendA<7>(buff1_A[7],txA7,rd>0&&rd<Total_rd+1);
            sendA<8>(buff1_A[8],txA8,rd>0&&rd<Total_rd+1);
            sendA<9>(buff1_A[9],txA9,rd>0&&rd<Total_rd+1);
            sendA<10>(buff1_A[10],txA10,rd>0&&rd<Total_rd+1);
            sendA<11>(buff1_A[11],txA11,rd>0&&rd<Total_rd+1);
            sendA<12>(buff1_A[12],txA12,rd>0&&rd<Total_rd+1);
            sendA<13>(buff1_A[13],txA13,rd>0&&rd<Total_rd+1);
            sendA<14>(buff1_A[14],txA14,rd>0&&rd<Total_rd+1);
            sendA<15>(buff1_A[15],txA15,rd>0&&rd<Total_rd+1);
            sendA<16>(buff1_A[16],txA16,rd>0&&rd<Total_rd+1);
            sendA<17>(buff1_A[17],txA17,rd>0&&rd<Total_rd+1);
            sendA<18>(buff1_A[18],txA18,rd>0&&rd<Total_rd+1);
            sendA<19>(buff1_A[19],txA19,rd>0&&rd<Total_rd+1);
            sendA<20>(buff1_A[20],txA20,rd>0&&rd<Total_rd+1);
            sendA<21>(buff1_A[21],txA21,rd>0&&rd<Total_rd+1);
            sendA<22>(buff1_A[22],txA22,rd>0&&rd<Total_rd+1);
            sendA<23>(buff1_A[23],txA23,rd>0&&rd<Total_rd+1);
            sendA<24>(buff1_A[24],txA24,rd>0&&rd<Total_rd+1);
            sendA<25>(buff1_A[25],txA25,rd>0&&rd<Total_rd+1);
            sendA<26>(buff1_A[26],txA26,rd>0&&rd<Total_rd+1);
            sendA<27>(buff1_A[27],txA27,rd>0&&rd<Total_rd+1);
            sendA<28>(buff1_A[28],txA28,rd>0&&rd<Total_rd+1);
            sendA<29>(buff1_A[29],txA29,rd>0&&rd<Total_rd+1);
            sendA<30>(buff1_A[30],txA30,rd>0&&rd<Total_rd+1);
            sendA<31>(buff1_A[31],txA31,rd>0&&rd<Total_rd+1);
            

            sendB<0>(buff1_B[0],txB0,rd>0&&rd<Total_rd+1);
            sendB<1>(buff1_B[1],txB1,rd>0&&rd<Total_rd+1);
            sendB<2>(buff1_B[2],txB2,rd>0&&rd<Total_rd+1);
            sendB<3>(buff1_B[3],txB3,rd>0&&rd<Total_rd+1);
            sendB<4>(buff1_B[4],txB4,rd>0&&rd<Total_rd+1);
            sendB<5>(buff1_B[5],txB5,rd>0&&rd<Total_rd+1);
            sendB<6>(buff1_B[6],txB6,rd>0&&rd<Total_rd+1);
            sendB<7>(buff1_B[7],txB7,rd>0&&rd<Total_rd+1);
            sendB<8>(buff1_B[8],txB8,rd>0&&rd<Total_rd+1);
            sendB<9>(buff1_B[9],txB9,rd>0&&rd<Total_rd+1);
            sendB<10>(buff1_B[10],txB10,rd>0&&rd<Total_rd+1);
            sendB<11>(buff1_B[11],txB11,rd>0&&rd<Total_rd+1);
            sendB<12>(buff1_B[12],txB12,rd>0&&rd<Total_rd+1);
            sendB<13>(buff1_B[13],txB13,rd>0&&rd<Total_rd+1);
            sendB<14>(buff1_B[14],txB14,rd>0&&rd<Total_rd+1);
            sendB<15>(buff1_B[15],txB15,rd>0&&rd<Total_rd+1);
            

            receiveC<0>(buff1_C[0],rxC0, rd>0&&rd<Total_rd+1);
            receiveC<1>(buff1_C[1],rxC1, rd>0&&rd<Total_rd+1);
            receiveC<2>(buff1_C[2],rxC2, rd>0&&rd<Total_rd+1);
            receiveC<3>(buff1_C[3],rxC3, rd>0&&rd<Total_rd+1);
            receiveC<4>(buff1_C[4],rxC4, rd>0&&rd<Total_rd+1);
            receiveC<5>(buff1_C[5],rxC5, rd>0&&rd<Total_rd+1);
            receiveC<6>(buff1_C[6],rxC6, rd>0&&rd<Total_rd+1);
            receiveC<7>(buff1_C[7],rxC7, rd>0&&rd<Total_rd+1);
            receiveC<8>(buff1_C[8],rxC8, rd>0&&rd<Total_rd+1);
            receiveC<9>(buff1_C[9],rxC9, rd>0&&rd<Total_rd+1);
            receiveC<10>(buff1_C[10],rxC10, rd>0&&rd<Total_rd+1);
            receiveC<11>(buff1_C[11],rxC11, rd>0&&rd<Total_rd+1);
            receiveC<12>(buff1_C[12],rxC12, rd>0&&rd<Total_rd+1);
            receiveC<13>(buff1_C[13],rxC13, rd>0&&rd<Total_rd+1);
            receiveC<14>(buff1_C[14],rxC14, rd>0&&rd<Total_rd+1);
            receiveC<15>(buff1_C[15],rxC15, rd>0&&rd<Total_rd+1);
            receiveC<16>(buff1_C[16],rxC16, rd>0&&rd<Total_rd+1);
            receiveC<17>(buff1_C[17],rxC17, rd>0&&rd<Total_rd+1);
            receiveC<18>(buff1_C[18],rxC18, rd>0&&rd<Total_rd+1);
            receiveC<19>(buff1_C[19],rxC19, rd>0&&rd<Total_rd+1);
            receiveC<20>(buff1_C[20],rxC20, rd>0&&rd<Total_rd+1);
            receiveC<21>(buff1_C[21],rxC21, rd>0&&rd<Total_rd+1);
            receiveC<22>(buff1_C[22],rxC22, rd>0&&rd<Total_rd+1);
            receiveC<23>(buff1_C[23],rxC23, rd>0&&rd<Total_rd+1);
            receiveC<24>(buff1_C[24],rxC24, rd>0&&rd<Total_rd+1);
            receiveC<25>(buff1_C[25],rxC25, rd>0&&rd<Total_rd+1);
            receiveC<26>(buff1_C[26],rxC26, rd>0&&rd<Total_rd+1);
            receiveC<27>(buff1_C[27],rxC27, rd>0&&rd<Total_rd+1);
            receiveC<28>(buff1_C[28],rxC28, rd>0&&rd<Total_rd+1);
            receiveC<29>(buff1_C[29],rxC29, rd>0&&rd<Total_rd+1);
            receiveC<30>(buff1_C[30],rxC30, rd>0&&rd<Total_rd+1);
            receiveC<31>(buff1_C[31],rxC31, rd>0&&rd<Total_rd+1);
            

            storeC(dataC_in, buff0_C, rd>TY&&s_flg==(TY-1));
        }
        else{ //if(rd%2==1&&c_flg==1)
            loadA(dataA_out,buff1_A,rd<Total_rd);
            loadB(dataB_out,buff1_B,rd<Total_rd);   

            sendA<0>(buff0_A[0],txA0,rd>0&&rd<Total_rd+1);
            sendA<1>(buff0_A[1],txA1,rd>0&&rd<Total_rd+1);
            sendA<2>(buff0_A[2],txA2,rd>0&&rd<Total_rd+1);
            sendA<3>(buff0_A[3],txA3,rd>0&&rd<Total_rd+1);
            sendA<4>(buff0_A[4],txA4,rd>0&&rd<Total_rd+1);
            sendA<5>(buff0_A[5],txA5,rd>0&&rd<Total_rd+1);
            sendA<6>(buff0_A[6],txA6,rd>0&&rd<Total_rd+1);
            sendA<7>(buff0_A[7],txA7,rd>0&&rd<Total_rd+1);
            sendA<8>(buff0_A[8],txA8,rd>0&&rd<Total_rd+1);
            sendA<9>(buff0_A[9],txA9,rd>0&&rd<Total_rd+1);
            sendA<10>(buff0_A[10],txA10,rd>0&&rd<Total_rd+1);
            sendA<11>(buff0_A[11],txA11,rd>0&&rd<Total_rd+1);
            sendA<12>(buff0_A[12],txA12,rd>0&&rd<Total_rd+1);
            sendA<13>(buff0_A[13],txA13,rd>0&&rd<Total_rd+1);
            sendA<14>(buff0_A[14],txA14,rd>0&&rd<Total_rd+1);
            sendA<15>(buff0_A[15],txA15,rd>0&&rd<Total_rd+1);
            sendA<16>(buff0_A[16],txA16,rd>0&&rd<Total_rd+1);
            sendA<17>(buff0_A[17],txA17,rd>0&&rd<Total_rd+1);
            sendA<18>(buff0_A[18],txA18,rd>0&&rd<Total_rd+1);
            sendA<19>(buff0_A[19],txA19,rd>0&&rd<Total_rd+1);
            sendA<20>(buff0_A[20],txA20,rd>0&&rd<Total_rd+1);
            sendA<21>(buff0_A[21],txA21,rd>0&&rd<Total_rd+1);
            sendA<22>(buff0_A[22],txA22,rd>0&&rd<Total_rd+1);
            sendA<23>(buff0_A[23],txA23,rd>0&&rd<Total_rd+1);
            sendA<24>(buff0_A[24],txA24,rd>0&&rd<Total_rd+1);
            sendA<25>(buff0_A[25],txA25,rd>0&&rd<Total_rd+1);
            sendA<26>(buff0_A[26],txA26,rd>0&&rd<Total_rd+1);
            sendA<27>(buff0_A[27],txA27,rd>0&&rd<Total_rd+1);
            sendA<28>(buff0_A[28],txA28,rd>0&&rd<Total_rd+1);
            sendA<29>(buff0_A[29],txA29,rd>0&&rd<Total_rd+1);
            sendA<30>(buff0_A[30],txA30,rd>0&&rd<Total_rd+1);
            sendA<31>(buff0_A[31],txA31,rd>0&&rd<Total_rd+1);
            

            sendB<0>(buff0_B[0],txB0,rd>0&&rd<Total_rd+1);
            sendB<1>(buff0_B[1],txB1,rd>0&&rd<Total_rd+1);
            sendB<2>(buff0_B[2],txB2,rd>0&&rd<Total_rd+1);
            sendB<3>(buff0_B[3],txB3,rd>0&&rd<Total_rd+1);
            sendB<4>(buff0_B[4],txB4,rd>0&&rd<Total_rd+1);
            sendB<5>(buff0_B[5],txB5,rd>0&&rd<Total_rd+1);
            sendB<6>(buff0_B[6],txB6,rd>0&&rd<Total_rd+1);
            sendB<7>(buff0_B[7],txB7,rd>0&&rd<Total_rd+1);
            sendB<8>(buff0_B[8],txB8,rd>0&&rd<Total_rd+1);
            sendB<9>(buff0_B[9],txB9,rd>0&&rd<Total_rd+1);
            sendB<10>(buff0_B[10],txB10,rd>0&&rd<Total_rd+1);
            sendB<11>(buff0_B[11],txB11,rd>0&&rd<Total_rd+1);
            sendB<12>(buff0_B[12],txB12,rd>0&&rd<Total_rd+1);
            sendB<13>(buff0_B[13],txB13,rd>0&&rd<Total_rd+1);
            sendB<14>(buff0_B[14],txB14,rd>0&&rd<Total_rd+1);
            sendB<15>(buff0_B[15],txB15,rd>0&&rd<Total_rd+1);
            

            receiveC<0>(buff1_C[0],rxC0, rd>0&&rd<Total_rd+1);
            receiveC<1>(buff1_C[1],rxC1, rd>0&&rd<Total_rd+1);
            receiveC<2>(buff1_C[2],rxC2, rd>0&&rd<Total_rd+1);
            receiveC<3>(buff1_C[3],rxC3, rd>0&&rd<Total_rd+1);
            receiveC<4>(buff1_C[4],rxC4, rd>0&&rd<Total_rd+1);
            receiveC<5>(buff1_C[5],rxC5, rd>0&&rd<Total_rd+1);
            receiveC<6>(buff1_C[6],rxC6, rd>0&&rd<Total_rd+1);
            receiveC<7>(buff1_C[7],rxC7, rd>0&&rd<Total_rd+1);
            receiveC<8>(buff1_C[8],rxC8, rd>0&&rd<Total_rd+1);
            receiveC<9>(buff1_C[9],rxC9, rd>0&&rd<Total_rd+1);
            receiveC<10>(buff1_C[10],rxC10, rd>0&&rd<Total_rd+1);
            receiveC<11>(buff1_C[11],rxC11, rd>0&&rd<Total_rd+1);
            receiveC<12>(buff1_C[12],rxC12, rd>0&&rd<Total_rd+1);
            receiveC<13>(buff1_C[13],rxC13, rd>0&&rd<Total_rd+1);
            receiveC<14>(buff1_C[14],rxC14, rd>0&&rd<Total_rd+1);
            receiveC<15>(buff1_C[15],rxC15, rd>0&&rd<Total_rd+1);
            receiveC<16>(buff1_C[16],rxC16, rd>0&&rd<Total_rd+1);
            receiveC<17>(buff1_C[17],rxC17, rd>0&&rd<Total_rd+1);
            receiveC<18>(buff1_C[18],rxC18, rd>0&&rd<Total_rd+1);
            receiveC<19>(buff1_C[19],rxC19, rd>0&&rd<Total_rd+1);
            receiveC<20>(buff1_C[20],rxC20, rd>0&&rd<Total_rd+1);
            receiveC<21>(buff1_C[21],rxC21, rd>0&&rd<Total_rd+1);
            receiveC<22>(buff1_C[22],rxC22, rd>0&&rd<Total_rd+1);
            receiveC<23>(buff1_C[23],rxC23, rd>0&&rd<Total_rd+1);
            receiveC<24>(buff1_C[24],rxC24, rd>0&&rd<Total_rd+1);
            receiveC<25>(buff1_C[25],rxC25, rd>0&&rd<Total_rd+1);
            receiveC<26>(buff1_C[26],rxC26, rd>0&&rd<Total_rd+1);
            receiveC<27>(buff1_C[27],rxC27, rd>0&&rd<Total_rd+1);
            receiveC<28>(buff1_C[28],rxC28, rd>0&&rd<Total_rd+1);
            receiveC<29>(buff1_C[29],rxC29, rd>0&&rd<Total_rd+1);
            receiveC<30>(buff1_C[30],rxC30, rd>0&&rd<Total_rd+1);
            receiveC<31>(buff1_C[31],rxC31, rd>0&&rd<Total_rd+1);
            

            storeC(dataC_in, buff0_C, rd>TY&&s_flg==(TY-1));
        }
    }
}
void dma(ap_uint<AXI_WIDTH_A>* ina, ap_uint<AXI_WIDTH_B>* inb, ap_uint<AXI_WIDTH_C>* out0, ap_uint<AXI_WIDTH_C>* bias,
             axis_stream& txA0, axis_stream& txA1, axis_stream& txA2, axis_stream& txA3, 
             axis_stream& txA4, axis_stream& txA5, axis_stream& txA6, axis_stream& txA7, 
             axis_stream& txA8, axis_stream& txA9, axis_stream& txA10, axis_stream& txA11, 
             axis_stream& txA12, axis_stream& txA13, axis_stream& txA14, axis_stream& txA15, 
             axis_stream& txA16, axis_stream& txA17, axis_stream& txA18, axis_stream& txA19, 
             axis_stream& txA20, axis_stream& txA21, axis_stream& txA22, axis_stream& txA23, 
             axis_stream& txA24, axis_stream& txA25, axis_stream& txA26, axis_stream& txA27, 
             axis_stream& txA28, axis_stream& txA29, axis_stream& txA30, axis_stream& txA31, 
             axis_stream& txB0, axis_stream& txB1, axis_stream& txB2, axis_stream& txB3, 
             axis_stream& txB4, axis_stream& txB5, axis_stream& txB6, axis_stream& txB7, 
             axis_stream& txB8, axis_stream& txB9, axis_stream& txB10, axis_stream& txB11, 
             axis_stream& txB12, axis_stream& txB13, axis_stream& txB14, axis_stream& txB15, 
             axis_stream& rxC0, axis_stream& rxC1, axis_stream& rxC2, axis_stream& rxC3, 
             axis_stream& rxC4, axis_stream& rxC5, axis_stream& rxC6, axis_stream& rxC7, 
             axis_stream& rxC8, axis_stream& rxC9, axis_stream& rxC10, axis_stream& rxC11, 
             axis_stream& rxC12, axis_stream& rxC13, axis_stream& rxC14, axis_stream& rxC15, 
             axis_stream& rxC16, axis_stream& rxC17, axis_stream& rxC18, axis_stream& rxC19, 
             axis_stream& rxC20, axis_stream& rxC21, axis_stream& rxC22, axis_stream& rxC23, 
             axis_stream& rxC24, axis_stream& rxC25, axis_stream& rxC26, axis_stream& rxC27, 
             axis_stream& rxC28, axis_stream& rxC29, axis_stream& rxC30, axis_stream& rxC31, 
             const int TX, const int TY, const int TZ){
    
    #pragma HLS interface m_axi offset=slave bundle=gmem0 port=ina max_read_burst_length=16 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=ina
    #pragma HLS interface m_axi offset=slave bundle=gmem1 port=inb max_read_burst_length=16 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=inb
    #pragma HLS interface m_axi offset=slave bundle=gmem2 port=out0 max_write_burst_length=16 num_write_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=out0
    #pragma HLS interface m_axi offset=slave bundle=gmem3 port=bias max_write_burst_length=16 num_write_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=bias
    #pragma HLS interface s_axilite bundle=control port=TX
    #pragma HLS interface s_axilite bundle=control port=TY
    #pragma HLS interface s_axilite bundle=control port=TZ
    #pragma HLS interface axis port=txA0
    #pragma HLS interface axis port=txA1
    #pragma HLS interface axis port=txA2
    #pragma HLS interface axis port=txA3
    #pragma HLS interface axis port=txA4
    #pragma HLS interface axis port=txA5
    #pragma HLS interface axis port=txA6
    #pragma HLS interface axis port=txA7
    #pragma HLS interface axis port=txA8
    #pragma HLS interface axis port=txA9
    #pragma HLS interface axis port=txA10
    #pragma HLS interface axis port=txA11
    #pragma HLS interface axis port=txA12
    #pragma HLS interface axis port=txA13
    #pragma HLS interface axis port=txA14
    #pragma HLS interface axis port=txA15
    #pragma HLS interface axis port=txA16
    #pragma HLS interface axis port=txA17
    #pragma HLS interface axis port=txA18
    #pragma HLS interface axis port=txA19
    #pragma HLS interface axis port=txA20
    #pragma HLS interface axis port=txA21
    #pragma HLS interface axis port=txA22
    #pragma HLS interface axis port=txA23
    #pragma HLS interface axis port=txA24
    #pragma HLS interface axis port=txA25
    #pragma HLS interface axis port=txA26
    #pragma HLS interface axis port=txA27
    #pragma HLS interface axis port=txA28
    #pragma HLS interface axis port=txA29
    #pragma HLS interface axis port=txA30
    #pragma HLS interface axis port=txA31
    #pragma HLS interface axis port=txB0
    #pragma HLS interface axis port=txB1
    #pragma HLS interface axis port=txB2
    #pragma HLS interface axis port=txB3
    #pragma HLS interface axis port=txB4
    #pragma HLS interface axis port=txB5
    #pragma HLS interface axis port=txB6
    #pragma HLS interface axis port=txB7
    #pragma HLS interface axis port=txB8
    #pragma HLS interface axis port=txB9
    #pragma HLS interface axis port=txB10
    #pragma HLS interface axis port=txB11
    #pragma HLS interface axis port=txB12
    #pragma HLS interface axis port=txB13
    #pragma HLS interface axis port=txB14
    #pragma HLS interface axis port=txB15
    #pragma HLS interface axis port=rxC0
    #pragma HLS interface axis port=rxC1
    #pragma HLS interface axis port=rxC2
    #pragma HLS interface axis port=rxC3
    #pragma HLS interface axis port=rxC4
    #pragma HLS interface axis port=rxC5
    #pragma HLS interface axis port=rxC6
    #pragma HLS interface axis port=rxC7
    #pragma HLS interface axis port=rxC8
    #pragma HLS interface axis port=rxC9
    #pragma HLS interface axis port=rxC10
    #pragma HLS interface axis port=rxC11
    #pragma HLS interface axis port=rxC12
    #pragma HLS interface axis port=rxC13
    #pragma HLS interface axis port=rxC14
    #pragma HLS interface axis port=rxC15
    #pragma HLS interface axis port=rxC16
    #pragma HLS interface axis port=rxC17
    #pragma HLS interface axis port=rxC18
    #pragma HLS interface axis port=rxC19
    #pragma HLS interface axis port=rxC20
    #pragma HLS interface axis port=rxC21
    #pragma HLS interface axis port=rxC22
    #pragma HLS interface axis port=rxC23
    #pragma HLS interface axis port=rxC24
    #pragma HLS interface axis port=rxC25
    #pragma HLS interface axis port=rxC26
    #pragma HLS interface axis port=rxC27
    #pragma HLS interface axis port=rxC28
    #pragma HLS interface axis port=rxC29
    #pragma HLS interface axis port=rxC30
    #pragma HLS interface axis port=rxC31
    #pragma HLS interface s_axilite bundle=control port=return

    #pragma HLS dataflow
    axis_stream_A dataA_out;
    axis_stream_B dataB_out;
    axis_stream_C dataC_in;
    // axis_stream_C dataBias_out;
// #pragma HLS STREAM depth=64 type=fifo variable=dataBias_out
    axis_stream_32 addrA_out;
    axis_stream_32 addrB_out;
    axis_stream_32 addrC_out;
    axis_stream_32 addrbias_out;

    address_A_ddr(addrA_out,TX,TY,TZ);
    loadA_ddr(ina, addrA_out,dataA_out,TX,TY,TZ);

    address_B_ddr(addrB_out,TX,TY,TZ);
    loadB_ddr(inb,addrB_out,dataB_out,TX,TY,TZ);

    // address_C_ddr(addrC_out, TX, TZ);
    // storeC_ddr(out0, addrC_out, dataC_in, TX, TZ);

    address_C_ddr(addrC_out, addrbias_out, TX, TZ);
    // loadBias_ddr(bias, addrbias_out, dataBias_out, TX, TZ);
    // storeC_ddr(out0, addrC_out, dataC_in, dataBias_out, TX, TZ);
    storeC_ddr(out0,bias,addrC_out,addrbias_out,dataC_in,TX,TZ);

    compute(dataA_out, dataB_out, dataC_in,
            txA0, txA1, txA2, txA3, 
            txA4, txA5, txA6, txA7, 
            txA8, txA9, txA10, txA11, 
            txA12, txA13, txA14, txA15, 
            txA16, txA17, txA18, txA19, 
            txA20, txA21, txA22, txA23, 
            txA24, txA25, txA26, txA27, 
            txA28, txA29, txA30, txA31, 
            txB0, txB1, txB2, txB3, 
            txB4, txB5, txB6, txB7, 
            txB8, txB9, txB10, txB11, 
            txB12, txB13, txB14, txB15, 
            rxC0, rxC1, rxC2, rxC3, 
            rxC4, rxC5, rxC6, rxC7, 
            rxC8, rxC9, rxC10, rxC11, 
            rxC12, rxC13, rxC14, rxC15, 
            rxC16, rxC17, rxC18, rxC19, 
            rxC20, rxC21, rxC22, rxC23, 
            rxC24, rxC25, rxC26, rxC27, 
            rxC28, rxC29, rxC30, rxC31, 
            TX, TY, TZ);

}
