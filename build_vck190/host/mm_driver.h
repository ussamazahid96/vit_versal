#ifndef MM_DRIVER_H
#define MM_DRIVER_H
#include <iostream>
#include <tuple>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "../aie/layer0/aie_top_L0.h"
#include "../aie/layer1/aie_top_L1.h"
#include "xrt/xrt.h"
#include "xrt/experimental/xrt_kernel.h"
#include "adf/adf_api/XRTConfig.h"

const unsigned long int H1=32;
const unsigned long int W1=32;
const unsigned long int W2=32;
const unsigned long int A=8;
const unsigned long int B=4;
const unsigned long int C=4;
const unsigned long int X=1;
const unsigned long int Y=2;
const unsigned long int Z=2;
const int M=H1*A*X;
const int K=W1*B*Y;
const int N=W2*C*Z;
const int M_SMALL = 64;
const int K_SMALL = 64;
const int N_SMALL = 128;
const int NUM_LARYERS = 12;
const long unsigned int heads = 12;
const long unsigned int dim = 64;
const long unsigned TOKENS = 197;
const long unsigned TOKENS_PADDED = 256;
const int EMB_DIM = 768;
typedef float lndatatype;
typedef float gldatatype;
typedef float smdatatype;
unsigned int PaddedSize(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}
static std::vector<char> load_xclbin(xrtDeviceHandle device, const std::string& fnm) {
    if (fnm.empty()) throw std::runtime_error("No xclbin specified");
    std::ifstream stream(fnm);
    stream.seekg(0, stream.end);
    size_t size = stream.tellg();
    stream.seekg(0, stream.beg);
    std::vector<char> header(size);
    stream.read(header.data(), size);
    auto top = reinterpret_cast<const axlf*>(header.data());
    if (xrtDeviceLoadXclbin(device, top)) throw std::runtime_error("Xclbin loading failed");
    return header;
}
class Timer {
    std::chrono::high_resolution_clock::time_point mTimeStart;
    std::chrono::high_resolution_clock::time_point mTimeEnd;
    double total_time = 0;
public:
    Timer(){}
    void start() { mTimeStart = std::chrono::high_resolution_clock::now(); }
    void stop() {
        mTimeEnd = std::chrono::high_resolution_clock::now();
        total_time = std::chrono::duration<double>(mTimeEnd - mTimeStart).count();
    }
    double get_time()
        {return total_time;}
};
template<typename T, const long unsigned int BATCH_SIZE = 1>
class AieMMDriver {
private:
    mm_x8_x4_x4_graph0  mm_graph0;
    mm_x2_x2_x4_graph1  mm_graph1;
    xrtDeviceHandle dhdl;
    xrtKernelHandle dma_khdl, ln_khdl, sm_khdl, gl_khdl, rt_khdl, dmas_khdl;
    xrtRunHandle dma_rhdl, ln_rhdl, sm_rhdl, gl_rhdl, rt_rhdl, dmas_rhdl;    
    bool buffers_init = false;
    Timer tmr;

public:
    xrtBufferHandle buffer0, buffer1, buffer2, zero_buffer; 
    xrtBufferHandle weibias_buffer[NUM_LARYERS+1][4][2], qbuffer[heads], kbuffer[heads], vbuffer[heads], attn_scores[heads], attn_probs[heads];
    xrtBufferHandle lni_buffer, lnsi_buffer, lnwb_buffer[NUM_LARYERS+1][2][2], lno_buffer, lnso_buffer; 
    xrtBufferHandle smi_buffer, smo_buffer, gli_buffer, glo_buffer;
    T *buffer0_mapped, *buffer1_mapped, *buffer2_mapped, *zero_buffer_mapped;
    T *weibias_buffer_mapped[NUM_LARYERS+1][4][2], *qbuffer_mapped[heads], *kbuffer_mapped[heads], *vbuffer_mapped[heads], *attn_scores_mapped[heads], *attn_probs_mapped[heads];
    lndatatype *lni_buffer_mapped, *lnsi_buffer_mapped, *lnwb_buffer_mapped[NUM_LARYERS+1][2][2], *lno_buffer_mapped, *lnso_buffer_mapped;
    smdatatype *smi_buffer_mapped, *smo_buffer_mapped;
    gldatatype *gli_buffer_mapped, *glo_buffer_mapped;

    AieMMDriver(char* xclbinFilename) {
        dhdl = xrtDeviceOpen(0);
        if (dhdl == nullptr) 
            throw std::runtime_error("No valid device handle found. Make sure using right xclOpen index.");
        auto xclbin = load_xclbin(dhdl, xclbinFilename);
        auto top = reinterpret_cast<const axlf*>(xclbin.data());
        adf::registerXRT(dhdl, top->m_header.uuid);       
        mm_graph0.init();                   
        mm_graph1.init();                   
        mm_graph0.run(-1);
        mm_graph1.run(-1);
        dma_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "dma");
        ln_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "layernorm");
        sm_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "softmax");
        gl_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "gelu");
        rt_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "rant");
        dmas_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "dma_small");
        std::cout << "Initializing base accelerator for A = {" << K << ", " << M << "} and B = {" << BATCH_SIZE << ", " << N << ", " << K << "}" << '\n';
    }

    void allocate_buffers(const int M_MAX, const int K_MAX, const int N_MAX) {
        std::cout << "Allocating Buffers..." << '\n';
        buffer0 = xrtBOAlloc(dhdl, M_MAX*K_MAX*sizeof(T), 0, 0);
        buffer1 = xrtBOAlloc(dhdl, K_MAX*N_MAX*sizeof(T), 0, 0);
        buffer2 = xrtBOAlloc(dhdl, M_MAX*N_MAX*sizeof(T), 0, 0);
        zero_buffer = xrtBOAlloc(dhdl, M_MAX*sizeof(T), 0, 0);
        buffer0_mapped = reinterpret_cast<T*>(xrtBOMap(buffer0));
        buffer1_mapped = reinterpret_cast<T*>(xrtBOMap(buffer1));
        buffer2_mapped = reinterpret_cast<T*>(xrtBOMap(buffer2));
        zero_buffer_mapped = reinterpret_cast<T*>(xrtBOMap(zero_buffer));
        memset(zero_buffer_mapped, 0x00000000, M_MAX * sizeof(float));
        xrtBOSync(zero_buffer, XCL_BO_SYNC_BO_TO_DEVICE, M_MAX*sizeof(T), 0);
        lni_buffer = xrtBOAlloc(dhdl, BATCH_SIZE*TOKENS_PADDED*EMB_DIM*sizeof(lndatatype), 0, 0);
        lni_buffer_mapped = reinterpret_cast<lndatatype*>(xrtBOMap(lni_buffer));
        lnsi_buffer = xrtBOAlloc(dhdl, BATCH_SIZE*TOKENS_PADDED*EMB_DIM*sizeof(lndatatype), 0, 0);
        lnsi_buffer_mapped = reinterpret_cast<lndatatype*>(xrtBOMap(lnsi_buffer));
        lno_buffer = xrtBOAlloc(dhdl, BATCH_SIZE*TOKENS_PADDED*EMB_DIM*sizeof(lndatatype), 0, 0);
        lno_buffer_mapped = reinterpret_cast<lndatatype*>(xrtBOMap(lno_buffer));
        lnso_buffer = xrtBOAlloc(dhdl, BATCH_SIZE*TOKENS_PADDED*EMB_DIM*sizeof(lndatatype), 0, 0);
        lnso_buffer_mapped = reinterpret_cast<lndatatype*>(xrtBOMap(lnso_buffer));
        for(long unsigned int i=0; i<heads; i++){
            qbuffer[i] = xrtBOAlloc(dhdl, TOKENS_PADDED*dim*sizeof(T), 0, 0);
            kbuffer[i] = xrtBOAlloc(dhdl, TOKENS_PADDED*dim*sizeof(T), 0, 0);
            vbuffer[i] = xrtBOAlloc(dhdl, TOKENS_PADDED*dim*sizeof(T), 0, 0);
            attn_scores[i] = xrtBOAlloc(dhdl, TOKENS_PADDED*TOKENS_PADDED*sizeof(T), 0, 0);
            attn_probs[i] = xrtBOAlloc(dhdl, TOKENS_PADDED*TOKENS_PADDED*sizeof(T), 0, 0);
            qbuffer_mapped[i] = reinterpret_cast<T*>(xrtBOMap(qbuffer[i]));
            kbuffer_mapped[i] = reinterpret_cast<T*>(xrtBOMap(kbuffer[i]));
            vbuffer_mapped[i] = reinterpret_cast<T*>(xrtBOMap(vbuffer[i]));
            attn_scores_mapped[i] = reinterpret_cast<T*>(xrtBOMap(attn_scores[i]));
            attn_probs_mapped[i] = reinterpret_cast<T*>(xrtBOMap(attn_probs[i]));
        }
        for(int i=0;i<NUM_LARYERS;i++){
            // every layer has 2 layernorms
            for(int j=0;j<2;j++){
                // weights
                lnwb_buffer[i][j][0] = xrtBOAlloc(dhdl, EMB_DIM*sizeof(lndatatype), 0, 0);
                lnwb_buffer_mapped[i][j][0] = reinterpret_cast<lndatatype*>(xrtBOMap(lnwb_buffer[i][j][0]));
                // bias
                lnwb_buffer[i][j][1] = xrtBOAlloc(dhdl, EMB_DIM*sizeof(lndatatype), 0, 0);
                lnwb_buffer_mapped[i][j][1] = reinterpret_cast<lndatatype*>(xrtBOMap(lnwb_buffer[i][j][1]));
            }
            weibias_buffer[i][0][0] = xrtBOAlloc(dhdl, 768*2304*sizeof(T), 0, 0);
            weibias_buffer[i][1][0] = xrtBOAlloc(dhdl, 768*768*sizeof(T), 0, 0);
            weibias_buffer[i][2][0] = xrtBOAlloc(dhdl, 768*3072*sizeof(T), 0, 0);
            weibias_buffer[i][3][0] = xrtBOAlloc(dhdl, 3072*768*sizeof(T), 0, 0);
            weibias_buffer[i][0][1] = xrtBOAlloc(dhdl, 2304*sizeof(T), 0, 0);
            weibias_buffer[i][1][1] = xrtBOAlloc(dhdl, 768*sizeof(T), 0, 0);
            weibias_buffer[i][2][1] = xrtBOAlloc(dhdl, 3072*sizeof(T), 0, 0);
            weibias_buffer[i][3][1] = xrtBOAlloc(dhdl, 768*sizeof(T), 0, 0);
            for(int j=0;j<4;j++) {
                weibias_buffer_mapped[i][j][0] = reinterpret_cast<T*>(xrtBOMap(weibias_buffer[i][j][0]));
                weibias_buffer_mapped[i][j][1] = reinterpret_cast<T*>(xrtBOMap(weibias_buffer[i][j][1]));
            }
        }
        // for the last layernorm weights
        lnwb_buffer[NUM_LARYERS][0][0] = xrtBOAlloc(dhdl, EMB_DIM*sizeof(lndatatype), 0, 0);
        lnwb_buffer_mapped[NUM_LARYERS][0][0] = reinterpret_cast<lndatatype*>(xrtBOMap(lnwb_buffer[NUM_LARYERS][0][0]));
        // for the last layernorm bias
        lnwb_buffer[NUM_LARYERS][0][1] = xrtBOAlloc(dhdl, EMB_DIM*sizeof(lndatatype), 0, 0);
        lnwb_buffer_mapped[NUM_LARYERS][0][1] = reinterpret_cast<lndatatype*>(xrtBOMap(lnwb_buffer[NUM_LARYERS][0][1]));
        // for the last linear weights
        weibias_buffer[NUM_LARYERS][0][0] = xrtBOAlloc(dhdl, 768*1152*sizeof(T), 0, 0);
        weibias_buffer_mapped[NUM_LARYERS][0][0] = reinterpret_cast<T*>(xrtBOMap(weibias_buffer[NUM_LARYERS][0][0]));
         // for the last linear bias
        weibias_buffer[NUM_LARYERS][0][1] = xrtBOAlloc(dhdl, 1152*sizeof(T), 0, 0);
        weibias_buffer_mapped[NUM_LARYERS][0][1] = reinterpret_cast<T*>(xrtBOMap(weibias_buffer[NUM_LARYERS][0][1]));
        smi_buffer = xrtBOAlloc(dhdl, BATCH_SIZE*TOKENS_PADDED*TOKENS_PADDED*sizeof(smdatatype), 0, 0);
        smo_buffer = xrtBOAlloc(dhdl, BATCH_SIZE*TOKENS_PADDED*TOKENS_PADDED*sizeof(smdatatype), 0, 0);
        smi_buffer_mapped = reinterpret_cast<smdatatype*>(xrtBOMap(smi_buffer));
        smo_buffer_mapped = reinterpret_cast<smdatatype*>(xrtBOMap(smo_buffer));
        gli_buffer = xrtBOAlloc(dhdl, BATCH_SIZE*TOKENS_PADDED*3072*sizeof(gldatatype), 0, 0);
        glo_buffer = xrtBOAlloc(dhdl, BATCH_SIZE*TOKENS_PADDED*3072*sizeof(gldatatype), 0, 0);
        gli_buffer_mapped = reinterpret_cast<gldatatype*>(xrtBOMap(gli_buffer));
        glo_buffer_mapped = reinterpret_cast<gldatatype*>(xrtBOMap(glo_buffer));
        this->buffers_init = true;
    }
    void copy_to_buffers(const xt::xarray<T> &matA, const xt::xarray<T> &matB) {
        memcpy(buffer0_mapped, matA.data(), matA.size()*sizeof(T));
        memcpy(buffer1_mapped, matB.data(), matB.size()*sizeof(T));
        xrtBOSync(buffer0, XCL_BO_SYNC_BO_TO_DEVICE, matA.size()*sizeof(T), 0);
        xrtBOSync(buffer1, XCL_BO_SYNC_BO_TO_DEVICE, matB.size()*sizeof(T), 0);
    }
    void copy_to_buffers(const xt::xarray<T> &input, T* buffer_mapped, xrtBufferHandle buffer){
        memcpy(buffer_mapped, input.data(), input.size()*sizeof(T));
        xrtBOSync(buffer, XCL_BO_SYNC_BO_TO_DEVICE, input.size()*sizeof(T), 0);
    }
    std::tuple<const long unsigned int,  const long unsigned int, const long unsigned int> 
    preprocess_matrixA(xt::xarray<T> &matA, const bool batched = false) {
        auto ashape = xt::adapt(matA.shape());
        const long unsigned int rows = (batched) ? ashape[2] : ashape[1];
        const long unsigned int dp   = (batched) ? ashape[1] : ashape[0];
        const long unsigned int r_padded  = PaddedSize(rows, M);
        const long unsigned int dp_padded = PaddedSize(dp, K);
        if(r_padded != rows || dp_padded != dp) {
            matA = (batched) ? xt::pad(matA, {{0,0},{0,dp_padded-dp},{0,r_padded-rows}}, xt::pad_mode::constant, 0) : 
                               xt::pad(matA, {{0,dp_padded-dp},{0,r_padded-rows}}, xt::pad_mode::constant, 0);
        }
        return {rows, r_padded, dp_padded};
    }
    std::tuple< const long unsigned int,  const long unsigned int> 
    preprocess_matrixB(xt::xarray<T> &matB) {
        auto bshape = xt::adapt(matB.shape());
        const long unsigned int dp   = bshape[2];
        const long unsigned int cols = bshape[1];
        const long unsigned int dp_padded = PaddedSize(dp, K);
        const long unsigned int c_padded  = PaddedSize(cols, N);
        if(c_padded != cols || dp_padded != dp)
            {matB = xt::pad(matB, { {0,0}, {0,c_padded-cols}, {0,dp_padded-dp} }, xt::pad_mode::constant, 0);}
        return {cols, c_padded};
    }
    std::tuple< const long unsigned int,  const long unsigned int> 
    preprocess_bias(xt::xarray<T> &bias) {
        auto bshape = xt::adapt(bias.shape());
        const long unsigned int cols = bshape[1];
        const long unsigned int c_padded  = PaddedSize(cols, M);
        if(c_padded != cols)
            {bias = xt::pad(bias, { {0,0}, {0,c_padded-cols} }, xt::pad_mode::constant, 0);}
        return {cols, c_padded};
    }
    xt::xarray<T> postprocess_buffer(T *buffer_mapped, const long unsigned int rows, const long unsigned int cols, const long unsigned int r_padded, const long unsigned int c_padded){
        xt::xarray<T> out_pad = xt::empty<T>({BATCH_SIZE, c_padded, r_padded});
        xt::view(out_pad, 0, xt::all()) = xt::adapt(buffer_mapped, r_padded*c_padded, xt::no_ownership(), 
                                                    std::vector<std::size_t> {c_padded, r_padded});
        if(r_padded != rows || c_padded != cols)
            {out_pad = xt::view(out_pad, xt::all(), xt::range(_, cols), xt::range(_, rows));}
        return out_pad;
    }
    std::tuple<xt::xarray<T>, int, double> mm_aie(xt::xarray<T> &matA, xt::xarray<T> &matB, const bool batched = false) {
        tmr.start();
        auto [rows, r_padded, dp_padded] = this->preprocess_matrixA(matA, batched);
        auto [cols, c_padded] = this->preprocess_matrixB(matB);
        this->copy_to_buffers(matA, matB);
        double hw_time = this->mm_aie(buffer0, buffer1, buffer2, r_padded, dp_padded, c_padded);
        xt::xarray<T> out = this->postprocess_buffer(buffer2_mapped, rows, cols, r_padded, c_padded);
        tmr.stop();
        int ops = r_padded*c_padded*dp_padded*2;
        return {out, ops, hw_time};
    }

    double mm_aie(xrtBufferHandle matA, xrtBufferHandle matB, xrtBufferHandle matC, int r_padded, int dp_padded, int c_padded) {
        tmr.start();
        dma_rhdl = xrtKernelRun(dma_khdl, matA, matB, matC, this->zero_buffer,
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                r_padded/M, dp_padded/K, c_padded/N);
        xrtRunWait(dma_rhdl);
        tmr.stop();
        return tmr.get_time();
    }

    double mm_aie_small(xrtBufferHandle matA, xrtBufferHandle matB, xrtBufferHandle matC, int r_padded, int dp_padded, int c_padded, int head_idx) {
        tmr.start();
        dmas_rhdl = xrtKernelRun(dmas_khdl, matA, matB, matC,
                nullptr, nullptr, 
                nullptr, nullptr, 
                nullptr, nullptr, 
                nullptr, nullptr, 
                nullptr, nullptr, 
                nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                r_padded/M_SMALL, dp_padded/K_SMALL, c_padded/N_SMALL, head_idx);
        xrtRunWait(dmas_rhdl);
        tmr.stop();
        return tmr.get_time();
    }

    double linear(xrtBufferHandle matA, xrtBufferHandle matB, xrtBufferHandle matC, xrtBufferHandle bias, int r_padded, int dp_padded, int c_padded) {
        tmr.start();
        dma_rhdl = xrtKernelRun(dma_khdl, matA, matB, matC, bias,
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                nullptr, nullptr, nullptr, nullptr, 
                r_padded/M, dp_padded/K, c_padded/N);
        xrtRunWait(dma_rhdl);
        tmr.stop();
        return tmr.get_time();
    }

    std::tuple<xt::xarray<T>, xt::xarray<T>, double> layernorm(xt::xarray<lndatatype> &input, xt::xarray<lndatatype> &skip_in, 
                                                xt::xarray<lndatatype> &weight, xt::xarray<lndatatype> &bias) {
        memcpy(lni_buffer_mapped, input.data(), input.size()*sizeof(lndatatype));
        memcpy(lnsi_buffer_mapped, skip_in.data(), input.size()*sizeof(lndatatype));
        memcpy(lnwb_buffer_mapped[0][0][0], weight.data(), weight.size()*sizeof(lndatatype));
        memcpy(lnwb_buffer_mapped[0][0][1], bias.data(), bias.size()*sizeof(lndatatype));
        tmr.start();
        ln_rhdl = xrtKernelRun(ln_khdl, lni_buffer, lnsi_buffer, lnwb_buffer[0][0][0], lnwb_buffer[0][0][1], lno_buffer, lnso_buffer, BATCH_SIZE);
        xrtRunWait(ln_rhdl);
        tmr.stop();
        xt::xarray<lndatatype> out = xt::adapt(lno_buffer_mapped, input.size(), xt::no_ownership(), input.shape());      
        xt::xarray<lndatatype> skip_out = xt::adapt(lnso_buffer_mapped, input.size(), xt::no_ownership(), input.shape());      
        return {out, skip_out, tmr.get_time()};
    }

    double layernorm(xrtBufferHandle input, xrtBufferHandle skip_in, xrtBufferHandle weight, xrtBufferHandle bias, xrtBufferHandle out, xrtBufferHandle skip_out){
        tmr.start();
        ln_rhdl = xrtKernelRun(ln_khdl, input, skip_in, weight, bias, out, skip_out, BATCH_SIZE);
        xrtRunWait(ln_rhdl);
        tmr.stop();
        return tmr.get_time();
    }

    std::tuple<xt::xarray<T>, double> softmax(xt::xarray<T> &input){
        tmr.start();
        memcpy(smi_buffer_mapped, input.data(), input.size()*sizeof(smdatatype));
        sm_rhdl = xrtKernelRun(sm_khdl, smi_buffer, smo_buffer, BATCH_SIZE);
        xrtRunWait(sm_rhdl);
        xt::xarray<smdatatype> out = xt::adapt(smo_buffer_mapped, input.size(), xt::no_ownership(), input.shape());
        tmr.stop();
        return {out, tmr.get_time()};
    }

    double softmax(xrtBufferHandle input, xrtBufferHandle output) {
        tmr.start();
        sm_rhdl = xrtKernelRun(sm_khdl, input, output, BATCH_SIZE);
        xrtRunWait(sm_rhdl);
        tmr.stop();
        return tmr.get_time();       
    }

    std::tuple<xt::xarray<T>, double> gelu(xt::xarray<T> &input){
        memcpy(gli_buffer_mapped, input.data(), input.size()*sizeof(gldatatype));
        tmr.start();
        gl_rhdl = xrtKernelRun(gl_khdl, gli_buffer, glo_buffer, BATCH_SIZE);
        xrtRunWait(gl_rhdl); 
        tmr.stop();
        xt::xarray<gldatatype> out = xt::adapt(glo_buffer_mapped, input.size(), xt::no_ownership(), input.shape());  
        return {out, tmr.get_time()};
    }

    double gelu(xrtBufferHandle input_buffer, xrtBufferHandle output_buffer){
        tmr.start();
        gl_rhdl = xrtKernelRun(gl_khdl, input_buffer, output_buffer, BATCH_SIZE);
        xrtRunWait(gl_rhdl); 
        tmr.stop();
        return tmr.get_time();
    }

    double rant(xrtBufferHandle input, xrtBufferHandle qout, xrtBufferHandle kout, xrtBufferHandle vout, int head_idx) {
        tmr.start();
        rt_rhdl = xrtKernelRun(rt_khdl, input, input, input, qout, kout, vout, head_idx);
        xrtRunWait(rt_rhdl);
        tmr.stop();
        return tmr.get_time();
    }

    xt::xarray<T> mm_sw(xt::xarray<T> &matA, xt::xarray<T> &matB) {
        auto ashape = xt::adapt(matA.shape());
        auto bshape = xt::adapt(matB.shape());
        std::cout << "Software MM for A = " << bshape << " B = " << ashape << '\n';
        xt::xarray<T> output = xt::empty<T>({bshape[0], ashape[1]});
        for(long unsigned int i=0;i<bshape[0];i++) {
            for(long unsigned int j=0;j<ashape[1];j++) {
                T accumulator = 0;
                for(long unsigned int k=0;k<ashape[0];k++) {
                    accumulator += matB(i,k)*matA(k,j);
                }
                output(i,j) = accumulator;
            }
        }
        return output;
    }

    ~AieMMDriver() {
        xrtBOFree(buffer0);
        xrtBOFree(buffer1);
        xrtBOFree(buffer2);
        for(int i=0; i<NUM_LARYERS;i++){
            for(int j=0;j<2;j++) {
                xrtBOFree(lnwb_buffer[i][j][0]);
                xrtBOFree(lnwb_buffer[i][j][1]);
            }
            for(int j=0;j<4;j++){
                xrtBOFree(weibias_buffer[i][j][0]);
                xrtBOFree(weibias_buffer[i][j][1]);
            }
        }
        for(long unsigned int i=0; i<heads; i++) {
            xrtBOFree(qbuffer[i]);
            xrtBOFree(kbuffer[i]);
            xrtBOFree(vbuffer[i]);
            xrtBOFree(attn_scores[i]);
            xrtBOFree(attn_probs[i]);
        }
        xrtBOFree(lnwb_buffer[NUM_LARYERS][0][0]);
        xrtBOFree(lnwb_buffer[NUM_LARYERS][0][1]);
        xrtBOFree(weibias_buffer[NUM_LARYERS][0][0]);
        xrtBOFree(weibias_buffer[NUM_LARYERS][0][1]);
        xrtBOFree(lni_buffer);
        xrtBOFree(lnsi_buffer);
        xrtBOFree(lno_buffer);
        xrtBOFree(lnso_buffer);
        xrtBOFree(smi_buffer);
        xrtBOFree(smo_buffer);
        xrtBOFree(gli_buffer);
        xrtBOFree(glo_buffer);
        xrtKernelClose(dma_khdl);
        xrtRunClose(dma_rhdl);
        xrtKernelClose(ln_khdl);
        xrtRunClose(ln_rhdl);
        xrtKernelClose(sm_khdl);
        xrtRunClose(sm_rhdl);
        xrtKernelClose(gl_khdl);
        xrtRunClose(gl_rhdl);
        xrtKernelClose(rt_khdl);
        xrtRunClose(rt_rhdl);
        xrtKernelClose(dmas_khdl);
        xrtRunClose(dmas_rhdl);
        xrtDeviceClose(dhdl);
        mm_graph0.end();
        mm_graph1.end();
    }
};


#endif // MM_DRIVER_H


