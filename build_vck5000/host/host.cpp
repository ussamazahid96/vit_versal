#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <string>
#include "mm_driver.h"
#include "utils.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#define PADDED_M_MAX 3072
#define PADDED_K_MAX 3072
#define PADDED_N_MAX 3072
int main(int argc, char** argv) {
    char* xclbinFilename;
    if(argc == 2) {
        xclbinFilename = argv[1];
    }
    AieMMDriver<float, 1>* platform = new AieMMDriver<float, 1>(xclbinFilename);
    platform->allocate_buffers(PADDED_M_MAX, PADDED_K_MAX, PADDED_N_MAX);
    std::cout << "--------------------------------------------------- DataLoading ---------------------------------------------------" << '\n';
    // input
    xt::xarray<float> input_raw = xt::load_npy<float>("./vit/input_raw.npy"); 
    // embedding weights
    xt::xarray<float> emb_w = xt::load_npy<float>("./vit/embedding_layer/patch_embeddings_w.npy");
    xt::xarray<float> emb_b = xt::load_npy<float>("./vit/embedding_layer/patch_embeddings_b.npy");
    xt::xarray<float> pos_emb = xt::load_npy<float>("./vit/embedding_layer/positional_embeddings.npy");
    xt::xarray<float> cls_tokens = xt::load_npy<float>("./vit/embedding_layer/cls_tokens.npy");
    // encoder weights
    xt::xarray<float> qkv_w, attn_out_w, intermediate_w, output_w;    
    for(int i=0; i<NUM_LARYERS; i++) {
        // first layernorm
        xt::xarray<float> lnbw  = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/layernorm_before_w.npy");
        xt::xarray<float> lnbb  = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/layernorm_before_b.npy");
        // attention 
        qkv_w = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/qkv_w.npy");
        xt::xarray<float> qkv_b = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/qkv_b.npy");        
        // concat heads out
        attn_out_w = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/attn_out_w.npy");
        xt::xarray<float> attn_out_b = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/attn_out_b.npy");
        // second layernorm
        xt::xarray<float> lnaw = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/layernorm_after_w.npy");
        xt::xarray<float> lnab = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/layernorm_after_b.npy");
        // intermediate layer
        intermediate_w = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/intermediate_w.npy");
        xt::xarray<float> intermediate_b = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/intermediate_b.npy");
        // output layer
        output_w = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/output_w.npy");
        xt::xarray<float> output_b = xt::load_npy<float>("./vit/layer_" + std::to_string(i) + "/output_b.npy");
        platform->preprocess_matrixA(qkv_w);
        platform->preprocess_matrixA(attn_out_w);
        platform->preprocess_matrixA(intermediate_w);
        platform->preprocess_matrixA(output_w);
        platform->preprocess_bias(qkv_b);
        platform->preprocess_bias(attn_out_b);
        platform->preprocess_bias(intermediate_b);
        platform->preprocess_bias(output_b);
        platform->copy_to_buffers(lnbw, platform->lnwb_buffer_mapped[i][0][0], platform->lnwb_buffer[i][0][0]);
        platform->copy_to_buffers(lnbb, platform->lnwb_buffer_mapped[i][0][1], platform->lnwb_buffer[i][0][1]);
        platform->copy_to_buffers(lnaw, platform->lnwb_buffer_mapped[i][1][0], platform->lnwb_buffer[i][1][0]);
        platform->copy_to_buffers(lnab, platform->lnwb_buffer_mapped[i][1][1], platform->lnwb_buffer[i][1][1]);
        platform->copy_to_buffers(qkv_w, platform->weibias_buffer_mapped[i][0][0], platform->weibias_buffer[i][0][0]);
        platform->copy_to_buffers(qkv_b, platform->weibias_buffer_mapped[i][0][1], platform->weibias_buffer[i][0][1]);
        platform->copy_to_buffers(attn_out_w, platform->weibias_buffer_mapped[i][1][0], platform->weibias_buffer[i][1][0]);
        platform->copy_to_buffers(attn_out_b, platform->weibias_buffer_mapped[i][1][1], platform->weibias_buffer[i][1][1]);
        platform->copy_to_buffers(intermediate_w, platform->weibias_buffer_mapped[i][2][0], platform->weibias_buffer[i][2][0]);
        platform->copy_to_buffers(intermediate_b, platform->weibias_buffer_mapped[i][2][1], platform->weibias_buffer[i][2][1]);
        platform->copy_to_buffers(output_w, platform->weibias_buffer_mapped[i][3][0], platform->weibias_buffer[i][3][0]);
        platform->copy_to_buffers(output_b, platform->weibias_buffer_mapped[i][3][1], platform->weibias_buffer[i][3][1]);
    }
    // last layernorm
    xt::xarray<float> llnw = xt::load_npy<float>("./vit/last_layernorm/llnw.npy");
    xt::xarray<float> llnb = xt::load_npy<float>("./vit/last_layernorm/llnb.npy");
    // classifier weights
    xt::xarray<float> c_w  = xt::load_npy<float>("./vit/classifier/weight.npy");
    xt::xarray<float> c_b  = xt::load_npy<float>("./vit/classifier/bias.npy");
    platform->preprocess_matrixA(c_w);
    platform->preprocess_bias(c_b);
    platform->copy_to_buffers(llnw, platform->lnwb_buffer_mapped[NUM_LARYERS][0][0], platform->lnwb_buffer[NUM_LARYERS][0][0]);
    platform->copy_to_buffers(llnb, platform->lnwb_buffer_mapped[NUM_LARYERS][0][1], platform->lnwb_buffer[NUM_LARYERS][0][1]);
    platform->copy_to_buffers(c_w, platform->weibias_buffer_mapped[NUM_LARYERS][0][0], platform->weibias_buffer[NUM_LARYERS][0][0]);
    platform->copy_to_buffers(c_b, platform->weibias_buffer_mapped[NUM_LARYERS][0][1], platform->weibias_buffer[NUM_LARYERS][0][1]);
    auto shape1 = xt::adapt(qkv_w.shape());
    auto shape2 = xt::adapt(attn_out_w.shape());
    auto shape3 = xt::adapt(intermediate_w.shape());
    auto shape4 = xt::adapt(output_w.shape());
    auto shape5 = xt::adapt(c_w.shape());
    double mm_time = 0, ln_time = 0, sm_time = 0, gl_time = 0, rt_time = 0, OPS = 0;
    Timer tmr, tmr2;
    for(int i=0; i<NUM_LARYERS;i++) {
        OPS += (shape1[1]*shape1[0]*TOKENS_PADDED*2);
        OPS += (shape2[1]*shape2[0]*TOKENS_PADDED*2);
        OPS += (shape3[1]*shape3[0]*TOKENS_PADDED*2);
        OPS += (shape4[1]*shape4[0]*TOKENS_PADDED*2);
    }
    OPS += (shape5[1]*shape5[0]*TOKENS_PADDED*2);
    std::cout << "---------------------------------------------------- Inference ----------------------------------------------------" << '\n';
    tmr.start();
    xt::xarray<float> input = Embedding<16, 16, 0, 0>(input_raw, emb_w, emb_b, pos_emb, cls_tokens);
    platform->preprocess_matrixB(input);
    auto shape = xt::adapt(input.shape());
    xt::xarray<float> skip_conn = xt::zeros<float>(input.shape());    
    platform->copy_to_buffers(input, platform->lni_buffer_mapped, platform->lni_buffer);
    platform->copy_to_buffers(skip_conn, platform->lnsi_buffer_mapped, platform->lnsi_buffer);
    tmr.stop();
    std::cout << "Pre-Processing   time = " << tmr.get_time() <<  "s/image\n";
    tmr.start();
    for(int i=0; i<NUM_LARYERS; i++) {
        // first LayerNorm
        auto ln_time1 = platform->layernorm(platform->lni_buffer, platform->lnsi_buffer, 
                                            platform->lnwb_buffer[i][0][0], platform->lnwb_buffer[i][0][1], 
                                            platform->lno_buffer, platform->lnso_buffer);
        ln_time += ln_time1;
        // qkv
        auto qkvtime = platform->linear(platform->weibias_buffer[i][0][0], platform->lno_buffer, platform->buffer0, 
                                        platform->weibias_buffer[i][0][1], shape1[1], shape1[0], TOKENS_PADDED);
        mm_time += qkvtime;
        // Reshape and Transpose
        for(unsigned long int head_idx = 0; head_idx<heads; head_idx++) {           
            auto rttime = platform->rant(platform->buffer0, platform->qbuffer[head_idx], platform->kbuffer[head_idx], platform->vbuffer[head_idx], head_idx);
            rt_time += rttime;
        }
        // QK^T
        for(unsigned long int head_idx=0; head_idx<heads; head_idx++) {
            auto mmstime1 = platform->mm_aie_small(platform->kbuffer[head_idx], platform->qbuffer[head_idx], 
                                                platform->attn_scores[head_idx], TOKENS_PADDED, dim, TOKENS_PADDED, -1);
            mm_time += mmstime1;
            OPS += (TOKENS_PADDED*TOKENS_PADDED*dim*2);
        }
        // Softmax
        for(unsigned long int head_idx=0; head_idx<heads; head_idx++) {
            auto smtimeh = platform->softmax(platform->attn_scores[head_idx], platform->attn_probs[head_idx]); 
            sm_time += smtimeh;
        }
        // Self Attention 
        for(unsigned long int head_idx=0; head_idx<heads; head_idx++) {
            auto mmstime2 = platform->mm_aie_small(platform->vbuffer[head_idx], platform->attn_probs[head_idx], 
                                                   platform->buffer0, dim, TOKENS_PADDED, TOKENS_PADDED, head_idx);
            mm_time += mmstime2;
            OPS += (TOKENS_PADDED*TOKENS_PADDED*dim*2);
        }
        // Concat heads
        auto time6 = platform->linear(platform->weibias_buffer[i][1][0], platform->buffer0, platform->buffer1, 
                                      platform->weibias_buffer[i][1][1], shape2[1], shape2[0], TOKENS_PADDED);
        mm_time += time6;
        // Second LayerNorm
        auto ln_time2 = platform->layernorm(platform->buffer1, platform->lnso_buffer, 
                                            platform->lnwb_buffer[i][1][0], platform->lnwb_buffer[i][1][1], 
                                            platform->lno_buffer, platform->lnsi_buffer);  
        ln_time += ln_time2;
        // Intermediate output
        auto time7 = platform->linear(platform->weibias_buffer[i][2][0], platform->lno_buffer, platform->buffer0, 
                                      platform->weibias_buffer[i][2][1], shape3[1], shape3[0], TOKENS_PADDED);
        mm_time += time7;        
        // GeLU
        double gltime = platform->gelu(platform->buffer0, platform->glo_buffer);
        gl_time += gltime;
        // Output Layer
        auto time8 = platform->linear(platform->weibias_buffer[i][3][0], platform->glo_buffer, platform->lni_buffer, 
                                      platform->weibias_buffer[i][3][1], shape4[1], shape4[0], TOKENS_PADDED);
        mm_time += time8;
    }  
    // LayerNorm after Encoder
    auto ln_time3 = platform->layernorm(platform->lni_buffer, platform->lnsi_buffer, 
                                        platform->lnwb_buffer[NUM_LARYERS][0][0], platform->lnwb_buffer[NUM_LARYERS][0][1], 
                                        platform->lno_buffer, platform->lnso_buffer);
    ln_time += ln_time3;
    // Classifier Layer
    auto time9 = platform->linear(platform->weibias_buffer[NUM_LARYERS][0][0], platform->lno_buffer, platform->buffer0, 
                                  platform->weibias_buffer[NUM_LARYERS][0][1], shape5[1], shape5[0], TOKENS_PADDED);
    mm_time += time9;
    tmr.stop();
    // using the CLS token for classification
    xrtBOSync(platform->buffer0, XCL_BO_SYNC_BO_FROM_DEVICE , shape5[1]*TOKENS_PADDED*sizeof(float), 0);
    xt::xarray<float> output = xt::adapt(platform->buffer0_mapped, shape5[1]*TOKENS_PADDED, xt::no_ownership(), std::vector<std::size_t> {TOKENS_PADDED, shape5[1]});   
    output = xt::view(output, 0, xt::range(_,1000)); 
    xt::xarray<int> idx = xt::argmax(output);
    double total_time = tmr.get_time();
    double offload_time = mm_time + ln_time + sm_time + gl_time;
    double pp_time = total_time - offload_time;
    double GOPS = (OPS*1e-9)/mm_time;
    std::cout << "total            time = " << total_time << "s/image\n";
    std::cout << "offload          time = " << offload_time << "s/image\n";
    std::cout << "Processor        time = " << pp_time << "s/image\n";
    std::cout << "GOPs                  = " << GOPS << '\n';
    std::cout << "total Matrix Mul time = " << mm_time << "s/image\n";
    std::cout << "total LayerNorm  time = " << ln_time << "s/image\n";
    std::cout << "total Softmax    time = " << sm_time << "s/image\n";
    std::cout << "total GeLU       time = " << gl_time << "s/image\n";
    std::cout << "total Transpose  time = " << rt_time << "s/image\n";
    std::cout << "Predicted Class index = " << idx << '\n';
    xt::dump_npy("./vit/output_aie.npy", output);
    delete platform;
    return 0;
}
