#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "weights.h"
#include "mobilenet.h"
#include <time.h>

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

float array_ping[24000];
float array_pong[12000];
float array_add[1500];


void print_matrix(float* data, int num_elements, int offset){
    for (size_t i = 0; i < num_elements; ++i) {
        printf("Data [%ld] = %.2f\n", i+offset,  data[i+offset]);
    }
}



static void relu_op_forward(nonlinear_op* op)
{
    for (int i = 0; i < (op->units) * (op->batchsize); i++)
    {
        if (op->input[i] <= 0.0)
        {
            op->output[i] = 0.0;
        }
        else
        {
            op->output[i] = op->input[i];
        }
    }
}

static void relu6_op_forward(nonlinear_op* op)
{

    for (int i = 0; i < op->units; i++)
    {
        if (op->input[i] <= 0.0)
        {
            op->output[i] = 0.0;
        }
        else if (op->input[i] >= 6.0)
        {
            op->output[i] = 6.0;
        }
        else
        {
            op->output[i] = op->input[i];
        }
    }
}

static void sigmoid_op_forward(nonlinear_op* op)
{
    for (int i = 0; i < (op->units) * (op->batchsize); i++)
        op->output[i] = 1.0f/ (1.0f + exp(0.0f - op->input[i]));
}



static void conv_weights_and_bias(mobilenet* net, conv_op* op, float* conv2d, int conv2d_size, float* bias, int bias_size) {
    op->weights = conv2d;
    op->bias = bias;
    op->filter = conv2d_size + bias_size;
}



void initializeWeights(mobilenet* net) {

    // conv_weights_and_bias(net, &net->f0_conv , conv2d_1, 56, bias_1, 8);
    conv_weights_and_bias(net, &net->f0_conv , (float*)conv2d_1, sizeof(conv2d_1) / sizeof(float), bias_1, sizeof(bias_1) / sizeof(float));

    net->f0_bn.weight = batch_normalization_1_gamma;
    net->f0_bn.bias= batch_normalization_1_beta;
    net->f0_bn.mean= batch_normalization_1_mean;
    net->f0_bn.var= batch_normalization_1_variance;


    /* ------------------------------------------ First Inverted block ---------------------------------- */


    //conv_weights_and_bias(net, &net->f1.conv[0], conv2d_2, bias_2);
    conv_weights_and_bias(net, &net->f1.conv[0] , (float*)conv2d_2, sizeof(conv2d_2) / sizeof(float), bias_2, sizeof(bias_2) / sizeof(float));

    batch_norm_op* f1_bn_1 = &net->f1.bn[0];

    f1_bn_1->weight= batch_normalization_2_gamma;
    f1_bn_1->bias= batch_normalization_2_beta;
    f1_bn_1->mean= batch_normalization_2_mean;
    f1_bn_1->var= batch_normalization_2_variance;


    conv_weights_and_bias(net, &net->f1.conv[1] , (float*)depthwise_conv2d, sizeof(depthwise_conv2d) / sizeof(float),
                          depthwise_conv2d_bias, sizeof(depthwise_conv2d_bias) / sizeof(float));


    batch_norm_op* f1_bn_2 = &net->f1.bn[1];

    f1_bn_2->weight = batch_normalization_3_gamma;
    f1_bn_2->bias = batch_normalization_3_beta;
    f1_bn_2->mean = batch_normalization_3_mean;
    f1_bn_2->var = batch_normalization_3_variance;

    conv_weights_and_bias(net, &net->f1.conv[2] , (float*)conv2d_4, sizeof(conv2d_4) / sizeof(float), bias_4, sizeof(bias_4) / sizeof(float));


    batch_norm_op* f1_bn_3 = &net->f1.bn[2];

    f1_bn_3->weight = batch_normalization_4_gamma;
    f1_bn_3->bias = batch_normalization_4_beta;
    f1_bn_3->mean = batch_normalization_4_mean;
    f1_bn_3->var = batch_normalization_4_variance;

    //    /* ------------------------------------------ Second Inverted block ---------------------------------- */
    conv_weights_and_bias(net, &net->f2.conv[0] , (float*)conv2d_5, sizeof(conv2d_5) / sizeof(float), bias_5, sizeof(bias_5) / sizeof(float));
    batch_norm_op* f2_bn_1 = &net->f2.bn[0];
    f2_bn_1->weight = batch_normalization_5_gamma;
    f2_bn_1->bias = batch_normalization_5_beta;
    f2_bn_1->mean = batch_normalization_5_mean;
    f2_bn_1->var = batch_normalization_5_variance;


    conv_weights_and_bias(net, &net->f2.conv[1] , (float*)conv2d_6, sizeof(conv2d_6) / sizeof(float), bias_6, sizeof(bias_6) / sizeof(float));
    batch_norm_op* f2_bn_2 = &net->f2.bn[1];
    f2_bn_2->weight = batch_normalization_6_gamma;
    f2_bn_2->bias = batch_normalization_6_beta;
    f2_bn_2->mean = batch_normalization_6_mean;
    f2_bn_2->var = batch_normalization_6_variance;


    conv_weights_and_bias(net, &net->f2.conv[2] , (float*)conv2d_7, sizeof(conv2d_7) / sizeof(float), bias_7, sizeof(bias_7) / sizeof(float));
    batch_norm_op* f2_bn_3 = &net->f2.bn[2];
    f2_bn_3->weight = batch_normalization_7_gamma;
    f2_bn_3->bias = batch_normalization_7_beta;
    f2_bn_3->mean = batch_normalization_7_mean;
    f2_bn_3->var = batch_normalization_7_variance;


    //    /* ------------------------------------------ Third Inverted block ---------------------------------- */
    //
    conv_weights_and_bias(net, &net->f3.conv[0] , (float*)conv2d_8, sizeof(conv2d_8) / sizeof(float), bias_8, sizeof(bias_8) / sizeof(float));
    batch_norm_op* f3_bn_1 = &net->f3.bn[0];
    f3_bn_1->weight = batch_normalization_8_gamma;
    f3_bn_1->bias = batch_normalization_8_beta;
    f3_bn_1->mean = batch_normalization_8_mean;
    f3_bn_1->var = batch_normalization_8_variance;


    conv_weights_and_bias(net, &net->f3.conv[1] , (float*)conv2d_9, sizeof(conv2d_9) / sizeof(float), bias_9, sizeof(bias_9) / sizeof(float));
    batch_norm_op* f3_bn_2 = &net->f3.bn[1];
    f3_bn_2->weight = batch_normalization_9_gamma;
    f3_bn_2->bias = batch_normalization_9_beta;
    f3_bn_2->mean = batch_normalization_9_mean;
    f3_bn_2->var = batch_normalization_9_variance;


    conv_weights_and_bias(net, &net->f3.conv[2] , (float*)conv2d_10, sizeof(conv2d_10) / sizeof(float), bias_10, sizeof(bias_10) / sizeof(float));
    batch_norm_op* f3_bn_3 = &net->f3.bn[2];
    f3_bn_3->weight = batch_normalization_10_gamma;
    f3_bn_3->bias = batch_normalization_10_beta;
    f3_bn_3->mean = batch_normalization_10_mean;
    f3_bn_3->var = batch_normalization_10_variance;


    //         /* ------------------------------------------ Forth Inverted block ---------------------------------- */


    conv_weights_and_bias(net, &net->f4.conv[0] , (float*)conv2d_11, sizeof(conv2d_11) / sizeof(float), bias_11, sizeof(bias_11) / sizeof(float));
    batch_norm_op* f4_bn_1 = &net->f4.bn[0];
    f4_bn_1->weight = batch_normalization_11_gamma;
    f4_bn_1->bias = batch_normalization_11_beta;
    f4_bn_1->mean = batch_normalization_11_mean;
    f4_bn_1->var = batch_normalization_11_variance;


    conv_weights_and_bias(net, &net->f4.conv[1] , (float*)conv2d_12, sizeof(conv2d_12) / sizeof(float), bias_12, sizeof(bias_12) / sizeof(float));
    batch_norm_op* f4_bn_2 = &net->f4.bn[1];
    f4_bn_2->weight = batch_normalization_12_gamma;
    f4_bn_2->bias = batch_normalization_12_beta;
    f4_bn_2->mean = batch_normalization_12_mean;
    f4_bn_2->var = batch_normalization_12_variance;


    conv_weights_and_bias(net, &net->f4.conv[2] , (float*)conv2d_13, sizeof(conv2d_13) / sizeof(float), bias_13, sizeof(bias_13) / sizeof(float));
    batch_norm_op* f4_bn_3 = &net->f4.bn[2];
    f4_bn_3->weight = batch_normalization_13_gamma;
    f4_bn_3->bias = batch_normalization_13_beta;
    f4_bn_3->mean = batch_normalization_13_mean;
    f4_bn_3->var = batch_normalization_13_variance;

    //     /* ------------------------------------------ Fifth Inverted block ---------------------------------- */

    conv_weights_and_bias(net, &net->f5.conv[0] , (float*)conv2d_14, sizeof(conv2d_14) / sizeof(float), bias_14, sizeof(bias_14) / sizeof(float));
    batch_norm_op* f5_bn_1 = &net->f5.bn[0];
    f5_bn_1->weight = batch_normalization_14_gamma;
    f5_bn_1->bias = batch_normalization_14_beta;
    f5_bn_1->mean = batch_normalization_14_mean;
    f5_bn_1->var = batch_normalization_14_variance;


    conv_weights_and_bias(net, &net->f5.conv[1] , (float*)conv2d_15, sizeof(conv2d_15) / sizeof(float), bias_15, sizeof(bias_15) / sizeof(float));
    batch_norm_op* f5_bn_2 = &net->f5.bn[1];
    f5_bn_2->weight = batch_normalization_15_gamma;
    f5_bn_2->bias = batch_normalization_15_beta;
    f5_bn_2->mean = batch_normalization_15_mean;
    f5_bn_2->var = batch_normalization_15_variance;


    conv_weights_and_bias(net, &net->f5.conv[2] , (float*)conv2d_16, sizeof(conv2d_16) / sizeof(float), bias_16, sizeof(bias_16) / sizeof(float));
    batch_norm_op* f5_bn_3 = &net->f5.bn[2];
    f5_bn_3->weight = batch_normalization_16_gamma;
    f5_bn_3->bias = batch_normalization_16_beta;
    f5_bn_3->mean = batch_normalization_16_mean;
    f5_bn_3->var = batch_normalization_16_variance;


    //     /* ------------------------------------------ Sixth Inverted block ---------------------------------- */

    conv_weights_and_bias(net, &net->f6.conv[0] , (float*)conv2d_17, sizeof(conv2d_17) / sizeof(float), bias_17, sizeof(bias_17) / sizeof(float));
    batch_norm_op* f6_bn_1 = &net->f6.bn[0];
    f6_bn_1->weight = batch_normalization_17_gamma;
    f6_bn_1->bias = batch_normalization_17_beta;
    f6_bn_1->mean = batch_normalization_17_mean;
    f6_bn_1->var = batch_normalization_17_variance;

    conv_weights_and_bias(net, &net->f6.conv[1] , (float*)conv2d_18, sizeof(conv2d_18) / sizeof(float), bias_18, sizeof(bias_18) / sizeof(float));
    batch_norm_op* f6_bn_2 = &net->f6.bn[1];
    f6_bn_2->weight = batch_normalization_18_gamma;
    f6_bn_2->bias = batch_normalization_18_beta;
    f6_bn_2->mean = batch_normalization_18_mean;
    f6_bn_2->var = batch_normalization_18_variance;

    conv_weights_and_bias(net, &net->f6.conv[2] , (float*)conv2d_19, sizeof(conv2d_19) / sizeof(float), bias_19, sizeof(bias_19) / sizeof(float));
    batch_norm_op* f6_bn_3 = &net->f6.bn[2];
    f6_bn_3->weight = batch_normalization_19_gamma;
    f6_bn_3->bias = batch_normalization_19_beta;
    f6_bn_3->mean = batch_normalization_19_mean;
    f6_bn_3->var = batch_normalization_19_variance;



   //     /* ------------------------------------------ Seventh Inverted block ---------------------------------- */

   conv_weights_and_bias(net, &net->f7.conv[0], (float*)conv2d_20, sizeof(conv2d_20) / sizeof(float), bias_20, sizeof(bias_20) / sizeof(float));
   batch_norm_op* f7_bn_1 = &net->f7.bn[0];
   f7_bn_1->weight = batch_normalization_20_gamma;
   f7_bn_1->bias = batch_normalization_20_beta;
   f7_bn_1->mean = batch_normalization_20_mean;
   f7_bn_1->var = batch_normalization_20_variance;

   conv_weights_and_bias(net, &net->f7.conv[1], (float*)conv2d_21, sizeof(conv2d_21) / sizeof(float), bias_21, sizeof(bias_21) / sizeof(float));
   batch_norm_op* f7_bn_2 = &net->f7.bn[1];
   f7_bn_2->weight = batch_normalization_21_gamma;
   f7_bn_2->bias = batch_normalization_21_beta;
   f7_bn_2->mean = batch_normalization_21_mean;
   f7_bn_2->var = batch_normalization_21_variance;

   conv_weights_and_bias(net, &net->f7.conv[2], (float*)conv2d_22, sizeof(conv2d_22) / sizeof(float), bias_22, sizeof(bias_22) / sizeof(float));
   batch_norm_op* f7_bn_3 = &net->f7.bn[2];
   f7_bn_3->weight = batch_normalization_22_gamma;
   f7_bn_3->bias = batch_normalization_22_beta;
   f7_bn_3->mean = batch_normalization_22_mean;
   f7_bn_3->var = batch_normalization_22_variance;


   //     /* ------------------------------------------ Eighth Inverted block ---------------------------------- */

   conv_weights_and_bias(net, &net->f8.conv[0], (float*)conv2d_23, sizeof(conv2d_23) / sizeof(float), bias_23, sizeof(bias_23) / sizeof(float));
   batch_norm_op* f8_bn_1 = &net->f8.bn[0];
   f8_bn_1->weight = batch_normalization_23_gamma;
   f8_bn_1->bias = batch_normalization_23_beta;
   f8_bn_1->mean = batch_normalization_23_mean;
   f8_bn_1->var = batch_normalization_23_variance;

   conv_weights_and_bias(net, &net->f8.conv[1], (float*)conv2d_24, sizeof(conv2d_24) / sizeof(float), bias_24, sizeof(bias_24) / sizeof(float));
   batch_norm_op* f8_bn_2 = &net->f8.bn[1];
   f8_bn_2->weight = batch_normalization_24_gamma;
   f8_bn_2->bias = batch_normalization_24_beta;
   f8_bn_2->mean = batch_normalization_24_mean;
   f8_bn_2->var = batch_normalization_24_variance;

   conv_weights_and_bias(net, &net->f8.conv[2], (float*)conv2d_25, sizeof(conv2d_25) / sizeof(float), bias_25, sizeof(bias_25) / sizeof(float));
   batch_norm_op* f8_bn_3 = &net->f8.bn[2];
   f8_bn_3->weight = batch_normalization_25_gamma;
   f8_bn_3->bias = batch_normalization_25_beta;
   f8_bn_3->mean = batch_normalization_25_mean;
   f8_bn_3->var = batch_normalization_25_variance;


   //     /* ------------------------------------------ Ninth Inverted block ---------------------------------- */

   conv_weights_and_bias(net, &net->f9.conv[0], (float*)conv2d_26, sizeof(conv2d_26) / sizeof(float), bias_26, sizeof(bias_26) / sizeof(float));
   batch_norm_op* f9_bn_1 = &net->f9.bn[0];
   f9_bn_1->weight = batch_normalization_26_gamma;
   f9_bn_1->bias = batch_normalization_26_beta;
   f9_bn_1->mean = batch_normalization_26_mean;
   f9_bn_1->var = batch_normalization_26_variance;

   conv_weights_and_bias(net, &net->f9.conv[1], (float*)conv2d_27, sizeof(conv2d_27) / sizeof(float), bias_27, sizeof(bias_27) / sizeof(float));
   batch_norm_op* f9_bn_2 = &net->f9.bn[1];
   f9_bn_2->weight = batch_normalization_27_gamma;
   f9_bn_2->bias = batch_normalization_27_beta;
   f9_bn_2->mean = batch_normalization_27_mean;
   f9_bn_2->var = batch_normalization_27_variance;

   conv_weights_and_bias(net, &net->f9.conv[2], (float*)conv2d_28, sizeof(conv2d_28) / sizeof(float), bias_28, sizeof(bias_28) / sizeof(float));
   batch_norm_op* f9_bn_3 = &net->f9.bn[2];
   f9_bn_3->weight = batch_normalization_28_gamma;
   f9_bn_3->bias = batch_normalization_28_beta;
   f9_bn_3->mean = batch_normalization_28_mean;
   f9_bn_3->var = batch_normalization_28_variance;

   //     /* ------------------------------------------ Tenth Inverted block ---------------------------------- */

   conv_weights_and_bias(net, &net->f10.conv[0], (float*)conv2d_29, sizeof(conv2d_29) / sizeof(float), bias_29, sizeof(bias_29) / sizeof(float));
   batch_norm_op* f10_bn_1 = &net->f10.bn[0];
   f10_bn_1->weight = batch_normalization_29_gamma;
   f10_bn_1->bias = batch_normalization_29_beta;
   f10_bn_1->mean = batch_normalization_29_mean;
   f10_bn_1->var = batch_normalization_29_variance;

   conv_weights_and_bias(net, &net->f10.conv[1], (float*)conv2d_30, sizeof(conv2d_30) / sizeof(float), bias_30, sizeof(bias_30) / sizeof(float));
   batch_norm_op* f10_bn_2 = &net->f10.bn[1];
   f10_bn_2->weight = batch_normalization_30_gamma;
   f10_bn_2->bias = batch_normalization_30_beta;
   f10_bn_2->mean = batch_normalization_30_mean;
   f10_bn_2->var = batch_normalization_30_variance;

   conv_weights_and_bias(net, &net->f10.conv[2], (float*)conv2d_31, sizeof(conv2d_31) / sizeof(float), bias_31, sizeof(bias_31) / sizeof(float));
   batch_norm_op* f10_bn_3 = &net->f10.bn[2];
   f10_bn_3->weight = batch_normalization_31_gamma;
   f10_bn_3->bias = batch_normalization_31_beta;
   f10_bn_3->mean = batch_normalization_31_mean;
   f10_bn_3->var = batch_normalization_31_variance;

   //     /* ------------------------------------------ Dense Layer ---------------------------------- */

   net->f20_fc.weights = Dense;
   net->f20_fc.bias = bias_Dense;
}

static void conv_op_forward(conv_op* op)
{
    op->batchsize = 1;
    float* input_p;
    int s = op->stride;
    int p = op->padding;
    int iw = op->in_w;
    int ih = op->in_h;
    int iwih = iw * ih;
    //int iwih = iw * ih;
    int iw1 = iw + p ; //strid = 2
    //int ih1 = ih;
    int iwih1 = iw1 ;
    int owoh  = op->out_w * 1;
    //int owoh  = op->out_w * op->out_h;
    int k = op->kernel_size;
    //int kk = op->kernel_size * kernel_size;
    int ikk = op->in_channels * k;
    //int ikk = op->in_channels * kk;
    int i_iwih = op->in_channels * iwih;
    int i_iwih1 = op->in_channels * iwih1;

    float sum;
    float mult;
    int pad = op->padding;
    int initial_start_index = pad%2;
    for (int w_n = 0; w_n < op->out_channels; w_n++) {
        for (int start_index = initial_start_index; start_index < op->in_w; start_index += op->stride) {
            sum = 0;
            for (int w_i = 0; w_i < op->kernel_size; w_i++) {
                for (int w_j = 0; w_j < op->in_channels; w_j++) {
                    if (start_index + w_i - pad < op->in_w && start_index + w_i - pad > -1) {
                        mult = mem3d(op->weights, op->kernel_size, op->in_channels, w_n, w_j, w_i) *
                                mem2d(op->input, op->in_w, w_j, start_index + w_i - pad);
                        sum += mult;
                    }
                }
            }
            sum += op->bias[w_n];
            mem2d(op->output, op->out_w, w_n, start_index / op->stride) = sum;
        }
    }

}

static void conv_dw_op_forward(conv_op* op)
{
    //op->batchsize = 1;
    float* input_p;
    int S = op->stride;
    int P = op->padding;
    int iw = op->in_w;
    int ih = 1;
    int iwih = iw * ih;
    int iw1 = iw + 2 * P;
    int ih1 = 1 ;
    int iwih1 = iw1 * ih1;
    int owoh = op->out_w;
    int k = op->kernel_size;
    //int kk = op->kernel_size * kernel_size;
    int ikk = op->in_channels * k;
    int i_iwih = op->in_channels * iwih;
    int i_iwih1 = op->in_channels * iwih1;


    float sum;
    float mult;
    int pad = op->padding;
    int initial_start_index = pad % (op->stride);
    for (int start_index = initial_start_index; start_index < op->in_w; start_index += op->stride) {
        for (int w_j = 0; w_j < op->in_channels; w_j++) {
            sum = 0;
            for (int w_i = 0; w_i < op->kernel_size; w_i++) {
                if (start_index + w_i - pad < op->in_w && start_index + w_i - pad > -1) {
                    mult = mem3d(op->weights, op->kernel_size, op->in_channels, 0, w_j, w_i) *
                            mem2d(op->input, op->in_w, w_j, start_index + w_i - pad);
                    sum += mult;
                }
            }

            sum += op->bias[w_j];
            mem2d(op->output, op->out_w, w_j, start_index / op->stride) = sum;
        }
    }

}

static void batch_norm_op_forward(batch_norm_op* op)
{
    int w = op->w;
    int h = 1;
    int wh = w * h;
    int c = op->channels;

    for (int i = 0; i < c; i++)
    {
        for (int j = 0; j < wh; j++)
        {
            op->output[i * wh + j] = op->weight[i] * (op->input[i * wh + j] - op->mean[i]) / (sqrt(op->var[i] + op->eps)) + op->bias[i];
        }
    }

}

static void res_connect_op_forward(res_connect_op* op)
{

    int units = op->units;
    for (int i = 0; i < units; i++)
    {
        op->output[i] = op->input[i] + op->add_input[i];
    }

}

static void bottleneck_op_forward(bottleneck_op* op)
{
    printf("############################Bottleneck#############################\n");

    if (op->stride == 1 && op->in_units == op->out_units)  // Keep the input for the skip connection
        {
        for (int i=0; i <op->out_units; i++){
            array_add[i] = op->input[i];
        }
        }

 
    if (op->t > 1)
    {
        op->conv[0].output = array_ping;
        op->conv[0].input = op->input;
        conv_op_forward(&(op->conv[0]));

        op->bn[0].input = op->conv[0].output;
        op->bn[0].output = op->conv[0].output;
        batch_norm_op_forward(&(op->bn[0]));

        op->relu6[0].input = op->bn[0].output;
        op->relu6[0].output = op->bn[0].output;
        relu6_op_forward(&(op->relu6[0]));

        op->conv[1].input = op->relu6[0].output;
    }
    else op->conv[1].input = op->input;
    //dw
    op->conv[1].output = array_pong;
    conv_dw_op_forward(&(op->conv[1]));

    op->bn[1].input = op->conv[1].output;
    op->bn[1].output = op->conv[1].output;
    batch_norm_op_forward(&(op->bn[1]));

    op->relu6[1].input = op->bn[1].output;
    op->relu6[1].output = op->bn[1].output;
    relu6_op_forward(&(op->relu6[1]));

    //pw
    op->conv[2].input = op->relu6[1].output;
    op->conv[2].output = array_ping;
    conv_op_forward(&(op->conv[2]));

    op->bn[2].input = op->conv[2].output;
    op->bn[2].output = array_pong;
    batch_norm_op_forward(&(op->bn[2]));


    if (op->stride == 1 && op->in_units == op->out_units)
    {
        op->res_con.add_input = array_add;
        op->res_con.input = op->bn[2].output;
        op->res_con.output = op->bn[2].output;
        res_connect_op_forward(&(op->res_con));
        op->output = op->res_con.output;
    }

    else op->output = op->bn[2].output;


}


static void avg_pooling_op_forward(avg_pooling_op* op)
{
    int channels = op->channels;
 
    printf("############################AVG Pooling############################\n");
    float mean;
    float sum;
    for (int c = 0; c < channels; c++)
    {
        mean = 0.0;
        sum = 0.0;
        for (int j = 0; j < op->in_w; j++)
        {
            sum += mem2d(op->input, op->in_w, c, j);
        }
        mean = sum / (float) op->in_w;
        op->output[c] = mean;
    }
}

static void fc_op_forward(fc_op* op)
{
    float sum = 0.0;
    for (int i = 0; i < op->out_units; i++)
    {
        sum = 0.0;
        for (int j = 0; j < op->in_units; j++)
        {
            sum += op->input[j] * op->weights[i * op->in_units + j];
        }
        op->output[i]  = sum;
    }

    for (int i = 0; i < op->out_units; i++){
        op->output[i] += op->bias[i];
        op->output[i] =  (op->output[i] > 0)? op->output[i] : 0;
    }
}




static void init_conv_op(conv_op* op,int in_channels,int out_channels, int stride, int padding, int kernel_size, int input_shape,bool is_dw)
{
    //op->batchsize = 10;
    op->in_channels = in_channels;
    op->out_channels = out_channels;
    op->stride = stride;
    op->padding = padding;
    op->kernel_size = kernel_size;
    op->in_w = input_shape;
    op->in_h = 1;
    int output_shape = (input_shape - kernel_size + 2 * padding) / stride + 1;
    op->out_w = output_shape;
    //op->out_h = output_shape;
    op->out_h = 1 ;
    //op->in_units = input_shape * input_shape * in_channels;
    op->in_units = input_shape * 1 * in_channels;
    //op->out_units = output_shape * output_shape * out_channels;
    op->out_units = output_shape * 1 * out_channels;
    
    if (!is_dw) op->filter = kernel_size * 1 * in_channels * out_channels;
    else op->filter = kernel_size * 1 * out_channels;
}

static void init_bn_op(batch_norm_op* op, int channels, int shape)
{
    op->batchsize = 1;
    op->w = shape;
    op->h = 1;
    op->channels = channels;
    op->units = shape * 1 * channels;
    op->eps = 1e-3;

}

static void init_bottleneck_op(bottleneck_op* op, int in_channels, int out_channels,int padding , int kernel ,int stride, int input_shape, int t) 
{
    op->in_channels = in_channels;
    op->out_channels = out_channels;
    op->in_w = input_shape;
    op->in_h = 1;
    op->stride = stride;
    op->t = t;
    if (t > 1)
    {
        init_conv_op(&(op->conv[0]), in_channels, in_channels * t, PW_STRIDES, PW_PADDING, PW_KERNEL_L, input_shape,false);
        init_bn_op(&(op->bn[0]), op->conv[0].out_channels, op->conv[0].out_w);
        op->relu6[0].units = op->bn[0].units;
        init_conv_op(&(op->conv[1]), in_channels * t, in_channels * t, stride, padding, kernel, op->conv[0].out_w,true);
    }
    else init_conv_op(&(op->conv[1]), in_channels * t, in_channels * t, stride, padding, kernel, input_shape,true);
    
    init_bn_op(&(op->bn[1]), op->conv[1].out_channels, op->conv[1].out_w);
    op->relu6[1].units = op->bn[1].units;

    init_conv_op(&(op->conv[2]), in_channels * t, out_channels, PW_STRIDES, PW_PADDING, PW_KERNEL_L, op->conv[1].out_w,false);
    init_bn_op(&(op->bn[2]), op->conv[2].out_channels, op->conv[2].out_w);

    int output_shape = op->conv[2].out_w;
    op->out_w = output_shape;
    op->out_h = 1;
    //op->in_units = input_shape * input_shape * in_channels;
    op->in_units = input_shape * 1 * in_channels;
    op->out_units = output_shape * 1 * out_channels;
    op->res_con.units = op->in_units;

}

static void init_avg_pool_op(avg_pooling_op* op, int channels, int stride, int kernel_size, int input_shape)
{
    op->batchsize = 1;
    op->channels = channels;
    op->kernel_size = kernel_size;
    op->stride = stride;
    
    op->in_w = input_shape;
    op->in_h = 1;
    int output_shape = (input_shape - kernel_size) / stride + 1;
    op->out_w = output_shape;
    op->out_h = 1;
    op->in_units = input_shape * 1 * channels;
    op->out_units = output_shape * 1 * channels;
}

void setup_mobilenet(mobilenet* net)
{
    net->batchsize = 1;
    init_conv_op(&(net->f0_conv), IN_CHANNELS, F0_CHANNELS, F0_STRIDES, F0_PADDING, F0_KERNEL_L, IN_L,false);
    init_bn_op(&(net->f0_bn), net->f0_conv.out_channels, net->f0_conv.out_w);
    net->f0_relu6.units = net->f0_bn.units;

    init_bottleneck_op(&(net->f1), F0_CHANNELS, F1_CHANNELS,DW_PADDING_1, DW_KERNEL_L_1 ,F1_STRIDES ,net->f0_conv.out_w, F1_T);

    init_bottleneck_op(&(net->f2), F1_CHANNELS, F2_CHANNELS,DW_PADDING_1, DW_KERNEL_L_1 , F2_STRIDES, net->f1.out_w,F2_T);

    init_bottleneck_op(&(net->f3), F2_CHANNELS, F3_CHANNELS,DW_PADDING_1, DW_KERNEL_L_1 , F3_STRIDES, net->f2.out_w, F3_T);

    init_bottleneck_op(&(net->f4), F3_CHANNELS, F4_CHANNELS,DW_PADDING_2, DW_KERNEL_L_2 , F4_STRIDES, net->f3.out_w, F4_T);
    
    init_bottleneck_op(&(net->f5), F4_CHANNELS, F5_CHANNELS,DW_PADDING_2, DW_KERNEL_L_2 , F5_STRIDES, net->f4.out_w, F5_T);
    
    init_bottleneck_op(&(net->f6), F5_CHANNELS, F6_CHANNELS,DW_PADDING_2, DW_KERNEL_L_2 , F6_STRIDES, net->f5.out_w, F6_T);
    
   init_bottleneck_op(&(net->f7), F6_CHANNELS, F7_CHANNELS,DW_PADDING_2, DW_KERNEL_L_2 , F7_STRIDES, net->f6.out_w, F7_T);

   init_bottleneck_op(&(net->f8), F7_CHANNELS, F8_CHANNELS,DW_PADDING_2, DW_KERNEL_L_2 , F8_STRIDES, net->f7.out_w, F8_T);

   init_bottleneck_op(&(net->f9), F8_CHANNELS, F9_CHANNELS,DW_PADDING_2, DW_KERNEL_L_2 , F9_STRIDES, net->f8.out_w, F9_T);

   init_bottleneck_op(&(net->f10), F9_CHANNELS, F10_CHANNELS,DW_PADDING_2, DW_KERNEL_L_2 , F10_STRIDES, net->f9.out_w, F10_T);

   init_avg_pool_op(&(net->f19_ap), F19_CHANNELS, F19_STRIDES, F19_KERNEL_L, net->f10.out_w);

   net->f20_fc.batchsize = 1;
   net->f20_fc.in_units = F19_CHANNELS;
   net->f20_fc.out_units = F20_CHANNELS;

}


void forward_mobilenet(mobilenet* net)
{   

    // --- Timing variables for each stage ---
    clock_t initial_conv_start, initial_conv_end;
    clock_t bottleneck_start, bottleneck_end;
    clock_t classifier_start, classifier_end;
    double initial_conv_time, bottleneck_time, classifier_time;

    int t = 0;
    initializeWeights(net);

    //=================================================
    // STAGE 1: Initial Convolution
    //=================================================
    initial_conv_start = clock();


    net->f0_conv.input = net->input;
    net->f0_conv.output = array_pong;

    conv_op_forward(&(net->f0_conv));
    net->f0_bn.input = net->f0_conv.output;
    net->f0_bn.output = net->f0_conv.output;

    batch_norm_op_forward(&(net->f0_bn));

    net->f0_relu6.input = net->f0_bn.output;
    net->f0_relu6.output = net->f0_bn.output;
    relu6_op_forward(&(net->f0_relu6));

    initial_conv_end = clock();

    //=================================================
    // STAGE 2: Bottleneck Layers
    //=================================================
    bottleneck_start = clock();

    net->f1.input = net->f0_relu6.output;
    bottleneck_op_forward(&(net->f1));

    net->f2.input = net->f1.output;
    bottleneck_op_forward(&(net->f2));

    net->f3.input = net->f2.output;
    bottleneck_op_forward(&(net->f3));

    net->f4.input = net->f3.output;
    bottleneck_op_forward(&(net->f4));

    net->f5.input = net->f4.output;
    bottleneck_op_forward(&(net->f5));

    net->f6.input = net->f5.output;
    bottleneck_op_forward(&(net->f6));

   net->f7.input = net->f6.output;
   bottleneck_op_forward(&(net->f7));

   net->f8.input = net->f7.output;
   bottleneck_op_forward(&(net->f8));

   net->f9.input = net->f8.output;
   bottleneck_op_forward(&(net->f9));

   net->f10.input = net->f9.output;
   bottleneck_op_forward(&(net->f10));

   
   bottleneck_end = clock();

   //=================================================
   // STAGE 3: Classifier Head
   //=================================================
   classifier_start = clock();


   net->f19_ap.input = net->f10.output;
   net->f19_ap.output = array_add;
   avg_pooling_op_forward(&(net->f19_ap));

   t = net->batchsize * net->f20_fc.out_units;
   net->f20_fc.input = net->f19_ap.output;
   net->f20_fc.output = array_pong;

   fc_op_forward(&(net->f20_fc));
   classifier_end = clock();


   net->output = net->f20_fc.output;
    print_matrix( net->f6.output, 20, 0);

    //=================================================
    // CALCULATE AND PRINT RESULTS
    //=================================================
    initial_conv_time = ((double)(initial_conv_end - initial_conv_start)) / CLOCKS_PER_SEC;
    bottleneck_time = ((double)(bottleneck_end - bottleneck_start)) / CLOCKS_PER_SEC;
    classifier_time = ((double)(classifier_end - classifier_start)) / CLOCKS_PER_SEC;

    printf("\n--- MobileNet Performance Breakdown ---\n");
    printf("Stage 1 (Initial Conv) Time: %f seconds\n", initial_conv_time);
    printf("Stage 2 (Bottlenecks) Time:  %f seconds\n", bottleneck_time);
    printf("Stage 3 (Classifier) Time:   %f seconds\n", classifier_time);
    printf("---------------------------------------\n");
    printf("Total Inference Time:        %f seconds\n", initial_conv_time + bottleneck_time + classifier_time);
    printf("---------------------------------------\n");

}
