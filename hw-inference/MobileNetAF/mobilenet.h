#ifndef MOBILENET_H
#define MOBILENET_H


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define IN_CHANNELS 1
#define F0_CHANNELS 8

#define F1_CHANNELS 8
#define F2_CHANNELS 12
#define F3_CHANNELS 12
#define F4_CHANNELS 16
#define F5_CHANNELS 16
#define F6_CHANNELS 16
#define F7_CHANNELS 24
#define F8_CHANNELS 24
#define F9_CHANNELS 24
#define F10_CHANNELS 24
// #define F11_CHANNELS 96
// #define F12_CHANNELS 96
// #define F13_CHANNELS 96
// #define F14_CHANNELS 160
// #define F15_CHANNELS 160
// #define F16_CHANNELS 160
// #define F17_CHANNELS 320
//#define F18_CHANNELS 1280
//#define F19_CHANNELS 1280
#define F19_CHANNELS 24
#define F20_CHANNELS 64

//�����˴�С
#define F0_KERNEL_L 7
#define PW_KERNEL_L 1
#define DW_KERNEL_L_1 7
#define DW_KERNEL_L_2 5
//#define F18_KERNEL_L 7
#define F19_KERNEL_L 1


#define PW_STRIDES 1
#define F0_STRIDES 2

#define F1_STRIDES 2
#define F2_STRIDES 2
#define F3_STRIDES 1
#define F4_STRIDES 2
#define F5_STRIDES 1
#define F6_STRIDES 1
#define F7_STRIDES 2
#define F8_STRIDES 1
#define F9_STRIDES 1
#define F10_STRIDES 1

#define F19_STRIDES 1

#define F0_PADDING 3
#define PW_PADDING 0
#define DW_PADDING_1 3
#define DW_PADDING_2 2
//#define F18_PADDING 0
#define F19_PADDING 0

#define F1_T 6
#define F2_T 6
#define F3_T 6
#define F4_T 6
#define F5_T 6
#define F6_T 6
#define F7_T 6
#define F8_T 6
#define F9_T 6
#define F10_T 6
// #define F11_T 6
// #define F12_T 6
// #define F13_T 6
// #define F14_T 6
// #define F15_T 6
// #define F16_T 6
// #define F17_T 6

#define IN_L 1000



#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))



typedef struct nonlinear_op {
    float* input;
    float* output;
    int units;

    int batchsize;
} nonlinear_op;

typedef struct conv_op {
    float* input;
    float* output;
    float* weights;
    float* bias;
    //float* input_col;

    int in_channels, out_channels;
    int kernel_size; int padding; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;
    int filter;
    int batchsize;
} conv_op;

typedef struct batch_norm_op {
    float* input;
    float* output;
    float* weight;
    float* bias;
    float* mean;
    float* var;
    int channels;
    int w,h;
    int units;
    float eps;
    int batchsize;

}batch_norm_op;

typedef struct res_connect_op {
    float* input;
    float* add_input;
    float* output;
    int units;
    int in_channels, out_channels;
}res_connect_op;


typedef struct bottleneck_op
{
    float* input;
    float* output;
    int stride;
    int in_channels, out_channels;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;
    int t;
    conv_op conv[3];
    batch_norm_op bn[3];
    nonlinear_op relu6[2];
    res_connect_op res_con;

}bottleneck_op;

typedef struct max_pooling_op {
    float* input;
    float* output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    int batchsize;
} max_pooling_op;

typedef struct avg_pooling_op {
    float* input;
    float* output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    int batchsize;
} avg_pooling_op;


typedef struct fc_op {
    float* input;
    float* output;
    float* weights;
    float* bias;
    int in_units, out_units;

    int batchsize;
} fc_op;


typedef struct mobilenet {
    float* input;
    float* output;
    int batchsize;

    conv_op f0_conv;
    batch_norm_op f0_bn;
    nonlinear_op f0_relu6;


    bottleneck_op f1;

    bottleneck_op f2;

    bottleneck_op f3;

    bottleneck_op f4;

    bottleneck_op f5;

    bottleneck_op f6;

    bottleneck_op f7;

    bottleneck_op f8;

    bottleneck_op f9;

    bottleneck_op f10;

    avg_pooling_op f19_ap;

    fc_op f20_fc;


}mobilenet;



static void relu_op_forward(nonlinear_op* op);

static void relu6_op_forward(nonlinear_op* op);

static void sigmoid_op_forward(nonlinear_op* op);


static void conv_op_forward(conv_op* op);

//static void conv_dw_op_forward(conv_op* op);

static void batch_norm_op_forward(batch_norm_op* op);

static void res_connect_op_forward(res_connect_op* op);

static void bottleneck_op_forward(bottleneck_op* op);

static void max_pooling_op_forward(max_pooling_op* op);

static void avg_pooling_op_forward(avg_pooling_op* op);

static void fc_op_forward(fc_op* op);


static void init_conv_op(conv_op* op, int in_channels, int out_channels, int stride, int padding, int kernel_size, int input_shape, bool is_dw);

static void init_bn_op(batch_norm_op* op, int channels, int shape);

static void init_bottleneck_op(bottleneck_op* op, int in_channels, int out_channels,int padding , int kernel ,int stride, int input_shape, int t);

static void init_avg_pool_op(avg_pooling_op* op, int channels, int stride, int kernel_size, int input_shape);

void setup_mobilenet(mobilenet* net);

void forward_mobilenet(mobilenet* net);
#endif
