#ifndef DATA_H
#define DATA_H

#include <stdint.h>

//========= Global Embeddings and Tokens =========//
extern int16_t pos_embedding[1936];
extern int16_t cls_token[16];

//========= Patch Embedding Layers =========//
extern int16_t to_patch_embedding_layer_norm1_weight[400];
extern int16_t to_patch_embedding_layer_norm1_bias[400];
extern int16_t to_patch_embedding_linear_weight[6400];
extern int16_t to_patch_embedding_linear_bias[16];
extern int16_t to_patch_embedding_layer_norm2_weight[16];
extern int16_t to_patch_embedding_layer_norm2_bias[16];

//========= Transformer Layer 0 =========//
// Attention Block
extern int16_t transformer_layers_0_0_norm_weight[16];
extern int16_t transformer_layers_0_0_norm_bias[16];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_Q_H0[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_Q_H1[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_Q_H2[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_Q_H3[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_K_H0[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_K_H1[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_K_H2[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_K_H3[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_V_H0[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_V_H1[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_V_H2[64];
extern int16_t transformer_layers_0_0_fn_to_qkv_weight_V_H3[64];
extern int16_t transformer_layers_0_0_fn_projection_weight[256];
extern int16_t transformer_layers_0_0_fn_projection_bias[16];
// Feed-forward Block
extern int16_t transformer_layers_0_1_norm_weight[16];
extern int16_t transformer_layers_0_1_norm_bias[16];
extern int16_t transformer_layers_0_1_fn_ff1_weight[64];
extern int16_t transformer_layers_0_1_fn_ff1_bias[4];
extern int16_t transformer_layers_0_1_fn_ff2_weight[64];
extern int16_t transformer_layers_0_1_fn_ff2_bias[16];

//========= Transformer Layer 1 =========//
// Attention Block
extern int16_t transformer_layers_1_0_norm_weight[16];
extern int16_t transformer_layers_1_0_norm_bias[16];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_Q_H0[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_Q_H1[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_Q_H2[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_Q_H3[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_K_H0[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_K_H1[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_K_H2[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_K_H3[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_V_H0[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_V_H1[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_V_H2[64];
extern int16_t transformer_layers_1_0_fn_to_qkv_weight_V_H3[64];
extern int16_t transformer_layers_1_0_fn_projection_weight[256];
extern int16_t transformer_layers_1_0_fn_projection_bias[16];
// Feed-forward Block
extern int16_t transformer_layers_1_1_norm_weight[16];
extern int16_t transformer_layers_1_1_norm_bias[16];
extern int16_t transformer_layers_1_1_fn_ff1_weight[64];
extern int16_t transformer_layers_1_1_fn_ff1_bias[4];
extern int16_t transformer_layers_1_1_fn_ff2_weight[64];
extern int16_t transformer_layers_1_1_fn_ff2_bias[16];

//========= Transformer Layer 2 =========//
// Attention Block
extern int16_t transformer_layers_2_0_norm_weight[16];
extern int16_t transformer_layers_2_0_norm_bias[16];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_Q_H0[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_Q_H1[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_Q_H2[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_Q_H3[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_K_H0[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_K_H1[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_K_H2[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_K_H3[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_V_H0[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_V_H1[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_V_H2[64];
extern int16_t transformer_layers_2_0_fn_to_qkv_weight_V_H3[64];
extern int16_t transformer_layers_2_0_fn_projection_weight[256];
extern int16_t transformer_layers_2_0_fn_projection_bias[16];
// Feed-forward Block
extern int16_t transformer_layers_2_1_norm_weight[16];
extern int16_t transformer_layers_2_1_norm_bias[16];
extern int16_t transformer_layers_2_1_fn_ff1_weight[64];
extern int16_t transformer_layers_2_1_fn_ff1_bias[4];
extern int16_t transformer_layers_2_1_fn_ff2_weight[64];
extern int16_t transformer_layers_2_1_fn_ff2_bias[16];

//========= Transformer Layer 3 =========//
// Attention Block
extern int16_t transformer_layers_3_0_norm_weight[16];
extern int16_t transformer_layers_3_0_norm_bias[16];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_Q_H0[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_Q_H1[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_Q_H2[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_Q_H3[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_K_H0[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_K_H1[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_K_H2[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_K_H3[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_V_H0[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_V_H1[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_V_H2[64];
extern int16_t transformer_layers_3_0_fn_to_qkv_weight_V_H3[64];
extern int16_t transformer_layers_3_0_fn_projection_weight[256];
extern int16_t transformer_layers_3_0_fn_projection_bias[16];
// Feed-forward Block
extern int16_t transformer_layers_3_1_norm_weight[16];
extern int16_t transformer_layers_3_1_norm_bias[16];
extern int16_t transformer_layers_3_1_fn_ff1_weight[64];
extern int16_t transformer_layers_3_1_fn_ff1_bias[4];
extern int16_t transformer_layers_3_1_fn_ff2_weight[64];
extern int16_t transformer_layers_3_1_fn_ff2_bias[16];

//========= MLP Head =========//
extern int16_t mlp_head_layer_norm_weight[16];
extern int16_t mlp_head_layer_norm_bias[16];
extern int16_t mlp_head_linear_weight[256];
extern int16_t mlp_head_linear_bias[16];

#endif // DATA_H
