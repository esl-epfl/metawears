//
// Created by alireza on 10/6/23.
//

#ifndef FVLLMONTITRANSFORMER_TRANSFORMERBLOCK_H
#define FVLLMONTITRANSFORMER_TRANSFORMERBLOCK_H

#include <stddef.h>
#include <stdint.h>
#include "selfattentionC.h"
#include "addNormC.h"
#include "dense_layerC.h"
#include "tokenPosEmbeddingC.h"
#include "../param.h"
#include "transposeC.h"

typedef struct {
    size_t num_heads_;
    size_t head_hidden_size_;
    size_t input_dim_;
    size_t ff_size_;
    SingleHeadSelfAttn* selfatten[NUM_LAYERS*NUM_HEAD];
    int16_t* multihead_out;
    int16_t* condense_out;
    int16_t* intermediateFF;
    int16_t* intermediateFFBlockWise;
    AddNormalize addNorm;
    AddNormalize addNorm2;
    AddNormalize transformer_layer_0_addNorm[NUM_LAYERS];
    AddNormalize transformer_layer_1_addNorm[NUM_LAYERS];
    AddNormalize mlp_head_norm;
    TokenPosEmbedding* token;
    Dense* condense[NUM_LAYERS];
    Dense* feedForward0[NUM_LAYERS];
    Dense* feedForward1[NUM_LAYERS];
    Dense* patchEmbedding;
    Dense* mlp_head_linear;
    #ifndef REARRANGE
    int16_t* multihead_out_reshape;
    #endif
} TransformerBlock;

TransformerBlock* createTransformerBlock(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, size_t num_heads, size_t ff_size, int16_t** weightVector, int16_t** biasVector, int16_t* clsTokenVector, int16_t* posMatrix);
void destroyTransformerBlock(TransformerBlock* transformerBlock);
void computeFixedPoint(TransformerBlock* transformerBlock, size_t seq_len, int16_t* input, int16_t* input_normalized, int16_t* output, int16_t* intermediate, int16_t* qkv);

#endif //FVLLMONTITRANSFORMER_TRANSFORMERBLOCK_H
