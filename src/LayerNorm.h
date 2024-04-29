//
// Created by Hamidreza Khazaei on 4/19/24.
//

#ifndef LLM_INFERENCE_LAYERNORM_H
#define LLM_INFERENCE_LAYERNORM_H

#include "Extension.h"
#include <numeric>

namespace llm {

/**
 * @brief Computes mean of a tensor.
 *
 * @param in is the input.
 * @return the computed mean.
 */
float mean(view<const float, 1> in);

/**
 * @brief Computes variance of a tensor.
 *
 * @param in is the input.
 * @param mean is the mean of the tensor.
 * @return the computed variance.
 */
float variance(view<const float, 1> in, float mean);

/**
 * @brief Does layer normalization.
 *
 * Computes the statistics for each (batch, token) separately and normalizes the tensor.
 * Then multiplies the tensor by the weight and adds the bias.
 *
 * @param out is [batch, token, embedding], the result of layer norm.
 * @param in is [batch, token, embedding], input to layer norm.
 * @param weights is embedding size, used to scale the normalized input.
 * @param bias is embedding size, used to shift the normalized input.
 */
void layerNorm(view<float, 3> out, view<const float, 3> in, view<const float, 1> weights,
               view<const float, 1> bias);

/**
 * @brief Add two matrices.
 *
 * @param out is [batch, token, embedding], the result of the matrix add.
 * @param in1 is [batch, token, embedding], first input.
 * @param in2 is [batch, token, embedding], second input.
 */
void matAdd(view<float, 3> out, view<const float, 3> in1, view<const float, 3> in2);

/**
 * @brief positional encode input token ids.
 *
 * @param out is [batch, token, embedding]. At each position (batch,token), a
 * C-dimensional vector summarizing the token & position.
 * @param tokenIndices is [batch, token] of integers. Holding the token ids at each
 * (batch,token) position.
 * @param tokenEmbedding is [vocabulary size, embedding dimension]. Maps a token id to an
 * embedding.
 * @param positionEmbedding positionEmbedding is [max sequence length, embedding
 * dimension] of position embeddings. Maps a token position to an position embedding.
 */
void positionalEncoding(view<float, 3> out, view<const int, 2> tokenIndices,
                        view<const float, 2> tokenEmbedding,
                        view<const float, 2> positionEmbedding);

} // namespace llm

#endif // LLM_INFERENCE_LAYERNORM_H
