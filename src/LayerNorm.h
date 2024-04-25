//
// Created by Hamidreza Khazaei on 4/19/24.
//

#ifndef LLM_INFERENCE_LAYERNORM_H
#define LLM_INFERENCE_LAYERNORM_H

#include "Extension.h"
#include <numeric>

namespace llm {

float mean(view<const float, 1> in);

float variance(view<const float, 1> in, float mean);

// both in and out are (B,T,C) of the activations
// weight and bias both length C
void layerNorm(view<float, 3> out, view<const float, 3> in,
               view<const float, 1> weights, view<const float, 1> bias);

void matAdd(view<float, 3> out, view<const float, 3> in1,
              view<const float, 3> in2);

// out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token &
// position
// tokenIndices is (B,T) of integers, holding the token ids at each (b,t) position
// tokenEmbedding is (V,C) of token embeddings, short for "weight token embeddings"
// positionEmbedding is (maxT,C) of position embeddings, short for "weight positional
// embedding"
void positionalEncoding(view<float, 3> out, view<const int, 2> tokenIndices,
                        view<const float, 2> tokenEmbedding,
                        view<const float, 2> positionEmbedding);

} // namespace llm

#endif // LLM_INFERENCE_LAYERNORM_H
