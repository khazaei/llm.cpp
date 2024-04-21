//
// Created by Hamidreza Khazaei on 4/19/24.
//

#ifndef LLM_INFERENCE_LAYERNORM_H
#define LLM_INFERENCE_LAYERNORM_H

#include "Extension.h"
#include <numeric>

namespace llm {

float mean(std::span<const float> in);

float variance(std::span<const float> in, float mean);

// both in and out are (B,T,C) of the activations
// weight and bias both length C
void layerNorm(view3d<float> out, view3d<const float> in, std::span<const float> weights,
               std::span<const float> bias);

} // namespace llm

#endif // LLM_INFERENCE_LAYERNORM_H
