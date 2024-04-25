//
// Created by Hamidreza Khazaei on 4/21/24.
//

#ifndef LLM_INFERENCE_ACTIVATION_H
#define LLM_INFERENCE_ACTIVATION_H

#include "Extension.h"

#include <span>

namespace llm {

void gelu(view<float, 3> out, view<const float,3> in);

// output: out are (B,T,V) of the probabilities (sums to 1.0 in each b,t position)
// input: logit is (B,T,V) of the un-normalized log probabilities
void softmax(view<float, 3> probabilities, view<const float, 3> logits);

// output: losses is (B,T) of the individual losses at each position
// input: probs are (B,T,V) of the probabilities
// input: targets is (B,T) of integers giving the correct index in logits
void crossEntropy(view<float, 2> losses, view<const float, 3> probs,
                  view<const int, 2> targets);

} // namespace llm

#endif // LLM_INFERENCE_ACTIVATION_H
