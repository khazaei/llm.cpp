//
// Created by Hamidreza Khazaei on 4/20/24.
//

#ifndef LLM_INFERENCE_ATTENTION_H
#define LLM_INFERENCE_ATTENTION_H

#include "Extension.h"

namespace llm {

// input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
// output is (B, T, C)
// attention is the only layer that mixes information across time
// every other operation is applied at every (b,t) position independently
// (and of course, no layer mixes information across batch)
void multiHeadAttentionCausal(view<float, 3> out, view<const float, 3> in, int numHeads);

} // namespace llm

#endif // LLM_INFERENCE_ATTENTION_H
