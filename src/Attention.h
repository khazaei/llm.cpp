//
// Created by Hamidreza Khazaei on 4/20/24.
//

#ifndef LLM_INFERENCE_ATTENTION_H
#define LLM_INFERENCE_ATTENTION_H

#include "Extension.h"

namespace llm {

/**
 * @brief Computes causal multi-head attention
 *
 * Attention is the only layer that mixes information across time. every other operation
 * is applied at every (b,t) position independently (and of course, no layer mixes
 * information across batch).
 *
 * @param out output is the probabilities (sums to 1.0 in each b,t position) in the shape
 * of [batch, token, vocabulary]
 * @param in input is in the shape of [batch, token, 3 * embedding dimension] holding the
 * the query, key, value (Q, K, V) vectors in the last dimension. For every batch and
 * every token the Q, K, V are laid out consecutively in memory.
 * @param numHeads number of heads.
 */
void multiHeadAttentionCausal(view<float, 3> out, view<const float, 3> in, int numHeads);

} // namespace llm

#endif // LLM_INFERENCE_ATTENTION_H
