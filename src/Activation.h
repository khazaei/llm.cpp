//
// Created by Hamidreza Khazaei on 4/21/24.
//

#ifndef LLM_INFERENCE_ACTIVATION_H
#define LLM_INFERENCE_ACTIVATION_H

#include "Extension.h"

namespace llm {

/**
 * @brief Computes element wise gelu on the tensor.
 *
 * Implementation is the tanh approximation implementation.
 * GELU(x) = 0.5 * x * (1+Tanh(sqrt(2/π))) * (x + 0.044715 * x^3)
 *
 * @param out output in the shape of [batch, token, embedding]
 * @param in input in the shape of [batch, token, embedding]
 */
void gelu(view<float, 3> out, view<const float, 3> in);

/**
 * @brief Computes softmax on the third dimension.
 *
 * @param probabilities output is the probabilities (sums to 1.0 in each b,t position) in
 * the shape of [batch, token, vocabulary]
 * @param logits input is the logits in the shape of [batch, token, vocabulary]
 */
void softmax(view<float, 3> probabilities, view<const float, 3> logits);

/**
 * @brief Computes cross-entropy loss on the tensor
 *
 * @param losses output in the shape of [batch, token], individual losses at each
 * position.
 * @param probs input probabilities in the shape of [batch, token, vocabulary]
 * @param targets ground truth targets in the shape of [batch, token]. These are the
 * integers giving the correct index in logits.
 */
void crossEntropy(view<float, 2> losses, view<const float, 3> probs,
                  view<const int, 2> targets);

} // namespace llm

#endif // LLM_INFERENCE_ACTIVATION_H
