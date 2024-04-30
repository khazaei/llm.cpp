//
// Created by Hamidreza Khazaei on 4/20/24.
//

#ifndef LLM_INFERENCE_MATOPS_H
#define LLM_INFERENCE_MATOPS_H

#include "Extension.h"

namespace llm {

/**
 * @brief matrix multiplication, out = weight * in + bias.
 *
 * weights are stored as row major.
 *
 * @param out output in the shape of [batch, token, out dimension].
 * @param in input in the shape of [batch, token, in dimension].
 * @param weight shape of [out dimension, in dimension].
 * @param bias shape of out dimension.
 */
void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
            view<const float, 1> bias);

/**
 * @brief matrix multiplication, out = weight * in.
 *
 * weights are stored as row major.
 *
 * @param out output in the shape of [batch, token, out dimension].
 * @param in input in the shape of [batch, token, in dimension].
 * @param weight shape of [out dimension, in dimension].
 */
void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight);

/**
 * @brief matrix multiplication using SIMD, out = weight * in + bias.
 *
 * weights are stored as row major.
 *
 * @param out output in the shape of [batch, token, out dimension].
 * @param in input in the shape of [batch, token, in dimension].
 * @param weight shape of [out dimension, in dimension].
 * @param bias shape of out dimension.
 */
void matMulNeon(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight);

/**
 * @brief matrix multiplication using SIMD, out = weight * in.
 *
 * weights are stored as row major.
 *
 * @param out output in the shape of [batch, token, out dimension].
 * @param in input in the shape of [batch, token, in dimension].
 * @param weight shape of [out dimension, in dimension].
 */
void matMulNeon(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
                view<const float, 1> bias);

} // namespace llm

#endif // LLM_INFERENCE_MATOPS_H
