//
// Created by Hamidreza Khazaei on 4/20/24.
//

#ifndef LLM_INFERENCE_MATMUL_H
#define LLM_INFERENCE_MATMUL_H

#include "Extension.h"

namespace llm {

// in is (B,T,C), weight is (OC, C), bias is (OC)
// out will be (B,T,OC)
// the weights need to be stored row major.
void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
            std::span<const float> bias);

} // namespace llm

#endif // LLM_INFERENCE_MATMUL_H
