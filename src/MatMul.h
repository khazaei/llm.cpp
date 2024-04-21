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
void matMul(view3d<float> out, view3d<const float> in, view2d<const float> weight,
            std::span<const float> bias);

} // namespace llm

#endif // LLM_INFERENCE_MATMUL_H
