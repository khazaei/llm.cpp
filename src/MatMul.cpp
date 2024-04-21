//
// Created by Hamidreza Khazaei on 4/20/24.
//

#include "MatMul.h"

namespace llm {

void matMul(view3d<float> out, view3d<const float> in, view2d<const float> weight,
            std::span<const float> bias) {
  const auto B = out.extent(0);
  const auto T = out.extent(1);
  const auto outChSize = weight.extent(0);
  const auto inChSize = weight.extent(1);

  LLM_ASSERT(in.extent(0) == B);
  LLM_ASSERT(in.extent(1) == T);
  LLM_ASSERT(in.extent(2) == inChSize);
  LLM_ASSERT(out.extent(2) == outChSize);
  LLM_ASSERT(bias.size() == outChSize);

  for(auto batch = 0; batch < B; ++batch) {
    for(auto token = 0; token < T; ++token) {
      auto inView = std::span{&in[batch, token, 0], inChSize};
      auto outView = std::span{&out[batch, token, 0], outChSize};
      for(auto outIdx = 0; outIdx < outChSize; ++outIdx) {
        outView[outIdx] = bias[outIdx];
        auto weightRow = std::span{&weight[outIdx, 0], inChSize};
        for (auto inIdx = 0; inIdx < inChSize; ++inIdx) {
          outView[outIdx] += weightRow[inIdx] * inView[inIdx];
        }
      }
    }
  }
}

} // namespace llm