//
// Created by Hamidreza Khazaei on 4/20/24.
//

#include "MatMul.h"

namespace llm {

void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
            view<const float, 1> bias) {
  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  LLM_ASSERT(in.extent(0) == batchSize);
  LLM_ASSERT(in.extent(1) == seqLen);
  LLM_ASSERT(in.extent(2) == inDim);
  LLM_ASSERT(out.extent(2) == outDim);
  LLM_ASSERT(bias.size() == outDim);

  for (auto batch = 0; batch < batchSize; ++batch) {
    for (auto token = 0; token < seqLen; ++token) {
      const auto inView = view<const float, 1>{&in[batch, token, 0], inDim};
      const auto outView = view<float, 1>{&out[batch, token, 0], outDim};
      for (auto outIdx = 0; outIdx < outDim; ++outIdx) {
        outView[outIdx] = bias[outIdx];
        const auto weightRow = view<const float, 1>{&weight[outIdx, 0], inDim};
        for (auto inIdx = 0; inIdx < inDim; ++inIdx) {
          outView[outIdx] += weightRow[inIdx] * inView[inIdx];
        }
      }
    }
  }
}

void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight) {
  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  LLM_ASSERT(in.extent(0) == batchSize);
  LLM_ASSERT(in.extent(1) == seqLen);
  LLM_ASSERT(in.extent(2) == inDim);
  LLM_ASSERT(out.extent(2) == outDim);

  for (auto batch = 0; batch < batchSize; ++batch) {
    for (auto token = 0; token < seqLen; ++token) {
      const auto inView = view<const float, 1>{&in[batch, token, 0], inDim};
      const auto outView = view<float, 1>{&out[batch, token, 0], outDim};
      for (auto outIdx = 0; outIdx < outDim; ++outIdx) {
        outView[outIdx] = 0;
        const auto weightRow = view<const float, 1>{&weight[outIdx, 0], inDim};
        for (auto inIdx = 0; inIdx < inDim; ++inIdx) {
          outView[outIdx] += weightRow[inIdx] * inView[inIdx];
        }
      }
    }
  }
}

} // namespace llm