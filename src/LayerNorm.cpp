//
// Created by Hamidreza Khazaei on 4/19/24.
//

#include "LayerNorm.h"

namespace {

void layerNormPerChannel(llm::view<float, 1> out, llm::view<const float, 1> in,
                         llm::view<const float, 1> weights,
                         llm::view<const float, 1> bias) {
  const auto m = llm::mean(in);
  const auto var = llm::variance(in, m);

  constexpr auto eps = 1e-5F;
  const auto inverseStd = 1.0F / sqrtf(var + eps);
  for (auto channel = 0; channel < in.size(); ++channel) {
    const auto normalized = (in[channel] - m) * inverseStd;
    out[channel] = normalized * weights[channel] + bias[channel];
  }
}

} // namespace

namespace llm {

float mean(view<const float, 1> in) {
  auto sum = 0.0F;
  for (auto i = 0; i < in.extent(0); ++i) {
    sum += in[i];
  }
  return sum / static_cast<float>(in.size());
}

float variance(view<const float, 1> in, const float mean) {
  auto sum = 0.0F;
  for (auto i = 0; i < in.extent(0); ++i) {
    const auto centered = (in[i] - mean);
    sum += (centered * centered);
  }
  return sum / static_cast<float>(in.size());
}

void layerNorm(view<float, 3> out, view<const float, 3> in, view<const float, 1> weights,
               view<const float, 1> bias) {
  const auto batchSize = out.extent(0);
  const auto sequenceLength = out.extent(1);
  const auto outDim = out.extent(2);

  LLM_ASSERT(batchSize == in.extent(0));
  LLM_ASSERT(sequenceLength == in.extent(1));
  LLM_ASSERT(outDim == in.extent(2));
  LLM_ASSERT(weights.size() == outDim);
  LLM_ASSERT(bias.size() == outDim);

  for (auto batch = 0; batch < batchSize; ++batch) {
    for (auto token = 0; token < sequenceLength; ++token) {
      const auto inView = view<const float, 1>{&in[batch, token, 0], outDim};
      const auto outView = view<float, 1>{&out[batch, token, 0], outDim};
      layerNormPerChannel(outView, inView, weights, bias);
    }
  }
}

void matAdd(view<float, 3> out, view<const float, 3> in1, view<const float, 3> in2) {
  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto embeddingDim = out.extent(2);

  LLM_ASSERT(in1.extent(0) == batchSize);
  LLM_ASSERT(in1.extent(1) == seqLen);
  LLM_ASSERT(in1.extent(2) == embeddingDim);
  LLM_ASSERT(in2.extent(0) == batchSize);
  LLM_ASSERT(in2.extent(1) == seqLen);
  LLM_ASSERT(in2.extent(2) == embeddingDim);

  for (auto batch = 0; batch < batchSize; ++batch) {
    for (auto token = 0; token < seqLen; ++token) {
      for (auto c = 0; c < embeddingDim; ++c) {
        out[batch, token, c] = in1[batch, token, c] + in2[batch, token, c];
      }
    }
  }
}

void positionalEncoding(view<float, 3> out, view<const int, 2> tokenIndices,
                        view<const float, 2> tokenEmbedding,
                        view<const float, 2> positionEmbedding) {

  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto embeddingDim = out.extent(2);
  LLM_ASSERT(tokenIndices.extent(0) == batchSize);
  LLM_ASSERT(tokenIndices.extent(1) == seqLen);
  LLM_ASSERT(tokenEmbedding.extent(1) == embeddingDim);
  LLM_ASSERT(positionEmbedding.extent(1) == embeddingDim);

  for (auto batch = 0; batch < batchSize; ++batch) {
    for (auto tokenPos = 0; tokenPos < seqLen; ++tokenPos) {
      const auto indexOfToken = tokenIndices[batch, tokenPos];
      LLM_ASSERT(indexOfToken < tokenEmbedding.extent(0));
      for (auto c = 0; c < embeddingDim; ++c) {
        out[batch, tokenPos, c] =
            tokenEmbedding[indexOfToken, c] + positionEmbedding[tokenPos, c];
      }
    }
  }
}

} // namespace llm