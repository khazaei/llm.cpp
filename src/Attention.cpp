//
// Created by Hamidreza Khazaei on 4/20/24.
//

#include "Attention.h"
#include <vector>

namespace llm {

float dotProduct(std::span<const float> in1, std::span<const float> in2) {
  auto sum = 0.0F;
  for (auto i = 0; i < in1.size(); ++i) {
    sum += (in1[i] * in2[i]);
  }
  return sum;
}

float scoreQueryKey(std::span<float> out, view2d<const float> in, const int currToken,
                    const int queryOffset, const int keyOffset, const int headOffset,
                    const size_t headSize) {

  const auto query = std::span{&in[currToken, headOffset + queryOffset], headSize};
  // XXX scale can be computed outside for optimizations.
  const auto scale = 1.0 / std::sqrtf(headSize);

  // find the dot product with all the prev tokens and store max (causal/masked)
  // score query key
  auto maxCorr = std::numeric_limits<float>::min();
  for (auto prevToken = 0; prevToken <= currToken; ++prevToken) {
    const auto key = std::span{&in[prevToken, headOffset + keyOffset], headSize};
    const auto corr = dotProduct(query, key) * scale;
    maxCorr = std::fmaxf(maxCorr, corr);
    out[prevToken] = corr;
  }

  // keeping track of maxCorr to be used in softmax for numerical stability.
  return maxCorr;
}

void softmax(std::span<float> out, std::span<const float> in, const int currTokens,
             const float maxCorr) {
  // calculate the exp and keep track of sum
  // maxCorr is being calculated and subtracted only for numerical stability
  auto expSum = 0.0F;
  for (auto token = 0; token <= currTokens; ++token) {
    const auto expV = expf(in[token] - maxCorr);
    expSum += expV;
    out[token] = expV;
  }
  auto expSumInv = expSum == 0.0F ? 0.0F : 1.0F / expSum;

  // normalize to get the softmax
  for (auto token = 0; token <= currTokens; ++token) {
    out[token] *= expSumInv;
  }
}

void weightedSumValues(std::span<float> out, view2d<const float> in,
                       std::span<const float> softmaxScores, const int currToken,
                       const int valueOffset, const int headOffset) {

  for(auto& elem : out) { elem = 0; }

  const auto headSize = out.size(); // same as the value size (embedding / numHeads)
  for (auto prevToken = 0; prevToken <= currToken; ++prevToken) {
    const auto value = std::span{&in[prevToken, headOffset + valueOffset], headSize};
    const auto valueScale = softmaxScores[prevToken];
    for (auto i = 0; i < headSize; i++) {
      out[i] += valueScale * value[i];
    }
  }
}

void attention(view3d<float> out, view3d<const float> in, const int numHeads) {
  const auto batchSize = in.extent(0);
  const auto seqLen = in.extent(1);
  const auto inChSize = in.extent(2);
  const auto embeddingDim = inChSize / 3;
  LLM_ASSERT(batchSize == out.extent(0));
  LLM_ASSERT(seqLen == out.extent(1));
  LLM_ASSERT(embeddingDim == out.extent(2));
  LLM_ASSERT(embeddingDim % numHeads == 0);

  // for each batch and token the input is stored as [Q, K, V] and each having embedding
  // dimension length
  const auto queryOffset = 0;
  const auto keyOffset = queryOffset + embeddingDim;
  const auto valueOffset = keyOffset + embeddingDim;
  const auto headSize = embeddingDim / numHeads;

  auto corrBuf = std::vector<float>(seqLen);    // xxx optimize out
  auto softmaxBuf = std::vector<float>(seqLen); // xxx optimize out
  for (auto batch = 0; batch < batchSize; ++batch) {
    const auto inBatch = view2d<const float>{&in[batch, 0, 0], seqLen, inChSize};
    for (auto currToken = 0; currToken < seqLen; ++currToken) {
      for (auto head = 0; head < numHeads; ++head) {
        const auto headOffset = headSize * head;

        // score key with queries
        const auto maxCorr = scoreQueryKey(corrBuf, inBatch, currToken, queryOffset,
                                           keyOffset, headOffset, headSize);
        // softmax of scores
        softmax(softmaxBuf, corrBuf, currToken, maxCorr);

        // weighted sum of value vectors to get attention for single head
        const auto outView = std::span{&out[batch, currToken, headOffset], headSize};
        weightedSumValues(outView, inBatch, softmaxBuf, currToken, valueOffset,
                          headOffset);
      }
    }
  }
}

} // namespace llm