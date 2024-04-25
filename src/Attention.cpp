//
// Created by Hamidreza Khazaei on 4/20/24.
//

#include "Attention.h"
#include <vector>

namespace llm {

float dotProduct(view<const float, 1> in1, view<const float, 1> in2) {
  auto sum = 0.0F;
  for (auto i = 0; i < in1.size(); ++i) {
    sum += (in1[i] * in2[i]);
  }
  return sum;
}

float scoreQueryKey(view<float, 1> out, view<const float, 2> in, const int currToken,
                    const int queryOffset, const int keyOffset, const int headOffset,
                    const int headDim) {

  const auto query =
      view<const float, 1>{&in[currToken, headOffset + queryOffset], headDim};
  // XXX scale can be computed outside for optimizations.
  const auto scale = 1.0F / std::sqrtf(static_cast<float>(headDim));

  // find the dot product with all the prev tokens and store max (causal/masked)
  // score query key
  auto maxScore = std::numeric_limits<float>::min();
  for (auto prevToken = 0; prevToken <= currToken; ++prevToken) {
    const auto key =
        view<const float, 1>{&in[prevToken, headOffset + keyOffset], headDim};
    const auto score = dotProduct(query, key) * scale;
    maxScore = std::fmaxf(maxScore, score);
    out[prevToken] = score;
  }

  // keeping track of maxCorr to be used in softmax for numerical stability.
  return maxScore;
}

void softmax(view<float, 1> out, view<const float, 1> in, const float maxScore) {
  // calculate the exp and keep track of sum
  // max score is being calculated and subtracted only for numerical stability
  auto expSum = 0.0F;
  const auto numElem = in.size();
  for (auto token = 0; token < numElem; ++token) {
    const auto expV = expf(in[token] - maxScore);
    expSum += expV;
    out[token] = expV;
  }
  const auto expSumInv = expSum == 0.0F ? 0.0F : 1.0F / expSum;

  // normalize over sum of exp
  for (auto token = 0; token < numElem; ++token) {
    out[token] *= expSumInv;
  }
}

void weightedSumValues(view<float, 1> out, view<const float, 2> in,
                       view<const float, 1> softmaxScores, const int currToken,
                       const int valueOffset, const int headOffset) {

  for (auto i = 0; i < out.extent(0); ++i) {
    out[i] = 0;
  }

  const auto headDim = out.size(); // same as the value size (embedding / numHeads)
  for (auto prevToken = 0; prevToken <= currToken; ++prevToken) {
    const auto value =
        view<const float, 1>{&in[prevToken, headOffset + valueOffset], headDim};
    const auto valueScale = softmaxScores[prevToken];
    for (auto i = 0; i < headDim; i++) {
      out[i] += valueScale * value[i];
    }
  }
}

void multiHeadAttentionCausal(view<float, 3> out, view<const float, 3> in,
                              const int numHeads) {
  const auto batchSize = in.extent(0);
  const auto seqLen = in.extent(1);
  const auto inDim = in.extent(2);
  const auto embeddingDim = inDim / 3;
  LLM_ASSERT(batchSize == out.extent(0));
  LLM_ASSERT(seqLen == out.extent(1));
  LLM_ASSERT(embeddingDim == out.extent(2));
  LLM_ASSERT(embeddingDim % numHeads == 0);

  // for each batch and token the input is stored as [Q, K, V] and each having embedding
  // dimension length
  const auto queryOffset = 0;
  const auto keyOffset = queryOffset + embeddingDim;
  const auto valueOffset = keyOffset + embeddingDim;
  const auto headDim = embeddingDim / numHeads;

  auto scoreBuf = std::vector<float>(seqLen);   // xxx optimize out
  auto softmaxBuf = std::vector<float>(seqLen); // xxx optimize out
  auto scoreBufV = view<float, 1>{scoreBuf.data(), scoreBuf.size()};
  auto softmaxBufV = view<float, 1>{softmaxBuf.data(), softmaxBuf.size()};

  for (auto batch = 0; batch < batchSize; ++batch) {
    const auto inBatch = view<const float, 2>{&in[batch, 0, 0], seqLen, inDim};
    for (auto currToken = 0; currToken < seqLen; ++currToken) {
      for (auto head = 0; head < numHeads; ++head) {
        const auto headOffset = headDim * head;

        // score key with queries
        const auto maxScore = scoreQueryKey(scoreBufV, inBatch, currToken, queryOffset,
                                            keyOffset, headOffset, headDim);
        // softmax of scores
        softmax(softmaxBufV,
                view<const float, 1>{scoreBuf.data(), currToken + 1},
                maxScore);

        // weighted sum of value vectors to get attention for single head
        const auto outView = view<float, 1>{&out[batch, currToken, headOffset],
                                            headDim};
        weightedSumValues(outView, inBatch, softmaxBufV, currToken, valueOffset,
                          headOffset);
      }
    }
  }
}

} // namespace llm