//
// Created by Hamidreza Khazaei on 4/21/24.
//

#include <numbers>

#include "Activation.h"
#include "Extension.h"

namespace llm {

void gelu(view<float, 3> out, view<const float, 3> in) {
  static const auto geluScalingFactor = std::sqrtf(2.0F / std::numbers::pi_v<float>);
  LLM_ASSERT(out.extent(0) == in.extent(0));
  LLM_ASSERT(out.extent(1) == in.extent(1));
  LLM_ASSERT(out.extent(2) == in.extent(2));

  for (auto batch = 0; batch < in.extent(0); ++batch) {
    for (int token = 0; token < in.extent(1); ++token) {
      for (int c = 0; c < in.extent(2); ++c) {
        const auto x = in[batch, token, c];
        const auto cube = 0.044715F * x * x * x;
        out[batch, token, c] = 0.5F * x * (1.0F + tanhf(geluScalingFactor * (x + cube)));
      }
    }
  }
}

void softmax(view<float, 3> probabilities, view<const float, 3> logits) {
  const auto batchSize = logits.extent(0);
  const auto seqLen = logits.extent(1);
  const auto embedDim = logits.extent(2);

  LLM_ASSERT(probabilities.extent(0) == batchSize);
  LLM_ASSERT(probabilities.extent(1) == seqLen);
  LLM_ASSERT(probabilities.extent(2) == embedDim);

  for (auto batch = 0; batch < batchSize; ++batch) {
    for (auto token = 0; token < seqLen; ++token) {
      // max is only calculated and subtracted for numerical stability
      auto maxVal = std::numeric_limits<float>::min();
      for (auto i = 0; i < embedDim; ++i) {
        maxVal = std::fmaxf(logits[batch, token, i], maxVal);
      }

      auto sum = 0.0F;
      for (auto i = 0; i < embedDim; ++i) {
        const auto expV = expf(logits[batch, token, i] - maxVal);
        sum += expV;
        probabilities[batch, token, i] = expV;
      }

      const auto normalization = 1.0F / sum;
      for (auto i = 0; i < embedDim; ++i) {
        probabilities[batch, token, i] *= normalization;
      }
    }
  }
}

void crossEntropy(view<float, 2> losses, view<const float, 3> probs,
                  view<const int, 2> targets) {

  const auto batchSize = losses.extent(0);
  const auto seqLen = losses.extent(1);

  LLM_ASSERT(batchSize == probs.extent(0));
  LLM_ASSERT(seqLen == probs.extent(1));
  LLM_ASSERT(batchSize == targets.extent(0));
  LLM_ASSERT(seqLen == targets.extent(1));

  for (auto batch = 0; batch < batchSize; ++batch) {
    for (auto token = 0; token < seqLen; ++token) {
      // loss = -log(probs[target])
      losses[batch, token] = -std::logf(probs[batch, token, targets[batch, token]]);
    }
  }
}

} // namespace llm