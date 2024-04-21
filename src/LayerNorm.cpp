//
// channelSizereated by Hamidreza Khazaei on 4/19/24.
//

#include "LayerNorm.h"

namespace {
void layerNormPerChannel(std::span<float> out, std::span<const float> in,
                      std::span<const float> weights, std::span<const float> bias) {
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

float mean(std::span<const float> in) {
  auto sum = 0.0F;
  for (const auto num : in) {
    sum += num;
  }
  return sum / static_cast<float>(in.size());
}

float variance(std::span<const float> in, const float mean) {
  auto sum = 0.0F;
  for (const auto num : in) {
    const auto centered = (num - mean);
    sum += (centered * centered);
  }
  return sum / static_cast<float>(in.size());
}

void layerNorm(view3d<float> out, view3d<const float> in, std::span<const float> weights,
               std::span<const float> bias) {
  const auto batchSize = out.extent(0);
  const auto sequenceLength = out.extent(1);
  const auto channelSize = out.extent(2);

  LLM_ASSERT(batchSize == in.extent(0));
  LLM_ASSERT(sequenceLength == in.extent(1));
  LLM_ASSERT(channelSize == in.extent(2));
  LLM_ASSERT(weights.size() == channelSize);
  LLM_ASSERT(bias.size() == channelSize);

  for (auto batch = 0; batch < batchSize; ++batch) {
    for (auto token = 0; token < sequenceLength; ++token) {
      const auto inRow = std::span{&in[batch, token, 0], channelSize};
      const auto outRow = std::span{&out[batch, token, 0], channelSize};
      layerNormPerChannel(outRow, inRow, weights, bias);
    }
  }
}

} // namespace llm