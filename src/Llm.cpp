//
// Created by Hamidreza Khazaei on 4/29/24.
//

#include "Llm.h"

namespace llm {

std::vector<int> genToken(const int EOT, gpt2::Module &model, const int numTokens,
                          std::mt19937_64 rng) {
  const auto B = 1;
  auto in = std::vector<int>(static_cast<size_t>(B) * numTokens, EOT);
  const auto inView = view<const int, 2>{in.data(), B, numTokens};

  for (auto t = 1; t < numTokens; ++t) {
    // below we're only using b=0 (i.e. the first row) of all B rows
    // we're in principle running B "inference streams" in parallel here
    // but only using position 0
    // get the Vp-dimensional vector probs[0, t-1, :]
    model.forward(inView);
    const auto &probs = model.getScratch().getMemory().probs;
    const auto V = probs.extent(2);
    const auto nextToken =
        sampleDiscreteDistribution(view<const float, 1>{&probs[0, t - 1, 0], V}, rng);
    in[t] = nextToken;
  }
  return in;
}

} // namespace llm