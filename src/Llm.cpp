//
// Created by Hamidreza Khazaei on 4/29/24.
//

#include "Llm.h"

namespace llm {

void genToken(gpt2::Module &model, const gpt2::Tokenizer &tokenizer,
              const int numTokens) {
  const auto B = 1;
  auto in = std::vector<int>(static_cast<size_t>(B) * numTokens, tokenizer.getEOT());
  const auto inView = view<const int, 2>{in.data(), B, numTokens};

  auto rng = std::mt19937_64{std::random_device{}()};
  for (auto t = 1; t < numTokens; ++t) {
    // below we're only using b=0 (i.e. the first row) of all B rows
    // we're in principle running B "inference streams" in parallel here
    // but only using position 0
    // get the Vp-dimensional vector probs[0, t-1, :]
    model.forward(inView);
    const auto &logits = model.getScratch().getMemory().logits;
    const auto V = logits.extent(2);
    const auto nextToken =
        sampleDiscreteDistribution(view<const float, 1>{&logits[0, t - 1, 0], V}, rng);
    std::cout << numTokens << '\n';
    in[t] = nextToken;
  }
}

} // namespace llm