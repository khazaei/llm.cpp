//
// Created by Hamidreza Khazaei on 4/24/24.
//

#include "Gpt2.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Sanity test for creating GPT2.") {

  auto maxSequenceLength = 1024;
  auto vocabularySize = 50257;
  auto numLayers = 12;
  auto numHeads = 12;
  auto channelDimension = 768;
  auto gpt = llm::gpt2::Module{maxSequenceLength, vocabularySize, numLayers, numHeads,
                               channelDimension};

  auto in = std::vector<int>(800);
  const auto v = llm::view<const int, 2>{in.data(), 8, 100};
  gpt.forward(v);
}