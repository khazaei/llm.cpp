//
// Created by Hamidreza Khazaei on 4/24/24.
//

#include "Gpt2.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Sanity test for creating GPT2.") {

  auto maxSequenceLength = 100;
  auto vocabularySize = 1000;
  auto numLayers = 2;
  auto numHeads = 2;
  auto channelDimension = 256;
  std::cout << "creating gpt" << std::endl;

  auto gpt = llm::gpt2::Module{maxSequenceLength, vocabularySize, numLayers, numHeads,
                               channelDimension};

  std::cout << "created gpt" << std::endl;

  auto in = std::vector<int>(800);
  const auto v = llm::view<const int, 2>{in.data(), 8, 100};
  gpt.forward(v);
}