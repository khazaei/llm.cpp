//
// Created by Hamidreza Khazaei on 4/29/24.
//

#ifndef LLM_INFERENCE_LLM_H
#define LLM_INFERENCE_LLM_H

#include "Gpt2.h"
#include <random>
#include <string>
#include <vector>

namespace llm {

// template <typename Model, typename Tokenizer>
std::vector<int> genToken(int EOT, gpt2::Module &model, int numTokens,
                          std::mt19937_64 rng = std::mt19937_64{std::random_device{}()});

} // namespace llm

#endif // LLM_INFERENCE_LLM_H
