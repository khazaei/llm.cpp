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
void genToken(gpt2::Module &model, const gpt2::Tokenizer &tokenizer, int numTokens);

} // namespace llm

#endif // LLM_INFERENCE_LLM_H
