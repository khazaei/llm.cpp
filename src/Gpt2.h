//
// Created by Hamidreza Khazaei on 4/23/24.
//

#ifndef LLM_INFERENCE_GPT2_H
#define LLM_INFERENCE_GPT2_H

#include <map>
#include <string>
#include <vector>

#include "Extension.h"

namespace llm::gpt2 {

using DimensionMap = std::map<std::string, std::vector<size_t>>;
[[nodiscard]] size_t getTotalSize(const DimensionMap &dims);

class Parameters {
public:
  struct Memory {
    view<const float, 2> wte{};      // (V, C)
    view<const float, 2> wpe{};      // (maxT, C)
    view<const float, 2> ln1w{};     // (L, C)
    view<const float, 2> ln1b{};     // (L, C)
    view<const float, 3> qkvw{};     // (L, 3*C, C)
    view<const float, 2> qkvb{};     // (L, 3*C)
    view<const float, 3> attprojw{}; // (L, C, C)
    view<const float, 2> attprojb{}; // (L, C)
    view<const float, 2> ln2w{};     // (L, C)
    view<const float, 2> ln2b{};     // (L, C)
    view<const float, 3> fcw{};      // (L, 4*C, C)
    view<const float, 2> fcb{};      // (L, 4*C)
    view<const float, 3> fcprojw{};  // (L, C, 4*C)
    view<const float, 2> fcprojb{};  // (L, C)
    std::span<const float> lnfw{};   // (C)
    std::span<const float> lnfb{};   // (C)

    void assignMemory(const float *data, int totalSize, const DimensionMap &dimMap);
  };

  Parameters(size_t vocabularySize, size_t channelDimension, size_t maxSequenceLength,
             size_t numLayers);

  [[nodiscard]] const Memory &getMemory() const { return memory; }
  [[nodiscard]] Memory &getMemory() { return memory; }
  [[nodiscard]] const DimensionMap &getDimMap() const { return dims; }

private:
  DimensionMap dims{};
  Memory memory{};
};

class Scratch {
  Scratch(size_t batchSize, size_t sequenceLength, size_t channelDimension,
          size_t numLayers, size_t numHeads, size_t vocabularySize);

  struct Memory {
    view<float, 3> encoded{};   // (B, T, C)
    view<float, 4> ln1{};       // (L, B, T, C)
    view<float, 3> ln1Mean{};   // (L, B, T)
    view<float, 3> ln1Rstd{};   // (L, B, T)
    view<float, 4> qkv{};       // (L, B, T, 3*C)
    view<float, 4> atty{};      // (L, B, T, C)
    view<float, 5> preatt{};    // (L, B, NH, T, T)
    view<float, 5> att{};       // (L, B, NH, T, T)
    view<float, 4> attproj{};   // (L, B, T, C)
    view<float, 4> residual2{}; // (L, B, T, C)
    view<float, 4> ln2{};       // (L, B, T, C)
    view<float, 3> ln2Mean{};   // (L, B, T)
    view<float, 3> ln2Rstd{};   // (L, B, T)
    view<float, 4> fch{};       // (L, B, T, 4*C)
    view<float, 4> fchGelu{};   // (L, B, T, 4*C)
    view<float, 4> fcproj{};    // (L, B, T, C)
    view<float, 4> residual3{}; // (L, B, T, C)
    view<float, 3> lnf{};       // (B, T, C)
    view<float, 2> lnfMean{};   // (B, T)
    view<float, 2> lnfRstd{};   // (B, T)
    view<float, 3> logits{};    // (B, T, V)
    view<float, 3> probs{};     // (B, T, V)
    view<float, 2> losses{};    // (B, T)

    void setupViews(float *data, int totalSize, const DimensionMap &dimMap);
  };

  [[nodiscard]] const Memory &getMemory() const { return memory; }
  [[nodiscard]] Memory &getMemory() { return memory; }
  [[nodiscard]] const DimensionMap &getDimMap() const { return dims; }

private:
  DimensionMap dims{};
  Memory memory;
};

} // namespace llm::gpt2

#endif // LLM_INFERENCE_GPT2_H
