//
// Created by Hamidreza Khazaei on 4/23/24.
//

#ifndef LLM_INFERENCE_GPT2_H
#define LLM_INFERENCE_GPT2_H

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "Extension.h"

namespace llm::gpt2 {

using DimensionMap = std::map<std::string, std::vector<int>>;
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
    view<const float, 3> fcw1{};     // (L, 4*C, C)
    view<const float, 2> fcb1{};     // (L, 4*C)
    view<const float, 3> fcw2{};     // (L, C, 4*C)
    view<const float, 2> fcb2{};     // (L, C)
    view<const float, 1> lnfw{};     // (C)
    view<const float, 1> lnfb{};     // (C)

    void setupViews(const float *data, size_t totalSize, const DimensionMap &dimMap);
    std::vector<float> buffer{}; // NOLINT
  };

  Parameters(int vocabularySize, int channelDimension, int maxSequenceLength,
             int numLayers);
  Parameters(std::ifstream &file, int vocabularySize, int channelDimension,
             int maxSequenceLength, int numLayers);
  Parameters() = default;

  [[nodiscard]] const Memory &getMemory() const { return memory; }
  [[nodiscard]] Memory &getMemory() { return memory; }
  [[nodiscard]] const DimensionMap &getDimMap() const { return dims; }

private:
  void computeDimension(int V, int C, int maxT, int L);
  DimensionMap dims{};
  Memory memory{};
};

class Scratch {
public:
  Scratch(int batchSize, int sequenceLength, int channelDimension, int numLayers,
          int vocabularySize);
  Scratch() = default;

  struct Memory {
    view<float, 3> encoded{};   // (B, T, C)
    view<float, 4> ln1{};       // (L, B, T, C)
    view<float, 4> qkv{};       // (L, B, T, 3*C)
    view<float, 4> atty{};      // (L, B, T, C)
    view<float, 4> attproj{};   // (L, B, T, C)
    view<float, 4> residual2{}; // (L, B, T, C)
    view<float, 4> ln2{};       // (L, B, T, C)
    view<float, 4> fch{};       // (L, B, T, 4*C)
    view<float, 4> fchGelu{};   // (L, B, T, 4*C)
    view<float, 4> fcproj{};    // (L, B, T, C)
    view<float, 4> residual3{}; // (L, B, T, C)
    view<float, 3> lnf{};       // (B, T, C)
    view<float, 3> logits{};    // (B, T, V)
    view<float, 3> probs{};     // (B, T, V)

    void setupViews(float *data, size_t totalSize, const DimensionMap &dimMap);
    std::vector<float> buffer{}; // NOLINT
  };

  [[nodiscard]] const Memory &getMemory() const { return memory; }
  [[nodiscard]] Memory &getMemory() { return memory; }
  [[nodiscard]] const DimensionMap &getDimMap() const { return dims; }

private:
  DimensionMap dims{};
  Memory memory{};
};

class Module {
public:
  struct Config {
    int maxSequenceLength{};
    int vocabularySize{};
    int numLayers{};
    int numHeads{};
    int channelDimension{};
  };

  explicit Module(const std::filesystem::path &file);
  Module(int maxSequenceLength, int vocabularySize, int numLayers, int numHeads,
         int channelDimension);
  void forward(view<const int, 2> inputTokenIndices);
  [[nodiscard]] Config getConfig() const { return config; }
  [[nodiscard]] Config getConfig() { return config; }
  [[nodiscard]] const Scratch &getScratch() const { return scratch; }
  [[nodiscard]] const Parameters &getParameters() const { return parameters; }

  // dangerous, use with care. Only use in unit testing, to reduce run times.
  void setConfig(const auto &conf) { config = conf; }

private:
  Config config{};
  Parameters parameters{};
  Scratch scratch{};
};

} // namespace llm::gpt2

#endif // LLM_INFERENCE_GPT2_H
