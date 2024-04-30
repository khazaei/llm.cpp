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

/**
 * @brief Type that stores the dimension of each tensor. The key is the name of the
 * tensor, and the value is a vector of sizes, representing the n-dimensional size of the
 * tensor.
 */
using DimensionMap = std::map<std::string, std::vector<int>>;

/**
 * @brief Computes the total size given a dimension map.
 *
 * @param dims dimension and sizes of all the tensors.
 * @return the total size of the tensors.
 */
[[nodiscard]] size_t getTotalSize(const DimensionMap &dims);

/**
 * @brief class representing parameters (weights and biases) for GPT2.
 *
 * This class defines the memory layout of the parameters for GPT2.
 */
class Parameters {
public:
  /**
   * @brief structure containing all the parameter tensors, and their dimensions.
   *
   * A single memory allocation is done, and the memory is "cast" to the memory layout.
   */
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

  /**
   * @brief Constructs parameters given the input dimensions.
   *
   * @param vocabularySize vocabulary size for model.
   * @param channelDimension embedding dimension for model.
   * @param maxSequenceLength maximum sequence length of the model.
   * @param numLayers number of layers for model.
   */
  Parameters(int vocabularySize, int channelDimension, int maxSequenceLength,
             int numLayers);

  /**
   * @brief Constructs parameters from a file, given the input dimensions.
   *
   * @param file filestream containing the parameters of the model.
   * @param vocabularySize vocabulary size for model.
   * @param channelDimension embedding dimension for model.
   * @param maxSequenceLength maximum sequence length of the model.
   * @param numLayers number of layers for model.
   */
  Parameters(std::ifstream &file, int vocabularySize, int channelDimension,
             int maxSequenceLength, int numLayers);

  /**
   * @brief Default constructor for parameter memory.
   */
  Parameters() = default;

  /**
   * @brief returns the memory view for parameters.
   *
   * @return memory view of parameter buffer.
   */
  [[nodiscard]] Memory &getMemory() { return memory; }

private:
  void computeDimension(int V, int C, int maxT, int L);
  DimensionMap dims{};
  Memory memory{};
};

/**
 * @brief class representing scratch memory for GPT2.
 *
 * This class defines the scratch memory and layout for all intermediate computations used
 * by GPT2.
 */
class Scratch {
public:
  /**
   * @brief structure containing all the intermediate tensors, and their dimensions.
   *
   * A single memory allocation is done, and the memory is "cast" to the memory layout.
   */
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
    std::vector<float> buffer{};
  };

  /**
   * @brief Constructs scratch given the input dimensions.
   *
   * @param batchSize batch size used for training.
   * @param sequenceLength sequence length of input tokens, also known as context window.
   * @param channelDimension embedding dimension for model.
   * @param numLayers number of layers for model.
   * @param vocabularySize vocabulary size for model.
   */
  Scratch(int batchSize, int sequenceLength, int channelDimension, int numLayers,
          int vocabularySize);

  /**
   * @brief Default constructor for scratch memory.
   */
  Scratch() = default;

  /**
   * @brief returns the memory view for const scratch.
   *
   * @return memory view of scratch buffer.
   */
  [[nodiscard]] const Memory &getMemory() const { return memory; }

  /**
   * @brief returns the memory view for scratch.
   *
   * @return memory view of scratch buffer.
   */
  [[nodiscard]] Memory &getMemory() { return memory; }

private:
  DimensionMap dims{};
  Memory memory{};
};

/**
 * @brief class representing GPT2 model.
 */
class Module {
public:
  /**
   * @brief model config for GPT2.
   */
  struct Config {
    int maxSequenceLength{};
    int vocabularySize{};
    int numLayers{};
    int numHeads{};
    int channelDimension{};
  };

  /**
   * @brief Constructs GPT2 from a file.
   *
   * @param file path to the file containing the model.
   */
  explicit Module(const std::filesystem::path &file);

  /**
   * @brief Constructs GPT2 given the input dimensions.
   *
   * @param maxSequenceLength maximum sequence length of the model.
   * @param vocabularySize vocabulary size for the model.
   * @param numLayers number of layers for the model.
   * @param numHeads number of heads for the model.
   * @param channelDimension embedding dimension for the model.
   */
  Module(int maxSequenceLength, int vocabularySize, int numLayers, int numHeads,
         int channelDimension);

  /**
   * @brief forward pass.
   *
   * @param inputTokenIndices indices for the input tokens, in the shape of [batch,
   * token].
   */
  void forward(view<const int, 2> inputTokenIndices);

  /**
   * @brief returns the config for GPT2.
   *
   * @return config for GPT2.
   */
  [[nodiscard]] Config getConfig() { return config; }

  /**
   * @brief returns the scratch memory for GPT2 (used for testing mostly).
   *
   * @return scratch memory for GPT2.
   */
  [[nodiscard]] const Scratch &getScratch() const { return scratch; }

  /**
   * @brief sets the config for GPT2 (used for testing mostly).
   *
   * This function is potentially dangerous, can create mismatch with config and
   * parameters/scratch. Use with care. Only used in unit testing, to reduce run times.
   *
   * @return scratch memory for GPT2.
   */
  void setConfig(const auto &conf) { config = conf; }

private:
  Config config{};
  Parameters parameters{};
  Scratch scratch{};
};

class Tokenizer{
public:
  explicit Tokenizer(const std::filesystem::path& filename);
  [[nodiscard]] std::string decode(int tokenId) const;
  [[nodiscard]] int getEOT() const { return eotToken; }
  [[nodiscard]] int getVocabSize() const { return vocabSize; }

private:
  int eotToken {};
  int vocabSize {};
  std::vector<std::string> tokenTable{};
};

} // namespace llm::gpt2

#endif // LLM_INFERENCE_GPT2_H
