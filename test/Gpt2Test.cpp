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

template <typename T> void readIntoVector(std::ifstream &file, std::vector<T> &vec) {
  file.read(reinterpret_cast<char *>(vec.data()), sizeof(T) * vec.size()); // NOLINT
}

TEST_CASE("Test GPT2.") {

  // the binary is usually run in PROJECTROOT/cmake-build-*/test/
  constexpr auto paramFile = "../../test/input/gpt2_124M.bin";
  LLM_ASSERT(std::filesystem::exists(paramFile));
  auto gpt2 = llm::gpt2::Module{paramFile};
  auto configChange = gpt2.getConfig();
  configChange.numLayers = 1;
  gpt2.setConfig(configChange);

  // load additional information that we will use for debugging and error checking
  constexpr auto debugFile = "../../test/input/gpt2_124M_debug_state.bin";
  LLM_ASSERT(std::filesystem::exists(debugFile));
  auto stateFile = std::ifstream{debugFile, std::ios_base::binary};
  auto stateHeader = std::vector<int>(256);
  readIntoVector(stateFile, stateHeader);

  CHECK(stateHeader[0] == 20240327); // "Bad magic state file"
  CHECK(stateHeader[1] == 1);        // "Bad version in state file"
  const auto B = stateHeader[2];     // batch size, e.g. 4
  const auto T = stateHeader[3];     // time / sequence length (e.g. 64, up to maxT)
  std::cout << "[State]\n";
  std::cout << "batch_size: " << B << "\n";
  std::cout << "seq_len: " << T << "\n";

  // inputs and expected outputs, only used for error checking
  auto x = std::vector<int>(static_cast<size_t>(B) * T);
  readIntoVector(stateFile, x);
  gpt2.forward(llm::view<int, 2>{x.data(), B, T});

  // check positional encoding
  constexpr auto intermediateFile = "../../test/input/act_debug.bin";
  LLM_ASSERT(std::filesystem::exists(intermediateFile));
  auto actFile = std::ifstream{intermediateFile, std::ios_base::binary};
  const auto C = gpt2.getConfig().channelDimension;
  auto posOut = std::vector<float>(static_cast<size_t>(B) * T * C);
  readIntoVector(actFile, posOut);
  const auto gptPosOut =
      std::span<float>{gpt2.getScratch().getMemory().encoded.data_handle(),
                       gpt2.getScratch().getMemory().encoded.size()};
  CHECK(llm::isTensorsEqual(gptPosOut, posOut));

  // check 1st layer norm
  auto lnOut = std::vector<float>(static_cast<size_t>(B) * T * C);
  readIntoVector(actFile, lnOut);
  const auto ln1out = std::span<float>{gpt2.getScratch().getMemory().ln1.data_handle(),
                                       gpt2.getScratch().getMemory().ln1.size()};
  CHECK(llm::isTensorsEqual(ln1out, lnOut, 1e-6));

  // check qkv projection
  auto qkvOutFile = std::vector<float>(static_cast<size_t>(B) * T * 3 * C);
  readIntoVector(actFile, qkvOutFile);
  const auto qkvOutCode =
      std::span<float>{gpt2.getScratch().getMemory().qkv.data_handle(),
                       gpt2.getScratch().getMemory().qkv.size()};
  CHECK(llm::isTensorsEqual(qkvOutFile, qkvOutCode, 1e-5));

  // check multi head attention
  auto mhOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  readIntoVector(actFile, mhOutFile);
  const auto mhOutCode =
      std::span<float>{gpt2.getScratch().getMemory().atty.data_handle(),
                       gpt2.getScratch().getMemory().atty.size()};
  CHECK(llm::isTensorsEqual(mhOutFile, mhOutCode, 1e-5));

  // check  attention proj
  auto attnOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  readIntoVector(actFile, attnOutFile);
  const auto attnOutCode =
      std::span<float>{gpt2.getScratch().getMemory().attproj.data_handle(),
                       gpt2.getScratch().getMemory().attproj.size()};
  CHECK(llm::isTensorsEqual(attnOutFile, attnOutCode, 1e-5));

  // check first residual
  auto resOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  readIntoVector(actFile, resOutFile);
  const auto resOutCode =
      std::span<float>{gpt2.getScratch().getMemory().residual2.data_handle(),
                       gpt2.getScratch().getMemory().residual2.size()};
  CHECK(llm::isTensorsEqual(resOutFile, resOutCode, 1e-5));

  // check out of mlp
  auto mlpOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  readIntoVector(actFile, mlpOutFile);
  const auto mlpOutCode =
      std::span<float>{gpt2.getScratch().getMemory().fcproj.data_handle(),
                       gpt2.getScratch().getMemory().fcproj.size()};
  // xxx investigate where resolution is dropping
  CHECK(llm::isTensorsEqual(mlpOutFile, mlpOutCode, 1e-4));

  // check logits
  // check out of layer norm + mlp + res
  auto logitOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  readIntoVector(actFile, logitOutFile);
  const auto logitOutCode =
      std::span<float>{gpt2.getScratch().getMemory().logits.data_handle(),
                       gpt2.getScratch().getMemory().logits.size()};
  // xxx investigate where resolution is dropping
  CHECK(llm::isTensorsEqual(logitOutFile, logitOutCode, 1e-4));
}