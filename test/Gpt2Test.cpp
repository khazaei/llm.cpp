//
// Created by Hamidreza Khazaei on 4/24/24.
//

#include "Gpt2.h"
#include "Llm.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Sanity test for creating GPT2.") {

  auto maxSequenceLength = 100;
  auto vocabularySize = 1000;
  auto numLayers = 2;
  auto numHeads = 2;
  auto channelDimension = 256;

  auto gpt = llm::gpt2::Module{maxSequenceLength, vocabularySize, numLayers, numHeads,
                               channelDimension};

  auto in = std::vector<int>(800);
  const auto v = llm::view<const int, 2>{in.data(), 8, 100};
  gpt.forward(v);
}

TEST_CASE("Test GPT2 single layer.") {

  // the binary is usually run in PROJECTROOT/cmake-build-*/test/
  constexpr auto paramFile = "../../test/input/gpt2_124M.bin";
  CHECK(std::filesystem::exists(paramFile));
  auto gpt2 = llm::gpt2::Module{paramFile};
  auto configChange = gpt2.getConfig();
  configChange.numLayers = 1;
  gpt2.setConfig(configChange);

  // load additional information that we will use for debugging and error checking
  constexpr auto debugFile = "../../test/input/gpt2_124M_debug_state.bin";
  CHECK(std::filesystem::exists(debugFile));
  auto stateFile = std::ifstream{debugFile, std::ios_base::binary};
  auto stateHeader = std::vector<int>(256);
  llm::readIntoVector(stateFile, stateHeader);

  CHECK(stateHeader[0] == 20240327); // "Bad magic state file"
  CHECK(stateHeader[1] == 1);        // "Bad version in state file"
  const auto B = stateHeader[2];     // batch size, e.g. 4
  const auto T = stateHeader[3];     // time / sequence length (e.g. 64, up to maxT)

  // inputs and expected outputs, only used for error checking
  auto x = std::vector<int>(static_cast<size_t>(B) * T);
  llm::readIntoVector(stateFile, x);
  gpt2.forward(llm::view<int, 2>{x.data(), B, T});

  constexpr auto intermediateFile = "../../test/input/single_layer_debug.bin";
  CHECK(std::filesystem::exists(intermediateFile));
  auto actFile = std::ifstream{intermediateFile, std::ios_base::binary};

  // check positional encoding
  const auto C = gpt2.getConfig().channelDimension;
  auto posOut = std::vector<float>(static_cast<size_t>(B) * T * C);
  llm::readIntoVector(actFile, posOut);
  const auto gptPosOut =
      std::span<float>{gpt2.getScratch().getMemory().encoded.data_handle(),
                       gpt2.getScratch().getMemory().encoded.size()};
  CHECK(llm::isTensorsEqual(gptPosOut, posOut));

  // check 1st layer norm
  auto lnOut = std::vector<float>(static_cast<size_t>(B) * T * C);
  llm::readIntoVector(actFile, lnOut);
  const auto ln1out = std::span<float>{gpt2.getScratch().getMemory().ln1.data_handle(),
                                       gpt2.getScratch().getMemory().ln1.size()};
  CHECK(llm::isTensorsEqual(ln1out, lnOut, 1e-6));

  // check qkv projection
  auto qkvOutFile = std::vector<float>(static_cast<size_t>(B) * T * 3 * C);
  llm::readIntoVector(actFile, qkvOutFile);
  const auto qkvOutCode =
      std::span<float>{gpt2.getScratch().getMemory().qkv.data_handle(),
                       gpt2.getScratch().getMemory().qkv.size()};
  CHECK(llm::isTensorsEqual(qkvOutFile, qkvOutCode, 1e-5));

  // check multi head attention
  auto mhOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  llm::readIntoVector(actFile, mhOutFile);
  const auto mhOutCode =
      std::span<float>{gpt2.getScratch().getMemory().atty.data_handle(),
                       gpt2.getScratch().getMemory().atty.size()};
  CHECK(llm::isTensorsEqual(mhOutFile, mhOutCode, 1e-5));

  // check  attention proj
  auto attnOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  llm::readIntoVector(actFile, attnOutFile);
  const auto attnOutCode =
      std::span<float>{gpt2.getScratch().getMemory().attproj.data_handle(),
                       gpt2.getScratch().getMemory().attproj.size()};
  CHECK(llm::isTensorsEqual(attnOutFile, attnOutCode, 1e-5));

  // check first residual
  auto resOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  llm::readIntoVector(actFile, resOutFile);
  const auto resOutCode =
      std::span<float>{gpt2.getScratch().getMemory().residual2.data_handle(),
                       gpt2.getScratch().getMemory().residual2.size()};
  CHECK(llm::isTensorsEqual(resOutFile, resOutCode, 1e-5));

  // check out of mlp
  auto mlpOutFile = std::vector<float>(static_cast<size_t>(B) * T * C);
  llm::readIntoVector(actFile, mlpOutFile);
  const auto mlpOutCode =
      std::span<float>{gpt2.getScratch().getMemory().fcproj.data_handle(),
                       gpt2.getScratch().getMemory().fcproj.size()};
  CHECK(llm::isTensorsEqual(mlpOutFile, mlpOutCode, 1e-4));

  // check logits
  // check out of layer norm + mlp + res
  const auto V = gpt2.getConfig().vocabularySize;
  auto logitOutFile = std::vector<float>(static_cast<size_t>(B) * T * V);
  llm::readIntoVector(actFile, logitOutFile);
  const auto logitOutCode =
      std::span<float>{gpt2.getScratch().getMemory().logits.data_handle(),
                       gpt2.getScratch().getMemory().logits.size()};
  CHECK(llm::isTensorsEqual(logitOutFile, logitOutCode, 1e-4));

  // check probs
  // check out of layer norm + mlp + res
  auto probOutFile = std::vector<float>(static_cast<size_t>(B) * T * V);
  llm::readIntoVector(actFile, probOutFile);
  const auto probOutCode =
      std::span<float>{gpt2.getScratch().getMemory().probs.data_handle(),
                       gpt2.getScratch().getMemory().probs.size()};

  CHECK(llm::isTensorsEqual(probOutFile, probOutCode, 1e-3));

  llm::readIntoVector(actFile, probOutFile);
}

TEST_CASE("Test GPT2 multi layer.") {
  // the binary is usually run in PROJECTROOT/cmake-build-*/test/
  constexpr auto paramFile = "../../test/input/gpt2_124M.bin";
  CHECK(std::filesystem::exists(paramFile));
  auto gpt2 = llm::gpt2::Module{paramFile};

  // load additional information that we will use for debugging and error checking
  constexpr auto debugFile = "../../test/input/gpt2_124M_debug_state.bin";
  CHECK(std::filesystem::exists(debugFile));
  auto stateFile = std::ifstream{debugFile, std::ios_base::binary};
  auto stateHeader = std::vector<int>(256);
  llm::readIntoVector(stateFile, stateHeader);

  CHECK(stateHeader[0] == 20240327); // "Bad magic state file"
  CHECK(stateHeader[1] == 1);        // "Bad version in state file"
  const auto B = stateHeader[2];     // batch size, e.g. 4
  const auto T = stateHeader[3];     // time / sequence length (e.g. 64, up to maxT)

  // inputs and expected outputs, only used for error checking
  auto x = std::vector<int>(static_cast<size_t>(B) * T);
  llm::readIntoVector(stateFile, x);
  gpt2.forward(llm::view<int, 2>{x.data(), B, T});

  constexpr auto intermediateFile = "../../test/input/multi_layer_debug.bin";
  CHECK(std::filesystem::exists(intermediateFile));
  auto actFile = std::ifstream{intermediateFile, std::ios_base::binary};

  // check logits
  const auto V = gpt2.getConfig().vocabularySize;
  auto logitOutFile = std::vector<float>(static_cast<size_t>(B) * T * V);
  llm::readIntoVector(actFile, logitOutFile);
  const auto logitOutCode =
      std::span<float>{gpt2.getScratch().getMemory().logits.data_handle(),
                       gpt2.getScratch().getMemory().logits.size()};
  CHECK(llm::isTensorsEqual(logitOutFile, logitOutCode, 1e-4));

  // check probs
  auto probOutFile = std::vector<float>(static_cast<size_t>(B) * T * V);
  llm::readIntoVector(actFile, probOutFile);
  const auto probOutCode =
      std::span<float>{gpt2.getScratch().getMemory().probs.data_handle(),
                       gpt2.getScratch().getMemory().probs.size()};

  CHECK(llm::isTensorsEqual(probOutFile, probOutCode, 1e-3));

  llm::readIntoVector(actFile, probOutFile);
}

// - basic implementation no optimizations 26 seconds.
// - with open mp running on 10 cores you get 5 seconds.
// - Neon intrinsics 9 second.
// - Neon + omp  2 second.
TEST_CASE("profile GPT2 124M param.") {

  // the binary is usually run in PROJECTROOT/cmake-build-*/test/
  constexpr auto paramFile = "../../test/input/gpt2_124M.bin";
  CHECK(std::filesystem::exists(paramFile));
  auto gpt2 = llm::gpt2::Module{paramFile};

  // load additional information that we will use for debugging and error checking
  constexpr auto debugFile = "../../test/input/gpt2_124M_debug_state.bin";
  CHECK(std::filesystem::exists(debugFile));
  auto stateFile = std::ifstream{debugFile, std::ios_base::binary};
  auto stateHeader = std::vector<int>(256);
  llm::readIntoVector(stateFile, stateHeader);

  CHECK(stateHeader[0] == 20240327); // "Bad magic state file"
  CHECK(stateHeader[1] == 1);        // "Bad version in state file"
  const auto B = stateHeader[2];     // batch size, e.g. 4
  const auto T = stateHeader[3];     // time / sequence length (e.g. 64, up to maxT)

  // inputs and expected outputs, only used for error checking
  auto x = std::vector<int>(static_cast<size_t>(B) * T);
  llm::readIntoVector(stateFile, x);
  gpt2.forward(llm::view<int, 2>{x.data(), B, T});
}

TEST_CASE("Test tokenizer.") {
  constexpr auto tokenizerFile = "../../test/input/gpt2_tokenizer.bin";
  CHECK(std::filesystem::exists(tokenizerFile));
  auto tokenizer = llm::gpt2::Tokenizer{tokenizerFile};

  CHECK(tokenizer.decode(0) == "!");
  CHECK(tokenizer.decode(11) == ",");
  CHECK(tokenizer.decode(1001) == " Se");
  CHECK(tokenizer.decode(2222) == " bring");
  CHECK(tokenizer.decode(10565) == "leton");
  CHECK(tokenizer.decode(37456) == " lambda");
  CHECK(tokenizer.decode(50000) == " grids");
  CHECK(tokenizer.decode(50256) == "<|endoftext|>");
}

TEST_CASE("GPT inference.") {

  // the binary is usually run in PROJECTROOT/cmake-build-*/test/
  constexpr auto paramFile = "../../test/input/gpt2_124M.bin";
  CHECK(std::filesystem::exists(paramFile));
  auto gpt2 = llm::gpt2::Module{paramFile};

  constexpr auto tokenizerFile = "../../test/input/gpt2_tokenizer.bin";
  CHECK(std::filesystem::exists(tokenizerFile));
  auto tokenizer = llm::gpt2::Tokenizer{tokenizerFile};

  llm::genToken(gpt2, tokenizer, 64);
}