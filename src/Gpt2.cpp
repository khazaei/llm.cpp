//
// Created by Hamidreza Khazaei on 4/23/24.
//

#include "Gpt2.h"
#include "Activation.h"
#include "Attention.h"
#include "LayerNorm.h"
#include "MatOps.h"

// one way to quickly switch the underlying matrix multiplication used.
#define matMul matMulNeon

// needs to be macros for stringify
// NOLINT(BEGIN)
#define CREATE_VIEW1D(varname, data, dims)                                               \
  do {                                                                                   \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data), dim.at(0)};                                    \
    std::advance((data), (varname).size());                                              \
  } while (0)

#define CREATE_VIEW2D(varname, data, dims)                                               \
  do {                                                                                   \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data), dim.at(0), dim.at(1)};                         \
    std::advance((data), (varname).size());                                              \
  } while (0)

#define CREATE_VIEW3D(varname, data, dims)                                               \
  do {                                                                                   \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data), dim.at(0), dim.at(1), dim.at(2)};              \
    std::advance((data), (varname).size());                                              \
  } while (0)

#define CREATE_VIEW4D(varname, data, dims)                                               \
  do {                                                                                   \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data), dim.at(0), dim.at(1), dim.at(2), dim.at(3)};   \
    std::advance((data), (varname).size());                                              \
  } while (0)

// NOLINT(END)

namespace llm::gpt2 {

size_t getTotalSize(const DimensionMap &dims) {
  auto total = size_t{};
  for (const auto &[name, dim] : dims) {
    auto prod = size_t{1};
    for (auto el : dim) {
      prod *= el;
    }
    total += prod;
  }
  return total;
}

Parameters::Parameters(const int vocabularySize, const int channelDimension,
                       const int maxSequenceLength, const int numLayers) {
  computeDimension(vocabularySize, channelDimension, maxSequenceLength, numLayers);

  const auto paramSize = getTotalSize(dims);
  memory.buffer = std::vector<float>(paramSize);

  memory.setupViews(memory.buffer.data(), paramSize, dims);
}

Parameters::Parameters(std::ifstream &in, const int vocabularySize,
                       const int channelDimension, const int maxSequenceLength,
                       const int numLayers)
    : Parameters(vocabularySize, channelDimension, maxSequenceLength, numLayers) {

  in.read(reinterpret_cast<char *>(memory.buffer.data()),         // NOLINT
          static_cast<long>(sizeof(float)) * getTotalSize(dims)); // NOLINT
}

void Parameters::computeDimension(const int V, const int C, const int maxT, const int L) {
  dims = {{"wte", {V, C}},         {"wpe", {maxT, C}},
          {"ln1w", {L, C}},        {"ln1b", {L, C}},
          {"qkvw", {L, 3 * C, C}}, {"qkvb", {L, 3 * C}},
          {"attprojw", {L, C, C}}, {"attprojb", {L, C}},
          {"ln2w", {L, C}},        {"ln2b", {L, C}},
          {"fcw1", {L, 4 * C, C}}, {"fcb1", {L, 4 * C}},
          {"fcw2", {L, C, 4 * C}}, {"fcb2", {L, C}},
          {"lnfw", {C}},           {"lnfb", {C}}};
}

void Parameters::Memory::setupViews(const float *data, const size_t totalSize,
                                    const DimensionMap &dimMap) {
  const auto *const initAddr = data;
  CREATE_VIEW2D(wte, data, dimMap);
  CREATE_VIEW2D(wpe, data, dimMap);
  CREATE_VIEW2D(ln1w, data, dimMap);
  CREATE_VIEW2D(ln1b, data, dimMap);
  CREATE_VIEW3D(qkvw, data, dimMap);
  CREATE_VIEW2D(qkvb, data, dimMap);
  CREATE_VIEW3D(attprojw, data, dimMap);
  CREATE_VIEW2D(attprojb, data, dimMap);
  CREATE_VIEW2D(ln2w, data, dimMap);
  CREATE_VIEW2D(ln2b, data, dimMap);
  CREATE_VIEW3D(fcw1, data, dimMap);
  CREATE_VIEW2D(fcb1, data, dimMap);
  CREATE_VIEW3D(fcw2, data, dimMap);
  CREATE_VIEW2D(fcb2, data, dimMap);
  CREATE_VIEW1D(lnfw, data, dimMap);
  CREATE_VIEW1D(lnfb, data, dimMap);
  const auto diff = data - initAddr;
  LLM_ASSERT(diff == totalSize);
}

Scratch::Scratch(const int batchSize, const int sequenceLength,
                 const int channelDimension, const int numLayers,
                 const int vocabularySize) {
  const auto B = batchSize;
  const auto T = sequenceLength;
  const auto C = channelDimension;
  const auto L = numLayers;
  const auto V = vocabularySize;

  dims = {
      {"encoded", {B, T, C}},        {"ln1", {L, B, T, C}},
      {"qkv", {L, B, T, 3 * C}},     {"atty", {L, B, T, C}},
      {"attproj", {L, B, T, C}},     {"residual2", {L, B, T, C}},
      {"ln2", {L, B, T, C}},         {"fch", {L, B, T, 4 * C}},
      {"fchGelu", {L, B, T, 4 * C}}, {"fcproj", {L, B, T, C}},
      {"residual3", {L, B, T, C}},   {"lnf", {B, T, C}},
      {"logits", {B, T, V}},         {"probs", {B, T, V}},
  };

  const auto paramSize = getTotalSize(dims);
  memory.buffer = std::vector<float>(paramSize);
  memory.setupViews(memory.buffer.data(), paramSize, dims);
}

void Scratch::Memory::setupViews(float *data, const size_t totalSize,
                                 const DimensionMap &dimMap) {
  const auto *const initAddr = data;
  CREATE_VIEW3D(encoded, data, dimMap);
  CREATE_VIEW4D(ln1, data, dimMap);
  CREATE_VIEW4D(qkv, data, dimMap);
  CREATE_VIEW4D(atty, data, dimMap);
  CREATE_VIEW4D(attproj, data, dimMap);
  CREATE_VIEW4D(residual2, data, dimMap);
  CREATE_VIEW4D(ln2, data, dimMap);
  CREATE_VIEW4D(fch, data, dimMap);
  CREATE_VIEW4D(fchGelu, data, dimMap);
  CREATE_VIEW4D(fcproj, data, dimMap);
  CREATE_VIEW4D(residual3, data, dimMap);
  CREATE_VIEW3D(lnf, data, dimMap);
  CREATE_VIEW3D(logits, data, dimMap);
  CREATE_VIEW3D(probs, data, dimMap);
  const auto diff = data - initAddr;
  LLM_ASSERT(diff == totalSize);
}

Module::Module(const std::filesystem::path &file) {
  LLM_ASSERT(std::filesystem::exists(file));

  auto in = std::ifstream{file, std::ios_base::binary};
  auto header = std::vector<int>(256);
  readIntoVector(in, header);

  LLM_ASSERT(header[0] == 20240326); // magic model number
  LLM_ASSERT(header[1] == 1);        // bad version

  // read in hyperparameters
  config.maxSequenceLength = header[2];
  config.vocabularySize = header[3];
  config.numLayers = header[4];
  config.numHeads = header[5];
  config.channelDimension = header[6];

  parameters = Parameters{in, config.vocabularySize, config.channelDimension,
                          config.maxSequenceLength, config.numLayers};
}

void Module::forward(view<const int, 2> inputTokenIndices) {

  const auto batchSize = inputTokenIndices.extent(0);
  const auto seqLen = inputTokenIndices.extent(1);
  LLM_ASSERT(parameters.getMemory().buffer.size() != 0);
  scratch = Scratch{batchSize, seqLen, config.channelDimension, config.numLayers,
                    config.vocabularySize};

  const auto &p = parameters.getMemory();
  auto &a = scratch.getMemory();

  positionalEncoding(a.encoded, inputTokenIndices, p.wte, p.wpe);
  auto residual = a.encoded;

  for (auto layer = 0; layer < config.numLayers; ++layer) {
    // get the views for layer norm
    const auto ln1W = view<const float, 1>{&p.ln1w[layer, 0], p.ln1w.extent(1)};
    const auto ln1B = view<const float, 1>{&p.ln1b[layer, 0], p.ln1b.extent(1)};
    const auto ln1 = view<float, 3>{&a.ln1[layer, 0, 0, 0], a.ln1.extent(1),
                                    a.ln1.extent(2), a.ln1.extent(3)};
    layerNorm(ln1, residual, ln1W, ln1B);

    // get the views for embedding -> qkv projection
    const auto qkv = view<float, 3>{&a.qkv[layer, 0, 0, 0], a.qkv.extent(1),
                                    a.qkv.extent(2), a.qkv.extent(3)};
    const auto qkvW =
        view<const float, 2>{&p.qkvw[layer, 0, 0], p.qkvw.extent(1), p.qkvw.extent(2)};
    const auto qkvB = view<const float, 1>{&p.qkvb[layer, 0], p.qkvb.extent(1)};
    // proj C -> 3C
    matMul(qkv, ln1, qkvW, qkvB);

    // get view for attention
    const auto attOut = view<float, 3>{&a.atty[layer, 0, 0, 0], a.atty.extent(1),
                                       a.atty.extent(2), a.atty.extent(3)};
    multiHeadAttentionCausal(attOut, qkv, config.numHeads);

    // get view attention projection
    const auto attProj = view<float, 3>{&a.attproj[layer, 0, 0, 0], a.attproj.extent(1),
                                        a.attproj.extent(2), a.attproj.extent(3)};
    const auto attProjW = view<const float, 2>{
        &p.attprojw[layer, 0, 0], p.attprojw.extent(1), p.attprojw.extent(2)};
    const auto attProjB =
        view<const float, 1>{&p.attprojb[layer, 0], p.attprojb.extent(1)};
    // proj C -> C
    matMul(attProj, attOut, attProjW, attProjB);

    // get view for residual add
    const auto residual2 =
        view<float, 3>{&a.residual2[layer, 0, 0, 0], a.residual2.extent(1),
                       a.residual2.extent(2), a.residual2.extent(3)};
    matAdd(residual2, residual, attProj);

    // get the views for layer norm
    const auto ln2W = view<const float, 1>{&p.ln2w[layer, 0], p.ln2w.extent(1)};
    const auto ln2B = view<const float, 1>{&p.ln2b[layer, 0], p.ln2b.extent(1)};
    const auto ln2 = view<float, 3>{&a.ln2[layer, 0, 0, 0], a.ln2.extent(1),
                                    a.ln2.extent(2), a.ln2.extent(3)};
    layerNorm(ln2, residual2, ln2W, ln2B);

    // get the views for fully connected projection
    const auto fch = view<float, 3>{&a.fch[layer, 0, 0, 0], a.fch.extent(1),
                                    a.fch.extent(2), a.fch.extent(3)};
    const auto fcW =
        view<const float, 2>{&p.fcw1[layer, 0, 0], p.fcw1.extent(1), p.fcw1.extent(2)};
    const auto fcB = view<const float, 1>{&p.fcb1[layer, 0], p.fcb1.extent(1)};
    // proj C -> 4C
    matMul(fch, ln2, fcW, fcB);

    // get view for gelu
    const auto fchGelu = view<float, 3>{&a.fchGelu[layer, 0, 0, 0], a.fchGelu.extent(1),
                                        a.fchGelu.extent(2), a.fchGelu.extent(3)};
    gelu(fchGelu, fch);

    // get view for fully connected proj
    const auto fcProj = view<float, 3>{&a.fcproj[layer, 0, 0, 0], a.fcproj.extent(1),
                                       a.fcproj.extent(2), a.fcproj.extent(3)};
    const auto fcProjW =
        view<const float, 2>{&p.fcw2[layer, 0, 0], p.fcw2.extent(1), p.fcw2.extent(2)};
    const auto fcProjB = view<const float, 1>{&p.fcb2[layer, 0], p.fcb2.extent(1)};
    // proj 4C -> C
    matMul(fcProj, fchGelu, fcProjW, fcProjB);

    // get view for residual
    const auto residual3 =
        view<float, 3>{&a.residual3[layer, 0, 0, 0], a.residual3.extent(1),
                       a.residual3.extent(2), a.residual3.extent(3)};
    matAdd(residual3, residual2, fcProj);

    residual = residual3;
  }

  layerNorm(a.lnf, residual, p.lnfw, p.lnfb);
  matMul(a.logits, a.lnf, p.wte);
  // needed for sampling from a distribution, can it be optimized?
  softmax(a.probs, a.logits);
}

Module::Module(const int maxSequenceLength, const int vocabularySize, const int numLayers,
               const int numHeads, const int channelDimension)
    : config{maxSequenceLength, vocabularySize, numLayers, numHeads, channelDimension},
      parameters{vocabularySize, channelDimension, maxSequenceLength, numLayers} {}

Tokenizer::Tokenizer(const std::filesystem::path &filename) {
  auto fileStream = std::ifstream{filename, std::ios::binary};
  auto header = std::vector<int>(256);
  readIntoVector(fileStream, header);

  LLM_ASSERT(header[0] == 20240328);
  LLM_ASSERT(header[1] == 2); // version
  vocabSize = header[2];
  LLM_ASSERT(vocabSize == 50257);

  eotToken = header[3];
  LLM_ASSERT(eotToken == 50256);

  tokenTable.reserve(vocabSize);
  for (auto i = 0; i < vocabSize; ++i) {
    unsigned char length = 0;
    fileStream.read(reinterpret_cast<char *>(&length), sizeof(char)); // NOLINT
    LLM_ASSERT(length > 0);

    // read length char into vector and add null terminator.
    auto stringVec = std::vector<char>(length);
    readIntoVector(fileStream, stringVec);

    tokenTable.emplace_back(stringVec.cbegin(), stringVec.cend());
  }
}

std::string Tokenizer::decode(const int tokenId) const {
  LLM_ASSERT(tokenId < vocabSize);
  return tokenTable[tokenId];
}

} // namespace llm::gpt2