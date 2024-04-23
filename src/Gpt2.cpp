//
// Created by Hamidreza Khazaei on 4/23/24.
//

#include "Gpt2.h"

// needs to be macros for stringify
// NOLINT(BEGIN)
#define CREATE_VIEW1D(varname, data, dims)                                               \
  {                                                                                      \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data), dim.at(0)};                                    \
    std::advance((data), (varname).size());                                              \
  }

#define CREATE_VIEW2D(varname, data, dims)                                               \
  {                                                                                      \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data), dim.at(0), dim.at(1)};                         \
    std::advance((data), (varname).size());                                              \
  }

#define CREATE_VIEW3D(varname, data, dims)                                               \
  {                                                                                      \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data), dim.at(0), dim.at(1), dim.at(2)};              \
    std::advance((data), (varname).size());                                              \
  }

#define CREATE_VIEW4D(varname, data, dims)                                               \
  {                                                                                      \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data), dim.at(0), dim.at(1), dim.at(2), dim.at(3)};   \
    std::advance((data), (varname).size());                                              \
  }

#define CREATE_VIEW5D(varname, data, dims)                                               \
  {                                                                                      \
    const auto &dim = (dims).at(#varname);                                               \
    (varname) = decltype(varname){(data),    dim.at(0), dim.at(1),                       \
                                  dim.at(2), dim.at(3), dim.at(4)};                      \
    std::advance((data), (varname).size());                                              \
  }
// NOLINT(END)

namespace llm::gpt2 {

size_t getTotalSize(const DimensionMap &dims) {
  auto total = size_t{};
  for (const auto &[name, dim] : dims) {
    for (auto el : dim) {
      total += el;
    }
  }
  return total;
}

Parameters::Parameters(const size_t vocabularySize, const size_t channelDimension,
                       const size_t maxSequenceLength, const size_t numLayers) {
  const auto C = channelDimension;
  const auto V = vocabularySize;
  const auto maxT = maxSequenceLength;
  const auto L = numLayers;
  dims = {{"wte", {V, C}},
          {"wpe", {maxT, C}},
          {"ln1w", {L, C}},
          {"ln1b", {L, C}},
          {"qkvw", {L, 3 * C, C}},
          {"qkvb", {L, 3 * C}},
          {"attprojw", {L, C, C}},
          {"attprojb", {L, C}},
          {"ln2w", {L, C}},
          {"ln2b", {L, C}},
          {"fcw", {L, 4 * C, C}},
          {"fcb", {L, 4 * C}},
          {"fcprojw", {L, C, 4 * C}},
          {"fcprojb", {L, C}},
          {"lnfw", {C}},
          {"lnfb", {C}}};
}

void Parameters::Memory::assignMemory(const float *data, const int totalSize,
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
  CREATE_VIEW3D(fcw, data, dimMap);
  CREATE_VIEW2D(fcb, data, dimMap);
  CREATE_VIEW3D(fcprojw, data, dimMap);
  CREATE_VIEW2D(fcprojb, data, dimMap);
  CREATE_VIEW1D(lnfw, data, dimMap);
  CREATE_VIEW1D(lnfb, data, dimMap);
  LLM_ASSERT(data - initAddr == totalSize);
}

Scratch::Scratch(size_t batchSize, size_t sequenceLength, size_t channelDimension,
                 size_t numLayers, size_t numHeads, size_t vocabularySize) {
  const auto B = batchSize;
  const auto T = sequenceLength;
  const auto C = channelDimension;
  const auto L = numLayers;
  const auto NH = numHeads;
  const auto V = vocabularySize;

  dims = {
      {"encoded", {B, T, C}},
      {"ln1", {L, B, T, C}},
      {"ln1Mean", {L, B, T}},
      {"ln1Rstd", {L, B, T}},
      {"qkv", {L, B, T, 3 * C}},
      {"atty", {L, B, T, C}},
      {"preatt", {L, B, NH, T, T}},
      {"att", {L, B, NH, T, T}},
      {"attproj", {L, B, T, C}},
      {"residual2", {L, B, T, C}},
      {"ln2", {L, B, T, C}},
      {"ln2Mean", {L, B, T}},
      {"ln2Rstd", {L, B, T}},
      {"fch", {L, B, T, 4 * C}},
      {"fchGelu", {L, B, T, 4 * C}},
      {"fcproj", {L, B, T, C}},
      {"residual3", {L, B, T, C}},
      {"lnf", {B, T, C}},
      {"lnfMean", {B, T}},
      {"lnfRstd", {B, T}},
      {"logits", {B, T, V}},
      {"probs", {B, T, V}},
      {"losses", {B, T}},
  };
}

void Scratch::Memory::setupViews(float *data, const int totalSize,
                                 const DimensionMap &dimMap) {
  const auto *const initAddr = data;
  CREATE_VIEW3D(encoded, data, dimMap);
  CREATE_VIEW4D(ln1, data, dimMap);
  CREATE_VIEW3D(ln1Mean, data, dimMap);
  CREATE_VIEW3D(ln1Rstd, data, dimMap);
  CREATE_VIEW4D(qkv, data, dimMap);
  CREATE_VIEW4D(atty, data, dimMap);
  CREATE_VIEW5D(preatt, data, dimMap);
  CREATE_VIEW5D(att, data, dimMap);
  CREATE_VIEW4D(attproj, data, dimMap);
  CREATE_VIEW4D(residual2, data, dimMap);
  CREATE_VIEW4D(ln2, data, dimMap);
  CREATE_VIEW3D(ln2Mean, data, dimMap);
  CREATE_VIEW3D(ln2Rstd, data, dimMap);
  CREATE_VIEW4D(fch, data, dimMap);
  CREATE_VIEW4D(fchGelu, data, dimMap);
  CREATE_VIEW4D(fcproj, data, dimMap);
  CREATE_VIEW4D(residual3, data, dimMap);
  CREATE_VIEW3D(lnf, data, dimMap);
  CREATE_VIEW2D(lnfMean, data, dimMap);
  CREATE_VIEW2D(lnfRstd, data, dimMap);
  CREATE_VIEW3D(logits, data, dimMap);
  CREATE_VIEW3D(probs, data, dimMap);
  CREATE_VIEW2D(losses, data, dimMap);
  LLM_ASSERT(data - initAddr == totalSize);
}
} // namespace llm::gpt2