//
// Created by Hamidreza Khazaei on 4/19/24.
//

#ifndef LLM_INFERENCE_EXTENSION_H
#define LLM_INFERENCE_EXTENSION_H

#include <cassert>
#include <iostream>
#include <mdspan>

namespace llm {

template <typename T, int dim> using view = std::mdspan<T, std::dextents<size_t, dim>>;

#define GET_MACRO(_1, _2, NAME, ...) NAME // NOLINT
#define LLM_ASSERT(...)                                                                  \
  GET_MACRO(__VA_ARGS__, LLM_ASSERT2, LLM_ASSERT1)(__VA_ARGS__) // NOLINT

#ifdef NDEBUG
#define LLM_ASSERT1(check)
#define LLM_ASSERT2(exp, msg)
#else
#define LLM_ASSERT1(check) assert((check))         // NOLINT
#define LLM_ASSERT2(exp, msg) assert((exp) && msg) // NOLINT
#endif

bool isApproximatelyEqual(float a, float b,
                          float tolerance = std::numeric_limits<float>::epsilon());

template <typename T>
static bool isTensorsEqual(const T &a, const T &b,
                           const float eps = std::numeric_limits<float>::epsilon()) {
  for (auto i = 0; i < a.size(); ++i) {
    if (!isApproximatelyEqual(a.at(i), b.at(i), eps)) {
      std::cout << "values not equal at index: " << i << " " << a.at(i) << " " << b.at(i)
                << std::endl;
      return false;
    }
  }

  return true;
}

} // namespace llm

#endif // LLM_INFERENCE_EXTENSION_H
