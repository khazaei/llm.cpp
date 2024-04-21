//
// Created by Hamidreza Khazaei on 4/19/24.
//

#ifndef LLM_INFERENCE_EXTENSION_H
#define LLM_INFERENCE_EXTENSION_H

#include <cassert>
#include <iostream>
#include <mdspan>

namespace llm {

template <typename T> using view3d = std::mdspan<T, std::dextents<size_t, 3>>;
template <typename T> using view2d = std::mdspan<T, std::dextents<size_t, 2>>;

#define GET_MACRO(_1, _2, NAME, ...) NAME // NOLINT
#define LLM_ASSERT(...)                                                                  \
  GET_MACRO(__VA_ARGS__, LLM_ASSERT2, LLM_ASSERT1)(__VA_ARGS__) // NOLINT

#ifdef NDEBUG
#define LLM_ASSERT1(check) ((check) ? ((void)0) : std::abort())
#define LLM_ASSERT2(exp, msg) ((exp) ? ((void)0) : std::abort())
#else
#define LLM_ASSERT1(check) assert((check))         // NOLINT
#define LLM_ASSERT2(exp, msg) assert((exp) && msg) // NOLINT
#endif

bool isApproximatelyEqual(float a, float b,
                          float tolerance = std::numeric_limits<float>::epsilon());

template <typename T>
static bool isTensorsEqual(const T &a, const T &b, const float eps = std::numeric_limits<float>::epsilon()) {
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
