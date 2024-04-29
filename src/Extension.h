//
// Created by Hamidreza Khazaei on 4/19/24.
//

#ifndef LLM_INFERENCE_EXTENSION_H
#define LLM_INFERENCE_EXTENSION_H

#include <cassert>
#include <iostream>
#include <mdspan>

namespace llm {

/**
 * @brief Type for creating a data view of different shapes for a raw array of data.
 */
template <typename T, int dim> using view = std::mdspan<T, std::dextents<int, dim>>;

/**
 * @brief Asserts for debugging.
 */
#define GET_MACRO(_1, _2, NAME, ...) NAME // NOLINT
#define LLM_ASSERT(...)                                                                  \
  GET_MACRO(__VA_ARGS__, LLM_ASSERT2, LLM_ASSERT1)(__VA_ARGS__) // NOLINT

#ifdef NDEBUG
#define LLM_ASSERT1(check) ((void)(check))  // NOLINT
#define LLM_ASSERT2(exp, msg) ((void)(exp)) // NOLINT
#else
#define LLM_ASSERT1(check) assert((check))         // NOLINT
#define LLM_ASSERT2(exp, msg) assert((exp) && msg) // NOLINT
#endif

/**
 * @brief Comparison function for two floats.
 *
 * @tparam a first float.
 * @tparam b second float.
 * @param tolerance the tolerance to compare the floats.
 * @return true if the are approximately equal, false otherwise.
 */
bool isApproximatelyEqual(float a, float b,
                          float tolerance = std::numeric_limits<float>::epsilon());

/**
 * @brief Compares two tensors and determines if they are the same.
 *
 * @tparam a first tensor container.
 * @tparam b second tensor container.
 * @param eps the tolerance to compare the tensors.
 * @return true if the are approximately equal, false otherwise.
 */
template <typename T1, typename T2>
static bool isTensorsEqual(T1 &a, T2 &b,
                           const float eps = std::numeric_limits<float>::epsilon()) {
  for (auto i = 0; i < a.size(); ++i) {
    if (!isApproximatelyEqual(a[i], b[i], eps)) {
      std::cout << "values not equal at index: " << i << " " << a[i] << " " << b[i]
                << std::endl;
      return false;
    }
  }

  return true;
}

} // namespace llm

#endif // LLM_INFERENCE_EXTENSION_H
