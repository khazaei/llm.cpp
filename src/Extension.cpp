//
// Created by Hamidreza Khazaei on 4/19/24.
//

#include "Extension.h"

namespace llm {

bool isApproximatelyEqual(const float a, const float b, const float tolerance) {
  const auto diff = std::fabs(a - b);
  if (diff <= tolerance) {
    return true;
  }

  if (diff < std::fmax(std::fabs(a), std::fabs(b)) * tolerance) {
    return true;
  }

  return false;
}

} // namespace llm