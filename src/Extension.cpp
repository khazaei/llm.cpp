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

int sampleDiscreteDistribution(view<const float, 1> probabilities, std::mt19937_64 &rng) {

  auto cdf = 0.0F;
  auto dist = std::uniform_real_distribution<float>{0, 1};
  const auto randVal = dist(rng);
  for (auto i = 0; i < probabilities.extent(0); ++i) {
    cdf += probabilities[i];
    if (randVal < cdf) {
      return i;
    }
  }
  return probabilities.extent(0) - 1; // in case of rounding errors
}

} // namespace llm