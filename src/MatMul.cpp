//
// Created by Hamidreza Khazaei on 4/20/24.
//

#include "MatMul.h"

#define ENABLE_OMP
#ifdef ENABLE_OMP
#include <omp.h>
constexpr auto numThreads = 1;
#endif

namespace llm {

void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
            view<const float, 1> bias) {
  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  LLM_ASSERT(in.extent(0) == batchSize);
  LLM_ASSERT(in.extent(1) == seqLen);
  LLM_ASSERT(in.extent(2) == inDim);
  LLM_ASSERT(out.extent(2) == outDim);
  LLM_ASSERT(bias.size() == outDim);

  for (auto batch = 0; batch < batchSize; ++batch) {
#ifdef ENABLE_OMP
#pragma omp parallel for num_threads(numThreads)
#endif
    for (auto token = 0; token < seqLen; ++token) {
      const auto inView = view<const float, 1>{&in[batch, token, 0], inDim};
      const auto outView = view<float, 1>{&out[batch, token, 0], outDim};
      for (auto outIdx = 0; outIdx < outDim; ++outIdx) {
        outView[outIdx] = bias[outIdx];
        const auto weightRow = view<const float, 1>{&weight[outIdx, 0], inDim};
        for (auto inIdx = 0; inIdx < inDim; ++inIdx) {
          outView[outIdx] += weightRow[inIdx] * inView[inIdx];
        }
      }
    }
  }
}

void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight) {
  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  LLM_ASSERT(in.extent(0) == batchSize);
  LLM_ASSERT(in.extent(1) == seqLen);
  LLM_ASSERT(in.extent(2) == inDim);
  LLM_ASSERT(out.extent(2) == outDim);

  for (auto batch = 0; batch < batchSize; ++batch) {
#ifdef ENABLE_OMP
#pragma omp parallel for num_threads(numThreads)
#endif
    for (auto token = 0; token < seqLen; ++token) {
      const auto inView = view<const float, 1>{&in[batch, token, 0], inDim};
      const auto outView = view<float, 1>{&out[batch, token, 0], outDim};
      for (auto outIdx = 0; outIdx < outDim; ++outIdx) {
        outView[outIdx] = 0;
        const auto weightRow = view<const float, 1>{&weight[outIdx, 0], inDim};
        for (auto inIdx = 0; inIdx < inDim; ++inIdx) {
          outView[outIdx] += weightRow[inIdx] * inView[inIdx];
        }
      }
    }
  }
}

#include <arm_neon.h>

void matMulNeon(view<float, 3> out, view<const float, 3> in,
                view<const float, 2> weight) {
  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  LLM_ASSERT(in.extent(0) == batchSize);
  LLM_ASSERT(in.extent(1) == seqLen);
  LLM_ASSERT(in.extent(2) == inDim);
  LLM_ASSERT(out.extent(2) == outDim);

  constexpr auto simdWidth = 4;
  for (auto batch = 0; batch < batchSize; ++batch) {
#ifdef ENABLE_OMP
#pragma omp parallel for num_threads(numThreads)
#endif
    for (auto token = 0; token < seqLen; ++token) {
      const auto inView = view<const float, 1>{&in[batch, token, 0], inDim};
      const auto outView = view<float, 1>{&out[batch, token, 0], outDim};
      for (auto outIdx = 0; outIdx < outDim; ++outIdx) {
        auto sum = vdupq_n_f32(0); // Set all elements to 0
        const auto weightRow = view<const float, 1>{&weight[outIdx, 0], inDim};

        // Handle multiples of 4 using SIMD
        auto inIdx = 0;
        for (; inIdx < inDim - simdWidth; inIdx += simdWidth) {
          const auto a = vld1q_f32(&weightRow[inIdx]);
          const auto b = vld1q_f32(&inView[inIdx]);
          sum = vmlaq_f32(sum, a, b);
        }

        // Handle the tail using scalar operations
        auto scalar_sum = vaddvq_f32(sum); // Sum up the vector elements
        for (; inIdx < inDim; ++inIdx) {
          scalar_sum += weightRow[inIdx] * inView[inIdx];
        }
        outView[outIdx] = scalar_sum;
      }
    }
  }
}

void matMulNeon(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
                view<const float, 1> bias) {
  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  LLM_ASSERT(in.extent(0) == batchSize);
  LLM_ASSERT(in.extent(1) == seqLen);
  LLM_ASSERT(in.extent(2) == inDim);
  LLM_ASSERT(out.extent(2) == outDim);

  constexpr auto simdWidth = 4;
  for (auto batch = 0; batch < batchSize; ++batch) {
#ifdef ENABLE_OMP
#pragma omp parallel for num_threads(numThreads)
#endif
    for (auto token = 0; token < seqLen; ++token) {
      const auto inView = view<const float, 1>{&in[batch, token, 0], inDim};
      const auto outView = view<float, 1>{&out[batch, token, 0], outDim};
      for (auto outIdx = 0; outIdx < outDim; ++outIdx) {
        auto sum = vdupq_n_f32(0); // Set all elements to 0
        const auto weightRow = view<const float, 1>{&weight[outIdx, 0], inDim};

        // Handle multiples of 4 using SIMD
        auto inIdx = 0;
        for (; inIdx < inDim - simdWidth; inIdx += simdWidth) {
          const auto a = vld1q_f32(&weightRow[inIdx]);
          const auto b = vld1q_f32(&inView[inIdx]);
          sum = vmlaq_f32(sum, a, b);
        }

        // Handle the tail using scalar operations
        auto scalar_sum = vaddvq_f32(sum); // Sum up the vector elements
        for (; inIdx < inDim; ++inIdx) {
          scalar_sum += weightRow[inIdx] * inView[inIdx];
        }
        outView[outIdx] = scalar_sum + bias[outIdx];
      }
    }
  }
}

} // namespace llm