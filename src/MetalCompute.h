//
// Created by Hamidreza Khazaei on 5/21/24.
//

#ifndef LLM_INFERENCE_METALCOMPUTE_H
#define LLM_INFERENCE_METALCOMPUTE_H

#include "Extension.h"
#include "Metal/Metal.hpp"

namespace llm {

class MetalCompute {
public:
  void setup();
  void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight);
  void matMul(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
           view<const float, 1> bias);

private:
  MTL::CommandQueue *commandQueue{};
  MTL::Device *device{};

  std::tuple<MTL::ComputeCommandEncoder *, MTL::CommandBuffer *,
             MTL::ComputePipelineState *>
  getComputeResources(const std::string &functNameString);

  static void launchAndWait(unsigned int numThreads,
                            MTL::ComputeCommandEncoder *computeEncoder,
                            MTL::CommandBuffer *commandBuffer,
                            const MTL::ComputePipelineState *computePipelineState);
};

} // namespace llm

#endif // LLM_INFERENCE_METALCOMPUTE_H
