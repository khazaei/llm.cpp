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
  void run(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight);
  void run(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
           view<const float, 1> bias);

private:
  MTL::CommandQueue *commandQueue{};
  MTL::Device *device{};
};

} // namespace llm

#endif // LLM_INFERENCE_METALCOMPUTE_H
