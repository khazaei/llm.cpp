//
// Created by Hamidreza Khazaei on 5/21/24.
//

#include "MetalCompute.h"
#include "Extension.h"

namespace llm {

constexpr auto metalLibName = "../shader/default.metallib";

void MetalCompute::setup() {
  device = MTL::CreateSystemDefaultDevice();
  const auto *libraryPath = NS::String::string(metalLibName, NS::ASCIIStringEncoding);

  NS::Error *error = nullptr;
  auto *lib = device->newLibrary(libraryPath, &error);
  LLM_ASSERT(error == nullptr);
  LLM_ASSERT(lib != nullptr);

  const auto *functionName =
      NS::String::string("matMul", NS::ASCIIStringEncoding);
  const auto *computeFunction = lib->newFunction(functionName);
  LLM_ASSERT(computeFunction != nullptr);

  computePipelineState = device->newComputePipelineState(computeFunction, &error);

  LLM_ASSERT(computePipelineState != nullptr);
  LLM_ASSERT(error == nullptr);

  commandQueueMatMul = device->newCommandQueue();
  LLM_ASSERT(commandQueueMatMul != nullptr);

  const auto *functionNameBias =
      NS::String::string("matMulWithBias", NS::ASCIIStringEncoding);
  const auto *computeFunctionBias = lib->newFunction(functionNameBias);
  LLM_ASSERT(computeFunctionBias != nullptr);

  computePipelineStateBias = device->newComputePipelineState(computeFunctionBias, &error);

  LLM_ASSERT(computePipelineStateBias != nullptr);
  LLM_ASSERT(error == nullptr);

  commandQueueMatMulBias = device->newCommandQueue();
  LLM_ASSERT(commandQueueMatMulBias != nullptr);
}

void MetalCompute::run(view<float, 3> out, view<const float, 3> in,
                       view<const float, 2> weight) {

  auto *commandBuffer = commandQueueMatMul->commandBuffer();
  LLM_ASSERT(commandBuffer != nullptr);

  auto *computeEncoder = commandBuffer->computeCommandEncoder();
  LLM_ASSERT(computeEncoder != nullptr);

  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  auto *outMetal =
      device->newBuffer(out.size() * sizeof(float), MTL::ResourceStorageModeShared);
  auto *inMetal =
      device->newBuffer(in.size() * sizeof(float), MTL::ResourceStorageModeShared);
  auto *weightMetal =
      device->newBuffer(weight.size() * sizeof(float), MTL::ResourceStorageModeShared);

  std::memcpy(static_cast<float *>(inMetal->contents()), in.data_handle(),
              in.size() * sizeof(float));
  std::memcpy(static_cast<float *>(weightMetal->contents()), weight.data_handle(),
              weight.size() * sizeof(float));

  computeEncoder->setComputePipelineState(computePipelineState);
  computeEncoder->setBuffer(outMetal, 0, 0);
  computeEncoder->setBuffer(inMetal, 0, 1);
  computeEncoder->setBuffer(weightMetal, 0, 2);
  computeEncoder->setBytes(&batchSize, sizeof(batchSize), 3);
  computeEncoder->setBytes(&seqLen, sizeof(seqLen), 4);
  computeEncoder->setBytes(&outDim, sizeof(outDim), 5);
  computeEncoder->setBytes(&inDim, sizeof(inDim), 6);

  const auto gridSize = MTL::Size(out.size(), 1, 1);
  const auto threadGroupSize = std::min(
      computePipelineState->maxTotalThreadsPerThreadgroup(), NS::UInteger{out.size()});
  const auto groupSize = MTL::Size(threadGroupSize, 1, 1);

  computeEncoder->dispatchThreads(gridSize, groupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  std::memcpy(out.data_handle(), static_cast<float *>(outMetal->contents()),
              out.size() * sizeof(float));
}

void MetalCompute::run(view<float, 3> out, view<const float, 3> in, view<const float, 2> weight,
         view<const float, 1> bias) {

  auto *commandBuffer = commandQueueMatMulBias->commandBuffer();
  LLM_ASSERT(commandBuffer != nullptr);

  auto *computeEncoder = commandBuffer->computeCommandEncoder();
  LLM_ASSERT(computeEncoder != nullptr);

  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  auto *outMetal =
      device->newBuffer(out.size() * sizeof(float), MTL::ResourceStorageModeShared);
  auto *inMetal =
      device->newBuffer(in.size() * sizeof(float), MTL::ResourceStorageModeShared);
  auto *weightMetal =
      device->newBuffer(weight.size() * sizeof(float), MTL::ResourceStorageModeShared);
  auto *biasMetal =
      device->newBuffer(bias.size() * sizeof(float), MTL::ResourceStorageModeShared);

  std::memcpy(static_cast<float *>(inMetal->contents()), in.data_handle(),
              in.size() * sizeof(float));
  std::memcpy(static_cast<float *>(weightMetal->contents()), weight.data_handle(),
              weight.size() * sizeof(float));
  std::memcpy(static_cast<float *>(biasMetal->contents()), bias.data_handle(),
              bias.size() * sizeof(float));

  computeEncoder->setComputePipelineState(computePipelineStateBias);
  computeEncoder->setBuffer(outMetal, 0, 0);
  computeEncoder->setBuffer(inMetal, 0, 1);
  computeEncoder->setBuffer(weightMetal, 0, 2);
  computeEncoder->setBuffer(biasMetal, 0, 3);
  computeEncoder->setBytes(&batchSize, sizeof(batchSize), 4);
  computeEncoder->setBytes(&seqLen, sizeof(seqLen), 5);
  computeEncoder->setBytes(&outDim, sizeof(outDim), 6);
  computeEncoder->setBytes(&inDim, sizeof(inDim), 7);

  const auto gridSize = MTL::Size(out.size(), 1, 1);
  const auto threadGroupSize = std::min(
      computePipelineStateBias->maxTotalThreadsPerThreadgroup(), NS::UInteger{out.size()});
  const auto groupSize = MTL::Size(threadGroupSize, 1, 1);

  computeEncoder->dispatchThreads(gridSize, groupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  std::memcpy(out.data_handle(), static_cast<float *>(outMetal->contents()),
              out.size() * sizeof(float));
}

} // namespace llm