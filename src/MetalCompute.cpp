//
// Created by Hamidreza Khazaei on 5/21/24.
//

#include "MetalCompute.h"
#include "Extension.h"

namespace llm {

constexpr auto metalLibName = "../shader/default.metallib";

void MetalCompute::setup() {
  device = MTL::CreateSystemDefaultDevice();

  // for some reason creating this inside run causes issues
  commandQueue = device->newCommandQueue();
  LLM_ASSERT(commandQueue != nullptr);
}

std::tuple<MTL::ComputeCommandEncoder *, MTL::CommandBuffer *,
           MTL::ComputePipelineState *>
MetalCompute::getComputeResources(const std::string &functNameString) {
  const auto *libraryPath = NS::String::string(metalLibName, NS::ASCIIStringEncoding);

  NS::Error *error = nullptr;
  auto *lib = device->newLibrary(libraryPath, &error);
  LLM_ASSERT(error == nullptr);
  LLM_ASSERT(lib != nullptr);

  const auto *functionName =
      NS::String::string(functNameString.c_str(), NS::ASCIIStringEncoding);
  const auto *computeFunction = lib->newFunction(functionName);
  LLM_ASSERT(computeFunction != nullptr);

  auto *computePipelineState = device->newComputePipelineState(computeFunction, &error);

  LLM_ASSERT(computePipelineState != nullptr);
  LLM_ASSERT(error == nullptr);

  auto *commandBuffer = commandQueue->commandBuffer();
  LLM_ASSERT(commandBuffer != nullptr);
  auto *computeEncoder = commandBuffer->computeCommandEncoder();
  LLM_ASSERT(computeEncoder != nullptr);
  computeEncoder->setComputePipelineState(computePipelineState);
  return {computeEncoder, commandBuffer, computePipelineState};
}

template <typename BufferType>
MTL::Buffer *mapToDevice(const BufferType &buff, MTL::Device *device) {
  return device->newBuffer(buff.data_handle(), buff.size() * sizeof(float),
                           MTL::ResourceStorageModeShared,
                           ^(void *pointer, NS::UInteger length){
                           });
}

void MetalCompute::run(view<float, 3> out, view<const float, 3> in,
                       view<const float, 2> weight) {

  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  const auto *outMetal = mapToDevice(out, device);
  const auto *inMetal = mapToDevice(in, device);
  const auto *weightMetal = mapToDevice(weight, device);

  const auto &[computeEncoder, commandBuffer, computePipelineState] =
      getComputeResources("matMul");
  computeEncoder->setBuffer(outMetal, 0, 0);
  computeEncoder->setBuffer(inMetal, 0, 1);
  computeEncoder->setBuffer(weightMetal, 0, 2);
  computeEncoder->setBytes(&batchSize, sizeof(batchSize), 3);
  computeEncoder->setBytes(&seqLen, sizeof(seqLen), 4);
  computeEncoder->setBytes(&outDim, sizeof(outDim), 5);
  computeEncoder->setBytes(&inDim, sizeof(inDim), 6);

  launchAndWait(out.size(), computeEncoder, commandBuffer, computePipelineState);
}

void MetalCompute::run(view<float, 3> out, view<const float, 3> in,
                       view<const float, 2> weight, view<const float, 1> bias) {

  const auto batchSize = out.extent(0);
  const auto seqLen = out.extent(1);
  const auto outDim = weight.extent(0);
  const auto inDim = weight.extent(1);

  const auto *outMetal = mapToDevice(out, device);
  const auto *inMetal = mapToDevice(in, device);
  const auto *weightMetal = mapToDevice(weight, device);
  const auto *biasMetal = mapToDevice(bias, device);

  const auto &[computeEncoder, commandBuffer, computePipelineState] =
      getComputeResources("matMulWithBias");
  computeEncoder->setBuffer(outMetal, 0, 0);
  computeEncoder->setBuffer(inMetal, 0, 1);
  computeEncoder->setBuffer(weightMetal, 0, 2);
  computeEncoder->setBuffer(biasMetal, 0, 3);
  computeEncoder->setBytes(&batchSize, sizeof(batchSize), 4);
  computeEncoder->setBytes(&seqLen, sizeof(seqLen), 5);
  computeEncoder->setBytes(&outDim, sizeof(outDim), 6);
  computeEncoder->setBytes(&inDim, sizeof(inDim), 7);

  launchAndWait(out.size(), computeEncoder, commandBuffer, computePipelineState);
}

void MetalCompute::launchAndWait(const unsigned int numThreads,
                                 MTL::ComputeCommandEncoder *computeEncoder,
                                 MTL::CommandBuffer *commandBuffer,
                                 const MTL::ComputePipelineState *computePipelineState) {
  const auto gridSize = MTL::Size(numThreads, 1, 1);
  const auto threadGroupSize = std::min(
      computePipelineState->maxTotalThreadsPerThreadgroup(), NS::UInteger{numThreads});
  const auto groupSize = MTL::Size(threadGroupSize, 1, 1);

  computeEncoder->dispatchThreads(gridSize, groupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

} // namespace llm