//
// Created by Hamidreza Khazaei on 5/5/24.
//

#include "catch2/catch_test_macros.hpp"
#include <Metal/Metal.hpp>
#include <iostream>
#include <random>

TEST_CASE("test metal libraries") {
  auto *device = MTL::CreateSystemDefaultDevice();
  std::cout << device->name()->cString(NS::ASCIIStringEncoding) << '\n';
  std::cout << device->architecture()->name()->cString(NS::ASCIIStringEncoding) << '\n';

  const auto *libraryPath =
      NS::String::string("../shader/default.metallib", NS::ASCIIStringEncoding);

  NS::Error *error = nullptr;
  auto *lib = device->newLibrary(libraryPath, &error);
  CHECK(error == nullptr);
  CHECK(lib != nullptr);

  const auto *functionName =
      NS::String::string("work_on_arrays", NS::ASCIIStringEncoding);
  const auto *computeFunction = lib->newFunction(functionName);
  CHECK(computeFunction != nullptr);

  const auto *computePipelineState =
      device->newComputePipelineState(computeFunction, &error);

  CHECK(computePipelineState != nullptr);
  CHECK(error == nullptr);

  auto *commandQueue = device->newCommandQueue();
  CHECK(commandQueue != nullptr);

  constexpr auto arrayLen = 100000000;
  constexpr auto bufferSize = arrayLen * sizeof(float);
  // Allocate three buffers to hold our initial data and the result.
  auto *bufferA = device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  auto *bufferB = device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  auto *bufferResult = device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);

  const auto seed = 0ULL;
  auto rng = std::mt19937_64{seed};
  auto generateRandomFloatData = [&rng](auto *buffer) {
    auto *dataPtr = static_cast<float *>(buffer->contents());
    auto dist = std::uniform_real_distribution<float>{0, 1};
    for (auto index = 0; index < arrayLen; ++index) {
      dataPtr[index] = dist(rng);
    }
  };

  generateRandomFloatData(bufferA);
  generateRandomFloatData(bufferB);

  // Create a command buffer to hold commands.
  auto *commandBuffer = commandQueue->commandBuffer();
  CHECK(commandBuffer != nullptr);

  // Start a compute pass.
  auto *computeEncoder = commandBuffer->computeCommandEncoder();
  CHECK(computeEncoder != nullptr);

  // Encode the pipeline state object and its parameters.
  computeEncoder->setComputePipelineState(computePipelineState);
  computeEncoder->setBuffer(bufferA, 0, 0);
  computeEncoder->setBuffer(bufferB, 0, 1);
  computeEncoder->setBuffer(bufferResult, 0, 2);

  const auto gridSize = MTL::Size(arrayLen, 1, 1);

  // Calculate a thread group size.
  const auto threadGroupSize = std::min(
      computePipelineState->maxTotalThreadsPerThreadgroup(), NS::UInteger{arrayLen});

  const auto groupSize = MTL::Size(threadGroupSize, 1, 1);

  // Encode the compute command.
  computeEncoder->dispatchThreads(gridSize, groupSize);

  // End the compute pass.
  computeEncoder->endEncoding();

  // Execute the command.
  commandBuffer->commit();

  // Normally, you want to do other work in your app while the GPU is running,
  // but in this example, the code simply blocks until the calculation is complete.
  commandBuffer->waitUntilCompleted();

  const auto *a = static_cast<float *>(bufferA->contents());
  const auto *b = static_cast<float *>(bufferB->contents());
  const auto *result = static_cast<float *>(bufferResult->contents());

  constexpr auto delta = 1.0e-6F;
  for (auto index = 0; index < arrayLen; index++) {
    if (abs(result[index] - (a[index] * b[index])) > delta) {
      std::cout << "Compute ERROR: index= " << index << " result= " << result[index]
                << " vs " << a[index] * b[index] << " = a * b\n";

      CHECK(abs(result[index] - (a[index] * b[index])) > delta);
    }
  }

  std::cout << "Compute results as expected.\n";
}