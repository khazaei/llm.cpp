//
// Created by Hamidreza Khazaei on 5/5/24.
//

#include "catch2/catch_test_macros.hpp"
#include <Metal/Metal.hpp>
#include <iostream>

TEST_CASE("test metal libraries") {
  auto *pDevice = MTL::CreateSystemDefaultDevice();
  const auto *libraryPath =
      NS::String::string("../shader/default.metallib", NS::ASCIIStringEncoding);

  NS::Error *error = nullptr;
  auto *lib = pDevice->newLibrary(libraryPath, &error);
  CHECK(error == nullptr);
  CHECK(lib != nullptr);

  const auto *functionName = NS::String::string("work_on_arrays", NS::ASCIIStringEncoding);
  const auto *computeFunction = lib->newFunction(functionName);
  CHECK(computeFunction != nullptr);
}