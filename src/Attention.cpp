//
// Created by Hamidreza Khazaei on 4/20/24.
//

#include "Attention.h"

namespace llm {


void attention(view3d<float> out, view3d<const float> in, const int numHeads){
  const auto batchSize = in.extent(0);
  const auto seqLen = in.extent(1);
  const auto inChSize = in.extent(2);
  const auto embeddingDim = inChSize / 3;
  LLM_ASSERT(batchSize == out.extent(0));
  LLM_ASSERT(seqLen == out.extent(1));
  LLM_ASSERT(embeddingDim == out.extent(2));
  LLM_ASSERT(embeddingDim % numHeads == 0);

  // for each batch and token, input is organized as [Q, K, V]
  // do this with views??
//  const auto keyOffset = embeddingDim;
//  const auto valueOffset = 2 * embeddingDim;
//  const auto headSize = embeddingDim / numHeads;
}

}