#include <metal_stdlib>
using namespace metal;

#include <metal_stdlib>
using namespace metal;

kernel void matMul(device float *out [[buffer(0)]],
                   device const float *in [[buffer(1)]],
                   device const float *weight [[buffer(2)]],
                   constant uint &batchSize [[buffer(3)]],
                   constant uint &seqLen [[buffer(4)]],
                   constant uint &outDim [[buffer(5)]],
                   constant uint &inDim [[buffer(6)]],
                   uint id [[thread_position_in_grid]])
{
  uint totalThreads = batchSize * seqLen * outDim;
  if (id >= totalThreads) return;

  uint batch = id / (seqLen * outDim);
  uint token = (id / outDim) % seqLen;
  uint outIdx = id % outDim;

  float sum = 0.0;
  for (uint inIdx = 0; inIdx < inDim; ++inIdx) {
    uint inIndex = batch * seqLen * inDim + token * inDim + inIdx;
    uint weightIndex = outIdx * inDim + inIdx;
    sum += in[inIndex] * weight[weightIndex];
  }

  uint outIndex = batch * seqLen * outDim + token * outDim + outIdx;
  out[outIndex] = sum;
}

kernel void matMulWithBias(device float *out [[ buffer(0) ]],
                   device const float *in [[ buffer(1) ]],
                   device const float *weight [[ buffer(2) ]],
                   device const float *bias [[ buffer(3) ]],
                   constant uint &batchSize [[ buffer(4) ]],
                   constant uint &seqLen [[ buffer(5) ]],
                   constant uint &outDim [[ buffer(6) ]],
                   constant uint &inDim [[ buffer(7) ]],
                   uint id [[ thread_position_in_grid ]])
{
  uint totalThreads = batchSize * seqLen * outDim;
  if (id >= totalThreads) return;

  uint batch = id / (seqLen * outDim);
  uint token = (id / outDim) % seqLen;
  uint outIdx = id % outDim;

  float sum = bias[outIdx];
  for (uint inIdx = 0; inIdx < inDim; ++inIdx) {
    uint inIndex = batch * seqLen * inDim + token * inDim + inIdx;
    uint weightIndex = outIdx * inDim + inIdx;
    sum += in[inIndex] * weight[weightIndex];
  }

  uint outIndex = batch * seqLen * outDim + token * outDim + outIdx;
  out[outIndex] = sum;
}