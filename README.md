# llm.cpp

C++ implementation of various language models optimized for Apple Silicon. Currently, only the forward pass of GPT2 is
implemented. GPT2 is bit exact with pytorch implementation.

Inspiration came from [llm.c](https://github.com/karpathy/llm.c).

Here are some rough profiling numbers on my Mac M1 Pro for generating a single token on the 124M parameter GPT2 model.

| technique          | seconds |
|--------------------|---------|
| no optimization    | 26.08   |
| openMP, 10 threads | 4.1     |
| Neon SIMD          | 9.1     |
| openMP + Neon      | 2.29    |

## TODO

- [ ]  Convert token id generated to string.
- [ ]  Make a standalone app for running inference.
- [ ]  Use c++ chronos for profiling.
- [ ]  Implement on Metal.
- [ ]  Implement backprop.
- [ ]  Implement different models (llama).
