# llm.c

A rewrite of [karpathy/llm.c](https://github.com/karpathy/llm.c) to brush-up on C/CUDA. CPU version is ~50% faster (more loops are parallelized).

For CPU build:

```shell
# Compile and run
make train_gpt2 && ./train_gpt2
```

### license
MIT