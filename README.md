# llm.c

A rewrite of [karpathy/llm.c](https://github.com/karpathy/llm.c) to brush-up on C/CUDA.

For CPU build:

```shell
clang -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes train_gpt2.c -o train_gpt2 -lm && ./train_gpt2
```

### license
MIT