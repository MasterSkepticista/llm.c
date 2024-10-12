/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings.
This is all we need for unconditional generation.
*/
#include <stdint.h>
#include <assert.h>

#include "utils.h"

typedef struct {
  uint32_t vocab_size;
  char **token_table;
  int init_ok;
  int eot_token;
} Tokenizer;

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("Failed to open the tokenizer file `%s`\n.", filename);
    tokenizer->init_ok = 0;
    return;
  }

  // verify header.
  uint32_t header[256];
  freadCheck(header, sizeof(uint32_t), 256, file);
  assert(header[0] == 20240328);
  int version = header[1];
  tokenizer->vocab_size = header[2];
  if (version == 1) {
    assert(tokenizer->vocab_size == 50257);
    tokenizer->eot_token = 50256;
  } else if (version == 2) {
    tokenizer->eot_token = header[3];
  } else {
    fprintf(stderr, "Tokenizer model file %s has bad version.\n", filename);
    exit(EXIT_FAILURE);
  }
  
  // read in all the tokens
  unsigned char length;
  // tokenizer->token_table
}