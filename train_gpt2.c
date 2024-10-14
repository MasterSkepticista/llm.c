/*
This file trains the GPT-2 model on CPU.
*/
#include <stdio.h>
#include <time.h>

#include "llmc/utils.h"
#include "llmc/dataloader.h"
#include "llmc/tokenizer.h"

// GPT-2 Model definition.
typedef struct {
  int max_seq_len;  // Max sequence length.
  int vocab_size;  // Vocabulary size, like 50257 for GPT2
  int padded_vocab_size;  // size padded to an even multiple, like 50304
  int num_layers;
  int num_heads;
  int channels;
} GPT2Config;

/*
Parameters of the model.
Notation:
V: Vocabulary size
C: Channel/feature dim
L: Layers
*/
#define NUM_PARAMETER_TENSORS 16
typedef struct {
  float *wte; // (V, C)
  float *wpe; // (maxT, C)
  float *ln1w; // (L, C)
  float *ln1b; // (L, C)
  float *qkvw; // (L, 3*C, C)
  float *qkvb; // (L, 3*C)
  float *attnprojw; // (L, C, C)
  float *attnprojb; // (L, C)
  float *ln2w; // (L, C)
  float *ln2b; // (L, C)
  float *fcw; // (L, 4*C, C)
  float *fcb; // (L, 4*C)
  float *fcprojw; // (L, C, 4*C)
  float *fcprojb; // (L, C)
  float *lnfw; // (C,)
  float *lnfb; // (C,)
} ParameterTensors;


/**
 * @brief Fills in the sizes of the parameters for the GPT-2 model based on the given configuration.
 *
 * This function calculates and assigns the sizes of various parameters required for the GPT-2 model
 * to the provided array `param_sizes`. The sizes are determined based on the configuration specified
 * in the `config` parameter.
 *
 * @param param_sizes A pointer to an array of size_t where the parameter sizes will be stored.
 *                    The array should have at least 16 elements.
 * @param config A GPT2Config structure containing the configuration parameters for the GPT-2 model.
 *               The structure should include the following fields:
 *               - padded_vocab_size: The size of the padded vocabulary.
 *               - channels: The number of channels.
 *               - max_seq_len: The maximum sequence length.
 *               - num_layers: The number of layers.
 */
void fill_in_parameter_sizes(size_t *param_sizes, GPT2Config config) {
  size_t Vp = config.padded_vocab_size;
  size_t C = config.channels;
  size_t maxT = config.max_seq_len;
  size_t L = config.num_layers;
  param_sizes[0] = Vp * C;  // wte
  param_sizes[1] = maxT * C;  // wpe
  param_sizes[2] = L * C; // ln1w
  param_sizes[3] = L * C; // ln1b
  param_sizes[4] = L * (3*C) * C; // qkvw
  param_sizes[5] = L * (3*C); // qkvb
  param_sizes[6] = L * C * C; // attnprojw
  param_sizes[7] = L * C; // attnprojb
  param_sizes[8] = L * C; // ln2w
  param_sizes[9] = L * C; // ln2b
  param_sizes[10] = L * (4*C) * C; // fcw
  param_sizes[11] = L * (4*C); // fcb
  param_sizes[12] = L * C * (4 * C); // fcprojw
  param_sizes[13] = L * C; // fcprojb
  param_sizes[14] = C; // lnfw
  param_sizes[15] = C; // lnfb
}

float* malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes) {
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }

  // malloc for all parameters.
  float *params_memory = (float *)mallocCheck(num_parameters * sizeof(float));
  // get addresses of all param tensors.
  float **ptrs[] = {
    &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
    &params->attnprojw, &params->attnprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
    &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
  };
  float *params_memory_iterator = params_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }
  return params_memory;
}

/*
Activations of the model.
Notation:
L: Number of layers
B: Batch size
NH: Number of attention heads
T: Sequence length
C: Channel dim (feature size)
V: Vocabulary size
*/
#define NUM_ACTIVATION_TENSORS 23
typedef struct {
  float *encoded; // (B, T, C)
  float *ln1; // (L, B, T, C)
  float *ln1_mean; // (L, B, T)
  float *ln1_rstd; // (L, B, T)
  float *qkv; // (L, B, T, 3*C)
  float *atty; // (L, B, T, C) what?
  float *preatt; // (L, B, NH, T, T)
  float *att; // (L, B, NH, T, T)
  float *attproj; // (L, B, T, C)
  float *residual2; // (L, B, T, C)
  float *ln2; // (L, B, T, C)
  float *ln2_mean; // (L, B, T)
  float *ln2_rstd; // (L, B, T)
  float *fch; // (L, B, T, 4*C)
  float *fch_gelu; // (L, B, T, 4*C)
  float *fcproj; // (L, B, T, C)
  float *residual3; // (L, B, T, C)
  float *lnf; // (B, T, C)
  float *lnf_mean; // (B, T)
  float *lnf_rstd; // (B, T)
  float *logits; // (B, T, V)
  float *probs; // (B, T, V)
  float *losses; // (B, T)
} ActivationTensors;

// A `typedef struct` is a class-like data structure of Python.
typedef struct {
  GPT2Config config;
  // the weights (parameters) of the model, and their sizes.
  ParameterTensors params;
  // size_t is an unsigned integer dtype, preferred for object size counting.
  // this is also defining a 1D array of given size, to hold param size info.
  size_t param_sizes[NUM_PARAMETER_TENSORS];
  float *params_memory;
  size_t num_parameters;
  // gradients of the weights.
  ParameterTensors grads;
  float *grads_memory;
  // buffers of the AdamW Optimizer.
  float *m_memory;
  float *v_memory;
  // the activations of the model, and their sizes.
  ActivationTensors acts;
  size_t act_sizes[NUM_ACTIVATION_TENSORS];
  float *act_memory;
  size_t num_activations;  // TODO: Why?
  // gradients of the activations.
  ActivationTensors grads_acts;
  float *grads_acts_memory;
  // other run state config
  int batch_size;  // the batch size (B) of current forward pass.
  int seq_len;  // the context length (T) for forward pass.
  int *inputs;  // input tokens.
  int *targets;  // target tokens.
  float mean_loss;  // average loss of the batch.
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {
  // read model data from a checkpoint file.
  FILE *model_file = fopenCheck(checkpoint_path, "rb");
  int model_header[256];
  freadCheck(model_header, sizeof(int), 256, model_file);

  // quirk from karpathy/llmc code.
  if (model_header[0] != 20240326) { printf("Bad magic model file.\n"); exit(1); }
  if (model_header[1] != 3) { printf("Bad version in model file.\n"); exit(1); }

  // read in hyperparameters from model_header.
  size_t maxT, V, Vp, L, NH, C;
  model->config.max_seq_len = maxT = model_header[2];
  model->config.vocab_size = V = model_header[3];
  model->config.num_layers = L = model_header[4];
  model->config.num_heads = NH = model_header[5];
  model->config.channels = C = model_header[6];
  model->config.padded_vocab_size = Vp = model_header[7];
  printf("[GPT-2]\n");
  printf("max_seq_len: %zu\n", maxT);
  printf("vocab_size: %zu\n", V);
  printf("padded_vocab_size: %zu\n", Vp);
  printf("num_layers: %zu\n", L);
  printf("num_heads: %zu\n", NH);
  printf("channels: %zu\n", C);

  // allocate space for all the parameters in memory + read.
  fill_in_parameter_sizes(model->param_sizes, model->config);

  // count the number of parameters.
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  printf("Total parameters: %zu\n", num_parameters);
  model->num_parameters = num_parameters;

  // read in all the parameters from the file.
  model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
  freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
  fcloseCheck(model_file);

  // other inits.
  model->act_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f;
}

int main() {
  
  // build the GPT-2 model from a checkpoint.
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

  // build the dataloaders.
  const char *train_tokens = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
  const char *val_tokens = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
  int B = 4; // batch size 4
  int T = 64; // sequence length 64
  DataLoader train_loader, val_loader;
  dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
  dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
  printf("Train batches: %zu\n", train_loader.num_tokens / (B * T));
  printf("Val batches: %zu\n", val_loader.num_tokens / (B * T));
  int val_num_batches = 5;

  // Build the Tokenizer.
  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  // some memory for generating samples from the model.
  uint64_t rng_state = 42;
  int *gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
  const int gen_steps = 64;

  // train
  struct timespec start, end;
  for (int step = 0; step <=40; step++) {

    // once in a while run eval.
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        // gpt2_forward(...)
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      printf("val_loss %f\n", val_loss);
    }

    // once in a while generate text.
    if (step > 0 && step % 20 == 0) {
      // fill up a buffer with EOT tokens
      for (int i = 0; i < B * T; i++) {
        gen_tokens[i] = tokenizer.eot_token;
      }

      // sample autoregressively.
      printf("generating:\n---\n");
      for (int i = 0; i < gen_steps; i++) {
        // forward 
      }
      printf("\n---\n");
    }

    // do a train step.
  }

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  // gpt2_free(&model);
  free(gen_tokens);
  return 0;
}