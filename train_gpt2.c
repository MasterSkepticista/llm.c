/*
This file trains the GPT-2 model on CPU.
*/
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef OMP
#include <omp.h>
#endif

#include "llmc/dataloader.h"
#include "llmc/tokenizer.h"
#include "llmc/utils.h"

// GPT-2 Model definition.
typedef struct {
  int max_seq_len;        // Max sequence length.
  int vocab_size;         // Vocabulary size, like 50257 for GPT2
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
  float *wte;        // (V, C)
  float *wpe;        // (maxT, C)
  float *ln1w;       // (L, C)
  float *ln1b;       // (L, C)
  float *qkvw;       // (L, 3*C, C)
  float *qkvb;       // (L, 3*C)
  float *attnprojw;  // (L, C, C)
  float *attnprojb;  // (L, C)
  float *ln2w;       // (L, C)
  float *ln2b;       // (L, C)
  float *fcw;        // (L, 4*C, C)
  float *fcb;        // (L, 4*C)
  float *fcprojw;    // (L, C, 4*C)
  float *fcprojb;    // (L, C)
  float *lnfw;       // (C,)
  float *lnfb;       // (C,)
} ParameterTensors;

/**
 * @brief Fills in the sizes of the parameters for the GPT-2 model based on the
 * given configuration.
 *
 * This function calculates and assigns the sizes of various parameters required
 * for the GPT-2 model to the provided array `param_sizes`. The sizes are
 * determined based on the configuration specified in the `config` parameter.
 *
 * @param param_sizes A pointer to an array of size_t where the parameter sizes
 * will be stored. The array should have at least 16 elements.
 * @param config A GPT2Config structure containing the configuration parameters
 * for the GPT-2 model. The structure should include the following fields:
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
  param_sizes[0] = Vp * C;            // wte
  param_sizes[1] = maxT * C;          // wpe
  param_sizes[2] = L * C;             // ln1w
  param_sizes[3] = L * C;             // ln1b
  param_sizes[4] = L * (3 * C) * C;   // qkvw
  param_sizes[5] = L * (3 * C);       // qkvb
  param_sizes[6] = L * C * C;         // attnprojw
  param_sizes[7] = L * C;             // attnprojb
  param_sizes[8] = L * C;             // ln2w
  param_sizes[9] = L * C;             // ln2b
  param_sizes[10] = L * (4 * C) * C;  // fcw
  param_sizes[11] = L * (4 * C);      // fcb
  param_sizes[12] = L * C * (4 * C);  // fcprojw
  param_sizes[13] = L * C;            // fcprojb
  param_sizes[14] = C;                // lnfw
  param_sizes[15] = C;                // lnfb
}

/**
 * Allocates memory for parameter tensors and sets their pointers.
 */
float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes) {
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }

  // malloc for all parameters.
  float *params_memory = (float *)mallocCheck(num_parameters * sizeof(float));
  // get addresses of all param tensors.
  float **ptrs[] = {&params->wte,       &params->wpe,       &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
                    &params->attnprojw, &params->attnprojb, &params->ln2w, &params->ln2b, &params->fcw,  &params->fcb,
                    &params->fcprojw,   &params->fcprojb,   &params->lnfw, &params->lnfb};
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
  float *encoded;    // (B, T, C)
  float *ln1;        // (L, B, T, C)
  float *ln1_mean;   // (L, B, T)
  float *ln1_rstd;   // (L, B, T)
  float *qkv;        // (L, B, T, 3*C)
  float *atty;       // (L, B, T, C)
  float *preatt;     // (L, B, NH, T, T)
  float *att;        // (L, B, NH, T, T)
  float *attproj;    // (L, B, T, C)
  float *residual2;  // (L, B, T, C)
  float *ln2;        // (L, B, T, C)
  float *ln2_mean;   // (L, B, T)
  float *ln2_rstd;   // (L, B, T)
  float *fch;        // (L, B, T, 4*C)
  float *fch_gelu;   // (L, B, T, 4*C)
  float *fcproj;     // (L, B, T, C)
  float *residual3;  // (L, B, T, C)
  float *lnf;        // (B, T, C)
  float *lnf_mean;   // (B, T)
  float *lnf_rstd;   // (B, T)
  float *logits;     // (B, T, V)
  float *probs;      // (B, T, V)
  float *losses;     // (B, T)
} ActivationTensors;

/**
 * Fills in the activation sizes for various layers and operations in the GPT-2
 * model.
 */
void fill_in_activation_sizes(size_t *act_sizes, GPT2Config config, int B, int T) {
  size_t C = config.channels;
  size_t L = config.num_layers;
  size_t NH = config.num_heads;
  size_t Vp = config.padded_vocab_size;
  act_sizes[0] = B * T * C;           // encoded
  act_sizes[1] = L * B * T * C;       // ln1
  act_sizes[2] = L * B * T;           // ln1_mean
  act_sizes[3] = L * B * T;           // ln1_rstd
  act_sizes[4] = L * B * T * 3 * C;   // qkv
  act_sizes[5] = L * B * T * C;       // atty
  act_sizes[6] = L * B * NH * T * T;  // preatt
  act_sizes[7] = L * B * NH * T * T;  // att
  act_sizes[8] = L * B * T * C;       // attproj
  act_sizes[9] = L * B * T * C;       // residual2
  act_sizes[10] = L * B * T * C;      // ln2
  act_sizes[11] = L * B * T;          // ln2_mean
  act_sizes[12] = L * B * T;          // ln2_rstd
  act_sizes[13] = L * B * T * 4 * C;  // fch
  act_sizes[14] = L * B * T * 4 * C;  // fch_gelu
  act_sizes[15] = L * B * T * C;      // fcproj
  act_sizes[16] = L * B * T * C;      // residual3
  act_sizes[17] = B * T * C;          // lnf
  act_sizes[18] = B * T;              // lnf_mean
  act_sizes[19] = B * T;              // lnf_rstd
  act_sizes[20] = B * T * Vp;         // logits
  act_sizes[21] = B * T * Vp;         // probs
  act_sizes[22] = B * T;              // losses
}

/**
 * Allocates memory for all activation tensors and sets pointers to the
 * respective memory locations.
 */
float *malloc_and_point_activations(ActivationTensors *acts, size_t *act_sizes) {
  // Do one giant malloc of all activations.
  size_t num_activations = 0;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    num_activations += act_sizes[i];
  }
  float *activations_memory = (float *)mallocCheck(num_activations * sizeof(float));

  // Demarcate bytes of each activation tensor.
  float **ptrs[] = {&acts->encoded, &acts->ln1,       &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv,
                    &acts->atty,    &acts->preatt,    &acts->att,      &acts->attproj,  &acts->residual2,
                    &acts->ln2,     &acts->ln2_mean,  &acts->ln2_rstd, &acts->fch,      &acts->fch_gelu,
                    &acts->fcproj,  &acts->residual3, &acts->lnf,      &acts->lnf_mean, &acts->lnf_rstd,
                    &acts->logits,  &acts->probs,     &acts->losses};
  float *memory_ptr = activations_memory;
  for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    *(ptrs[i]) = memory_ptr;
    memory_ptr += act_sizes[i];
  }
  return activations_memory;
}

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
  int batch_size;   // the batch size (B) of current forward pass.
  int seq_len;      // the context length (T) for forward pass.
  int *inputs;      // input tokens.
  int *targets;     // target tokens.
  float mean_loss;  // average loss of the batch.
} GPT2;

// --------------------------------Forward/Backward functions----------------------
/**
 * @brief Performs wte(inp) + wpe(inp).
 *
 * This function takes input token indices and combines their corresponding
 * token embeddings (from the word token embedding matrix) and position
 * embeddings (from the word position embedding matrix) to produce the output
 * tensor.
 *
 * @param out Pointer to the output tensor of shape (B, T, C).
 * @param inp Pointer to the input tensor of token indices of shape (B, T).
 * @param wte Pointer to the word token embedding matrix of shape (V, C).
 * @param wpe Pointer to the word position embedding matrix of shape (maxT, C).
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 */
void encoder_forward(float *out, int *inp, float *wte, float *wpe, int B, int T, int C) {
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the output position [b,t,:]
      float *out_bt = out + b * T * C + t * C;
      // fetch the input token value at [b, t]
      int ix = inp[b * T + t];
      // fetch relevant embedding start address of wpe and wte
      float *wte_ix = wte + ix * C;
      float *wpe_t = wpe + t * C;
      // fetch C corresponding embedding floats from wte and wpe
      for (int c = 0; c < C; c++) {
        out_bt[c] = wte_ix[c] + wpe_t[c];
      }
    }
  }
}

/**
 * Backward on token and position embeddings.
 *
 * @param dwte Pointer to store token embedding gradient. Shape (Vp, C).
 * @param dwpe Pointer to store position embedding gradient. Shape (T, C).
 * @param dout Output-side gradient. Shape (B, T, C).
 * @param inp Input token batch. Shape (B, T).
 */
void encoder_backward(float *dwte, float *dwpe, float *dout, int *inp, int B, int T, int C) {
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dout_bt = dout + b * T * C + t * C;
      int ix = inp[b * T + t];
      float *dwte_ix = dwte + ix * C;
      float *dwpe_t = dwpe + t * C;
      for (int i = 0; i < C; i++) {
        dwte_ix[i] += dout_bt[i];
        dwpe_t[i] += dout_bt[i];
      }
    }
  }
}

/**
 * @brief LayerNorm forward pass.
 *
 * @param out Pointer to output activation tensor of shape (B, T, C)
 * @param inp Pointer to input activation tensor of shape (B, T, C)
 * @param mean Pointer to store intermediate `mean`, of shape (B, T)
 * @param rstd Pointer to store intermediate `rstd`, of shape (B, T)
 * @param weight Pointer to weight parameter of shape (C,)
 * @param bias Pointer to bias parameter of shape (C,)
 * @param B Batch Size
 * @param T Sequence length
 * @param C Channel dimensions
 */
void layernorm_forward(float *out, float *inp, float *mean, float *rstd, float *weight, float *bias, int B, int T,
                       int C) {
  float eps = 1e-5f;
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to relevant row.
      float *x = inp + b * T * C + t * C;

      // calculate mean
      float mu = 0.0f;
      for (int c = 0; c < C; c++) {
        mu += x[c];
      }
      mu /= C;

      // calculate rstd (reciprocal std)
      float var = 0.0f;
      for (int c = 0; c < C; c++) {
        float x_minus_mu = x[c] - mu;
        var += x_minus_mu * x_minus_mu;
      }
      var /= C;
      float rs = 1.0f / sqrtf(var + eps);

      // seek to relevant output row.
      float *out_bt = out + b * T * C + t * C;
      for (int c = 0; c < C; c++) {
        float n = (x[c] - mu) * rs;
        out_bt[c] = n * weight[c] + bias[c];
      }

      // cache mean and rstd for backward.
      mean[b * T + t] = mu;
      rstd[b * T + t] = rs;
    }
  }
}

/**
 * @brief Computes the backward pass for layer normalization.
 * This was particularly tricky. Good exercise to derive the backward equations
 * yourself first!
 *
 * @param dinp Pointer to store gradient of the input, size (B, T, C).
 * @param dweight Pointer to store the gradient of the weight, size (C,).
 * @param dbias Pointer to store the gradient of the bias, size (C,).
 * @param dout Pointer to the gradient of the output, size (B, T, C).
 * @param inp Pointer to the input data to the layernorm, size (B, T, C).
 * @param weight Pointer to the weight data, size (C,).
 * @param mean Pointer to the mean of the input data, size (B, T).
 * @param rstd Pointer to the reciprocal standard deviation of the input data, size (B, T).
 */
void layernorm_backward(float *dinp, float *dweight, float *dbias, float *dout, float *inp, float *weight, float *mean,
                        float *rstd, int B, int T, int C) {
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dout_bt = dout + b * T * C + t * C;
      float *inp_bt = inp + b * T * C + t * C;
      float *dinp_bt = dinp + b * T * C + t * C;
      float mean_bt = mean[b * T + t];
      float rstd_bt = rstd[b * T + t];

      // reduce ops (required in the loop below).
      float dnorm_mean = 0.0f;
      float dnorm_norm_mean = 0.0f;
      for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
      }
      dnorm_mean /= C;
      dnorm_norm_mean /= C;

      // accumulate input, weight and bias gradients
      for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dbias[i] += dout_bt[i];
        dweight[i] += dout_bt[i] * norm_bti;
        dinp_bt[i] += rstd_bt * (dnorm_i - dnorm_mean - (dnorm_norm_mean * norm_bti));
      }
    }
  }
}

/**
 * Naive loop matmul for unfriendly input shapes. Performs:
 * (B, T, C) @ (C, OC) + (OC,) -> (B, T, OC)
 *
 * @param out Pointer to tensor of shape (B, T, OC) to store result.
 * @param inp Pointer to tensor of shape (B, T, C).
 * @param weight Pointer to tensor of shape (OC, C).
 * @param bias Optional, pointer to tensor of shape (OC,).
 * @param B Batch Size.
 * @param T Sequence length.
 * @param C Channel dims of input.
 * @param OC Channel dims after matmul with weight matrix.
 */
void matmul_forward_naive(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C,
                          int OC) {
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      int m = b * T + t;
      for (int o = 0; o < OC; o++) {
        float sum = (bias != NULL) ? bias[o] : 0.0f;
        for (int c = 0; c < C; c++) {
          sum += inp[m * C + c] * weight[o * C + c];
        }
        out[m * OC + o] = sum;
      }
    }
  }
}

/**
 * Performs a MatMul of shapes:
 * (B, T, C) @ (C, OC) + (OC,) -> (B, T, OC)
 *
 * @param out Pointer to tensor of shape (B, T, OC) to store result.
 * @param inp Pointer to tensor of shape (B, T, C).
 * @param weight Pointer to tensor of shape (OC, C).
 * @param bias Optional, pointer to tensor of shape (OC,).
 * @param B Batch Size.
 * @param T Sequence length.
 * @param C Channel dims of input.
 * @param OC Channel dims after matmul with weight matrix.
 */
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, int OC) {
  const int TILE_SIZE = 8;
  if (B * T % TILE_SIZE != 0) {
    matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
    return;
  }

#pragma omp parallel for
  for (int obt = 0; obt < B * T; obt += TILE_SIZE) {
    for (int o = 0; o < OC; o++) {
      // initialize the tile.
      float result[TILE_SIZE];
      for (int t = 0; t < TILE_SIZE; t++) {
        result[t] = (bias != NULL) ? bias[o] : 0.0f;
      }

      // compute tiled inner dot product
      for (int c = 0; c < C; c++) {
        float w = weight[o * C + c];
        for (int t = 0; t < TILE_SIZE; t++) {
          result[t] += inp[(obt + t) * C + c] * w;
        }
      }

      // store
      for (int t = 0; t < TILE_SIZE; t++) {
        out[(obt + t) * OC + o] = result[t];
      }
    }
  }
}

/**
 * Matmul backward. Approximately, this does
 *
 * ```
 * dW = inp * dout
 * db = dout
 * din = w * dout
 * ```
 * @param din Resulting input activation gradient, shape (B, T, C).
 * @param dweight Resulting weight matrix gradient, shape (OC, C).
 * @param dbias Resulting bias vector gradient, shape (OC,).
 * @param dout Inward activation gradient, shape (B, T, OC).
 * @param inp Inputs in the matmul_forward, shape (B, T, C).
 * @param weight Weight matrix, shape (OC, C).
 *
 */
void matmul_backward(float *din, float *dweight, float *dbias, const float *dout, const float *inp, const float *weight,
                     int B, int T, int C, int OC) {
#pragma omp parallel for collapse(2)
  // backward into inputs.
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      const float *dout_bt = dout + b * T * OC + t * OC;
      float *din_bt = din + b * T * C + t * C;
      for (int o = 0; o < OC; o++) {
        const float *wrow = weight + o * C;
        for (int i = 0; i < C; i++) {
          din_bt[i] += wrow[i] * dout_bt[o];
        }
      }
    }
  }

#pragma omp parallel for
  // backward into weights
  for (int o = 0; o < OC; o++) {
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        const float *dout_bt = dout + b * T * OC + t * OC;
        const float *inp_bt = inp + b * T * C + t * C;
        if (dbias != NULL) {
          dbias[o] += dout_bt[o];
        }
        for (int i = 0; i < C; i++) {
          dweight[o * C + i] += inp_bt[i] * dout_bt[o];
        }
      }
    }
  }
}

/**
 * @brief Computes dot-product attention.
 * out = softmax(preatt)
 *
 * @param out Result tensor of shape (B, T, C)
 * @param preatt Result tensor of shape (B, NH, T, T), output of QK.T
 * @param att Result tensor with softmax applied on preatt.
 * @param inp Input tensor of shape (B, T, 3C) comprising QKV.
 *
 * Rest are trivial parameters.
 */
void attention_forward(float *out, float *preatt, float *att, float *inp, int B, int T, int C, int NH) {
  int C3 = C * 3;
  int head_dim = C / NH;
  float scale = 1.0f / sqrtf(head_dim);

#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        float *query_t = inp + (b * T * C3) + (t * C3) + h * head_dim;
        float *preatt_bth = preatt + (b * NH * T * T) + (h * T * T) + t * T;
        float *att_bth = att + (b * NH * T * T) + (h * T * T) + t * T;

        // preatt = (Q @ K.T) / sqrt(d_k)
        float maxval = -10000.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          // key = inp[b, 0:t, c:2c]
          float *key_t2 = inp + (b * T * C3) + (t2 * C3) + (h * head_dim) + C;
          float val = 0.0f;
          for (int i = 0; i < head_dim; i++) {
            val += query_t[i] * key_t2[i];
          }
          val *= scale;
          maxval = (val > maxval) ? val : maxval;
          preatt_bth[t2] = val;
        }

        // att = softmax(preatt)
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float expv = expf(preatt_bth[t2] - maxval);
          expsum += expv;
          att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        for (int t2 = 0; t2 < T; t2++) {
          // causal attention mask
          att_bth[t2] = (t2 <= t) ? att_bth[t2] * expsum_inv : 0.0f;
        }

        // att @ V
        float *out_bth = out + b * T * C + t * C + h * head_dim;
        for (int i = 0; i < head_dim; i++) {
          out_bth[i] = 0.0f;
        }
        for (int t2 = 0; t2 <= t; t2++) {
          float *value = inp + b * T * C3 + t2 * C3 + (2 * C + h * head_dim);
          for (int i = 0; i < head_dim; i++) {
            out_bth[i] += att_bth[t2] * value[i];
          }
        }
      }
    }
  }
}

/**
 * Backward pass on dot-product attention operation.
 *
 * @param dinp Pointer to store input-side gradient. Shape (B, T, 3C).
 * @param dpreatt Pointer to store pre-softmax score gradient. Shape (B, NH, T, T).
 * @param datt Pointer to store softmax score gradient. Shape (B, NH, T, T).
 * @param dout Output-side gradient that will be backpropagated. Shape (B, T, C).
 * @param inp QKV Input to attention operation during forward pass. Shape (B, T, 3C).
 * @param att Softmax Attention scores during forward pass. Shape (B, NH, T, T).
 */
void attention_backward(float *dinp, float *dpreatt, float *datt, float *dout, float *inp, float *att, int B, int T,
                        int C, int NH) {
  int C3 = 3 * C;
  int head_dim = C / NH;
  float scale = 1.0f / sqrtf(head_dim);

#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
        float *query_bth = inp + b * T * C3 + t * C3 + h * head_dim;
        float *dquery_bth = dinp + b * T * C3 + t * C3 + h * head_dim;

        // att @ V = atty
        // datt = dout @ V; dinp[..., 2C:3C] = dout @ att
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;
        float *datt_bth = datt + b * NH * T * T + h * T * T + t * T;
        float *dout_bth = dout + b * T * C + t * C + h * head_dim;
        for (int t2 = 0; t2 < t; t2++) {
          float *value_t2 = inp + b * T * C3 + t2 * C3 + 2 * C + h * head_dim;
          float *dvalue_t2 = dinp + b * T * C3 + t2 * C3 + 2 * C + h * head_dim;
          for (int i = 0; i < head_dim; i++) {
            datt_bth[t2] += dout_bth[i] * value_t2[i];
            dvalue_t2[i] += dout_bth[i] * att_bth[t2];
          }
        }

        // softmax backward
        for (int t2 = 0; t2 <= t; t2++) {
          for (int t3 = 0; t3 <= t; t3++) {
            float indicator = (t2 == t3) ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
            dpreatt_bth[t3] += local_derivative * datt_bth[t2];
          }
        }

        // query/key backward
        for (int t2 = 0; t2 <= t; t2++) {
          float *key_bt2h = inp + b * T * C3 + t2 * C3 + h * head_dim + C;
          float *dkey_bt2h = dinp + b * T * C3 + t2 * C3 + h * head_dim + C;
          for (int i = 0; i < head_dim; i++) {
            dquery_bth[i] += key_bt2h[i] * dpreatt_bth[t2] * scale;
            dkey_bt2h[i] += query_bth[i] * dpreatt_bth[t2] * scale;
          }
        }
      }
    }
  }
}

/**
 * Residual forward, simple add of two matrices/vectors.
 */
void residual_forward(float *out, float *inp1, float *inp2, int N) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    out[i] = inp1[i] + inp2[i];
  }
}

/**
 * Residual backward, copy output-side gradient to inputs.
 */
void residual_backward(float *dout, float *dinp1, float *dinp2, int N) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    // we accumulate because this is "spiritually" correct in DL - grads
    // accumulate on each `backward()` call.
    dinp1[i] += dout[i];
    dinp2[i] += dout[i];
  }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

/**
 * Applies elementwise approximate GELU nonlinearity.
 */
void gelu_forward(float *out, float *inp, int N) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
  }
}

#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float *dout, float *dinp, float *inp, int N) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech2_out = 1.0f / (coshf_out * coshf_out);
    float local_grad =
        0.5f * (1.0f + tanh_out) + x * 0.5f * sech2_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] += dout[i] * local_grad;
  }
}
#pragma float_control(pop)

/**
 * Applies softmax on the outer dimension of `logits`.
 *
 * @param probs Pointer to tensor of shape (B, T, Vp).
 * @param logits Pointer to logits of shape (B, T, Vp).
 */
void softmax_forward(float *probs, float *logits, int B, int T, int V, int Vp) {
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *logits_bt = logits + b * T * Vp + t * Vp;
      float *probs_bt = probs + b * T * Vp + t * Vp;

      // compute maxval for numerical stability.
      float maxval = -10000.0f;
      for (int i = 0; i < V; i++) {
        maxval = (logits_bt[i] > maxval) ? logits_bt[i] : maxval;
      }

      // compute probability, ignore padded dimensions (Vp - V)
      float sum = 0.0f;
      for (int i = 0; i < V; i++) {
        probs_bt[i] = expf(logits_bt[i] - maxval);
        sum += probs_bt[i];
      }
      for (int i = 0; i < V; i++) {
        probs_bt[i] /= sum;
      }

      // for safety.
      for (int i = V; i < Vp; i++) {
        probs_bt[i] = 0.0f;
      }
    }
  }
}

/**
 * Compute CE loss.
 *
 * @param losses Output tensor pointer of shape (B, T).
 * @param probs Input tensor pointer of shape (B, T, V).
 * @param targets Input tensor pointer of shape (B, T).
 */
void crossentropy_forward(float *losses, float *probs, int *targets, int B, int T, int Vp) {
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *probs_bt = probs + b * T * Vp + t * Vp;
      float p_x = probs_bt[targets[b * T + t]];
      losses[b * T + t] = -logf(p_x);
    }
  }
}

/**
 * CE Backward pass from losses to logits.
 *
 * @param dlogits Output tensor of shape (B, T, Vp).
 * @param dlosses Input tensor of shape (B, T).
 * @param probs Input probabilities of shape (B, T, Vp).
 * @param targets Input tensor of shape (B, T).
 */
void crossentropy_softmax_backward(float *dlogits, float *dlosses, float *probs, int *targets, int B, int T, int V,
                                   int Vp) {
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dlogits_bt = dlogits + b * T * Vp + t * Vp;
      float *probs_bt = probs + b * T * Vp + t * Vp;
      float dloss = dlosses[b * T + t];
      int tgt = targets[b * T + t];
      // V->Vp will remain zero as they are padding.
      for (int i = 0; i < V; i++) {
        float indicator = (i == tgt) ? 1.0f : 0.0f;
        dlogits_bt[i] += (probs_bt[i] - indicator) * dloss;
      }
    }
  }
}

/**
 * @brief Performs forward pass on the model, records activations, and loss if
 * targets are provided.
 *
 * @param model Pointer to GPT2 model
 * @param inputs Pointer to input token buffer of size (B, T)
 * @param targets Pointer to targets token buffer of size (B, T), optional
 * @param B Batch Size
 * @param T Token sequence length
 */
void gpt2_forward(GPT2 *model, int *inputs, int *targets, size_t B, size_t T) {
  // targets are optional.

  if (model->params_memory == NULL) {
    printf("Error: model was not initialized properly.\n");
    exit(1);
  }

  // convenience parameters.
  size_t V = model->config.vocab_size;
  size_t Vp = model->config.padded_vocab_size;
  size_t C = model->config.channels;
  size_t L = model->config.num_layers;
  size_t NH = model->config.num_heads;

  // tokens must be in the range [0, V)
  for (int i = 0; i < B * T; i++) {
    assert(0 <= inputs[i] && inputs[i] < V);
    if (targets != NULL) {
      assert(0 <= targets[i] && targets[i] < V);
    }
  }

  // allocate space for all activations
  if (model->act_memory == NULL) {
    model->batch_size = B;
    model->seq_len = T;
    fill_in_activation_sizes(model->act_sizes, model->config, B, T);
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      num_activations += model->act_sizes[i];
    }
    printf("num_activations: %zu\n", num_activations);
    model->num_activations = num_activations;
    model->act_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
    // buffers for inputs and targets.
    model->inputs = (int *)mallocCheck(B * T * sizeof(int));
    model->targets = (int *)mallocCheck(B * T * sizeof(int));
  } else {
    // validate B, T are consistent with memory allocated.
    if (B != model->batch_size || T != model->seq_len) {
      printf("Model: B=%d T=%d, Expected: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
      exit(EXIT_FAILURE);
    }
  }

  // Cache the inputs and targets
  memcpy(model->inputs, inputs, B * T * sizeof(int));
  if (targets != NULL) {
    memcpy(model->targets, targets, B * T * sizeof(int));
  }

  // forward pass
  ParameterTensors params = model->params;
  ActivationTensors acts = model->acts;
  encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);

  // forward on each layer
  float *residual;
  for (int l = 0; l < L; l++) {
    residual = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

    // get the pointers of weights for this layer
    float *l_ln1w = params.ln1w + l * C;
    float *l_ln1b = params.ln1b + l * C;
    float *l_qkvw = params.qkvw + l * 3 * C * C;
    float *l_qkvb = params.qkvb + l * 3 * C;
    float *l_attnprojw = params.attnprojw + l * C * C;
    float *l_attnprojb = params.attnprojb + l * C;
    float *l_ln2w = params.ln2w + l * C;
    float *l_ln2b = params.ln2b + l * C;
    float *l_fcw = params.fcw + l * 4 * C * C;
    float *l_fcb = params.fcb + l * 4 * C;
    float *l_fcprojw = params.fcprojw + l * C * 4 * C;
    float *l_fcprojb = params.fcprojb + l * C;

    // get the pointers of activations for this layer
    float *l_ln1 = acts.ln1 + l * B * T * C;
    float *l_ln1_mean = acts.ln1_mean + l * B * T;
    float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
    float *l_qkv = acts.qkv + l * B * T * 3 * C;
    float *l_atty = acts.atty + l * B * T * C;
    float *l_preatt = acts.preatt + l * B * NH * T * T;
    float *l_att = acts.att + l * B * NH * T * T;
    float *l_attproj = acts.attproj + l * B * T * C;
    float *l_residual2 = acts.residual2 + l * B * T * C;
    float *l_ln2 = acts.ln2 + l * B * T * C;
    float *l_ln2_mean = acts.ln2_mean + l * B * T;
    float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
    float *l_fch = acts.fch + l * B * T * 4 * C;
    float *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
    float *l_fcproj = acts.fcproj + l * B * T * C;
    float *l_residual3 = acts.residual3 + l * B * T * C;

    // now do the forward pass
    layernorm_forward(l_ln1, residual, l_ln1_mean, l_ln1_rstd, l_ln1w, l_ln1b, B, T, C);
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
    attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
    matmul_forward(l_attproj, l_atty, l_attnprojw, l_attnprojb, B, T, C, C);
    residual_forward(l_residual2, residual, l_attproj, B * T * C);
    layernorm_forward(l_ln2, l_residual2, l_ln2_mean, l_ln2_rstd, l_ln2w, l_ln2b, B, T, C);
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
    gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
    residual_forward(l_residual3, l_fcproj, l_residual2, B * T * C);
  }
  residual = acts.residual3 + (L - 1) * B * T * C;
  layernorm_forward(acts.lnf, residual, acts.lnf_mean, acts.lnf_rstd, params.lnfw, params.lnfb, B, T, C);
  matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
  softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

  // compute loss if targets available.
  if (targets != NULL) {
    crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
    float mean_loss = 0.0f;
    for (int i = 0; i < B * T; i++) {
      mean_loss += model->acts.losses[i];
    }
    mean_loss /= B * T;
    model->mean_loss = mean_loss;
  } else {
    model->mean_loss = -1.0f;
  }
}

/**
 * Zeroes parameter gradients and activation gradients.
 */
void gpt2_zero_grad(GPT2 *model) {
  if (model->grads_memory != NULL) {
    memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
  }
  if (model->grads_acts_memory != NULL) {
    memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
  }
}

void gpt2_backward(GPT2 *model) {
  assert(model->mean_loss != -1.0f);

  // lazily allocate memory for (parameter + activation) gradients.
  if (model->grads_memory == NULL) {
    model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
    model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
    gpt2_zero_grad(model);
  }

  // convenience parameters
  size_t B = model->batch_size;
  size_t T = model->seq_len;
  size_t V = model->config.vocab_size;
  size_t Vp = model->config.padded_vocab_size;
  size_t L = model->config.num_layers;
  size_t NH = model->config.num_heads;
  size_t C = model->config.channels;

  ParameterTensors params = model->params;
  ParameterTensors grads = model->grads;
  ActivationTensors acts = model->acts;
  ActivationTensors grads_acts = model->grads_acts;

  // backward pass
  float dloss_mean = 1.0f / (B * T);
  for (int i = 0; i < B * T; i++) {
    grads_acts.losses[i] = dloss_mean;
  }
  crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
  matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
  float *residual = acts.residual3 + (L - 1) * B * T * C;
  float *dresidual = grads_acts.residual3 + (L - 1) * B * T * C;
  layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean,
                     acts.lnf_rstd, B, T, C);
  for (int l = L - 1; l >= 0; l--) {
    residual = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
    dresidual = (l == 0) ? grads_acts.encoded : grads_acts.residual3 + (l - 1) * B * T * C;

    // get weight ptrs of this layer. we ignore biases because their gradient
    // does not depend on their actual values. It adds up incoming gradients.
    float *l_ln1w = params.ln1w + l * C;
    float *l_qkvw = params.qkvw + l * 3 * C * C;
    float *l_attnprojw = params.attnprojw + l * C * C;
    float *l_ln2w = params.ln2w + l * C;
    float *l_fcw = params.fcw + l * 4 * C * C;
    float *l_fcprojw = params.fcprojw + l * C * 4 * C;
    // get gradient ptrs of this layer.
    float *dl_ln1w = grads.ln1w + l * C;
    float *dl_ln1b = grads.ln1b + l * C;
    float *dl_qkvw = grads.qkvw + l * 3 * C * C;
    float *dl_qkvb = grads.qkvb + l * 3 * C;
    float *dl_attnprojw = grads.attnprojw + l * C * C;
    float *dl_attnprojb = grads.attnprojb + l * C;
    float *dl_ln2w = grads.ln2w + l * C;
    float *dl_ln2b = grads.ln2b + l * C;
    float *dl_fcw = grads.fcw + l * 4 * C * C;
    float *dl_fcb = grads.fcb + l * 4 * C;
    float *dl_fcprojw = grads.fcprojw + l * C * 4 * C;
    float *dl_fcprojb = grads.fcprojb + l * C;
    // get activation ptrs of this layer.
    float *l_ln1 = acts.ln1 + l * B * T * C;
    float *l_ln1_mean = acts.ln1_mean + l * B * T;
    float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
    float *l_qkv = acts.qkv + l * B * T * 3 * C;
    float *l_atty = acts.atty + l * B * T * C;
    float *l_att = acts.att + l * B * NH * T * T;
    float *l_residual2 = acts.residual2 + l * B * T * C;
    float *l_ln2 = acts.ln2 + l * B * T * C;
    float *l_ln2_mean = acts.ln2_mean + l * B * T;
    float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
    float *l_fch = acts.fch + l * B * T * 4 * C;
    float *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
    // get activation gradient ptrs of this layer.
    float *dl_ln1 = grads_acts.ln1 + l * B * T * C;
    float *dl_qkv = grads_acts.qkv + l * B * T * 3 * C;
    float *dl_atty = grads_acts.atty + l * B * T * C;
    float *dl_preatt = grads_acts.preatt + l * B * NH * T * T;
    float *dl_att = grads_acts.att + l * B * NH * T * T;
    float *dl_attproj = grads_acts.attproj + l * B * T * C;
    float *dl_residual2 = grads_acts.residual2 + l * B * T * C;
    float *dl_ln2 = grads_acts.ln2 + l * B * T * C;
    float *dl_fch = grads_acts.fch + l * B * T * 4 * C;
    float *dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4 * C;
    float *dl_fcproj = grads_acts.fcproj + l * B * T * C;
    float *dl_residual3 = grads_acts.residual3 + l * B * T * C;

    // backprop.
    residual_backward(dl_residual3, dl_residual2, dl_fcproj, B * T * C);
    matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
    gelu_backward(dl_fch_gelu, dl_fcproj, l_fch, B * T * 4 * C);
    matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
    layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
    residual_backward(dl_residual2, dresidual, dl_attproj, B * T * C);
    matmul_backward(dl_atty, dl_attnprojw, dl_attnprojb, dl_attproj, l_atty, l_attnprojw, B, T, C, C);
    attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
    matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
    layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
  }
  encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
  if (model->m_memory == NULL) {
    model->m_memory = (float *)calloc(model->num_parameters, sizeof(float));
    model->v_memory = (float *)calloc(model->num_parameters, sizeof(float));
  }

#pragma omp parallel for
  for (size_t i = 0; i < model->num_parameters; i++) {
    float param = model->params_memory[i];
    float grad = model->grads_memory[i];

    // update first moment (momentum)
    float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
    // update second moment (RMSprop)
    float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
    // bias-correct both moments
    float m_hat = m / (1.0f - powf(beta1, t));
    float v_hat = v / (1.0f - powf(beta2, t));

    // update
    model->m_memory[i] = m;
    model->v_memory[i] = v;
    model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
  }
}

void gpt2_free(GPT2 *model) {
  free(model->params_memory);
  free(model->grads_memory);
  free(model->m_memory);
  free(model->v_memory);
  free(model->act_memory);
  free(model->grads_acts_memory);
  free(model->inputs);
  free(model->targets);
}

void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path) {
  // read model data from a checkpoint file.
  FILE *model_file = fopenCheck(checkpoint_path, "rb");
  int model_header[256];
  freadCheck(model_header, sizeof(int), 256, model_file);

  // quirk from karpathy/llmc code.
  if (model_header[0] != 20240326) {
    printf("Bad magic model file.\n");
    exit(1);
  }
  if (model_header[1] != 3) {
    printf("Bad version in model file.\n");
    exit(1);
  }

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

unsigned int random_u32(uint64_t *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

/**
 * Generates a random float in the range [0, 1).
 */
float random_f32(uint64_t *state) { return (random_u32(state) >> 8) / 16777216.0f; }

/**
 * Sample a token from a multinomial distribution.
 */
int sample_mult(float *probabilities, int n, float coin) {
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1;
}

int main() {
  // build the GPT-2 model from a checkpoint.
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

  // build the dataloaders.
  const char *train_tokens = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
  const char *val_tokens = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
  int B = 4;   // batch size 4
  int T = 64;  // sequence length 64
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
  int *gen_tokens = (int *)mallocCheck(B * T * sizeof(int));
  const int gen_steps = 64;

  // train
  struct timespec start, end;
  for (int step = 0; step <= 40; step++) {
    // once in a while run eval.
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
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
      for (int t = 1; t < gen_steps; t++) {
        gpt2_forward(&model, gen_tokens, NULL, B, T);
        float *probs = model.acts.probs + (t - 1) * model.config.padded_vocab_size;
        float coin = random_f32(&rng_state);
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;

        if (tokenizer.init_ok) {
          const char *token_str = tokenizer_decode(&tokenizer, next_token);
          safe_printf(token_str);
        } else {
          printf("%d ", next_token);
        }
        fflush(stdout);
      }
      printf("\n---\n");
    }

    // do a train step.
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1e3f);
  }

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  gpt2_free(&model);
  free(gen_tokens);
  return 0;
}