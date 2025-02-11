/*
 GPT-2 Training loop for CUDA.
 $> nvcc train_gpt2_fp32.cu -o train_gpt2_fp32 -lcublas && ./train_gpt2_fp32
*/
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "llmc/dataloader.h"
#include "llmc/logger.h"
#include "llmc/tokenizer.h"
#include "llmc/utils.h"

/**
 * CUDA error checking.
 */
void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

/**
 * cuBLAS error checking.
 */
void cublasCheck(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", status, file, line);
    exit(EXIT_FAILURE);
  }
}
#define cublasCheck(status) (cublasCheck(status, __FILE__, __LINE__))

/**
 * Ceildiv helper
 */
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// Kernels

__device__ inline float4 add_float4(const float4 &a, const float4 &b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void encoder_forward_kernel3(float4 *out, const int *inp, const float4 *wte, const float4 *wpe, int B, int T,
                                        int C) {
  int C4 = C / 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = B * T * C4;
  if (idx < N) {
    int bt = idx / C4;
    int b = bt / T;
    int t = bt % T;
    int c4 = idx % C4;
    int ix = inp[b * T + t];
    out[b * T * C4 + t * C4 + c4] = add_float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
  }
}
// Launchers

/**
 * Encoder forward pass kernel launcher.
 * @param out: ptr to store output tensor of shape (B, T, C).
 * @param inp: ptr to input tokens tensor of shape (B, T).
 * @param wte: ptr to token embedding weights.
 * @param wpe: ptr to positional embedding weights.
 */
void encoder_forward(float *out, const int *inp, const float *wte, const float *wpe, int B, int T, int C) {
  assert(C % 4 == 0);
  const int block_size = 512;
  const int N = B * T * C;
  const int grid_size = CEIL_DIV(N / 4, block_size);
  encoder_forward_kernel3<<<grid_size, block_size>>>((float4 *)out, inp, (float4 *)wte, (float4 *)wpe, B, T, C);
  cudaCheck(cudaGetLastError());
}

/**
 * Program global cuBLAS handle.
 */
static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;

/**
 * GPT2 config and model.
 */
typedef struct {
  int max_seq_len;        // max sequence length supported by the model.
  int vocab_size;         // vocabulary size.
  int padded_vocab_size;  // vocab size padded to some nearest multiple of 64/128.
  int num_layers;         // number of transformer layers.
  int num_heads;          // number of attention heads.
  int channels;           // hidden size of the model.
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
  float *wte;      // (Vp, C)
  float *wpe;      // (T, C)
  float *ln1w;     // (L, C)
  float *ln1b;     // (L, C)
  float *qkvw;     // (L, 3*C, C)
  float *qkvb;     // (L, 3*C)
  float *projw;    // (L, C, C)
  float *projb;    // (L, C)
  float *ln2w;     // (L, C)
  float *ln2b;     // (L, C)
  float *fcw;      // (L, 4*C, C)
  float *fcb;      // (L, 4*C)
  float *fcprojw;  // (L, C, 4*C)
  float *fcprojb;  // (L, C)
  float *lnfw;     // (C,)
  float *lnfb;     // (C,)
} ParameterTensors;

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
  float *encoded;    // (B, T, C)
  float *ln1;        // (L, B, T, C)
  float *ln1_mean;   // (L, B, T)
  float *ln1_rstd;   // (L, B, T)
  float *qkv;        // (L, B, T, 3*C)
  float *atty;       // (L, B, T, C)
  float *preatt;     // (L, B, NH, T, T)
  float *attn;       // (L, B, NH, T, T)
  float *attn_proj;  // (L, B, T, C)
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
  float *logits;     // (B, T, Vp)
  float *probs;      // (B, T, Vp)
  float *losses;     // (B, T)
} ActivationTensors;

typedef struct {
  GPT2Config config;

  ParameterTensors params;
  size_t param_sizes[NUM_PARAMETER_TENSORS];
  float *params_memory;
  size_t num_parameters;

  // Optimizer moments
  float *m_memory;
  float *v_memory;

  // Intermediates
  ActivationTensors acts;
  size_t acts_sizes[NUM_ACTIVATION_TENSORS];
  float *acts_memory;
  size_t num_activations;

  float *grads_memory;
  float *grads_acts_memory;

  // Inputs/outputs
  int seq_len;
  int batch_size;
  int *inputs;
  int *targets;
  float mean_loss;
  float *cpu_losses;

} GPT2;

void fill_in_parameter_sizes(size_t *param_sizes, GPT2Config config) {
  int Vp = config.padded_vocab_size;
  int C = config.channels;
  int T = config.max_seq_len;
  int L = config.num_layers;
  param_sizes[0] = Vp * C;            // wte
  param_sizes[1] = T * C;             // wpe
  param_sizes[2] = L * C;             // ln1w
  param_sizes[3] = L * C;             // ln1b
  param_sizes[4] = L * (3 * C) * C;   // qkvw
  param_sizes[5] = L * (3 * C);       // qkvb
  param_sizes[6] = L * C * C;         // projw
  param_sizes[7] = L * C;             // projb
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
 * Allocate memory for model parameters.
 *
 * @param params: Pointer to the parameter tensors.
 * @param param_sizes: Array of sizes of each parameter tensor.
 * @param on_device: Allocate memory on device (1) or host (0).
 * @return: Pointer to the allocated memory.
 */
float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes, int on_device) {
  // malloc at once on device/host for all params.
  size_t num_parameters = 0;
  for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  float *params_memory;
  size_t total_bytes = num_parameters * sizeof(float);
  if (on_device) {
    cudaCheck(cudaMalloc((void **)&params_memory, total_bytes));
  } else {
    params_memory = (float *)mallocCheck(total_bytes);
  }

  // point each param to its relevant memory block.
  float **ptrs[] = {&params->wte,     &params->wpe,     &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
                    &params->projw,   &params->projb,   &params->ln2w, &params->ln2b, &params->fcw,  &params->fcb,
                    &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb};
  float *params_memory_iterator = params_memory;
  for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }

  return params_memory;
}

void fill_in_activation_sizes(GPT2Config config, size_t *acts_sizes, int B, int T) {
  int Vp = config.padded_vocab_size;
  int V = config.vocab_size;
  int NH = config.num_heads;
  int L = config.num_layers;
  int C = config.channels;
  acts_sizes[0] = B * T * C;             // encoded
  acts_sizes[1] = L * B * T * C;         // ln1
  acts_sizes[2] = L * B * T;             // ln1_mean
  acts_sizes[3] = L * B * T;             // ln1_rstd
  acts_sizes[4] = L * B * T * (3 * C);   // qkv
  acts_sizes[5] = L * B * T * C;         // atty
  acts_sizes[6] = L * B * NH * T * T;    // preatt
  acts_sizes[7] = L * B * NH * T * T;    // attn
  acts_sizes[8] = L * B * T * C;         // attn_proj
  acts_sizes[9] = L * B * T * C;         // residual2
  acts_sizes[10] = L * B * T * C;        // ln2
  acts_sizes[11] = L * B * T;            // ln2_mean
  acts_sizes[12] = L * B * T;            // ln2_rstd
  acts_sizes[13] = L * B * T * (4 * C);  // fch
  acts_sizes[14] = L * B * T * (4 * C);  // fch_gelu
  acts_sizes[15] = L * B * T * C;        // fc_proj
  acts_sizes[16] = L * B * T * C;        // residual3
  acts_sizes[17] = L * B * T * C;        // lnf
  acts_sizes[18] = L * B * T;            // lnf_mean
  acts_sizes[19] = L * B * T;            // lnf_rstd
  acts_sizes[20] = B * T * Vp;           // logits
  acts_sizes[21] = B * T * Vp;           // probs
  acts_sizes[22] = B * T;                // losses
}

float *malloc_and_point_activations(ActivationTensors *acts, size_t *acts_sizes) {
  // Count total activation size.
  size_t num_activations = 0;
  for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    num_activations += acts_sizes[i];
  }
  size_t total_bytes = sizeof(float) * num_activations;

  // Allocate one-shot.
  float *acts_memory;
  cudaCheck(cudaMalloc((void **)&acts_memory, total_bytes));

  // Point each activation block to its respective area.
  float **ptrs[] = {&acts->encoded, &acts->ln1,       &acts->ln1_mean, &acts->ln1_rstd,  &acts->qkv,
                    &acts->atty,    &acts->preatt,    &acts->attn,     &acts->attn_proj, &acts->residual2,
                    &acts->ln2,     &acts->ln2_mean,  &acts->ln2_rstd, &acts->fch,       &acts->fch_gelu,
                    &acts->fcproj,  &acts->residual3, &acts->lnf,      &acts->lnf_mean,  &acts->lnf_rstd,
                    &acts->logits,  &acts->probs,     &acts->losses};
  float *acts_memory_iterator = acts_memory;
  for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    *ptrs[i] = acts_memory_iterator;
    acts_memory_iterator += acts_sizes[i];
  }
  return acts_memory;
}

/**
 * Load model parameters from a checkpoint file.
 * @param model: Pointer to the GPT2 model.
 * @param checkpoint: Path to the checkpoint file.
 * @return: None.
 */
void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint) {
  FILE *file = fopenCheck(checkpoint, "rb");
  int model_header[256];
  freadCheck(model_header, sizeof(int), 256, file);
  if (model_header[0] != 20240326 || model_header[1] != 3) {
    fprintf(stderr, "Bad magic number \n");
    exit(EXIT_FAILURE);
  }

  // read in hyperparameters
  model->config.max_seq_len = model_header[2];
  model->config.vocab_size = model_header[3];
  model->config.num_layers = model_header[4];
  model->config.num_heads = model_header[5];
  model->config.channels = model_header[6];
  model->config.padded_vocab_size = model_header[7];

  // allocate space for all the parameters and read them in.
  fill_in_parameter_sizes(model->param_sizes, model->config);
  size_t num_parameters = 0;
  for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  model->num_parameters = num_parameters;

  // Allocate memory for model parameters on the device.
  model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

  // Copy model params from file to the params_memory pointer.
  float *params_memory_cpu = (float *)mallocCheck(num_parameters * sizeof(float));
  freadCheck(params_memory_cpu, sizeof(float), num_parameters, file);
  cudaCheck(
      cudaMemcpy(model->params_memory, params_memory_cpu, sizeof(float) * num_parameters, cudaMemcpyHostToDevice));
  free(params_memory_cpu);
  fcloseCheck(file);

  // other inits.
  model->acts_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->cpu_losses = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f;
}

/**
 * GPT2 forward pass
 * @param model: Pointer to GPT2 Model instance
 * @param inputs: Pointer to inputs tensor of shape (B, T).
 * @param targets: Pointer to targets tensor of shape (B, T). Optional.
 * @param B: Batch size
 * @param T: Seq length
 */
void gpt2_forward(GPT2 *model, int *inputs, int *targets, int B, int T) {
  if (model->params_memory == NULL) {
    printf("Error: model was not initialized properly.");
    exit(EXIT_FAILURE);
  }

  // Shorthands.
  int V = model->config.vocab_size;
  int Vp = model->config.padded_vocab_size;
  int L = model->config.num_layers;
  int NH = model->config.num_heads;
  int C = model->config.channels;

  // Validate token values.
  for (int i = 0; i < B * T; i++) {
    assert(0 <= inputs[i] && inputs[i] < V);
    if (targets != NULL) {
      assert(0 <= targets[i] && targets[i] < V);
    }
  }

  // allocate space for all the activations lazily.
  if (model->acts_memory == NULL) {
    model->batch_size = B;
    model->seq_len = T;
    fill_in_activation_sizes(model->config, model->acts_sizes, B, T);
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      num_activations += model->acts_sizes[i];
    }
    model->num_activations = num_activations;
    model->acts_memory = malloc_and_point_activations(&model->acts, model->acts_sizes);
    printf("Allocating %zu MiB for activations.\n", (num_activations * sizeof(float)) >> 20);
    // allocate for input/output
    cudaCheck(cudaMalloc((void **)&model->inputs, sizeof(int) * B * T));
    cudaCheck(cudaMalloc((void **)&model->targets, sizeof(int) * B * T));
    cudaCheck(cudaMallocHost((void **)&model->cpu_losses, sizeof(float) * B * T));
  } else {
    // validate B, T.
    if (B != model->batch_size || T != model->seq_len) {
      printf("Model Expected: B=%d T=%d, Got: B=%d T=%d", model->batch_size, model->seq_len, B, T);
      exit(EXIT_FAILURE);
    }
  }
  // Copy inputs/targets to device.
  cudaCheck(cudaMemcpy(model->inputs, inputs, sizeof(int) * B * T, cudaMemcpyHostToDevice));
  if (targets != NULL) {
    cudaCheck(cudaMemcpy(model->inputs, inputs, sizeof(int) * B * T, cudaMemcpyHostToDevice));
  }

  // foward pass
  ParameterTensors params = model->params;
  ActivationTensors acts = model->acts;
  float *residual;
  encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);
  // encoder forward
  // loop - layers forward
  // final ln forward
  // decode to logits
  // ce forward?
  // loss
}

/**
 * Something like argparse.
 */
void error_usage() {
  fprintf(stderr, "Usage: ./train_gpt2_fp32.cu [options]\n");
  fprintf(stderr, "Options: \n");
  fprintf(stderr,
          " -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
  fprintf(stderr,
          " -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
  fprintf(stderr, " -o <string> output log file\n");
  fprintf(stderr, " -b <int>    batch size B (default = 4)\n");
  fprintf(stderr, " -t <int>    seq length T (default = 1024)\n");
  fprintf(stderr, " -l <float>  learning rate (default = 3e-4f)\n");
  fprintf(stderr, " -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
  fprintf(stderr, " -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
  fprintf(stderr, " -s <int>    sample_every, how often we inference the model (default = 20)\n");
  fprintf(stderr, " -g <int>    genT, how many steps of inference we do (default = 64)\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  const char *train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
  const char *val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
  const char *output_log_file = NULL;
  int B = 4;
  int T = 1024;
  float learning_rate = 3e-4f;
  int val_loss_every = 20;
  int val_max_steps = 20;
  int sample_every = 20;
  int genT = 64;

  // clang-format off
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) { error_usage(); }
    if (argv[i][0] != '-') { error_usage(); }
    if (strlen(argv[i]) != 2) { error_usage(); }
    // read in args
    if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
    else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
    else if (argv[i][1] == 'o') { output_log_file = argv[i+1]; }
    else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); }
    else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
    else if (argv[i][1] == 'l') { learning_rate = atof(argv[i+1]); }
    else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
    else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
    else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
    else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
    else { error_usage(); }
  }
  // clang-format on

  printf("+---------------------+----------------------------------------------------+\n");
  printf("| Parameter           | Value                                              |\n");
  printf("+---------------------+----------------------------------------------------+\n");
  printf("| train_data_pattern  | %-50s |\n", train_data_pattern);
  printf("| val_data_pattern    | %-50s |\n", val_data_pattern);
  printf("| output_log_file     | %-50s |\n", output_log_file);
  printf("| batch size B        | %-50d |\n", B);
  printf("| seq length T        | %-50d |\n", T);
  printf("| learning rate       | %-50f |\n", learning_rate);
  printf("| val_loss_every      | %-50d |\n", val_loss_every);
  printf("| val_max_steps       | %-50d |\n", val_max_steps);
  printf("| sample_every        | %-50d |\n", sample_every);
  printf("| genT                | %-50d |\n", genT);
  printf("+---------------------+----------------------------------------------------+\n");

  // set up the device
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceIdx));
  // Setup cuBLAS and cuBLASLt
  cublasCheck(cublasCreate(&cublas_handle));
  // TF32 precision is equivalent to torch.set_float32_matmul_precision("high")
  int enable_tf32 = prop.major >= 8 ? 1 : 0;
  cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
  printf("| Device Name         | %-50s |\n", prop.name);
  printf("| TensorFloat32       | %-50s |\n", prop.major >= 8 ? "Yes" : "No");
  printf("+---------------------+----------------------------------------------------+\n");

  // Build GPT2 model from a checkpoint.
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
  printf("| max_sequence_len T  | %-50d |\n", model.config.max_seq_len);
  printf("| vocab_size V        | %-50d |\n", model.config.vocab_size);
  printf("| padded_vocab_size Vp| %-50d |\n", model.config.padded_vocab_size);
  printf("| num_layers L        | %-50d |\n", model.config.num_layers);
  printf("| num_heads NH        | %-50d |\n", model.config.num_heads);
  printf("| channels C          | %-50d |\n", model.config.channels);
  printf("| num_parameters      | %-50zu |\n", model.num_parameters);
  printf("+---------------------+----------------------------------------------------+\n");
  printf("Allocated %d MiB for model parameters\n", (int)round(model.num_parameters * sizeof(float) / (1024 * 1024)));

  // Build dataloaders.
  DataLoader train_loader, val_loader;
  dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
  dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
  int train_num_batches = train_loader.num_tokens / (B * T);
  int val_num_batches = val_loader.num_tokens / (B * T);
  if (val_num_batches > val_max_steps) {
    val_num_batches = val_max_steps;
  }
  printf("| train_num_batches   | %-50d |\n", train_num_batches);
  printf("| val_num_batches     | %-50d |\n", val_num_batches);
  printf("+---------------------+----------------------------------------------------+\n");

  // Set up the logger.
  Logger logger;
  logger_init(&logger, output_log_file);

  // Build tokenizer.
  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  // Some memory for generating samples from the model.

  // train
  struct timespec start, end;
  for (int step = 0; step <= train_num_batches; step++) {
    int last_step = step == train_num_batches;
    if (last_step) {
      break;
    }

    // do a train step.
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double train_step_time = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    int tokens_per_second = B * T / train_step_time;
    printf("Step %d/%d, train_loss: %f (%f ms, %d tok/s)\n", step + 1, train_num_batches, model.mean_loss,
           train_step_time * 1000, tokens_per_second);
  }

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);

  cublasCheck(cublasDestroy(cublas_handle));
  return 0;
}