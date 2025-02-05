/*
 GPT-2 Training loop for CUDA.
 $> nvcc train_gpt2_fp32.cu -o train_gpt2_fp32 -lcublas && ./train_gpt2_fp32
*/
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

typedef struct {
  GPT2Config config;
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint) {
  printf("Loading checkpoint from %s\n", checkpoint);
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
  cublasCheck(cublasDestroy(cublas_handle));
}