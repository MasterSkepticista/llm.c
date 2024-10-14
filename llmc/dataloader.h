/*
Implements:
- DataLoader for model training. Reads and serves data shards.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <assert.h>
#include <glob.h>
#include <stdint.h>

#include "rand.h"
#include "utils.h"

#define HEADER_SIZE 256

typedef struct {
  // file info.
  FILE *tokens_file;
  glob_t glob_result;
  int *shard_indices;
  int *intra_shard_indices;

  // shard related
  int current_shard_idx;
  int current_sample_idx;

  // for distributed training.
  int process_rank;
  int num_processes;

  // token-related.
  size_t B;
  size_t T;
  size_t num_tokens;
  size_t shard_num_samples;

  // randomness-related.
  int should_shuffle;
  mt19937_state shuffle_rng;

  // state.
  size_t header_bytes;
  size_t file_size_bytes;
  size_t total_batch_size_bytes;
  size_t local_batch_offset_bytes;

  // data buffers.
  uint16_t *buffer;
  int *inputs;
  int *targets;

} DataLoader;

/**
 * @brief Loads a shard of data into the DataLoader.
 *
 * This function loads a specific shard of data into the DataLoader, optionally
 * shuffling the shard index if required. It validates the header of the data
 * file, checks the file size, and calculates the number of samples that can be
 * obtained from the shard.
 *
 * @param loader Pointer to the DataLoader structure.
 * @param shard_index Index of the shard to be loaded.
 * @return The number of tokens in the loaded shard.
 *
 * @note The function will exit the program if the header validation fails or if
 * the file size is inconsistent with the expected number of tokens.
 */
int64_t dataloader_load_shard_(DataLoader *loader, int shard_index) {
  if (loader->should_shuffle) {
    shard_index = loader->shard_indices[shard_index];
  }
  const char *filename = loader->glob_result.gl_pathv[shard_index];

  // Close any previously open shard file.
  if (loader->tokens_file != NULL) {
    fcloseCheck(loader->tokens_file);
  }
  loader->tokens_file = fopenCheck(filename, "rb");

  // validate the header
  int header[HEADER_SIZE];
  freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
  if (header[0] != 20240520) {
    fprintf(stderr, "Bad magic value in the data file.\n");
    exit(EXIT_FAILURE);
  }
  if (header[1] != 1) {
    printf("Bad version in the data file.\n");
    exit(EXIT_FAILURE);
  }
  int64_t ntok = header[2];
  assert(ntok > 0);

  // determine file size and make sure it is consistent with number of tokens.
  fseekCheck(loader->tokens_file, 0, SEEK_END);          // seek to end of file.
  loader->file_size_bytes = ftell(loader->tokens_file);  // read the offset, i.e. file size.
  fseekCheck(loader->tokens_file, 0, SEEK_SET);          // seek to start of the file.
  int64_t expected_file_size = HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
  if (loader->file_size_bytes != expected_file_size) {
    printf("Error: file size is not as expected.\n");
    printf("Expected %zu bytes, found %zu bytes.\n", expected_file_size, loader->file_size_bytes);
  }
  // number of batches that can be sampled from this shard.
  loader->shard_num_samples = ((ntok - 1) * sizeof(uint16_t)) / loader->total_batch_size_bytes;
  return ntok;
}

/**
 * @brief Shuffles examples within a shard.
 *
 * @param loader Pointer to the DataLoader structure.
 */
void prepare_intra_shard_indices_(DataLoader *loader) {
  // shuffle examples within the shard.
  if (loader->intra_shard_indices != NULL) {
    free(loader->intra_shard_indices);
  }
  loader->intra_shard_indices = (int *)mallocCheck(loader->shard_num_samples * sizeof(int));
  init_identity_permutation(loader->intra_shard_indices, (int)loader->shard_num_samples);
  random_permutation(loader->intra_shard_indices, (int)loader->shard_num_samples, &loader->shuffle_rng);
}

/**
 * @brief Resets the DataLoader to its initial state.
 *
 * This function resets the current shard and sample indices to zero.
 * If shuffling is enabled, it shuffles the shard indices and prepares
 * intra-shard indices.
 *
 * @param loader Pointer to the DataLoader instance to reset.
 */
void dataloader_reset(DataLoader *loader) {
  loader->current_shard_idx = 0;
  loader->current_sample_idx = 0;

  if (loader->should_shuffle) {
    random_permutation(loader->shard_indices, (int)loader->glob_result.gl_pathc, &loader->shuffle_rng);
  }

  dataloader_load_shard_(loader, (int)loader->current_shard_idx);

  if (loader->should_shuffle) {
    prepare_intra_shard_indices_(loader);
  }
}

/**
 * @brief Advances the DataLoader to the next shard.
 *
 * If the DataLoader is at the final shard, it resets to the first shard.
 * Otherwise, it advances to the next shard, resets the sample index, and
 * loads the new shard. If shuffling is enabled, it prepares intra-shard
 * indices.
 *
 * @param loader Pointer to the DataLoader instance.
 */
void dataloader_advance_(DataLoader *loader) {
  // if at the final shard, reset to first shard.
  if (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
    dataloader_reset(loader);
    return;
  }
  // advance the loader by loading the next shard and resetting its position.
  loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
  loader->current_sample_idx = 0;
  dataloader_load_shard_(loader, (int)loader->current_shard_idx);

  if (loader->should_shuffle) {
    prepare_intra_shard_indices_(loader);
  }
}

/**
 * @brief Loads a batch of data into the DataLoader.
 *
 * This function reads a batch of data from the tokens file and populates
 * the inputs and targets buffers. It supports shuffling of data if enabled.
 *
 * @param loader Pointer to the DataLoader structure.
 *
 * Preconditions:
 * - If shuffling is enabled, intra_shard_indices must not be NULL.
 * - current_sample_idx must be less than shard_num_samples.
 *
 * The function calculates the appropriate offset in the file based on
 * whether shuffling is enabled and reads the data into a buffer. It then
 * transfers the data from the buffer to the inputs and targets arrays.
 */
void dataloader_load_batch(DataLoader *loader) {
  assert(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != NULL));
  assert(loader->current_sample_idx < loader->shard_num_samples);  // check batches are available.
  size_t idx =
      loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
  size_t global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
  // each process would have a local offset. for single-process, local offset is
  // zero.
  int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

  size_t B = loader->B;
  size_t T = loader->T;
  // read BT+1 uint16_t tokens from the file to buffer.
  fseekCheck(loader->tokens_file, current_offset, SEEK_SET);
  freadCheck(loader->buffer, sizeof(uint16_t), B * T + 1, loader->tokens_file);
  // move from buffer to inputs/targets.
  for (int i = 0; i < B * T; i++) {
    loader->inputs[i] = (int)loader->buffer[i];
    loader->targets[i] = (int)loader->buffer[i + 1];
  }
}

/**
 * @brief Loads the next batch of data from the DataLoader.
 *
 * If the current sample index is at the end of the shard, it advances to the
 * next shard. Then, it loads the batch and increments the current sample index.
 *
 * @param loader Pointer to the DataLoader instance.
 */
void dataloader_next_batch(DataLoader *loader) {
  // if at the end of the shard, move to the next shard.
  if (loader->current_sample_idx >= loader->shard_num_samples) {
    dataloader_advance_(loader);
  }
  dataloader_load_batch(loader);
  loader->current_sample_idx += 1;
}

/**
 * @brief Initializes the DataLoader with the specified parameters.
 *
 * This function sets up the DataLoader by initializing its fields,
 * performing file globbing to match the given filename pattern,
 * shuffling if required, and validating all shards to avoid runtime errors.
 *
 * @param loader Pointer to the DataLoader to initialize.
 * @param filename_pattern Pattern to match filenames for loading data.
 * @param B Batch size.
 * @param T Sequence length.
 * @param process_rank Rank of the current process, for distributed mode.
 * @param num_processes Total number of processes, for distributed mode.
 * @param should_shuffle Flag indicating whether to shuffle the shards.
 */
void dataloader_init(DataLoader *loader, const char *filename_pattern, size_t B, size_t T, int process_rank,
                     int num_processes, int should_shuffle) {
  loader->process_rank = process_rank;
  loader->num_processes = num_processes;
  loader->B = B;
  loader->T = T;
  loader->tokens_file = NULL;
  loader->should_shuffle = should_shuffle;
  loader->header_bytes = HEADER_SIZE * sizeof(int);
  loader->total_batch_size_bytes = loader->num_processes * (loader->B * loader->T) * sizeof(uint16_t);
  loader->local_batch_offset_bytes = loader->process_rank * (loader->B * loader->T) * sizeof(uint16_t);

  // glob to get a list of all files matching the pattern.
  int glob_status = glob(filename_pattern, 0, NULL, &loader->glob_result);
  if (glob_status != 0) {
    fprintf(stderr, "Failed to glob pattern `%s`\n", filename_pattern);
    exit(EXIT_FAILURE);
  }
  if (loader->glob_result.gl_pathc == 0) {
    fprintf(stderr, "No files matched pattern `%s`\n", filename_pattern);
    exit(EXIT_FAILURE);
  }

  if (should_shuffle) {
    mt19937_state shuffle_rng;
    manual_seed(&shuffle_rng, 42 + process_rank);
    loader->shuffle_rng = shuffle_rng;
    // Create an array of shard indices, and fill those values with
    // range(len(shard_indices)).
    loader->shard_indices = (int *)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
    init_identity_permutation(loader->shard_indices, (int)loader->glob_result.gl_pathc);
    loader->intra_shard_indices = NULL;
  }

  // inspect and validate all shards to avoid runtime errors.
  int64_t ntok_total = 0;
  for (int shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
    int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);
    assert(shard_ntok >= (int64_t)(num_processes * B * T + 1));
    ntok_total += shard_ntok;
  }
  printf("DataLoader: filename_pattern %s\n", filename_pattern);
  printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

  // allocate necessary space.
  loader->buffer = (uint16_t *)mallocCheck((B * T + 1) * sizeof(uint16_t));
  loader->inputs = (int *)mallocCheck(B * T * sizeof(int));
  loader->targets = (int *)mallocCheck(B * T * sizeof(int));
  loader->num_tokens = ntok_total;

  // reset the loader
  dataloader_reset(loader);
}

void dataloader_free(DataLoader *loader) {
  free(loader->buffer);
  free(loader->inputs);
  free(loader->targets);
  if (loader->should_shuffle) {
    free(loader->shard_indices);
    free(loader->intra_shard_indices);
  }
  fcloseCheck(loader->tokens_file);
  globfree(&loader->glob_result);
}

#endif