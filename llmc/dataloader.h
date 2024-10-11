/*
Implements:
- DataLoader for model training. Reads and serves data shards.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

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

  // for distributed training.
  int process_rank;
  int num_processes;

  // token-related.
  size_t B;
  size_t T;

  // randomness-related.
  int should_shuffle;
  mt19937_state shuffle_rng;
  
  // state.
  size_t header_bytes;
  size_t total_batch_size_bytes;
  size_t local_batch_offset_bytes;

} DataLoader;

void dataloader_init(DataLoader *loader, 
                     const char *filename_pattern, 
                     size_t B, 
                     size_t T, 
                     int process_rank, 
                     int num_processes, 
                     int should_shuffle) {
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
    // Create an array of shard indices, and fill those values with range(len(shard_indices)).
    loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
    init_identity_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc);
    //
  }
}

#endif