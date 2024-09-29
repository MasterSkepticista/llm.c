/*
  This file contains utilities shared between different training scripts.
  We define a series of macros xxxCheck that call the corresponding C stdlib
  functions and check their return code, with additional debug information
  wherever applicable.  
*/
#include <stdio.h>
#include <stdlib.h>


/**
 * @brief Opens a file and checks for errors.
 *
 * This function attempts to open a file with the specified path and mode.
 * If the file cannot be opened, it prints an error message to stderr and
 * terminates the program.
 *
 * @param path The path to the file to be opened.
 * @param mode The mode in which to open the file.
 * @param file The name of the source file where this function is called.
 * @param line The line number in the source file where this function is called.
 * @return A pointer to the opened file.
 */
extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
  FILE *fp = fopen(path, mode);
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file `%s` at %s:%d\n", path, file, line);
    exit(EXIT_FAILURE);
  }
  return fp;
}
#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)


/**
 * @brief Reads data from a file stream and checks for errors.
 *
 * This function attempts to read `nmemb` elements of data, each `size` bytes long, from the given
 * file stream `stream` into the buffer pointed to by `ptr`. If the read operation does not
 * successfully read the expected number of elements, it checks for end-of-file or file read errors
 * and prints an appropriate error message along with the file name and line number where the error
 * occurred. The program will exit with a failure status if an error is detected.
 *
 * @param ptr Pointer to a block of memory with a size of at least (`size` * `nmemb`) bytes.
 * @param size Size in bytes of each element to be read.
 * @param nmemb Number of elements, each one with a size of `size` bytes.
 * @param stream Pointer to a FILE object that specifies an input stream.
 * @param file Name of the file where the fread_check function is called.
 * @param line Line number in the file where the fread_check function is called.
 */
extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
  size_t result = fread(ptr, size, nmemb, stream);
  if (result != nmemb) {
    if (feof(stream)) {
      fprintf(stderr, "Unexpected EOF at %s:%d\n", file, line);
    } else if (ferror(stream)) {
      fprintf(stderr, "File read error at %s:%d\n", file, line);
    } else {
      fprintf(stderr, "Partial read error at %s:%d, expected %zu elements, found %zu.", 
        file, line, nmemb, result);
    }
    exit(EXIT_FAILURE);
  }
}
#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)