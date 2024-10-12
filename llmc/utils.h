/*
  This file contains utilities shared between different training scripts.
  We define a series of macros xxxCheck that call the corresponding C stdlib
  functions and check their return code, with additional debug information
  wherever applicable.  
*/
#ifndef UTILS_H
#define UTILS_H

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


/**
 * @brief Closes a file and checks for errors.
 *
 * This function attempts to close the given file pointer. If the file
 * fails to close, it prints an error message to stderr with the file
 * name and line number where the function was called, and then exits
 * the program with a failure status.
 *
 * @param fp The file pointer to be closed.
 * @param file The name of the file where the function was called.
 * @param line The line number where the function was called.
 */
extern inline void fclose_check(FILE *fp, const char *file, int line) {
  if (fclose(fp) != 0) {
    fprintf(stderr, "Failed to close file at %s:%d\n", file, line);
    exit(EXIT_FAILURE);
  }
}
#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)


/**
 * @brief Allocates memory and checks for allocation failure.
 *
 * This function attempts to allocate a block of memory of the specified size.
 * If the allocation fails, it prints an error message to stderr with the size
 * of the requested memory, the file name, and the line number where the 
 * allocation was attempted, and then exits the program with a failure status.
 *
 * @param size The size of the memory block to allocate, in bytes.
 * @param file The name of the file where the allocation is attempted.
 * @param line The line number in the file where the allocation is attempted.
 * @return A pointer to the allocated memory block.
 */
extern inline void *malloc_check(size_t size, const char *file, int line) {
  void *ptr = malloc(size);
  if (ptr == NULL) {
    fprintf(stderr, "Failed to allocate %zu bytes at %s:%d\n", size, file, line);
    exit(EXIT_FAILURE);
  }
  return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

/**
 * @brief Safely seeks to a specified position in a file and checks for errors.
 *
 * This function attempts to move the file position indicator for the given file stream
 * to a new position defined by the offset and whence parameters. If the operation fails,
 * it prints an error message to stderr and terminates the program.
 *
 * @param fp Pointer to the FILE object that identifies the stream.
 * @param off The offset in bytes to move the file position indicator.
 * @param whence The position from where offset is added. It can be one of the following:
 *               - SEEK_SET: Beginning of file
 *               - SEEK_CUR: Current position of the file pointer
 *               - SEEK_END: End of file
 * @param file The name of the source file where the function is called (usually passed as __FILE__).
 * @param line The line number in the source file where the function is called (usually passed as __LINE__).
 */
extern inline void fseek_check(FILE *fp, long off, int whence, const char *file, int line) {
  if (fseek(fp, off, whence) != 0) {
    fprintf(stderr, "Failed to seek file to offset %ld, whence %d %s:%d\n", off, whence, file, line);
    exit(EXIT_FAILURE);
  }
}
#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

#endif