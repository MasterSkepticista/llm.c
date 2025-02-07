#pragma once
#include <stdio.h>
#include "utils.h"

typedef struct {
    FILE *logfile;
    int flush_every;
} Logger;

void logger_init(Logger *logger, const char *filename) {
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL) {
        logger->logfile = fopenCheck(filename, "w");
    }
}