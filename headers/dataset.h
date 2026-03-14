#ifndef DATASET_H
#define DATASET_H

#include "types.h"
#include <stddef.h>

typedef struct Dataset {
        u8 *images, *labels;
        size_t n_samples, pixels_per_image;
} Dataset;

void dataset_load_mnist(Dataset *dataset);
void dataset_free(Dataset *dataset);

#endif
