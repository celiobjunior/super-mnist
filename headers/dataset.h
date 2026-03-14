#ifndef DATASET_H
#define DATASET_H

#include "types.h"
#include <stddef.h>

/**
 * @brief In-memory representation of the MNIST training dataset.
 *
 * Ownership and lifecycle:
 * - initialize with `Dataset dataset = {0};`
 * - load with `dataset_load_mnist(&dataset);`
 * - release with `dataset_free(&dataset);`
 *
 * Reload behavior:
 * - `dataset_load_mnist()` may be called again on an already loaded dataset
 * - if the dataset already owns buffers, they are released before reloading
 *
 * After a successful load:
 * - `images` points to a contiguous buffer of
 *   `n_samples * pixels_per_image` bytes
 * - `labels` points to a contiguous buffer of `n_samples` bytes
 * - image `i` starts at `images + i * pixels_per_image`
 * - label `i` is stored at `labels[i]`
 *
 * After `dataset_free()`:
 * - all owned memory is released
 * - the struct is reset to zero
 */
typedef struct Dataset {
        u8 *images, *labels;
        size_t n_samples, pixels_per_image;
} Dataset;

/**
 * @brief Loads the MNIST training dataset configured by the project.
 *
 * This function reads both the configured image file and label file, validates
 * that they describe the same number of samples, and stores the resulting
 * buffers and metadata in `dataset`.
 *
 * Guarantees on success:
 * - `dataset->images` and `dataset->labels` are heap-allocated and owned by
 *   `dataset`
 * - `dataset->n_samples` is the number of images and labels
 * - `dataset->pixels_per_image` is the number of pixels in one image
 *
 * Reload behavior:
 * - if `dataset` already owns buffers, they are released before loading again
 * - this makes repeated calls safe for a previously loaded dataset
 *
 * Expected usage:
 * - pass a zero-initialized dataset object before the first load
 * - call this function before training
 * - later call `dataset_free(&dataset)` when the dataset is no longer needed
 *
 * @param dataset Dataset object to initialize or reload.
 */
void dataset_load_mnist(Dataset *dataset);

/**
 * @brief Releases all memory owned by a dataset and resets it to zero.
 *
 * Safe to call with a zero-initialized dataset. After this call, the dataset
 * no longer owns any buffers.
 *
 * @param dataset Dataset object to clear.
 */
void dataset_free(Dataset *dataset);

#endif
