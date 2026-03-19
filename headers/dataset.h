#ifndef DATASET_H
#define DATASET_H

#include "types.h"
#include <stddef.h>

/**
 * @brief In-memory representation of an MNIST dataset split.
 *
 * Ownership and lifecycle:
 * - initialize with `Dataset dataset = {0};`
 * - load with `dataset_load_mnist(&dataset, image_path, label_path);`
 * - optionally shuffle in place with `dataset_shuffle(&dataset, dataset.n_samples);`
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
 * @brief Loads an MNIST dataset split from the provided image and label files.
 *
 * This function reads the given MNIST image file and label file, validates that
 * they describe the same number of samples, and stores the resulting buffers
 * and metadata in `dataset`.
 *
 * The file paths must point to raw uncompressed IDX files.
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
 * - call this function before training or evaluation
 * - later call `dataset_free(&dataset)` when the dataset is no longer needed
 *
 * @param dataset Dataset object to initialize or reload.
 * @param img_path Path to the MNIST image file for the desired split.
 * @param label_path Path to the MNIST label file for the desired split.
 */
void dataset_load_mnist(Dataset *dataset, const char *img_path, const char *label_path);

/**
 * @brief Randomly shuffles a prefix of loaded dataset samples in place.
 *
 * Images and labels remain aligned after shuffling, so each label continues to
 * describe the image stored at the same sample index.
 *
 * The shuffle is applied to the half-open range `[0, end_index)`.
 *
 * Range handling:
 * - if `end_index > dataset->n_samples`, it is clamped to `dataset->n_samples`
 * - if `end_index < 2`, the function does nothing
 *
 * Safe no-op behavior:
 * - if `dataset` is NULL
 * - if the dataset is not fully loaded
 * - if the dataset contains fewer than two total samples
 *
 * @param dataset Dataset object to shuffle.
 * @param end_index Exclusive end of the prefix range to shuffle.
 */
void dataset_shuffle(Dataset *dataset, size_t end_index);

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
