#include "../headers/dataset.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static u32 read_be_u32(FILE *file)
{
        u8 bytes[4];

        if (fread(bytes, sizeof(u8), 4, file) != 4)
        {
                printf("Failed to read MNIST header.\n");
                exit(1);
        }

        return ((u32) bytes[0] << 24) |
               ((u32) bytes[1] << 16) |
               ((u32) bytes[2] << 8)  |
               ((u32) bytes[3]);
}

/**
 * @brief Reads image data from an MNIST IDX3 file.
 *
 * The MNIST image file contains a 16-byte header followed by raw pixel bytes.
 * All 32-bit header fields are stored in big-endian order (MSB first):
 * - bytes 0..3   : magic number
 * - bytes 4..7   : number of images
 * - bytes 8..11  : number of rows per image
 * - bytes 12..15 : number of columns per image
 * - bytes 16..   : pixel data
 *
 * Pixel organization in the file:
 * - each pixel is stored as one `u8`
 * - valid pixel range is `0..255`
 * - pixels are stored row-wise inside each image
 * - images are stored sequentially, with no padding between them
 *
 * Buffer organization after loading:
 * - `*pixels_per_image = rows * cols`
 * - image `i` starts at offset `i * (*pixels_per_image)`
 * - pixel `(row, col)` from image `i` is stored at:
 *   `(*images)[i * (*pixels_per_image) + row * cols + col]`
 *
 * @param filename Path to the MNIST IDX3 image file.
 * @param images Output parameter that receives a heap-allocated contiguous
 *               buffer containing all pixels from all images. The caller must
 *               free `*images` with `free()`.
 * @param n_samples Output parameter that receives the total number of images
 *                  stored in the file header.
 * @param pixels_per_image Output parameter that receives the number of pixels
 *                         per image (`rows * cols`).
 */
static void dataset_load_mnist_images(const char *filename,
                                      u8 **images,
                                      size_t *n_samples,
                                      size_t *pixels_per_image)
{
        u32 magic_number, n_images_u32, n_rows, n_cols, pixels_per_image_u32;
        size_t total_bytes;
        FILE *images_file;

        printf("Reading MNIST images...\n");

        images_file = fopen(filename, "rb");
        if (!images_file)
        {
                printf("The file does not exist!\n");
                exit(1);
        }

        magic_number = read_be_u32(images_file);
        n_images_u32 = read_be_u32(images_file);
        n_rows = read_be_u32(images_file);
        n_cols = read_be_u32(images_file);

        if (magic_number != 2051U)
        {
                printf("Invalid MNIST image file.\n");
                fclose(images_file);
                exit(1);
        }

        if (n_images_u32 > (u32) INT32_MAX ||
            n_rows > (u32) INT32_MAX ||
            n_cols > (u32) INT32_MAX)
        {
                printf("MNIST image file contains unsupported dimensions.\n");
                fclose(images_file);
                exit(1);
        }

        if (n_rows != 0 && n_cols > ((u32) INT32_MAX / n_rows))
        {
                printf("MNIST image size is too large.\n");
                fclose(images_file);
                exit(1);
        }

        pixels_per_image_u32 = n_rows * n_cols;

        if (pixels_per_image_u32 != 0 &&
            n_images_u32 > ((u32) SIZE_MAX / pixels_per_image_u32))
        {
                printf("MNIST image buffer is too large.\n");
                fclose(images_file);
                exit(1);
        }

        *n_samples = (size_t) n_images_u32;
        *pixels_per_image = (size_t) pixels_per_image_u32;
        total_bytes = (size_t) n_images_u32 * (size_t) pixels_per_image_u32;

        *images = malloc(total_bytes);
        if (!*images)
        {
                printf("Failed to allocate MNIST image buffer.\n");
                fclose(images_file);
                exit(1);
        }

        if (fread(*images, sizeof(u8), total_bytes, images_file) != total_bytes)
        {
                printf("Failed to read MNIST image data.\n");
                fclose(images_file);
                free(*images);
                *images = NULL;
                exit(1);
        }

        fclose(images_file);

        printf("MNIST images completely loaded...\n\n");
}

/**
 * @brief Reads label data from an MNIST IDX1 file.
 *
 * The MNIST label file contains an 8-byte header followed by one byte per
 * label. All 32-bit header fields are stored in big-endian order (MSB first):
 * - bytes 0..3 : magic number
 * - bytes 4..7 : number of labels
 * - bytes 8..  : label data
 *
 * Label organization in the file:
 * - each label is stored as one `u8`
 * - valid label range is `0..9`
 * - labels are stored sequentially, with no padding
 * - label for image `i` is stored at `(*labels)[i]`
 *
 * @param filename Path to the MNIST IDX1 label file.
 * @param labels Output parameter that receives a heap-allocated contiguous
 *               buffer containing all labels. The caller must free `*labels`
 *               with `free()`.
 * @param n_labels Output parameter that receives the total number of labels
 *                 stored in the file header.
 */
static void dataset_load_mnist_labels(const char *filename,
                                      u8 **labels,
                                      size_t *n_labels)
{
        u32 magic_number, n_labels_u32;
        FILE *labels_file;

        printf("Reading MNIST labels...\n");

        labels_file = fopen(filename, "rb");
        if (!labels_file)
        {
                printf("The file does not exist!\n");
                exit(1);
        }

        magic_number = read_be_u32(labels_file);
        n_labels_u32 = read_be_u32(labels_file);

        if (magic_number != 2049U)
        {
                printf("Invalid MNIST label file.\n");
                fclose(labels_file);
                exit(1);
        }

        if (n_labels_u32 > (u32) INT32_MAX)
        {
                printf("MNIST label file contains too many labels.\n");
                fclose(labels_file);
                exit(1);
        }

        *n_labels = (size_t) n_labels_u32;
        *labels = malloc((size_t) n_labels_u32);

        if (!*labels)
        {
                printf("Failed to allocate MNIST label buffer.\n");
                fclose(labels_file);
                exit(1);
        }

        if (fread(*labels, sizeof(u8), (size_t) n_labels_u32, labels_file) != (size_t) n_labels_u32)
        {
                printf("Failed to read MNIST label data.\n");
                fclose(labels_file);
                free(*labels);
                *labels = NULL;
                exit(1);
        }

        fclose(labels_file);

        printf("MNIST labels completely loaded...\n\n");
}

static void dataset_swap_labels(u8 *labels, size_t i, size_t j)
{
        u8 swap;

        if (i == j) return;

        swap = labels[i];
        labels[i] = labels[j];
        labels[j] = swap;
}

static void dataset_swap_image_samples(u8 *images,
                                       size_t pixels_per_image,
                                       size_t i,
                                       size_t j)
{
        size_t base_i, base_j;
        u8 swap;

        if (i == j) return;

        base_i = i * pixels_per_image;
        base_j = j * pixels_per_image;

        for (size_t k = 0; k < pixels_per_image; k++)
        {
                swap = images[base_i + k];
                images[base_i + k] = images[base_j + k];
                images[base_j + k] = swap;
        }
}

void dataset_shuffle(Dataset *dataset, size_t end_index)
{
        if (!dataset ||
            !dataset->images ||
            !dataset->labels ||
            dataset->pixels_per_image == 0 ||
            dataset->n_samples < 2)
                return;

        if (end_index > dataset->n_samples)
                end_index = dataset->n_samples;

        if (end_index < 2) return;

        for (size_t i = end_index - 1; i > 0; i--)
        {
                size_t j = (size_t) rand() % (i + 1);

                dataset_swap_image_samples(dataset->images,
                                           dataset->pixels_per_image,
                                           i,
                                           j);
                dataset_swap_labels(dataset->labels, i, j);
        }
}

void dataset_free(Dataset *dataset)
{
        if (!dataset) return;

        free(dataset->images);
        free(dataset->labels);

        *dataset = (Dataset){0};
}

void dataset_load_mnist(Dataset *dataset, const char *img_path, const char *label_path)
{
        size_t n_images;
        size_t n_labels;

        if (!dataset || !img_path || !label_path) return;

        if (dataset->images || dataset->labels)
                dataset_free(dataset);

        dataset_load_mnist_images(img_path,
                                  &dataset->images,
                                  &n_images,
                                  &dataset->pixels_per_image);

        dataset_load_mnist_labels(label_path,
                                  &dataset->labels,
                                  &n_labels);

        if (n_images != n_labels)
        {
                printf("Image/label count mismatch.\n");
                dataset_free(dataset);
                exit(1);
        }

        dataset->n_samples = n_images;
}
