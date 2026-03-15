#include "./headers/config.h"
#include "./headers/dataset.h"
#include "./headers/network.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
        Network net = {0};
        Dataset dataset = {0};
        f32 batch_img[MINI_BATCH_SIZE][MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE];
        f32 img_test[MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE];
        u8 batch_label[MINI_BATCH_SIZE];
        f32 learning_rate = LEARNING_RATE;
        clock_t start, end;
        double cpu_time_used;
        srand(time(NULL));

        dataset_load_mnist(&dataset);

        if (dataset.pixels_per_image != MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE)
        {
                printf("Unexpected image dimensions.\n");
                dataset_free(&dataset);
                return 1;
        }

        dataset_shuffle(&dataset, dataset.n_samples);
        network_init(&net, dataset.pixels_per_image);

        size_t train_samples = (size_t) (dataset.n_samples * TRAIN_SPLIT);
        size_t test_samples = dataset.n_samples - train_samples;
        
        for (i32 epoch = 0; epoch < EPOCHS; epoch++)
        {
                start = clock();
                dataset_shuffle(&dataset, train_samples);
                for (size_t i = 0; i < train_samples; i += MINI_BATCH_SIZE)
                {
                        size_t current_batch_size = train_samples - i;
                        if (current_batch_size > MINI_BATCH_SIZE)
                                current_batch_size = MINI_BATCH_SIZE;

                        for (size_t j = 0; j < current_batch_size; j++)
                                for (size_t k = 0; k < dataset.pixels_per_image; k++)
                                        batch_img[j][k] = dataset.images[(i + j) * dataset.pixels_per_image + k] / 255.0f;

                        for (size_t j = 0; j < current_batch_size; j++)
                                batch_label[j] = dataset.labels[i + j];

                        /* Pass the batch as a flat contiguous float buffer starting at the first pixel. */
                        network_train(&net, &batch_img[0][0], batch_label, current_batch_size, learning_rate);
                }
                
                i32 correct_predicted = 0;
                for (size_t i = train_samples; i < dataset.n_samples; i++)
                {
                        for (size_t j = 0; j < dataset.pixels_per_image; j++)
                                img_test[j] = dataset.images[i * dataset.pixels_per_image + j] / 255.0f;

                        if (network_predict(&net, img_test, dataset.labels[i]))
                        {
                                correct_predicted++;
                        }
                }
                end = clock();
                cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
                
                printf("Epoch %02d: %d / %d (%.2f%%) - Time: %.2f seconds\n", 
                    epoch + 1, correct_predicted, (i32) test_samples, (f32) correct_predicted / test_samples * 100, cpu_time_used);
        }

        network_free(&net);
        dataset_free(&dataset);

        return 0;
}
