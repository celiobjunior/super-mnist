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
        Dataset train_dataset = {0};
        Dataset test_dataset = {0};
        f32 batch_img[MINI_BATCH_SIZE][MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE];
        f32 img_test[MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE];
        u8 batch_label[MINI_BATCH_SIZE];
        f32 learning_rate = LEARNING_RATE;
        clock_t start, end;
        double cpu_time_used;

        srand(time(NULL));

        dataset_load_mnist(&train_dataset, TRAIN_IMG_PATH, TRAIN_LBL_PATH);
        dataset_load_mnist(&test_dataset, TEST_IMG_PATH, TEST_LBL_PATH);

        if (train_dataset.pixels_per_image != MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE ||
            test_dataset.pixels_per_image != MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE)
        {
                printf("Unexpected image dimensions.\n");
                dataset_free(&train_dataset);
                dataset_free(&test_dataset);
                return 1;
        }

        network_init(&net, train_dataset.pixels_per_image);

        for (i32 epoch = 0; epoch < EPOCHS; epoch++)
        {
                start = clock();

                dataset_shuffle(&train_dataset, train_dataset.n_samples);

                for (size_t i = 0; i < train_dataset.n_samples; i += MINI_BATCH_SIZE)
                {
                        size_t current_batch_size = train_dataset.n_samples - i;

                        if (current_batch_size > MINI_BATCH_SIZE)
                                current_batch_size = MINI_BATCH_SIZE;

                        for (size_t j = 0; j < current_batch_size; j++)
                                for (size_t k = 0; k < train_dataset.pixels_per_image; k++)
                                        batch_img[j][k] =
                                                train_dataset.images[(i + j) * train_dataset.pixels_per_image + k] / 255.0f;

                        for (size_t j = 0; j < current_batch_size; j++)
                                batch_label[j] = train_dataset.labels[i + j];

                        network_train(&net, &batch_img[0][0], batch_label, current_batch_size, learning_rate);
                }

                i32 correct_predicted = 0;

                for (size_t i = 0; i < test_dataset.n_samples; i++)
                {
                        for (size_t j = 0; j < test_dataset.pixels_per_image; j++)
                                img_test[j] = test_dataset.images[i * test_dataset.pixels_per_image + j] / 255.0f;

                        if (network_predict(&net, img_test, test_dataset.labels[i]))
                                correct_predicted++;
                }

                end = clock();
                cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

                printf("Epoch %02d: %d / %d (%.2f%%) - Time: %.2f seconds\n",
                       epoch + 1,
                       correct_predicted,
                       (i32) test_dataset.n_samples,
                       (f32) correct_predicted / test_dataset.n_samples * 100.0f,
                       cpu_time_used);
        }

        network_free(&net);
        dataset_free(&train_dataset);
        dataset_free(&test_dataset);

        return 0;
}
