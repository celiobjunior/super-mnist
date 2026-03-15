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
        f32 img[MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE];
        f32 learning_rate = LEARNING_RATE;
        clock_t start, end;
        double cpu_time_used;
        srand(time(NULL));

        network_init(&net, MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE);

        dataset_load_mnist(&dataset);

        if (dataset.pixels_per_image != MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE)
        {
                printf("Unexpected image dimensions.\n");
                network_free(&net);
                dataset_free(&dataset);
                return 1;
        }

        size_t train_samples = (size_t) (dataset.n_samples * TRAIN_SPLIT);
        size_t test_samples = dataset.n_samples - train_samples;
        
        for (i32 epoch = 0; epoch < EPOCHS; epoch++)
        {
                start = clock();
                for (size_t i = 0; i < train_samples; i++)
                {
                        for (size_t j = 0; j < dataset.pixels_per_image; j++)
                                img[j] = dataset.images[i * dataset.pixels_per_image + j] / 255.0f;

                        network_train(&net, img, dataset.labels[i], learning_rate);
                }
                
                i32 correct_predicted = 0;
                for (size_t i = train_samples; i < dataset.n_samples; i++)
                {
                        for (size_t j = 0; j < dataset.pixels_per_image; j++)
                                img[j] = dataset.images[i * dataset.pixels_per_image + j] / 255.0f;
        
                        if (network_predict(&net, img, dataset.labels[i]))
                        {
                                correct_predicted++;
                        }
                }
                end = clock();
                cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
                
                printf("Epoch %d, Accuracy: %.2f%%, Time: %.2f seconds\n", 
                    epoch + 1, (f32) correct_predicted / test_samples * 100, cpu_time_used);
        }

        network_free(&net);
        dataset_free(&dataset);

        return 0;
}
