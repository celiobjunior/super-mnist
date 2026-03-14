#include "./headers/config.h"
#include "./headers/dataset.h"
#include "./headers/network.h"

#include <stddef.h>
#include <stdio.h>

int main(void)
{
        Network net = {0};
        Dataset dataset = {0};
        size_t pixels_per_image = MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE;
        f32 img[pixels_per_image], learning_rate = LEARNING_RATE;

        network_init(&net, pixels_per_image);

        dataset_load_mnist(&dataset);

        if (dataset.pixels_per_image != pixels_per_image)
        {
                printf("Unexpected image dimensions.\n");
                network_free(&net);
                dataset_free(&dataset);
                return 1;
        }

        for (i32 epoch = 0; epoch < EPOCHS; epoch++)
        {
                printf("EPOCH #%d\n", epoch + 1);
                for (size_t i = 0; i < dataset.n_samples; i++)
                {
                        for (size_t j = 0; j < dataset.pixels_per_image; j++)
                                img[j] = dataset.images[i * dataset.pixels_per_image + j] / 255.0f;

                        network_train(&net, img, dataset.labels[i], learning_rate);
                }
        }

        network_free(&net);
        dataset_free(&dataset);

        return 0;
}
