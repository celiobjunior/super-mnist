#ifndef CONFIG_H
#define CONFIG_H

#define MNIST_IMAGE_SIDE 28
#define HIDDEN_LAYER_SIZE 30
#define OUTPUT_LAYER_SIZE 10
#define MINI_BATCH_SIZE 64
#define EPOCHS 30
#define LEARNING_RATE 3.0f

#define TRAIN_IMG_PATH "data/train-images-idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels-idx1-ubyte"
#define TEST_IMG_PATH "data/t10k-images-idx3-ubyte"
#define TEST_LBL_PATH "data/t10k-labels-idx1-ubyte"

#endif
