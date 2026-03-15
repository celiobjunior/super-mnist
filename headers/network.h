#ifndef NETWORK_H
#define NETWORK_H

#include "types.h"
#include <stddef.h>

/**
 * @brief Fixed-size fully connected layer.
 *
 * The buffers referenced by `biases` and `weights` are owned by the network
 * module after successful initialization and must be released with
 * `network_free()`.
 */
typedef struct Layer {
        f32 *biases, *weights;
        size_t input_count, output_count;
} Layer;

/**
 * @brief Fixed neural network used by this project.
 *
 * This project uses a fixed architecture:
 * - input size is provided to `network_init()`
 * - hidden layer size is `HIDDEN_LAYER_SIZE`
 * - output layer size is `OUTPUT_LAYER_SIZE`
 *
 * Lifecycle contract:
 * 1. Zero-initialize the object before first use, for example:
 *    `Network net = {0};`
 * 2. Call `network_init(&net, input_size)` before training.
 * 3. Call `network_train(...)` only while the network is initialized.
 * 4. Call `network_free(&net)` when finished.
 *
 * Safe reinitialization:
 * - calling `network_init()` on an already initialized network is supported
 * - any previously owned layer buffers are released before new ones are
 *   allocated
 * - after `network_free()`, the network may be initialized again
 */
typedef struct Network {
        Layer hidden, output;
} Network;

/**
 * @brief Allocates and initializes the fixed network layers.
 *
 * If `net` already owns layer buffers, they are released before the new
 * initialization is performed.
 *
 * @param net Network instance to initialize. It should be zero-initialized
 *            before the first call.
 * @param input_size Number of input features expected by the hidden layer.
 */
void network_init(Network *net, size_t input_size);

/**
 * @brief Performs one training step for a single labeled sample.
 *
 * @param net Initialized network instance.
 * @param input Normalized input sample with `input_size` elements.
 * @param label Expected class label for the sample.
 * @param learning_rate Gradient descent learning rate.
 */
void network_train(Network *net, const f32 *input, u8 label, f32 learning_rate);

/**
 * @brief Releases all heap memory owned by the network.
 *
 * Safe to call with a non-NULL pointer whose layers were previously
 * initialized. After this call, the network should not be trained again
 * unless it is reinitialized.
 *
 * @param net Network instance to release.
 */
void network_free(Network *net);

/**
 * @brief Makes a prediction for a given input sample.
 *
 * @param net Initialized network instance.
 * @param input Normalized input sample with `input_size` elements.
 * @return Predicted class label.
 */
b32 network_predict(Network *net, const f32 *input, u8 correct_label);

#endif
