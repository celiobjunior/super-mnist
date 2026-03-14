#include "../headers/network.h"
#include "../headers/config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static f32 sigmoid(f32 z)
{
        return 1.0f / (1.0f + expf(-z));
}

static void feed_forward(const Layer *layer, const f32 *input, f32 *output)
{
        // Initializing bias
        for (size_t i = 0; i < layer->output_count; i++)
                output[i] = layer->biases[i];

        // Matrix multiplication
        for (size_t i = 0; i < layer->input_count; i++)
                for (size_t j = 0; j < layer->output_count; j++)
                        output[j] += input[i] * layer->weights[i * layer->output_count + j];

        // Activation function - Sigmoid
        for (size_t i = 0; i < layer->output_count; i++)
                output[i] = sigmoid(output[i]);
}

static void backprop(Network *net,
                     const f32 *input,
                     const f32 *hidden_output,
                     const f32 *final_output,
                     u8 label,
                     f32 learning_rate)
{
        f32 error_output[OUTPUT_LAYER_SIZE] = {0}, error_hidden[HIDDEN_LAYER_SIZE] = {0};

        /*
        * error2 = (a2 - y) hProduct d_sig(z2)
        *
        * v1 = (a2 - y)
        * v2 = d_sig(z2) = a2 * (1.0f - a2)
        *
        * v3 = v1 hProduct v2
        */
        for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        {
                // v1
                error_output[i] = final_output[i] - ((size_t) label == i);
                // v3
                error_output[i] *= final_output[i] * (1.0f - final_output[i]);
        }

        /*
        * error1 = (w2)T * error2 hProduct d_sig(z1)
        *
        * v1 = (w2)T * error2
        * v2 = d_sig(z1) = (a1 * (1.0f - a1))
        *
        * v3 = v1 hProduct v2
        */
        for(size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
        {
                for(size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                {
                        // v1
                        error_hidden[i] += error_output[j] * net->output.weights[i * OUTPUT_LAYER_SIZE + j];
                }
                // v3 = v1 hProduct v2
                error_hidden[i] *= hidden_output[i] * (1.0f - hidden_output[i]);
        }
        
        // Updating weights for output layer
        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
                for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                        net->output.weights[i * OUTPUT_LAYER_SIZE + j] -= learning_rate * error_output[j] * hidden_output[i];
        
        // Updating biases for output layer
        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                net->output.biases[j] -= learning_rate * error_output[j];
        
        // Updating weights for hidden layer
        for (size_t i = 0; i < net->hidden.input_count; i++)
                for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                        net->hidden.weights[i * HIDDEN_LAYER_SIZE + j] -= learning_rate * error_hidden[j] * input[i];

        // Updating biases for hidden layer
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                net->hidden.biases[j] -= learning_rate * error_hidden[j];
}

static void layer_init(Layer *layer, size_t input_count, size_t output_count)
{
        size_t weight_count;
        f32 scale;

        if (!layer) return;

        printf("Loading layer...\n");

        weight_count = input_count * output_count;
        scale = sqrtf(2.0f / (f32) input_count);

        layer->input_count = input_count;
        layer->output_count = output_count;
        layer->weights = (f32 *) malloc(weight_count * sizeof(f32));
        layer->biases = (f32 *) calloc(output_count, sizeof(f32));

        for (size_t i = 0; i < weight_count; i++)
                layer->weights[i] = ((f32) rand() / RAND_MAX - 0.5f) * 2.0f * scale;

        printf("Layer loaded!\n\n");
}

static void layer_free(Layer *layer)
{
        if (!layer) return;

        free(layer->weights);
        free(layer->biases);

        layer->weights = NULL;
        layer->biases = NULL;
        layer->input_count = 0;
        layer->output_count = 0;
}

void network_train(Network *net, const f32 *input, u8 label, f32 learning_rate)
{
        f32 hidden_output[HIDDEN_LAYER_SIZE], final_output[OUTPUT_LAYER_SIZE];

        if (!net || !input) return;

        // a1 = sig(w1 * a0 + b1)
        feed_forward(&net->hidden, input, hidden_output);
        // a2 = sig(w2 * a1 + b2)
        feed_forward(&net->output, hidden_output, final_output);
        
        backprop(net, input, hidden_output, final_output, label, learning_rate);
}

void network_init(Network *net, size_t input_size)
{
        if (!net) return;

        layer_init(&net->hidden, input_size, HIDDEN_LAYER_SIZE);
        layer_init(&net->output, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
}

void network_free(Network *net)
{
        if (!net) return;

        layer_free(&net->hidden);
        layer_free(&net->output);
}
