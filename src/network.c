#include "../headers/network.h"
#include "../headers/config.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static f32 sigmoid(f32 z)
{
        return 1.0f / (1.0f + expf(-z));
}

static void feed_forward(const Layer *layer, const f32 *input, f32 *output)
{
        for (size_t i = 0; i < layer->output_count; i++)
                output[i] = layer->biases[i];

        for (size_t i = 0; i < layer->input_count; i++)
                for (size_t j = 0; j < layer->output_count; j++)
                        output[j] += input[i] * layer->weights[i * layer->output_count + j];

        for (size_t i = 0; i < layer->output_count; i++)
                output[i] = sigmoid(output[i]);
}

static void backprop(Network *net,
                     const f32 *input,
                     Gradient *grad_output,
                     Gradient *grad_hidden,
                     u8 label)
{
        f32 hidden_output[HIDDEN_LAYER_SIZE];
        f32 final_output[OUTPUT_LAYER_SIZE];
        f32 error_output[OUTPUT_LAYER_SIZE] = {0};
        f32 error_hidden[HIDDEN_LAYER_SIZE] = {0};

        feed_forward(&net->hidden, input, hidden_output);
        feed_forward(&net->output, hidden_output, final_output);

        for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        {
                error_output[i] = final_output[i] - ((size_t) label == i);
                error_output[i] *= final_output[i] * (1.0f - final_output[i]);
        }

        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
        {
                for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                        error_hidden[i] += error_output[j] * net->output.weights[i * OUTPUT_LAYER_SIZE + j];

                error_hidden[i] *= hidden_output[i] * (1.0f - hidden_output[i]);
        }

        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                grad_output->bias_grad[j] = error_output[j];

        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                grad_hidden->bias_grad[j] = error_hidden[j];

        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                for (size_t k = 0; k < OUTPUT_LAYER_SIZE; k++)
                        grad_output->weight_grad[j * OUTPUT_LAYER_SIZE + k] = error_output[k] * hidden_output[j];

        for (size_t j = 0; j < net->hidden.input_count; j++)
                for (size_t k = 0; k < HIDDEN_LAYER_SIZE; k++)
                        grad_hidden->weight_grad[j * HIDDEN_LAYER_SIZE + k] = error_hidden[k] * input[j];
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

static void gradient_free(Gradient *grad)
{
        if (!grad) return;

        free(grad->bias_grad);
        free(grad->weight_grad);

        grad->bias_grad = NULL;
        grad->weight_grad = NULL;
}

static void layer_init(Layer *layer, size_t input_count, size_t output_count)
{
        size_t weight_count;
        f32 scale;

        if (!layer) return;

        weight_count = input_count * output_count;
        scale = sqrtf(2.0f / (f32) input_count);

        layer->input_count = input_count;
        layer->output_count = output_count;
        layer->weights = (f32 *) malloc(weight_count * sizeof(f32));
        layer->biases = (f32 *) calloc(output_count, sizeof(f32));

        if (!layer->weights || !layer->biases)
        {
                printf("Failed to allocate layer parameters.\n");
                layer_free(layer);
                exit(1);
        }

        for (size_t i = 0; i < weight_count; i++)
                layer->weights[i] = ((f32) rand() / RAND_MAX - 0.5f) * 2.0f * scale;
}

static void gradient_init(Gradient *grad, size_t bias_count, size_t weight_count)
{
        grad->bias_grad = (f32 *) calloc(bias_count, sizeof(f32));
        grad->weight_grad = (f32 *) calloc(weight_count, sizeof(f32));

        if (!grad->bias_grad || !grad->weight_grad)
        {
                printf("Failed to allocate gradient parameters.\n");
                gradient_free(grad);
                exit(1);
        }
}

void network_train(Network *net, const f32 *input, const u8 *label, size_t batch_size, f32 learning_rate)
{
        Gradient grad_output, grad_hidden;
        Gradient sample_grad_output, sample_grad_hidden;
        f32 batch_scale;

        if (!net || !input || !label || batch_size == 0) return;

        if (!net->hidden.weights || !net->hidden.biases ||
            !net->output.weights || !net->output.biases)
                return;

        gradient_init(&grad_output, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE);
        gradient_init(&grad_hidden, HIDDEN_LAYER_SIZE, net->hidden.input_count * HIDDEN_LAYER_SIZE);
        gradient_init(&sample_grad_output, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE);
        gradient_init(&sample_grad_hidden, HIDDEN_LAYER_SIZE, net->hidden.input_count * HIDDEN_LAYER_SIZE);

        for (size_t i = 0; i < batch_size; i++)
        {
                /* `input` stores the whole mini-batch as one flat buffer, so this points to sample `i`. */
                const f32 *sample_input = input + i * net->hidden.input_count;

                backprop(net, sample_input, &sample_grad_output, &sample_grad_hidden, label[i]);

                for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                        grad_output.bias_grad[j] += sample_grad_output.bias_grad[j];

                for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                        grad_hidden.bias_grad[j] += sample_grad_hidden.bias_grad[j];

                for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                        for (size_t k = 0; k < OUTPUT_LAYER_SIZE; k++)
                                grad_output.weight_grad[j * OUTPUT_LAYER_SIZE + k] +=
                                        sample_grad_output.weight_grad[j * OUTPUT_LAYER_SIZE + k];

                for (size_t j = 0; j < net->hidden.input_count; j++)
                        for (size_t k = 0; k < HIDDEN_LAYER_SIZE; k++)
                                grad_hidden.weight_grad[j * HIDDEN_LAYER_SIZE + k] +=
                                        sample_grad_hidden.weight_grad[j * HIDDEN_LAYER_SIZE + k];
        }

        batch_scale = learning_rate / (f32) batch_size;

        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
                for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                        net->output.weights[i * OUTPUT_LAYER_SIZE + j] -=
                                batch_scale * grad_output.weight_grad[i * OUTPUT_LAYER_SIZE + j];

        for (size_t i = 0; i < net->hidden.input_count; i++)
                for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                        net->hidden.weights[i * HIDDEN_LAYER_SIZE + j] -=
                                batch_scale * grad_hidden.weight_grad[i * HIDDEN_LAYER_SIZE + j];

        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                net->output.biases[j] -= batch_scale * grad_output.bias_grad[j];

        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                net->hidden.biases[j] -= batch_scale * grad_hidden.bias_grad[j];

        gradient_free(&grad_output);
        gradient_free(&grad_hidden);
        gradient_free(&sample_grad_output);
        gradient_free(&sample_grad_hidden);
}

void network_init(Network *net, size_t input_size)
{
        if (!net) return;

        if (net->hidden.weights || net->hidden.biases ||
            net->output.weights || net->output.biases)
                network_free(net);

        layer_init(&net->hidden, input_size, HIDDEN_LAYER_SIZE);
        layer_init(&net->output, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
}

void network_free(Network *net)
{
        if (!net) return;

        layer_free(&net->hidden);
        layer_free(&net->output);
}

b32 network_predict(Network *net, const f32 *input, u8 correct_label) 
{
        f32 hidden_output[HIDDEN_LAYER_SIZE], final_output[OUTPUT_LAYER_SIZE];

        if (!net || !input) return 0;

        if (!net->hidden.weights || !net->hidden.biases ||
            !net->output.weights || !net->output.biases)
                return 0;

        feed_forward(&net->hidden, input, hidden_output);
        feed_forward(&net->output, hidden_output, final_output);

        u8 predicted_label = 0;
        f32 max_output = final_output[0];
        for (size_t i = 1; i < OUTPUT_LAYER_SIZE; i++)
        {
                if (final_output[i] > max_output)
                {
                        max_output = final_output[i];
                        predicted_label = i;
                }
        }

        return predicted_label == correct_label;
}
