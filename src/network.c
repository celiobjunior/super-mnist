#include "../headers/network.h"
#include "../headers/config.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static f32 sigmoid(f32 z)
{
        return 1.0f / (1.0f + expf(-z));
}

/* Softmax is the mathematical tool that allows transforming rows of matrices into probabilities
 * forming a perfect pair with Cross-Entropy to simplify the derivative during backpropagation
 * enabling learning to improve more quickly in the early epochs. 
 * We subtract the 'max_eval' before computing the exponential to prevent floating-point limits from causing overflow and producing Inf/NaN results.
 */
static void softmax(f32 *output, size_t count)
{
        f32 max_val = output[0];
        for (size_t i = 1; i < count; i++) {
                if (output[i] > max_val) max_val = output[i];
        }

        f32 sum = 0.0f;
        for (size_t i = 0; i < count; i++) {
                output[i] = expf(output[i] - max_val);
                sum += output[i];
        }

        for (size_t i = 0; i < count; i++) {
                output[i] /= sum;
        }
}

static void feed_forward(const Layer *layer, const f32 *input, f32 *output)
{
        for (size_t i = 0; i < layer->output_count; i++)
                output[i] = layer->biases[i];

        for (size_t i = 0; i < layer->input_count; i++)
                for (size_t j = 0; j < layer->output_count; j++)
                        output[j] += input[i] * layer->weights[i * layer->output_count + j];
}

/* Previously, backpropagation used MSE (Mean Squared Error) with the Sigmoid activation,
 * which caused the vanishing gradient problem when the network made highly confident mistakes.
 * Now, we use Cross-Entropy with Softmax activation, causing the derivative of Cross-Entropy to perfectly cancel out the derivative of Softmax.
 * As a result, the loss gradient at the final output simplifies to:
 * Error = Prediction - Target (final_output[i] - label).
 */
static void backprop(Network *net,
                     const f32 *input,
                     NetworkGradient *grad_output,
                     NetworkGradient *grad_hidden,
                     u8 label)
{
        f32 hidden_output[HIDDEN_LAYER_SIZE];
        f32 final_output[OUTPUT_LAYER_SIZE];
        f32 error_output[OUTPUT_LAYER_SIZE] = {0};
        f32 error_hidden[HIDDEN_LAYER_SIZE] = {0};

/* The feedforward becomes much cleaner now
 * since it no longer forces the Sigmoid activation and goes back 
 * to being just a linear transformation (Z = Wx + b) 
 */
        feed_forward(&net->hidden, input, hidden_output);
        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
                hidden_output[i] = sigmoid(hidden_output[i]);
        feed_forward(&net->output, hidden_output, final_output);
        softmax(final_output, OUTPUT_LAYER_SIZE);

        for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        {
                error_output[i] = final_output[i] - ((size_t) label == i);
        }

        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
        {
                for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                        error_hidden[i] += error_output[j] * net->output.weights[i * OUTPUT_LAYER_SIZE + j];

                error_hidden[i] *= hidden_output[i] * (1.0f - hidden_output[i]);
        }

        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                grad_output->bias_grad[j] += error_output[j];

        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                grad_hidden->bias_grad[j] += error_hidden[j];

        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                for (size_t k = 0; k < OUTPUT_LAYER_SIZE; k++)
                        grad_output->weight_grad[j * OUTPUT_LAYER_SIZE + k] += error_output[k] * hidden_output[j];

        for (size_t j = 0; j < net->hidden.input_count; j++)
                for (size_t k = 0; k < HIDDEN_LAYER_SIZE; k++)
                        grad_hidden->weight_grad[j * HIDDEN_LAYER_SIZE + k] += error_hidden[k] * input[j];
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

static void gradient_free(NetworkGradient *grad)
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

/* Now we use Xavier scaling (Xavier initialization): sqrt(2 / (input_count + output_count)).
 * Xavier initialization has been mathematically shown to be the best way to keep gradient variance
 * stable when dealing with exponential-based functions, such as Sigmoid
 */
        scale = sqrtf(2.0f / (f32) (input_count + output_count));

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

static void gradient_init(NetworkGradient *grad, size_t bias_count, size_t weight_count)
{
        if (!grad) return;

        grad->bias_grad = (f32 *) calloc(bias_count, sizeof(f32));
        grad->weight_grad = (f32 *) calloc(weight_count, sizeof(f32));

        if (!grad->bias_grad || !grad->weight_grad)
        {
                printf("Failed to allocate gradient parameters.\n");
                gradient_free(grad);
                exit(1);
        }
}

static void gradient_zero(NetworkGradient *grad, size_t bias_count, size_t weight_count)
{
        memset(grad->bias_grad, 0, bias_count * sizeof(f32));
        memset(grad->weight_grad, 0, weight_count * sizeof(f32));
}

void network_train(Network *net, const f32 *input, const u8 *label, size_t batch_size, f32 learning_rate)
{
        f32 batch_scale;

        if (!net || !input || !label || batch_size == 0) return;

        if (!net->hidden.weights || !net->hidden.biases ||
            !net->output.weights || !net->output.biases)
                return;

        gradient_zero(&net->grad_output, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE);
        gradient_zero(&net->grad_hidden, HIDDEN_LAYER_SIZE, net->hidden.input_count * HIDDEN_LAYER_SIZE);

        for (size_t i = 0; i < batch_size; i++)
        {
                /* `input` stores the whole mini-batch as one flat buffer, so this points to sample `i`. */
                const f32 *sample_input = input + i * net->hidden.input_count;

                backprop(net, sample_input, &net->grad_output, &net->grad_hidden, label[i]);
        }

        batch_scale = learning_rate / (f32) batch_size;

        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
                for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                        net->output.weights[i * OUTPUT_LAYER_SIZE + j] -=
                                batch_scale * net->grad_output.weight_grad[i * OUTPUT_LAYER_SIZE + j];

        for (size_t i = 0; i < net->hidden.input_count; i++)
                for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                        net->hidden.weights[i * HIDDEN_LAYER_SIZE + j] -=
                                batch_scale * net->grad_hidden.weight_grad[i * HIDDEN_LAYER_SIZE + j];

        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
                net->output.biases[j] -= batch_scale * net->grad_output.bias_grad[j];

        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
                net->hidden.biases[j] -= batch_scale * net->grad_hidden.bias_grad[j];
}

void network_init(Network *net, size_t input_size)
{
        if (!net) return;

        if (net->hidden.weights || net->hidden.biases ||
            net->output.weights || net->output.biases)
                network_free(net);

        layer_init(&net->hidden, input_size, HIDDEN_LAYER_SIZE);
        layer_init(&net->output, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

        gradient_init(&net->grad_output, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE);
        gradient_init(&net->grad_hidden, HIDDEN_LAYER_SIZE, input_size * HIDDEN_LAYER_SIZE);
}

void network_free(Network *net)
{
        if (!net) return;

        layer_free(&net->hidden);
        layer_free(&net->output);
        gradient_free(&net->grad_output);
        gradient_free(&net->grad_hidden);
}

b32 network_predict(Network *net, const f32 *input, u8 correct_label) 
{
        f32 hidden_output[HIDDEN_LAYER_SIZE], final_output[OUTPUT_LAYER_SIZE];

        if (!net || !input) return 0;
        if (!net->hidden.weights || !net->hidden.biases ||
            !net->output.weights || !net->output.biases)
                return 0;

/* Once again, the same simplification as in the feedforward */
        feed_forward(&net->hidden, input, hidden_output);
        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
                hidden_output[i] = sigmoid(hidden_output[i]);
        feed_forward(&net->output, hidden_output, final_output);
        softmax(final_output, OUTPUT_LAYER_SIZE);

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
