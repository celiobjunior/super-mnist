# superMNIST-c

`superMNIST-c` is a neural network written in C for classifying handwritten digits from the [MNIST dataset](https://github.com/sunsided/mnist). Inspired by [`miniMNIST-c`](https://github.com/konrad-gajdus/miniMNIST-c), which focuses on a minimal single-file approach, `superMNIST-c` explores the same problem with a cleaner, more **modular and robust architecture**.

## Features

- Two-layer neural network (input → hidden → output)
- Sigmoid activation function
- Mini-batch Gradient Descent optimizer
- Pre-allocated gradient buffers reused across batches (no per-epoch heap allocation)
- Modular architecture split across multiple files

## Project Structure

- `main.c` — Entry point, training loop, and evaluation on the MNIST test set
- `headers/config.h` — Project-wide constants (layer sizes, learning rate, epochs, batch size, dataset paths)
- `headers/dataset.h` / `src/dataset.c` — MNIST IDX loading, memory lifecycle, and dataset shuffling
- `headers/network.h` / `src/network.c` — Network data structures, initialization, mini-batch training, prediction, and cleanup
- `headers/types.h` — Fixed-width type aliases

## Performance

```
Epoch 01: 9181 / 10000 (91.81%) - Time: 3.99 seconds
Epoch 02: 9326 / 10000 (93.26%) - Time: 3.95 seconds
Epoch 03: 9401 / 10000 (94.01%) - Time: 4.00 seconds
Epoch 04: 9413 / 10000 (94.13%) - Time: 3.99 seconds
Epoch 05: 9464 / 10000 (94.64%) - Time: 4.00 seconds
Epoch 06: 9483 / 10000 (94.83%) - Time: 4.00 seconds
...
Epoch 25: 9599 / 10000 (95.99%) - Time: 4.10 seconds
Epoch 26: 9621 / 10000 (96.21%) - Time: 4.09 seconds
Epoch 27: 9611 / 10000 (96.11%) - Time: 4.07 seconds
Epoch 28: 9604 / 10000 (96.04%) - Time: 4.08 seconds
Epoch 29: 9623 / 10000 (96.23%) - Time: 4.07 seconds
Epoch 30: 9613 / 10000 (96.13%) - Time: 4.11 seconds
```

## Prerequisites

- GCC or Clang compiler
- MNIST dataset files in the `data/` directory:
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`
  - `t10k-images-idx3-ubyte`
  - `t10k-labels-idx1-ubyte`

## Compilation

```bash
make build
```

To force Clang explicitly:

```bash
make build CC=clang
```

## Usage

1. Place the MNIST dataset files in the `data/` directory.
2. Compile the program.
3. Run the executable:

   ```bash
   ./app
   ```

## Configuration

You can adjust the following parameters in `headers/config.h`:

- `HIDDEN_LAYER_SIZE`: Number of neurons in the hidden layer
- `LEARNING_RATE`: Learning rate for gradient descent
- `EPOCHS`: Number of training epochs
- `MINI_BATCH_SIZE`: Number of samples per mini-batch

## License

This project is open-source and available under the [MIT License](LICENSE).
