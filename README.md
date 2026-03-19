# superMNIST-c (WIP)

`superMNIST-c` is a neural network project written in C to classify handwritten digits from the MNIST dataset.

The name is inspired by [`mini-mnist`](https://github.com/konrad-gajdus/miniMNIST-c): while [`mini-mnist`](https://github.com/konrad-gajdus/miniMNIST-c) focuses on a minimal single-file implementation, `superMNIST-c` explores the same problem with a cleaner, more modular and robust architecture.

## Project structure

- `main.c`
  - Application entry point, training loop orchestration, and evaluation on the official MNIST test set.
- `headers/config.h`
  - Project-wide constants such as dataset paths, layer sizes, epochs, learning rate, and MNIST file locations.
- `headers/dataset.h` and `src/dataset.c`
  - MNIST IDX loading, dataset memory lifecycle management, and training-set shuffling.
- `headers/network.h` and `src/network.c`
  - Neural network data structures, parameter initialization, mini-batch training, prediction, and cleanup.
- `headers/types.h`
  - Fixed-width aliases and project numeric types.

## Build

Example build on macOS/Linux:

```sh
clang -std=c11 -Wall -Wextra -pedantic main.c ./src/*.c -lm -o app
```

## Run

```sh
./app
```

## MNIST Documentation

The dataset used in this project is the [MNIST Database of handwritten digits](https://github.com/sunsided/mnist).