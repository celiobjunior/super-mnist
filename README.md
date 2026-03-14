# superMNIST-c (WIP)

`superMNIST-c` is a neural network project written in C to classify handwritten digits from the MNIST dataset.

The name is inspired by [`mini-mnist`](https://github.com/konrad-gajdus/miniMNIST-c): while [`mini-mnist`](https://github.com/konrad-gajdus/miniMNIST-c) focuses on a minimal single-file implementation, `superMNIST-c` explores the same problem with a cleaner, more modular and robust architecture.

## Project structure

- `main.c`  
  - Application entry point and training loop orchestration.
- `headers/config.h`  
  - Project-wide constants such as dataset paths, layer sizes, epochs, and learning rate.
- `headers/dataset.h` and `src/dataset.c`  
  - MNIST dataset loading and memory lifecycle management.
- `headers/network.h` and `src/network.c`  
  - Neural network data structures, initialization, training, and cleanup.
- `headers/types.h`  
  - Fixed-width aliases and project numeric types.

## Build

```sh
clang -std=c11 -Wall -Wextra -pedantic main.c ./src/*.c -lm
```

## Run

```sh
./a.out
```
## MNIST Documentation

The dataset used in this project is the [MNIST Database of handwritten digits](https://github.com/sunsided/mnist).