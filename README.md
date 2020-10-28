### Minimal Multimodel Training

A simple reference implementation to train multiple models on one or more GPUs in parallel with PyTorch.

#### What's inside:
An minimal example of training multiple networks on MNIST in parallel with a multiprocessing Queue structure:
- `single_gpu.py` parallel training of multiple models with a single GPU
- `multi_gpu.py` as above but with multiple GPUs
- `main.py` a simple implementation of [population based training](https://deepmind.com/blog/population-based-training-neural-networks/), from [here](https://github.com/voiler/PopulationBasedTraining).

### Requirements
- PyTorch >= 1.0.0
