# Simple Manifold Visualizer

Manifold visualization plots the prediction/error surface of some estimator (e.g., neural network) as a function of a subset of the dataset (a "manifold"). It consists of two components:

1) Manifold generation, where a manifold is a [manifold_size, *features]-shaped ndarray. Typically generation will be done by a GAN, VAE, or other latent variable model that can interpolate between datapoints in the latent space. An example generator for MNIST (simple autoencoder) is provided in this repo. 

2) Manifold-prediction visualization: accepts a function mapping data manifolds (i.e., ndarrays of shape [manifold_size, *features]) to predictions in R^n (e.g., class probabilities), and visualizes the function in two dimensions. An example of how to do this for MNIST is provided in this repo.

### Instructions

See ipython notebook.

### Some possibly related papers

- [Visualizing the Loss Landscape of Neural Nets (NIPS 2018)](https://arxiv.org/abs/1712.09913)
- [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs (NIPS 2018)](https://arxiv.org/abs/1802.10026)
- [Averaging Weights Leads to Wider Optima and Better Generalization (UAI 2018)](https://arxiv.org/abs/1803.05407)

(taken from [this](https://www.reddit.com/r/MachineLearning/comments/9u519m/d_visualizing_and_analyzing_error_landscapes/) reddit thread)