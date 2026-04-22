# Self-Pruning CNN for CIFAR-10

A PyTorch implementation of a self-pruning convolutional neural network for CIFAR-10 using learnable sigmoid gates, STE-based hard masking, and L1 sparsity regularization to optimize the accuracy–compression tradeoff during training.

## Overview

This project implements the Tredence AI Engineering internship case study: a CNN that learns to prune its own weights while training, instead of relying on a post-training pruning step.

The model uses a custom `PrunableLinear` layer where each weight has an associated learnable gate score. During training, gates are converted with sigmoid, hard-thresholded with a Straight-Through Estimator (STE), and optimized with a sparsity penalty so that redundant connections are driven toward zero.

## Key Features

- Custom `PrunableLinear` layer with learnable `gate_scores`.
- Sigmoid gates with STE-based hard masking.
- L1 sparsity regularization on gate values.
- Separate learning rates for weights and gates.
- CIFAR-10 training with data augmentation and normalization.
- Evaluation across three lambda values to study sparsity vs accuracy.
- Plots for training curves, gate distributions, and trade-off analysis.

## Results

| Lambda | Test Accuracy | Sparsity |
|---|---:|---:|
| 0.001 | 89.22% | 69.91% |
| 0.01 | 88.96% | 76.53% |
| 0.05 | 89.23% | 84.31% |

The best result came from `lambda = 0.05`, which achieved the highest sparsity while keeping accuracy nearly unchanged.

## Why This Works

The sigmoid function maps gate scores into the range `(0, 1)`. Applying an L1 penalty to these gates encourages them to shrink toward zero, which makes more weights fall below the pruning threshold. The STE allows the model to use a hard binary mask in the forward pass while still receiving useful gradients during backpropagation.

## Repository Structure

```text
.
├── output_cnn/
│   ├── results_summary.json
│   ├── training_curves.png
│   ├── gate_distributions.png
│   └── lambda_tradeoff.png
├── self_pruning_cnn.py
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- NumPy
- Matplotlib

## Installation

```bash
git clone <your-repo-url>
cd self-pruning-cnn-cifar10
pip install torch torchvision numpy matplotlib
```

## Run the Project

```bash
python self_pruning_cnn.py
```

The script will download CIFAR-10 automatically, train the model for each lambda value, and save the results and plots in `output_cnn/`.

## Outputs

- Final accuracy and sparsity summary
- Training curves over 30 epochs
- Gate value histograms for the best model
- Sparsity vs accuracy trade-off bar chart

## Assignment Alignment

This project satisfies the case study requirements by providing:

- A custom `PrunableLinear` implementation.
- A self-pruning neural network trained on CIFAR-10.
- A sparsity regularization objective.
- Comparison across three lambda values.
- A report-ready visualization of the gate distributions and trade-off.


