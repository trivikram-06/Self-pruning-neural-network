# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune itself during training using learnable gate parameters.

## Key Idea
Each weight is multiplied by a sigmoid gate (0–1).
If the gate becomes 0, the weight is effectively pruned.

## Loss Function
Total Loss = CrossEntropy + λ * Sparsity Loss

Sparsity Loss = sum of all gate values (L1)

## Results
- Test Accuracy: ~91%
- Sparsity: ~0.01%

## Observations
Observed low sparsity due to small lambda value.
Increasing lambda would improve pruning but may reduce accuracy,
demonstrating the sparsity–accuracy trade-off.

## Simplifications
- Used MNIST instead of CIFAR-10 for faster experimentation
- Trained for limited epochs due to time constraints

## Future Improvements
- Train on CIFAR-10
- Compare multiple lambda values
- Visualize gate distribution
