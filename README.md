# Self-Pruning Neural Network

## Overview
This project implements a self-pruning neural network where each weight learns whether it is important or not during training.

Instead of pruning after training, the network uses **learnable gates** to dynamically remove unnecessary weights.

---

## Key Idea

Each weight is multiplied by a gate:

weight × sigmoid(gate_score)

- Gate values are between 0 and 1
- If gate → 0 → weight is effectively removed (pruned)
- If gate → 1 → weight is kept

---

## Loss Function

Total Loss = Classification Loss + λ × Sparsity Loss

- Classification Loss → CrossEntropy
- Sparsity Loss → Sum of all gate values (L1 regularization)

This encourages the network to:
- Reduce unnecessary connections
- Keep only important weights

---

## Implementation Details

- Custom `PrunableLinear` layer
- Gate mechanism using sigmoid
- L1-based sparsity regularization
- Training and evaluation pipeline

---

## Experiments

### MNIST Dataset
- Accuracy: ~91–92%
- Fast convergence
- Used to validate correctness of pruning mechanism

### CIFAR-10 Dataset
- Accuracy: ~45%
- More complex dataset
- Demonstrates scalability of approach

---

## Observations

- MNIST gives higher accuracy due to simplicity
- CIFAR-10 is harder and requires more complex models (e.g., CNN)

- Sparsity remained low because:
  - Training was limited to 1 epoch
  - Lambda value was moderate

- Increasing lambda would:
  - Increase sparsity (more pruning)
  - Reduce accuracy (trade-off)

---

## Key Insight

This project demonstrates the **sparsity–accuracy trade-off**:
- More pruning → simpler model but lower accuracy
- Less pruning → better accuracy but larger model

---

## Simplifications

- Used MNIST for quick validation
- Limited training epochs due to time constraints
- Used simple fully connected model instead of CNN

---

## Future Improvements

- Train for more epochs
- Experiment with different lambda values
- Use CNN for better performance on CIFAR-10
- Visualize gate distribution
- Measure model compression

---

## How to Run

```bash
pip install -r requirements.txt
python mnist_model.py
python cifar_model.py
