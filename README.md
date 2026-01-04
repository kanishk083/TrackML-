# TrackML Particle Tracking with Graph Neural Networks (GNN) : 

This project implements a Graph Neural Network (GNN) using **PyTorch Geometric** and **PyTorch Lightning** to solve the [TrackML Particle Identification Challenge](https://www.kaggle.com/c/trackml-particle-identification). The goal is to reconstruct particle tracks by classifying edges between hits in a 3D detector space.

##  Current Status: Model Evaluation
The model has completed training for 5 epochs. Below is the performance summary from the latest validation run:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **54.98%** | Percentage of correct edge classifications. |
| **Recall** | **59.73%** | Percentage of real tracks successfully found. |
| **Precision** | **2.69%** | Accuracy of the model when it predicts a "real" track. |
| **F1-Score** | **5.16%** | Harmonic mean of Precision and Recall. |

> **NOTE :** The `val_loss` reported as `nan`. This indicates numerical instability during training which must be addressed to improve results.

---

## Model Architecture: `TrackTransformer`
A memory-efficient GNN designed to fit within 15GB of VRAM:
* **GNN Layers:** 3x `TransformerConv` layers with multi-head attention (2 heads).
* **Memory Management:** Utilized `16-mixed` precision and gradient accumulation (`accumulate_grad_batches=2`).
* **Graph Construction:** KNN-based graph building ($k=8$) to limit the combinatorial explosion of edges.

---

##  Improving the Model
To move beyond the current **5% F1-score** and fix the **`nan` loss**, the following strategies should be implemented:

### 1. Fix Numerical Instability (`nan` loss)
The `nan` loss occurs when gradients "explode" or values become too large for the floating-point range.
* **Reduce Learning Rate:** Lower the current `1e-3` to `1e-4` or `5e-5` to allow for more stable convergence.
* **Stricter Gradient Clipping:** Change `gradient_clip_val` from `1.0` to `0.5` in the Trainer.
* **Layer Normalization:** Add `torch.nn.LayerNorm` between the `TransformerConv` layers to keep feature scales consistent.

### 2. Solve Class Imbalance (Boost Precision)
In TrackML, "Fake" edges outnumber "Real" edges by roughly 1000:1. The model currently over-predicts tracks, resulting in low Precision.
* **Weighted Loss:** Use `BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))`. This forces the model to treat missing a real track as a much larger error than misclassifying a fake one.
* **Hard Negative Mining:** Train only on a subset of the most "
