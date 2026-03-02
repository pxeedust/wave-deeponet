# Wave DeepONet

A Deep Operator Network (DeepONet) for learning and predicting spatiotemporal wave field dynamics, with support for bilevel optimization training strategies.

## Repository Structure

```
wave-deeponet/
├── Models/                             # Saved PyTorch model weights
│   ├── deeponet_best.pth
│   ├── deeponet_adam+mu_ifef_0.30.pth
│   ├── deeponet_adam+mu_ifef_0.70.pth
│   ├── deeponet_l1_loss.pth
│   ├── deeponet_pure_ifef.pth
│   └── deeponet_same_scale_trunk_branch.pth
├── square_field_analysis.ipynb         # Main training and evaluation notebook
├── norm_square_field_ifef_mu.ipynb     # Bilevel optimization training
└── norm_square_field.ipynb             # Data preprocessing and baseline training
```

## Architecture

The DeepONet consists of two sub-networks whose outputs are combined via a dot product:

**Branch Network** — encodes the input wave field U(t, x) (1666 spatial points):

- Tunable Fourier Feature Mapping (128-dim, low-frequency: ω₀=0.5, σ=1.0)
- MLP: 128 → 256 → 256 → 50 modes

**Trunk Network** — encodes query coordinates (t, x):

- Tunable Fourier Feature Mapping (128-dim, high-frequency: ω₀=5.0, σ=10.0)
- MLP: 128 → 256 → 256 → 50 modes

Activation: GCU (Gated Cosine Unit): `f(x) = x · cos(x)`

Output: dot product of branch and trunk outputs

## Data

Data is stored as MATLAB/HDF5 `.mat` files (not included in this repo) with the following structure:

- `t_out`: temporal grid (8276 time steps)
- `x`: spatial grid (5120 points)
- `U_out`: complex wave field (5120 × 8276)

**Preprocessing pipeline:**

1. Compute intensity |U|² from complex field
2. Mask to region t > 200, x ∈ [150, 650]
3. Checkerboard train/test split (~5.5M samples each)
4. MinMax normalization of t, x, and |U|² to [0, 1]
5. Reshape to 2D matrices: (3293 time steps × 1666 spatial points)

## Training

### Standard Training (`square_field_analysis.ipynb`)

Adam optimizer, MSE loss, teacher forcing with autoregressive evaluation.

```
Optimizer: Adam (lr=1e-4)
Batch size: 32
Scheduler: ReduceLROnPlateau
```

### Bilevel Optimization (`norm_square_field_ifef_mu.ipynb`)

Two-stage training with weight blending:

1. **Stage 1** — Full Adam gradient descent → weights W₁
2. **Stage 2** — Lower-level least-squares solve on final layer → weights W₂ (with ridge regularization λ=10⁻⁶)
3. **Blend** — W_final = μ · W₁ + (1−μ) · W₂

Variants saved: μ ∈ {0.3, 0.7}, pure least-squares (μ=0.0), and baseline Adam.

## Evaluation

- **One-step relative L2 error**: `||pred - truth||₂ / ||truth||₂`
- **Energy conservation error**: deviation in total field energy
- **Autoregressive rollout**: error accumulation over 3,292 time steps
- **2D DFT analysis**:
  - Frequency ball error vs radius δ
  - Top-Q dominant frequency magnitude/phase correlation
  - Cumulative spectral energy curves
