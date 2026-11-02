# CABLE: Efficient Local Interaction for Edge Ultrasound

**Preliminary implementation — research code, not a stable release.**

CABLE is a small residual feature-interaction operator for compact ultrasound neural networks. It is motivated by a practical design question: **can inexpensive local feature mixing stand in for expensive high-resolution aggregation while preserving useful spatial evidence?**

This repository provides the **core CABLE local operator** and a minimal example for integrating it into PyTorch models. It intentionally does **not** package the complete training pipelines, study datasets, pretrained checkpoints, or experimental results. The repository will be expanded and documented for a future stable release.

## Motivation

Ultrasound models often operate on fine spatial details, including weak boundaries and subtle tissue patterns. At high feature-map resolutions, all-to-all token interaction can be expensive. Lightweight channel or spatial attention primarily rescales existing features, whereas local convolutions form new combinations of nearby responses.

CABLE takes the second approach. Rather than using full-resolution self-attention, it combines a depthwise spatial convolution with a grouped intermediate channel projection and a dense output projection. An outer residual connection preserves the input feature map.

## What is provided

```text
cable/
  __init__.py             Public import for the operator
  operator.py             CABLE2D implementation
examples/
  minimal_usage.py        Small shape/parameter demonstration
notebooks/
  cable_operator_demo.ipynb  Minimal CPU notebook; no data downloads
tests/
  test_operator.py        Basic shape, gradient and residual checks
requirements.txt
README.md
.gitignore
```

Only the **local CABLE operator** is exposed as the method here. Some development experiments explored other interaction variants; those are not part of this preliminary interface.

## Operator definition

For an input tensor `X` with shape `(batch, channels, height, width)`,

```text
D = DepthwiseConv3x3(X)
H = GroupedPointwiseConv1x1(D)
M = DensePointwiseConv1x1(H)
Y = X + M
```

The reference configuration uses **32 channels** and **four groups**. The depthwise and pointwise convolutions use no biases; there is **no activation or normalization inserted between the two pointwise projections**. For this reference configuration the operator contains **1,568 trainable weights** (operator only, excluding the host network).

The two pointwise projections are linear and can be algebraically combined into a single equivalent pointwise mapping at inference. Here they remain separate to reproduce the studied parameterization.

## Installation

Python 3.10+ and PyTorch are required. In a fresh environment, install a PyTorch build appropriate for your CPU or CUDA setup, then run:

```bash
pip install -r requirements.txt
```

No datasets, GPUs, Hugging Face access or model weights are needed for the included demonstration.

## Quick start

```python
import torch
from cable import CABLE2D

block = CABLE2D(channels=32, groups=4)
x = torch.randn(2, 32, 64, 64)
y = block(x)
assert y.shape == x.shape
```

From the repository root:

```bash
python examples/minimal_usage.py
python -m unittest discover -s tests
```

Alternatively, open `notebooks/cable_operator_demo.ipynb` in Jupyter or Google Colab using a CPU runtime.

## Using CABLE in a host network

Place `CABLE2D` after an existing feature-extraction stage with the matching channel count. It preserves spatial dimensions and channels when used with the default odd kernel and stride 1. The exact insertion point is an experimental choice and is **not automatically optimized** by this repository.

This minimal release does not implement the complete MiT-B0/SwiftMSeg segmentation host, MobileNetV2 POCUS study, patient-disjoint folds, training scripts, or evaluation protocol. Reproducing the paper's full experiments therefore requires additional code and the respective datasets; those components are planned for a later release.

## Reproducibility and scope

- This is an **operator-level implementation**, not a full-paper reproduction package.
- No training data, patient information, test predictions, or checkpoints are included.
- The example runs with random tensors and is **not a performance benchmark**.
- The code is intended for research and prototyping, **not for clinical use**.
- Numerical findings are deliberately omitted from this preliminary README; please refer to the accompanying paper when available.

## Roadmap

The intended stable release may add a pinned environment, complete model integrations, dataset preparation instructions, experiment manifests, training and evaluation scripts, and reproducibility checks. These are plans, not features of this version.

## Citation

A citation entry will be added when the corresponding paper metadata are finalized. Until then, please reference the repository by its name and URL.

## License

**No license has been assigned in this preliminary package.** Review and add a license before public release if you intend to grant reuse or redistribution permissions. Do not assume that absence of a license grants permission for reuse.
