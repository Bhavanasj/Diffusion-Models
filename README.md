# Diffusion Models from Scratch - MNIST

Implementation of a denoising diffusion probabilistic model (DDPM) on MNIST, built for CSC 8851: Deep Learning (Fall 2025). Covers the full pipeline from forward noising through conditional sampling with classifier-free guidance.

---

## What's in this project

**Part 1: Linear Noise Schedule and Forward Noising**
Implemented a linear beta schedule across 1000 timesteps and visualized the forward diffusion process on a single MNIST digit. Watching a clean image dissolve into pure noise step by step is what made the mechanics click.

**Part 2: Model and Schedulers**
Set up a UNet2DModel from HuggingFace Diffusers with 11 input channels (1 image + 10 one-hot class planes) and configured both DDPMScheduler and DDIMScheduler. Implemented `make_model_input` with classifier-free guidance dropout so the model learns both conditional and unconditional denoising during training.

**Part 3: Training**
Trained the UNet for 50 epochs on MNIST to predict x0 directly (prediction_type="sample"). Training objective is MSE between the predicted clean image and the actual clean image at a randomly sampled noise level.

**Part 4: Visualizing Predictions Across Noise Levels**
Fixed a single digit and corrupted it at 12 different timesteps. Ran the trained model on each with and without the class label. The conditional model recovers a clean digit even from pure noise at t=999. Without the label, the model starts drifting around t=600 and by t=999 it's guessing a completely different digit. That gap is what classifier-free guidance exploits.

**Part 5: DDPM vs DDIM Sampling with CFG**
Compared full DDPM sampling (1000 steps, ~64 seconds) with DDIM (200 steps, ~12 seconds). The speed-up is real. The quality gap on a model this size is also real — DDIM's larger jumps through the noise schedule compound errors when the denoiser isn't strong enough to take them confidently.

---

## Key results

| Sampler | Steps | Time | Quality |
|---|---|---|---|
| DDPM | 1000 | ~64s | Clean, all digits legible |
| DDIM | 200 | ~12s | Faster, visible quality degradation |

The quality gap isn't a DDIM flaw — it's a model capacity issue. DDIM works well when the denoiser is highly accurate (e.g., Stable Diffusion). On a small UNet trained for 50 epochs, skipping steps compounds errors instead of saving them.

---

## Stack

- PyTorch
- HuggingFace Diffusers (UNet2DModel, DDPMScheduler, DDIMScheduler)
- MNIST dataset
- Python 3.10+

---

## Setup

```bash
pip install torch torchvision diffusers
```

Then run the notebook top to bottom. All parts are self-contained and sequential.

---

## Files

```
CSC8851_F2025_HW3_Bhavaan_Jayappa.ipynb   # Full implementation
HW3.pdf                                    # Assignment spec
```

---

## Key concepts covered

- Forward diffusion process and linear noise schedules
- x0-prediction vs noise-prediction training objectives
- Class conditioning with one-hot spatial maps
- Classifier-free guidance (CFG) and dropout-based training
- DDPM vs DDIM sampling and the speed/quality trade-off
