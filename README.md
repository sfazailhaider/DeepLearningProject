# Deepfake Detection via Generative Adversarial Networks

A foundational GAN-based framework for deepfake image detection, benchmarked across three progressively complex photorealistic datasets. Built as part of coursework at Rice University (COMP 576 — Introduction to Deep Learning).

---

## Overview

This project investigates whether a GAN's discriminator — the component trained to distinguish real from fake — can serve as an effective deepfake detector. We engineer GAN architectures from scratch, train them on three datasets of increasing visual complexity, and analyze how regularization techniques affect convergence, stability, and detection quality.

**Key result:** Achieved rapid visual convergence across all three datasets in just 20 epochs. Resolved discriminator dominance and mode collapse via label smoothing and dropout, reducing training time by 50%.

---

## Datasets

| Dataset | Description | Complexity |
|---|---|---|
| **MNIST** | Handwritten digits | Low |
| **Anime Faces** | Illustrated anime portraits | Medium |
| **CelebFaces** | Real human celebrity photos | High |

---

## Architecture

Each notebook implements a GAN with:
- **Generator** — upsamples latent noise vectors into synthetic images
- **Discriminator** — classifies images as real or generated (repurposed as deepfake detector)
- **Loss** — Binary cross-entropy with label smoothing to prevent discriminator dominance
- **Regularization** — Dropout layers to reduce overfitting and mode collapse

### Key Design Decisions

- **Label smoothing** on discriminator targets (real → 0.9 instead of 1.0) — critical for training stability
- **Dropout** in the discriminator — prevents it from dominating too early, allowing the generator to learn
- **Latent dimensionality tuning** — reducing latent space size improved convergence speed by 50%

---

## Results

- ✅ Stable convergence on all 3 datasets within 20 epochs
- ✅ No mode collapse after regularization fixes
- ✅ Generated images visually indistinguishable from real samples by epoch 15–20
- ✅ Discriminator generalizes well as a real/fake binary classifier

---

## Project Structure

- ├── MNIST_Project.ipynb         # GAN on MNIST digits
- ├── Anime_Project.ipynb         # GAN on anime face dataset
- ├── CelebFaces_Project.ipynb    # GAN on CelebFaces (most complex)
- ├── CelebFaces Images/          # Sample dataset images
- └── Project Report.pdf          # Full written report with analysis



---

## Tech Stack

`Python` · `TensorFlow` · `NumPy` · `Matplotlib` · `Jupyter Notebook`

---

## How to Run

1. Clone the repo
```bash
git clone https://github.com/sfazailhaider/Deepfake-Detection.git
