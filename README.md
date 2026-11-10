# AI Seminar [S06]: Diffusion Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiVan222/AI_Seminar_Diffusion/blob/main/experiments/task5_6_ldm_4_experiments.ipynb)

---

## Google Colab Notebooks (1-Click Run on T4 GPU)

1. **Tasks 5 & 6 (Latent Diffusion Models Suite):**
   * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiVan222/AI_Seminar_Diffusion/blob/main/experiments/task5_6_ldm_4_experiments.ipynb)
   * Covers:
     * **Exp 1:** VAE Compression Fidelity Test ($32\times 32 \to 8\times 8$ spatial latent).
     * **Exp 2:** Latent vs. Pixel Diffusion Training Efficiency Benchmark.
     * **Exp 3:** Latent Space Interpolation & Manifold Continuity Walk.
     * **Exp 4:** Classifier-Free Guidance (CFG) Scaling Test with Multi-Head Cross-Attention.

2. **Task 1 & Task 4 (Paradigms Comparison & Trajectories):**
   * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiVan222/AI_Seminar_Diffusion/blob/main/experiments/task1_generative_paradigms_comparison.ipynb)
   * Covers Discriminative vs. Generative 2D toy manifold, VAE vs. PixelCNN vs. DCGAN vs. DDPM on MNIST, forward corruption, and reverse image generation.

---

## Running in Google Colab Terminal

```bash
!git clone https://github.com/ChiVan222/AI_Seminar_Diffusion.git
%cd AI_Seminar_Diffusion
!python experiments/run_ldm_4_experiments.py
```

---

## Report Artifacts

* **LaTeX Report Source:** [`report/S06_Diffusion_Models_Report.tex`](report/S06_Diffusion_Models_Report.tex)
* **Compiled PDF:** [`report/S06_Diffusion_Models_Report.pdf`](report/S06_Diffusion_Models_Report.pdf)
