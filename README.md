# AI Seminar [S06]: Diffusion Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiVan222/AI_Seminar_Diffusion/blob/main/experiments/task5_6_ldm_4_experiments.ipynb)

---

## Google Colab Notebooks & Interactive Simulator

1. **Interactive Diffusion & Latent Space Simulator (Streamlit on Colab T4):**
   * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiVan222/AI_Seminar_Diffusion/blob/main/experiments/diffusion_simulator_colab.ipynb)
   * Uses pretrained **Stable Diffusion v1.5 FP16** (`stable-diffusion-v1-5/stable-diffusion-v1-5`) and **Pixel DDPM CelebA-HQ 256** (`google/ddpm-celebahq-256`) to interactively simulate:
     * **Unconditioned ($\varnothing$) vs. Conditioned LDM (CFG)** from identical initial Gaussian noise $\mathbf{z}_T$.
     * **Step-by-Step Latent Space Inspector:** 4 latent channels `z[0..3]`, linear RGB latent projection, VAE decoded $\mathcal{D}(\mathbf{z}_t)$ vs. predicted $\mathcal{D}(\hat{\mathbf{z}}_0)$, 2D PCA trajectories, and histograms vs. $\mathcal{N}(0, 1)$.
     * **Forward vs. Backward DDPM** (Closed-form jump vs. step-by-step Markov chain) in both latent and pixel space, with one-click PNG filmstrip and animated GIF export.
   * Run locally:
     ```bash
     streamlit run experiments/diffusion_simulator/app.py
     ```

2. **Tasks 5 & 6 (Latent Diffusion Models Suite):**
   * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiVan222/AI_Seminar_Diffusion/blob/main/experiments/task5_6_ldm_4_experiments.ipynb)
   * Covers:
     * **Exp 1:** VAE Compression Fidelity Test ($32\times 32 \to 8\times 8$ spatial latent).
     * **Exp 2:** Latent vs. Pixel Diffusion Training Efficiency Benchmark.
     * **Exp 3:** Latent Space Interpolation & Manifold Continuity Walk.
     * **Exp 4:** Classifier-Free Guidance (CFG) Scaling Test with Multi-Head Cross-Attention.

3. **Task 1 & Task 4 (Paradigms Comparison & Trajectories):**
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
