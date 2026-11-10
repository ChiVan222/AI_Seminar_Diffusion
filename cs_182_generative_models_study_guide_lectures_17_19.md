# UC Berkeley CS 182: Deep Generative Models (Lectures 17, 18, & 19)
**Comprehensive Technical Reference & Discussion Synthesis**  
*Instructor: Prof. Sergey Levine | Synthesized with Diffusion & Latent Diffusion Extensions*

---

## Table of Contents
1. [Lecture 17: Autoencoders, Autoregressive Models & Latent Variable Foundations](#1-lecture-17-autoencoders-autoregressive-models--latent-variable-foundations)
   - 1.1 Unsupervised Learning & Generative Modeling Taxonomy
   - 1.2 Autoregressive Generative Models (PixelRNN, PixelCNN)
   - 1.3 Deterministic Autoencoders, Bottlenecks & Denoising Autoencoders (DAEs)
   - 1.4 Transition to Probabilistic Latent Variable Models ($p(x) = \int p(x|z)p(z)dz$)
   - 1.5 Why Direct Marginal Likelihood $\log p_\theta(x_i)$ is Intractable
   - 1.6 Deriving the Evidence Lower Bound (ELBO) via Jensen's Inequality & KL Divergence
   - 1.7 Amortized Variational Inference: Why $q_\phi(z|x)$ is an Encoder Network
2. [Lecture 18: Variational Autoencoders (VAEs) & Invertible Models (Normalizing Flows)](#2-lecture-18-variational-autoencoders-vaes--invertible-models-normalizing-flows)
   - 2.1 Optimizing the ELBO: Decoder Gradient $\nabla_\theta$ vs. Encoder Gradient $\nabla_\phi$
   - 2.2 The Reinforcement Learning Connection: Policy Gradient (REINFORCE) & The Log-Derivative Trick
   - 2.3 Why Direct Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ Breaks Backpropagation
   - 2.4 The Reparameterization Trick ($z = \mu_\phi(x_i) + \epsilon \odot \sigma_\phi(x_i)$)
   - 2.5 Where Noise is Injected: DAE vs. VAE vs. DDPM vs. LDM
   - 2.6 Understanding the Role of $z$ in Training vs. Inference (End-to-End Signal Flows)
   - 2.7 Reconciling Theory ($\max$ ELBO) vs. Code ($\min$ Loss) & Closed-Form Gaussian KL
   - 2.8 The Great Tug-of-War: Why $q_\phi(z|x_i)$ Cannot Just Equal $\mathcal{N}(0, I)$
   - 2.9 Individual Posteriors $q_\phi(z|x_i)$ vs. The Aggregated Posterior $\frac{1}{N}\sum_i q_\phi(z|x_i)$
   - 2.10 Fully Convolutional VAEs vs. Vector VAEs
   - 2.11 VAE Failure Modes: Posterior Collapse ($p_\theta(x|z) \to p(x)$) & Why Ignored Latents Produce Blurry Average Images
   - 2.12 Invertible Generative Models (Normalizing Flows) & Change of Variables
   - 2.13 Volume-Preserving Flows ($\left|\det\left(\frac{df(z)}{dz}\right)\right| = 1$): Geometry, Squeeze Mappings & RealNVP
3. [Lecture 19: Generative Adversarial Networks (GANs)](#3-lecture-19-generative-adversarial-networks-gans)
   - 3.1 Implicit Density Models: Likelihood-Free Training via a Two-Player Game
   - 3.2 The 5-Step GAN Training Algorithm
   - 3.3 Why $G(z)$ Wants $D(x) = 0.5$ Globally (Nash Equilibrium) Despite Minimizing $-\log D(G(z))$ Locally
   - 3.4 Mathematical Proof: Optimal Discriminator $D^*(x)$ & Jensen-Shannon Divergence
   - 3.5 Vanishing Gradients, Non-Saturating Generator Loss & Mode Collapse
   - 3.6 Architectural & Representation Advances (DCGAN, WGAN, CycleGAN, StyleGAN)
4. [Master Synthesis: Connecting VAEs, Normalizing Flows, GANs, DDPMs, and Latent Diffusion](#4-master-synthesis-connecting-vaes-normalizing-flows-gans-ddpms-and-latent-diffusion)

---

## 1. Lecture 17: Autoencoders, Autoregressive Models & Latent Variable Foundations

### 1.1 Unsupervised Learning & Generative Modeling Taxonomy
In supervised learning, we learn a mapping $p(y|x)$ from labeled pairs $(x_i, y_i)$. In **generative modeling**, we are given only an unlabeled dataset $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$ drawn from an unknown true underlying data distribution $p_{\text{data}}(x)$, and our goal is to learn a model $p_\theta(x)$ that approximates $p_{\text{data}}(x)$.

Generative models serve three primary purposes in CS 182:
1. **Density Estimation:** Given an input $x$, evaluate $p_\theta(x)$ to determine how likely or anomalous $x$ is.
2. **Sampling / Generation:** Draw new samples $x_{\text{new}} \sim p_\theta(x)$ (e.g., image synthesis, text generation, planning futures in RL).
3. **Unsupervised Representation Learning:** Extract compact semantic features $z$ that capture the underlying factors of variation in $x$.

---

### 1.2 Autoregressive Generative Models (PixelRNN, PixelCNN)
Before introducing latent variables, Lecture 17 reviews how to model $p_\theta(x)$ directly using the **chain rule of probability**. If an image $x$ consists of $n$ pixels $(x^{(1)}, x^{(2)}, \dots, x^{(n)})$, any joint distribution can be factored exactly as a product of 1D conditionals:
$$p_\theta(x^{(1)}, x^{(2)}, \dots, x^{(n)}) = \prod_{t=1}^n p_\theta\big(x^{(t)} \mid x^{(1)}, \dots, x^{(t-1)}\big)$$

* **Training Objective:** Exact Maximum Likelihood Estimation (MLE):
  $$\max_\theta \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^n \log p_\theta\big(x_i^{(t)} \mid x_i^{(1)}, \dots, x_i^{(t-1)}\big)$$
* **Architectures:**
  * **PixelRNN:** Uses recurrent layers (e.g., diagonal BiLSTMs) scanning pixel-by-pixel in raster order (top-to-bottom, left-to-right).
  * **PixelCNN:** Replaces slow RNNs with masked 2D convolutions (zeroing out filter weights corresponding to "future" pixels below and to the right of the center pixel) so all conditionals can be evaluated in parallel during training.
* **Pros & Cons:**
  * *Pro:* Exact log-likelihood $\log p_\theta(x)$ with no variational approximation.
  * *Con:* **Extremely slow sequential generation** ($\mathcal{O}(n)$ forward passes, one per pixel/sub-pixel), and **no single compressed latent code $z$** representing the global semantics of the image.

---

### 1.3 Deterministic Autoencoders, Bottlenecks & Denoising Autoencoders (DAEs)
To learn a low-dimensional semantic representation $z \in \mathbb{R}^D$ (where $D \ll \dim(x)$), we introduce an **Autoencoder** consisting of two networks:
1. **Encoder $f_\phi(x)$:** Compresses high-dimensional input $x$ into a bottleneck vector $z = f_\phi(x)$.
2. **Decoder $g_\theta(z)$:** Reconstructs the input $\hat{x} = g_\theta(z)$ from the bottleneck $z$.

$$\min_{\theta, \phi} \frac{1}{N} \sum_{i=1}^N \big\| x_i - g_\theta(f_\phi(x_i)) \big\|^2$$

#### Why a Naive Autoencoder Cannot Generate New Images
If $z$ has lower dimensionality than $x$, the encoder compresses training images into isolated, arbitrary points $z_i = f_\phi(x_i)$ scattered across $\mathbb{R}^D$.
* We do **not** know the distribution $p(z)$ of these encoded points.
* There are massive empty gaps between training encodings $z_i$ and $z_j$. If you sample a random vector $z$ and pass it into $g_\theta(z)$, it lands in an unvisited gap and produces garbage.

#### Regularized Autoencoders
To prevent the autoencoder from simply learning an identity mapping or brittle lookup table, Lecture 17 explores ways to restrict the bottleneck:
* **Undercomplete Bottleneck:** Force $\dim(z) \ll \dim(x)$.
* **Sparse Autoencoders:** Add an $L_1$ sparsity penalty $\lambda \|z\|_1$ or KL sparsity penalty on hidden unit activations so only a few neurons fire per image.
* **Denoising Autoencoders (DAEs):** Corrupt the clean input image $x_i$ with noise at the very start ($\tilde{x}_i \sim q(\tilde{x}_i | x_i)$, e.g., Gaussian noise $\tilde{x}_i = x_i + \epsilon$ or masking) and train the network to reconstruct the **clean** $x_i$:
  $$\min_{\theta, \phi} \frac{1}{N} \sum_{i=1}^N \mathbb{E}_{\tilde{x}_i \sim q(\tilde{x}_i | x_i)} \Big[ \big\| x_i - g_\theta(f_\phi(\tilde{x}_i)) \big\|^2 \Big]$$
  *Note:* Even a Denoising Autoencoder is primarily a **feature extractor**, not a proper generative model from scratch, because it still lacks a known prior $p(z)$ from which to sample new latent codes.

---

### 1.4 Transition to Probabilistic Latent Variable Models ($p(x) = \int p(x|z)p(z)dz$)
To build a true generative model with a compact latent space, we frame generation probabilistically:
1. **Prior Distribution $p(z)$:** Assume the unobserved latent factors $z$ (e.g., pose, lighting, object identity, stroke thickness) follow a simple, known prior distribution:
   $$p(z) = \mathcal{N}(0, I)$$
2. **Conditional Generative Model (Decoder) $p_\theta(x|z)$:** A deep neural network parameterized by $\theta$ maps any latent sample $z$ to a probability distribution over data space $x$. For continuous pixel values, we typically model $p_\theta(x|z)$ as a Gaussian whose mean (and optionally variance) is output by the decoder neural network:
   $$p_\theta(x|z) = \mathcal{N}\big(x; \, \mu_\theta(z), \, \sigma_\theta^2(z)\big)$$

Even though both $p(z)$ and $p_\theta(x|z)$ are simple Gaussians, passing $z \sim \mathcal{N}(0, I)$ through a highly non-linear neural network $\mu_\theta(z)$ induces an arbitrarily complex, multi-modal marginal distribution over $x$:
$$p_\theta(x) = \int p_\theta(x|z) p(z) \, dz$$
Think of $p_\theta(x)$ as an **infinite Gaussian Mixture Model (GMM)**: every infinitesimal point $z$ weighted by $p(z)$ spawns a local Gaussian $\mathcal{N}(\mu_\theta(z), \sigma_\theta^2(z))$ in pixel space, and their continuous integral forms the complex data manifold.

---

### 1.5 Why Direct Marginal Likelihood $\log p_\theta(x_i)$ is Intractable
To train the generative model via Maximum Likelihood Estimation, we want to maximize the log marginal likelihood of all $N$ training images:
$$\max_\theta \frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i) = \max_\theta \frac{1}{N} \sum_{i=1}^N \log \left( \int p_\theta(x_i | z) p(z) \, dz \right)$$

**Why can't we just evaluate this integral directly?**
1. **Analytical Intractability:** Because $\mu_\theta(z)$ is a deep neural network inside the exponent of $p_\theta(x_i|z)$, the integral $\int p_\theta(x_i|z)p(z)\,dz$ has no closed-form antiderivative.
2. **Failure of Naive Monte Carlo Sampling:** What if we try to approximate the integral by drawing $M$ random samples $z_1, \dots, z_M \sim p(z) = \mathcal{N}(0, I)$ and averaging?
   $$p_\theta(x_i) = \mathbb{E}_{z \sim p(z)}\big[p_\theta(x_i | z)\big] \approx \frac{1}{M} \sum_{m=1}^M p_\theta(x_i | z_m)$$
   In a high-dimensional latent space (say, $z \in \mathbb{R}^{64}$), for any specific training image $x_i$ (e.g., a specific golden retriever), the region of $z$-space that decodes to something closely resembling $x_i$ is a microscopic needle in a 64-dimensional haystack. If you sample $z_m \sim \mathcal{N}(0, I)$ blindly, $99.999999\%$ of your samples will decode to random unrelated images, yielding $p_\theta(x_i | z_m) \approx 0$ and providing zero useful gradient signal unless $M$ is astronomically large (e.g., $M > 10^{20}$).

---

### 1.6 Deriving the Evidence Lower Bound (ELBO) via Jensen's Inequality & KL Divergence
Because blind sampling from the prior $p(z)$ fails, we introduce an auxiliary **variational posterior distribution $q_i(z) \approx p_\theta(z|x_i)$** whose sole job is to guess the tiny region of $z$-space most likely to have generated $x_i$.

In Lecture 17, Professor Levine derives the **Evidence Lower Bound (ELBO)** in two complementary ways:

#### Derivation A: Via Importance Sampling & Jensen's Inequality
Multiply and divide the integrand inside $\log p_\theta(x_i)$ by $q_i(z)$:
$$\begin{aligned}
\log p_\theta(x_i) &= \log \int p_\theta(x_i | z) p(z) \, dz \\
&= \log \int \frac{q_i(z)}{q_i(z)} p_\theta(x_i | z) p(z) \, dz \\
&= \log \mathbb{E}_{z \sim q_i(z)} \left[ \frac{p_\theta(x_i | z) p(z)}{q_i(z)} \right]
\end{aligned}$$
Since $\log(\cdot)$ is a strictly concave function, **Jensen's Inequality** states that $\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$. Pushing the logarithm inside the expectation gives:
$$\begin{aligned}
\log p_\theta(x_i) &\ge \mathbb{E}_{z \sim q_i(z)} \left[ \log \frac{p_\theta(x_i | z) p(z)}{q_i(z)} \right] \\
&= \mathbb{E}_{z \sim q_i(z)} \big[ \log p_\theta(x_i | z) + \log p(z) - \log q_i(z) \big] \\
&= \mathbb{E}_{z \sim q_i(z)} \big[ \log p_\theta(x_i | z) + \log p(z) \big] + \mathcal{H}(q_i)
\end{aligned}$$
where $\mathcal{H}(q_i) = \mathbb{E}_{z \sim q_i(z)}[-\log q_i(z)]$ is the **differential entropy** of $q_i(z)$.

Equivalently, grouping the prior $p(z)$ and variational distribution $q_i(z)$ together yields the classic **KL Divergence form of the ELBO**:
$$\begin{aligned}
\mathcal{L}_i(p_\theta, q_i) &= \mathbb{E}_{z \sim q_i(z)}\big[\log p_\theta(x_i | z)\big] + \int q_i(z) \log \frac{p(z)}{q_i(z)} \, dz \\
&= \underbrace{\mathbb{E}_{z \sim q_i(z)}\big[\log p_\theta(x_i | z)\big]}_{\text{Expected Reconstruction Log-Likelihood}} - \underbrace{D_{\text{KL}}\big(q_i(z) \,\|\, p(z)\big)}_{\text{KL Divergence Penalty to Prior}}
\end{aligned}$$

#### Derivation B: Exact Decomposition via $D_{\text{KL}}\big(q_i(z) \,\|\, p_\theta(z|x_i)\big)$ (Why Tightness Requires $q_i(z) \approx p_\theta(z|x_i)$)
How much slack is there between the true log-likelihood $\log p_\theta(x_i)$ and the lower bound $\mathcal{L}_i(p_\theta, q_i)$? Write out the KL divergence between our approximation $q_i(z)$ and the true unknown posterior $p_\theta(z|x_i) = \frac{p_\theta(x_i|z)p(z)}{p_\theta(x_i)}$:
$$\begin{aligned}
D_{\text{KL}}\big(q_i(z) \,\|\, p_\theta(z|x_i)\big) &= \mathbb{E}_{z \sim q_i(z)} \left[ \log \frac{q_i(z)}{p_\theta(z|x_i)} \right] \\
&= \mathbb{E}_{z \sim q_i(z)} \left[ \log \frac{q_i(z) p_\theta(x_i)}{p_\theta(x_i|z) p(z)} \right] \\
&= \mathbb{E}_{z \sim q_i(z)} \big[ \log q_i(z) - \log p_\theta(x_i|z) - \log p(z) \big] + \log p_\theta(x_i) \\
&= -\mathcal{L}_i(p_\theta, q_i) + \log p_\theta(x_i)
\end{aligned}$$
Rearranging this identity gives the fundamental equation of Variational Inference:
$$\log p_\theta(x_i) = \underbrace{\mathcal{L}_i(p_\theta, q_i)}_{\text{ELBO}} + \underbrace{D_{\text{KL}}\big(q_i(z) \,\|\, p_\theta(z|x_i)\big)}_{\ge 0}$$

* Because KL divergence is always non-negative ($D_{\text{KL}} \ge 0$), $\mathcal{L}_i(p_\theta, q_i)$ is guaranteed to be a lower bound on $\log p_\theta(x_i)$ for **any** choice of distribution $q_i(z)$.
* The bound becomes **tight** ($\log p_\theta(x_i) = \mathcal{L}_i$) if and only if $D_{\text{KL}}\big(q_i(z) \,\|\, p_\theta(z|x_i)\big) = 0$, which happens when $q_i(z) = p_\theta(z|x_i)$.
* Therefore, **maximizing $\mathcal{L}_i(p_\theta, q_i)$ with respect to $q_i$ directly minimizes $D_{\text{KL}}\big(q_i(z) \,\|\, p_\theta(z|x_i)\big)$** (forcing $q_i(z)$ to approximate the true posterior), while **maximizing $\mathcal{L}_i(p_\theta, q_i)$ with respect to $\theta$ pushes up the true data log-likelihood $\log p_\theta(x_i)$!**

---

### 1.7 Amortized Variational Inference: Why $q_\phi(z|x)$ is an Encoder Network
Initially, one could parameterize a separate Gaussian distribution $q_i(z) = \mathcal{N}(\mu_i, \sigma_i^2)$ for every single training image $x_i$ ($i = 1, \dots, N$).
* **Why per-image parameters $(\mu_i, \sigma_i)$ are impractical:** If your dataset has $N = 10,000,000$ images, storing and optimizing a separate lookup table of $(\mu_i, \sigma_i)$ vectors for each image scales poorly with $N$, and if a new test image $x_{\text{test}}$ arrives, you have no way to infer its latent code without running an expensive iterative optimization loop from scratch.
* **The Solution (Amortized Variational Inference):** At the end of Lecture 18 Part 1 ([00:27:14]), Professor Levine replaces the per-image lookup table $q_i(z)$ with a single shared neural network, **the Probabilistic Encoder $q_\phi(z|x_i)$** with weights $\phi$, that takes any image $x_i$ as input and outputs two vectors: its mean $\mu_\phi(x_i)$ and standard deviation $\sigma_\phi(x_i)$:
  $$q_\phi(z|x_i) = \mathcal{N}\big(\mu_\phi(x_i), \, \text{diag}(\sigma_\phi^2(x_i))\big)$$
  *(In practice, the encoder network outputs $\mu_\phi(x_i)$ and $\log \sigma_\phi^2(x_i)$ so the network outputs can be unconstrained real numbers in $(-\infty, +\infty)$, and we exponentiate to obtain $\sigma_\phi(x_i) = \exp(\frac{1}{2}\log \sigma_\phi^2(x_i)) > 0$).*

---

## 2. Lecture 18: Variational Autoencoders (VAEs) & Invertible Models (Normalizing Flows)

### 2.1 Optimizing the ELBO: Decoder Gradient $\nabla_\theta$ vs. Encoder Gradient $\nabla_\phi$
Substituting the encoder $q_\phi(z|x_i)$ into our ELBO gives the per-datapoint objective $\mathcal{L}_i(\theta, \phi)$:
$$\mathcal{L}_i(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z|x_i)} \big[ \log p_\theta(x_i | z) + \log p(z) \big] + \mathcal{H}\big(q_\phi(z|x_i)\big)$$
or equivalently:
$$\mathcal{L}_i(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z|x_i)} \big[ \log p_\theta(x_i | z) \big] - D_{\text{KL}}\big(q_\phi(z|x_i) \,\|\, p(z)\big)$$

Let's examine how to take gradients with respect to $\theta$ (Decoder) and $\phi$ (Encoder):
1. **Gradient with respect to Decoder $\theta$ ($\nabla_\theta \mathcal{L}_i$) is Easy:**
   Only $\log p_\theta(x_i | z)$ depends on $\theta$, and the sampling distribution $q_\phi(z|x_i)$ does **not** depend on $\theta$. Thus, we can push $\nabla_\theta$ right inside the expectation and approximate it with a single Monte Carlo sample $z \sim q_\phi(z|x_i)$:
   $$\nabla_\theta \mathcal{L}_i(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z|x_i)}\big[\nabla_\theta \log p_\theta(x_i | z)\big] \approx \nabla_\theta \log p_\theta(x_i | z)$$
2. **Gradient with respect to Encoder $\phi$ ($\nabla_\phi \mathcal{L}_i$) is Hard:**
   In $\nabla_\phi \mathbb{E}_{z \sim q_\phi(z|x_i)}\big[\log p_\theta(x_i|z)\big]$, the parameters $\phi$ live **in the subscript of the expectation** (the distribution from which $z$ is sampled), while the function inside the expectation, $r(x_i, z) = \log p_\theta(x_i|z) + \log p(z)$, does not explicitly contain $\phi$ at all!

---

### 2.2 The Reinforcement Learning Connection: Policy Gradient (REINFORCE) & The Log-Derivative Trick
In Lecture 18 Part 2 ([00:03:51]–[00:05:30]), Professor Levine shows why differentiating $\mathbb{E}_{z \sim q_\phi(z|x_i)}[r(x_i, z)]$ with respect to $\phi$ is mathematically identical to **Reinforcement Learning**:

| Concept | Reinforcement Learning (Policy Gradient) | Variational Autoencoder (Encoder Optimization) |
| :--- | :--- | :--- |
| **Input ("State")** | Environment state $s$ | Training image $x_i$ |
| **Stochastic Policy / Encoder** | Policy $\pi_\phi(a | s)$ | Variational Encoder $q_\phi(z | x_i)$ |
| **Sampled Decision ("Action")** | Action $a \sim \pi_\phi(a | s)$ | Latent code $z \sim q_\phi(z | x_i)$ |
| **Scalar Score ("Reward")** | Environment reward $r(s, a)$ | Reconstruction + Prior score $r(x_i, z) = \log p_\theta(x_i|z) + \log p(z)$ |
| **Objective** | $\max_\phi \mathbb{E}_{a \sim \pi_\phi(a|s)}[r(s, a)]$ | $\max_\phi \mathbb{E}_{z \sim q_\phi(z|x_i)}[r(x_i, z)]$ |

#### Why $\nabla_\phi$ Cannot Pass Directly Inside $\int q_\phi(z|x_i) r(x_i, z) dz$
Writing the expectation as an integral and differentiating with respect to $\phi$:
$$\nabla_\phi \mathbb{E}_{z \sim q_\phi(z|x_i)}\big[r(x_i, z)\big] = \int \nabla_\phi q_\phi(z|x_i) \, r(x_i, z) \, dz$$
Because $\nabla_\phi q_\phi(z|x_i)$ is a derivative of a density (it takes negative values and integrates to $0$, not $1$), it is **not a valid probability distribution**, so this integral cannot be directly estimated by sampling $z \sim q_\phi(z|x_i)$.

#### The Log-Derivative Trick (REINFORCE / Score Function Estimator)
Using the calculus chain-rule identity $\nabla_\phi \log q_\phi(z|x_i) = \frac{\nabla_\phi q_\phi(z|x_i)}{q_\phi(z|x_i)} \implies \nabla_\phi q_\phi(z|x_i) = q_\phi(z|x_i) \nabla_\phi \log q_\phi(z|x_i)$, we substitute back into the integral:
$$\begin{aligned}
\nabla_\phi \mathbb{E}_{z \sim q_\phi(z|x_i)}\big[r(x_i, z)\big] &= \int q_\phi(z|x_i) \nabla_\phi \log q_\phi(z|x_i) \, r(x_i, z) \, dz \\
&= \mathbb{E}_{z \sim q_\phi(z|x_i)} \Big[ r(x_i, z) \, \nabla_\phi \log q_\phi(z|x_i) \Big] \\
&\approx \frac{1}{m} \sum_{j=1}^m r(x_i, z_j) \nabla_\phi \log q_\phi(z_j | x_i)
\end{aligned}$$
* **How Policy Gradient works ("Trial and Error"):** It draws random samples $z_j \sim q_\phi(z|x_i)$, evaluates the scalar reward $r(x_i, z_j)$, and pushes $\phi$ along $\nabla_\phi \log q_\phi(z_j|x_i)$ to make high-reward samples more probable and low-reward samples less probable.
* **Why RL must use this, while VAEs do not:** In RL, the reward $r(s, a)$ comes from a black-box environment (a physics simulator or real world), so $\nabla_a r(s, a)$ is unknown and non-differentiable. Policy gradient treats $r(x_i, z)$ as a black-box scalar, which causes **extremely high variance** because it never knows *which direction* in $z$-space improves $r(x_i, z)$. In a VAE, however, $r(x_i, z) = \log p_\theta(x_i|z) + \log p(z)$ is a **known, differentiable neural network**! *(Note: If $z$ is discrete/categorical, you cannot differentiate with respect to $z$, so discrete latent models must either use Policy Gradient/REINFORCE or continuous relaxations like Gumbel-Softmax).*

---

### 2.3 Why Direct Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ Breaks Backpropagation
Why can't automatic differentiation (PyTorch/TensorFlow) just backpropagate through `z = torch.normal(mean=mu, std=sigma)`?

To update the encoder weights $\phi$ using the analytical derivative of the decoder $\frac{\partial r(x_i, z)}{\partial z}$, the chain rule requires:
$$\frac{\partial r(x_i, z)}{\partial \phi} = \frac{\partial r(x_i, z)}{\partial z} \cdot \underbrace{\frac{\partial z}{\partial \mu_\phi}}_{\text{Blocked!}} \cdot \frac{\partial \mu_\phi}{\partial \phi}$$
* **Forward Pass with Direct Sampling:** The encoder outputs $\mu_\phi = 2.0, \sigma_\phi = 1.5$. These parameters enter a stochastic Random Number Generator (RNG) node, which outputs a sample, say $z = 3.12$.
* **Backward Pass with Direct Sampling:** When `.backward()` reaches $z$ and tries to compute $\frac{\partial z}{\partial \mu_\phi}$, it hits the black-box RNG node. Because a random sampling operation is not a deterministic mathematical function of $\mu_\phi$ (calling it again with $\mu_\phi = 2.0001$ draws a completely unrelated random number), the partial derivative $\frac{\partial z}{\partial \mu_\phi}$ is undefined, blocking gradient flow to $\phi$.

---

### 2.4 The Reparameterization Trick ($z = \mu_\phi(x_i) + \epsilon \odot \sigma_\phi(x_i)$)
To make $z$ a differentiable, deterministic function of $\mu_\phi(x_i)$ and $\sigma_\phi(x_i)$ while preserving its exact Gaussian distribution $\mathcal{N}(\mu_\phi(x_i), \sigma_\phi^2(x_i))$, we factor out the randomness into an auxiliary parameter-free noise variable $\epsilon \sim \mathcal{N}(0, I)$:
$$z = \mu_\phi(x_i) + \epsilon \odot \sigma_\phi(x_i), \qquad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

1. **Mathematical Expectation View:**
   We change the variable of integration from $z \sim q_\phi(z|x_i)$ (which depends on $\phi$) to $\epsilon \sim \mathcal{N}(0, I)$ (which is **completely independent of $\phi$**):
   $$\begin{aligned}
   \nabla_\phi \mathbb{E}_{z \sim q_\phi(z|x_i)}\big[r(x_i, z)\big] &= \nabla_\phi \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \Big[ r\big(x_i, \, \mu_\phi(x_i) + \epsilon \odot \sigma_\phi(x_i)\big) \Big] \\
   &= \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \Big[ \nabla_\phi r\big(x_i, \, \mu_\phi(x_i) + \epsilon \odot \sigma_\phi(x_i)\big) \Big]
   \end{aligned}$$
   Because the distribution $\mathcal{N}(0, I)$ has no $\phi$ dependence, $\nabla_\phi$ passes cleanly inside the expectation!
2. **Computational Graph View:**
   In code, $\epsilon$ is sampled on an isolated leaf branch (`eps = torch.randn_like(std)`). Once drawn (say $\epsilon = 0.746$), $\epsilon$ is treated as a fixed constant during that pass. Now $z = \mu_\phi + \sigma_\phi \odot \epsilon$ is just elementary addition and multiplication, with exact, low-variance analytical derivatives:
   $$\frac{\partial z}{\partial \mu_\phi} = I, \qquad \frac{\partial z}{\partial \sigma_\phi} = \text{diag}(\epsilon)$$
   Even a **single sample** ($M = 1$) of $\epsilon \sim \mathcal{N}(0, I)$ per training step gives a remarkably low-variance gradient estimate because $\nabla_z r(x_i, z)$ tells the encoder the exact direction in latent space that improves reconstruction!

---

### 2.5 Where Noise is Injected: DAE vs. VAE vs. DDPM vs. LDM
In Lecture 18 Part 2 ([00:12:36]–[00:12:42]), Professor Levine remarks: *"In the denoising autoencoder, we added noise to $x$; in the variational autoencoder, we add noise to $z$."* (Note: YouTube's auto-captions famously mistranscribe this as *"denoising audio code"* and *"very short encoder"*).

Comparing where noise is injected across generative architectures reveals a clear evolutionary progression:

| Architecture | Where is Noise Injected? | Nature of the Latent Variable | Generative Quality & Speed |
| :--- | :--- | :--- | :--- |
| **Denoising Autoencoder (DAE)** | Added directly to the input **$x$** at the very start ($\tilde{x} = x + \text{noise}$). | Deterministic bottleneck vector $z$; used for representation learning. | **Poor generator**: No prior $p(z)$ defined to sample new images from scratch. |
| **Variational Autoencoder (VAE)** | Added to the compressed bottleneck **$z$** in the middle of the network ($z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$). | 1-step compressed latent $z$ ($\dim(z) \ll \dim(x)$). | **Fast (1 step), but blurry**: A single Gaussian step struggles to capture sharp multi-modal high-frequency details. |
| **Diffusion Models (DDPM)** | Added gradually over $T=1000$ steps directly in pixel space ($\mathbf{x}_0 \to \mathbf{x}_1 \to \dots \to \mathbf{x}_T$). | $T$-step Markov chain of latents with the **exact same dimensionality** as the image. | **Very sharp, but slow**: Running a full-resolution U-Net $1000$ times sequentially is computationally expensive. |
| **Latent Diffusion (LDM / Stable Diffusion)** | First uses a **Fully Convolutional VAE** to compress $x \to z_0$, then runs the $T$-step **DDPM** noise process inside **$z$-space** ($z_0 \to z_1 \dots \to z_T$). | Spatial compression via VAE ($64 \times 64 \times 4$) + $T$-step Markov diffusion trajectory in latent space. | **Best of both worlds**: Sharp multi-step generation with fast computation inside the compact VAE latent space. |

---

### 2.6 Understanding the Role of $z$ in Training vs. Inference (End-to-End Signal Flows)
A common point of confusion is whether $z$ is "pure noise" or "semantic features." The answer depends on whether you are in **Training** or **Inference (Generation)**:
* **During Training:** $z$ is **NOT pure noise**. It is the **clean semantic feature vector $\mu_\phi(x_i)$ perturbed by scaled random noise $\sigma_\phi(x_i) \odot \epsilon$**:
  $$\underbrace{z}_{\text{Noisy Latent Code}} = \underbrace{\mu_\phi(x_i)}_{\text{Clean Semantic Feature}} + \underbrace{\sigma_\phi(x_i) \odot \epsilon}_{\text{Injected Gaussian Noise}}$$
* **During Inference (Generation):** You **discard the encoder $q_\phi$**, sample $z \sim \mathcal{N}(0, I)$ directly from pure standard normal noise, and pass it through the decoder $p_\theta(x|z)$. *Why does pure noise $z \sim \mathcal{N}(0, I)$ produce a meaningful image at test time?* Because during training, the KL divergence penalty forced all $N$ training images' semantic clouds $\mathcal{N}(\mu_\phi(x_i), \sigma_\phi^2(x_i))$ to pack tightly inside $\mathcal{N}(0, I)$, so any random dart $z \sim \mathcal{N}(0, I)$ lands inside one of the organized semantic neighborhoods learned by the decoder!

#### Complete Signal Flow Diagrams

##### 1. Variational Autoencoder (VAE)
* **Training Flow (Encoder + Decoder Active):**
```text
                       ϵ ~ N(0, I)  (Random standard normal noise on side branch)
                            │
                            ▼
[Image x_i] ──► [Encoder q_ϕ] ──► μ_ϕ(x_i), σ_ϕ(x_i) ──► z = μ_ϕ + σ_ϕ ⊙ ϵ ──► [Decoder p_θ] ──► [Reconstructed x̂_i]
                                        │                                                              │
                                        ▼                                                              ▼
                           Term 2: KL Divergence                                          Term 1: Reconstruction
                         D_KL( N(μ_ϕ, σ_ϕ²) || N(0, I) )                                 -log p_θ(x_i | z) ≈ ||x_i - x̂_i||²
```
* **Inference / Generation Flow (Only Decoder Active):**
```text
[Sample z ~ N(0, I)] ──► [Trained Decoder p_θ] ──► [Brand New Generated Image x_new]
```

##### 2. Standard Diffusion Model (DDPM)
* **Training Flow (1 Random Timestep $t$ per Image using $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$):**
```text
[Real Image x_0] ──┐
                   ├──► x_t = √(ᾱ_t)·x_0 + √(1 - ᾱ_t)·ϵ ──► [U-Net ϵ_θ(x_t, t)] ──► [Predicted Noise ϵ_θ]
[Noise ϵ ~ N(0,I)] ┘                                                                          │
         │                                                                                    ▼
         └───────────────────────────────────────────────────────────────────────► Loss: ||ϵ - ϵ_θ(x_t, t)||²
```
* **Inference / Generation Flow ($T$-Step Sequential Reverse Loop):**
```text
[x_T ~ N(0, I)] ──► (Step T: U-Net subtracts noise) ──► x_{T-1} ──► ... ──► (Step 1: U-Net) ──► [Clean Image x_0]
```
  At each step $t = T, \dots, 1$:
  $$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

##### 3. Latent Diffusion Model (LDM / Stable Diffusion)
* **Training Flow:**
```text
[Image x_0] ──► [Frozen VAE Encoder ℰ] ──► [Clean Latent z_0] ──► Add noise to get z_t ──► [U-Net ϵ_θ(z_t, t)] ──► Loss: ||ϵ - ϵ_θ(z_t, t)||²
(512×512×3)                                    (64×64×4)              (64×64×4)
```
* **Inference / Generation Flow:**
```text
[Sample z_T ~ N(0, I)] ──► [Run T-step Reverse Diffusion Loop in z-space] ──► [Clean Latent z_0] ──► [Frozen VAE Decoder 𝒟] ──► [Final Image x_0]
       (64×64×4)                        (U-Net runs T times)                      (64×64×4)                                       (512×512×3)
```

---

### 2.7 Reconciling Theory ($\max$ ELBO) vs. Code ($\min$ Loss) & Closed-Form Gaussian KL
In Lecture 18 Part 2 ([00:10:09]), Professor Levine writes the complete VAE training objective on the slide as:
$$\max_{\theta, \phi} \frac{1}{N} \sum_{i=1}^N \Big( \log p_\theta\big(x_i \,\big|\, \mu_\phi(x_i) + \epsilon \sigma_\phi(x_i)\big) - D_{\text{KL}}\big(q_\phi(z|x_i) \,\|\, p(z)\big) \Big)$$

Here is how every piece of this slide equation translates into practical PyTorch code:
1. **Expanded $z$ Notation:** Instead of writing $\log p_\theta(x_i | z)$, the slide substitutes $z = \mu_\phi(x_i) + \epsilon \sigma_\phi(x_i)$ directly inside $\log p_\theta(x_i | \dots)$ so you can see that **both $\theta$ and $\phi$ are optimized through the first term**.
2. **Maximizing Objective vs. Minimizing Loss:** Deep learning frameworks minimize a loss (`loss.backward()`). Multiplying the ELBO by $-1$ flips $\max$ to $\min$ and flips the signs of both terms:
   $$\min_{\theta, \phi} \frac{1}{N} \sum_{i=1}^N \Big( -\log p_\theta\big(x_i \,\big|\, \mu_\phi(x_i) + \epsilon \sigma_\phi(x_i)\big) + D_{\text{KL}}\big(q_\phi(z|x_i) \,\|\, p(z)\big) \Big)$$
3. **Why $D_{\text{KL}}\big(q_\phi(z|x_i) \,\|\, p(z)\big)$ IS $D_{\text{KL}}\big(\mathcal{N}(\mu_\phi, \sigma_\phi^2) \,\|\, \mathcal{N}(0, I)\big)$:**
   By definition, $q_\phi(z|x_i) = \mathcal{N}(\mu_\phi(x_i), \sigma_\phi^2(x_i))$ and $p(z) = \mathcal{N}(0, I)$. Because both distributions are diagonal Gaussians in $\mathbb{R}^D$, their KL divergence integrates analytically in closed form without needing any random sampling:
   $$D_{\text{KL}}\Big(\mathcal{N}\big(\mu_\phi(x_i), \sigma_\phi^2(x_i)\big) \,\Big\|\, \mathcal{N}(0, I)\Big) = \frac{1}{2} \sum_{d=1}^D \Big( \mu_{\phi, d}^2(x_i) + \sigma_{\phi, d}^2(x_i) - \log \sigma_{\phi, d}^2(x_i) - 1 \Big)$$
   *Check:* When $\mu_{\phi,d} = 0$ and $\sigma_{\phi,d}^2 = 1$, each term is $\frac{1}{2}(0 + 1 - \log(1) - 1) = 0$.

---

### 2.8 The Great Tug-of-War: Why $q_\phi(z|x_i)$ Cannot Just Equal $\mathcal{N}(0, I)$
**Question:** *If we want $D_{\text{KL}}\big(q_\phi(z|x_i) \,\|\, \mathcal{N}(0, I)\big)$ to be as small as possible, why don't we just set $q_\phi(z|x_i) = \mathcal{N}(0, I)$ (i.e., $\mu_\phi(x_i) = 0, \sigma_\phi(x_i) = 1$) for every image $x_i$?*

**Answer:** Because all $q_\phi(z|x_i)$ constantly **tend toward $\mathcal{N}(0, I)$** under the pull of the KL penalty, **and the only force preventing them from collapsing into $\mathcal{N}(0, I)$ is the reconstruction term $\log p_\theta\big(x_i \,\big|\, \mu_\phi(x_i) + \epsilon \sigma_\phi(x_i)\big)$ fighting back!**

If $q_\phi(z|x_i) = \mathcal{N}(0, I)$ for every $x_i$:
* Then $z = 0 + 1 \cdot \epsilon = \epsilon \sim \mathcal{N}(0, I)$ contains **zero information** about which image $x_i$ was fed into the encoder.
* The decoder $p_\theta(x_i | z)$ is handed pure random static $z$ and asked to reconstruct a specific dog on step 1, and a specific sports car on step 2, from the exact same noise distribution, causing the reconstruction error to explode!

| Force in the ELBO | What it pushes $\mu_\phi(x_i)$ and $\sigma_\phi(x_i)$ to do | What happens if ONLY this term wins? |
| :--- | :--- | :--- |
| **Term 1 (Reconstruction):**<br>$\log p_\theta(x_i \mid \mu_\phi(x_i) + \epsilon\sigma_\phi(x_i))$ | Pushes centers $\mu_\phi(x_i)$ **far apart** for different images and squeezes noise **$\sigma_\phi(x_i) \to 0$** so the decoder never confuses two images. | **Deterministic Autoencoder:** Sharp reconstruction, but latent space is full of empty gaps; sampling $z \sim \mathcal{N}(0, I)$ fails. |
| **Term 2 (KL Regularizer):**<br>$-D_{\text{KL}}\big(q_\phi(z|x_i) \,\|\, \mathcal{N}(0, I)\big)$ | Pulls every center **$\mu_\phi(x_i) \to 0$** ("gravity" via $\frac{1}{2}\mu^2$) and inflates every spread **$\sigma_\phi(x_i) \to 1$** so $q_\phi(z|x_i) = \mathcal{N}(0, I)$. | **Posterior Collapse:** $z$ becomes pure noise independent of $x_i$; decoder outputs a blurry average image. |

---

### 2.9 Individual Posteriors $q_\phi(z|x_i)$ vs. The Aggregated Posterior $\frac{1}{N}\sum_i q_\phi(z|x_i)$
In a healthy, well-trained VAE, how do those two opposing forces settle into equilibrium?

```text
Probability
  ▲                     p(z) = N(0, 1)  (The Global Container: mean=0, std=1)
  │                        _.-""""-._
  │                      .'          '.
  │                     /   x_1    x_2 \       x_1 = q_ϕ(z|dog):  mean = -1.2, std = 0.15
  │                    ;    /\     /\   ;      x_2 = q_ϕ(z|cat):  mean = +0.8, std = 0.15
  │                   /    /  \   /  \   \
──┴───────────────────┴───┴────┴─┴────┴───┴──────► z
                    -3   -1.2   0 +0.8   +3
```

1. **For a Single Data Sample $x_i$ (Individual Posterior $q_\phi(z|x_i)$):**
   Each image $x_i$ is mapped to a **small, localized Gaussian sub-cloud** parked at a distinct coordinate $\mu_\phi(x_i)$ inside the $[-3, +3]$ region of $\mathcal{N}(0, I)$ (e.g., $\mu_\phi(x_1) = -1.2$ for a dog, $\mu_\phi(x_2) = +0.8$ for a cat) with a relatively narrow spread ($\sigma_\phi(x_i) \approx 0.15 < 1$). It gets pulled as close to $\mathcal{N}(0, I)$ as possible without overlapping with neighboring images so much that reconstruction breaks. *(If a latent dimension is completely useless for reconstruction, the KL spring wins completely on that dimension and sets $\mu_{\phi,d}(x_i) = 0, \sigma_{\phi,d}(x_i) = 1$).*
2. **Across All $N$ Data Samples Combined (Aggregated / Marginal Posterior $q_\phi(z)$):**
   When you average the thousands of small, slightly overlapping individual clouds across the entire dataset, they tile the space and blend together to match the global prior $\mathcal{N}(0, I)$:
   $$q_\phi(z) = \frac{1}{N} \sum_{i=1}^N \mathcal{N}\big(\mu_\phi(x_i), \sigma_\phi^2(x_i)\big) \approx \mathcal{N}(0, I)$$

---

### 2.10 Fully Convolutional VAEs vs. Vector VAEs
In Lecture 18 Part 2, Professor Levine notes that VAEs can be built with **fully convolutional** encoders and decoders.

```text
Standard Vector VAE:
[Image: 512×512×3] ──(Conv)──► [16×16×128] ──(Flatten + Linear)──► μ, σ ∈ ℝ^256 (1D Vector)

Fully Convolutional VAE (Used in Latent Diffusion / Stable Diffusion):
[Image: 512×512×3] ──(Strided Convs)──► [64×64×8] ──(Split Channels)──► μ ∈ ℝ^{64×64×4}, σ ∈ ℝ^{64×64×4}
```

* **How a Fully Convolutional VAE Works:** It removes all `Flatten` and `Linear` (dense) layers. An input image of shape $H \times W \times 3$ (e.g., $512 \times 512 \times 3$) is downsampled via strided convolutions into a 3D feature map of shape $64 \times 64 \times 8$. The first $4$ channels represent the 3D mean tensor $\mu_\phi(x_i) \in \mathbb{R}^{64 \times 64 \times 4}$, and the last $4$ channels represent $\log \sigma_\phi^2(x_i) \in \mathbb{R}^{64 \times 64 \times 4}$.
* **Element-wise Reparameterization & KL:** We sample noise $\epsilon \sim \mathcal{N}(0, I)$ of shape $64 \times 64 \times 4$, compute $z_{h,w,c} = \mu_{\phi,h,w,c} + \epsilon_{h,w,c}\sigma_{\phi,h,w,c}$ element-wise, and sum the closed-form Gaussian KL divergence over all $64 \times 64 \times 4$ spatial-channel entries.
* **Why This Matters:**
  1. **Preserves 2D Spatial Topology:** The top-left latent patch $z_{0,0,:}$ directly corresponds to the top-left corner of the image.
  2. **Resolution Agnostic:** Can encode/decode any image resolution without changing weights.
  3. **Enables Latent Diffusion:** Because $z_0 \in \mathbb{R}^{64 \times 64 \times 4}$ is still a 2D spatial grid with $4$ channels, a 2D convolutional U-Net can run diffusion directly on $z_t$!

---

### 2.11 VAE Failure Modes: Posterior Collapse ($p_\theta(x|z) \to p(x)$) & Why Ignored Latents Produce Blurry Average Images
In Lecture 18 Part 3, Professor Levine highlights two classic failure modes when training VAEs:

#### Problem 1: Latent Code is Ignored ($p_\theta(x|z) \to p(x)$, Posterior Collapse)
If the KL divergence penalty dominates early in training (or if the decoder is an overly powerful autoregressive model like a PixelCNN that can model $x$ without needing $z$), the encoder collapses to $q_\phi(z|x_i) = p(z) = \mathcal{N}(0, I)$ to achieve $D_{\text{KL}} = 0$, and the decoder ignores $z$ completely: $p_\theta(x|z) \to p_\theta(x)$.

**Mathematical Proof: Why Ignoring $z$ Produces a Blurry "Average" Image When Reconstructing**
When the decoder $p_\theta(x_i | z) = \mathcal{N}(x_i; \, \hat{x}_\theta(z), \sigma^2 I)$ ignores the latent code $z$, its predicted mean image $\hat{x}_\theta(z) = \hat{x}$ is forced to be a **single constant image $\hat{x}$** regardless of which training image $x_i$ is being reconstructed.
Maximizing the Gaussian log-likelihood $\frac{1}{N}\sum_{i=1}^N \log p_\theta(x_i|z)$ is equivalent to minimizing the Mean Squared Error across all $N$ training images:
$$\min_{\hat{x}} \frac{1}{N} \sum_{i=1}^N \big\| x_i - \hat{x} \big\|^2$$
Take the gradient with respect to the constant output $\hat{x}$ and set it to zero:
$$\nabla_{\hat{x}} \left( \frac{1}{N} \sum_{i=1}^N \| x_i - \hat{x} \|^2 \right) = \frac{2}{N} \sum_{i=1}^N (\hat{x} - x_i) = 0 \implies \hat{x}^* = \frac{1}{N} \sum_{i=1}^N x_i$$
The optimal constant prediction $\hat{x}^*$ under $L_2$ loss is literally the **pixel-by-pixel arithmetic mean of the entire dataset**! Because the decoder receives no information from $z$ about which specific dog, cat, or car $x_i$ it is supposed to draw, it hedges its bets by outputting the superimposed average of all images, which looks like a blurry, ghostly gray blob.

#### Problem 2: Prior Hole / Loose KL Constraint (Poor Generation vs. Good Reconstruction)
If the KL weight is too weak (or $q_\phi(z|x_i)$ fails to cover $p(z)$), the encoder shrinks $\sigma_\phi(x_i) \to 0$ and scatters $\mu_\phi(x_i)$ leaving empty gaps inside $\mathcal{N}(0, I)$. Reconstructions look sharp, but sampling $z \sim \mathcal{N}(0, I)$ at test time hits empty holes and produces broken samples. *(Common fix: **$\beta$-VAE** or **KL Annealing**, where a weight $\beta$ on $- \beta D_{\text{KL}}$ is gradually warmed up from $0 \to 1$ during training).*

---

### 2.12 Invertible Generative Models (Normalizing Flows) & Change of Variables
In the second half of Lecture 18 (Parts 3 & 4), Professor Levine asks: *What if we want **exact** log-likelihood $\log p_\theta(x)$ without any variational lower bound or encoder approximation?*

Instead of having a compressed latent $z$ ($\dim(z) < \dim(x)$) and a stochastic decoder $p_\theta(x|z)$, **Normalizing Flows (Invertible Models)** set $\dim(z) = \dim(x)$ and define $x = f_\theta(z)$ as a **deterministic, strictly invertible (bijective) function** with inverse $z = f_\theta^{-1}(x)$.

#### The Change of Variables Theorem
Because total probability mass must integrate to $1$ ($\int p(x)\,dx = \int p(z)\,dz = 1$), transforming $z \sim p(z)$ via $x = f_\theta(z)$ stretches or compresses local volume elements by the absolute determinant of the $D \times D$ Jacobian matrix $\frac{df_\theta(z)}{dz}$:
$$p_\theta(x) = p\big(f_\theta^{-1}(x)\big) \left|\det\left(\frac{df_\theta(z)}{dz}\right)\right|^{-1} = p\big(f_\theta^{-1}(x)\big) \left|\det\left(\frac{df_\theta^{-1}(x)}{dx}\right)\right|$$

Taking the logarithm gives the exact log-likelihood training objective:
$$\log p_\theta(x) = \log p\big(f_\theta^{-1}(x)\big) - \log \left|\det\left(\frac{df_\theta(z)}{dz}\right)\right|$$

* **Why Naive Invertible Networks Fail:** Computing the determinant of an arbitrary $D \times D$ Jacobian matrix takes $\mathcal{O}(D^3)$ time! For a small $64 \times 64 \times 3$ image ($D = 12,288$), $D^3 \approx 1.85 \times 10^{12}$ operations per image, which is completely intractable.
* **The Triangular Jacobian Solution:** In linear algebra, the determinant of an upper- or lower-triangular matrix is simply the **product of its diagonal entries** ($\mathcal{O}(D)$ time instead of $\mathcal{O}(D^3)$)!

---

### 2.13 Volume-Preserving Flows ($\left|\det\left(\frac{df(z)}{dz}\right)\right| = 1$): Geometry, Squeeze Mappings & RealNVP

#### 1. Additive Coupling Layers (NICE) & Why $\left|\det\left(\frac{df(z)}{dz}\right)\right| = 1$
To guarantee a triangular Jacobian with $1$s on the diagonal, **NICE** splits the vector $z$ into two partitions $(z_1, z_2)$ and defines an **additive coupling layer**:
$$\begin{aligned}
x_1 &= z_1 \\
x_2 &= z_2 + m_\theta(z_1)
\end{aligned} \qquad \Longleftrightarrow \qquad \text{Inverse: } \begin{aligned}
z_1 &= x_1 \\
z_2 &= x_2 - m_\theta(x_1)
\end{aligned}$$
Notice that $m_\theta(z_1)$ can be an **arbitrarily complex, non-invertible neural network** (e.g., a deep ResNet or CNN), because when inverting the layer, $x_1 = z_1$ is already known so we just run $m_\theta(x_1)$ forward and subtract it!

Look at the block Jacobian matrix of this transformation:
$$\frac{df(z)}{dz} = \begin{bmatrix} \frac{\partial x_1}{\partial z_1} & \frac{\partial x_1}{\partial z_2} \\ \frac{\partial x_2}{\partial z_1} & \frac{\partial x_2}{\partial z_2} \end{bmatrix} = \begin{bmatrix} I & 0 \\ \frac{\partial m_\theta(z_1)}{\partial z_1} & I \end{bmatrix} \implies \left|\det\left(\frac{df(z)}{dz}\right)\right| = 1$$

#### 2. Geometric Meaning of $\left|\det\left(\frac{df(z)}{dz}\right)\right| = 1$ (Incompressible / Volume-Preserving Flow)
When $\left|\det\left(\frac{df(z)}{dz}\right)\right| = 1$, the log-determinant term vanishes ($\log(1) = 0$), leaving $p_\theta(x) = p(z)$. Geometrically, this means $f(z)$ is **strictly volume-preserving**:

```text
      LATENT SPACE: z ~ N(0, I)                        DATA SPACE: x = f(z) with |det(J)| = 1
      (Round Gaussian Cloud)                           (Twisted "Banana / Sine Wave" Cloud)

             z_2 ▲                                            x_2 ▲
                 │      .  *  *  .                                │                . * * .
                 │   *    ┌────┐    *                             │              *   /   / *
                 │  *     │▒▒▒▒│     *      x_1 = z_1             │   . * .     *   /▒▒▒/   *
        ─────────┼────────┼────┼──────┼───► z_1   ──────────►   ──┼─ *─────*───*───/───/─────*─► x_1
                 │  *     └────┘     *      x_2 = z_2 + sin(z_1)  │ *       * *   /   /     *
                 │   *              *                             │  *       .   . * *     *
                 │      .  *  *  .                                │    . * .             .
```

* **What $\det = 1$ CAN do (Shearing & Squeeze Mappings):**
  * A single additive coupling layer **shears** slices of space (tilting a $1 \times 1$ square `▒▒▒▒` of area $1$ into a parallelogram `/▒▒▒/` of base $1$, vertical height $1$, and area $1$).
  * What about compressing one axis while stretching another so the product is still $1$ (e.g., transforming a $1 \times 1$ square into a $0.2 \times 5$ or $0.5 \times 2$ rectangle)? A single additive layer has $1$s on the diagonal, so it only shears; however, **stacking multiple additive layers that alternate which coordinates are updated** ($z_1$ then $z_2$) multiplies shear matrices together, which *can* produce squeeze mappings like $\begin{bmatrix} 5 & 0 \\ 0 & 0.2 \end{bmatrix}$ ($\det = 5 \times 0.2 = 1$).
* **Why $\det = 1$ is Still "Representationally Limiting" (Slide Note):**
  Even if you squeeze one axis by $0.2\times$ and stretch another by $5\times$ ($0.2 \times 5 = 1$), the **total $D$-dimensional hyper-volume of the region is still $1$**, which means the **number of sample points per unit volume (the probability density $p(x) = p(z)$) never changes!**
  * You can never compress all $D$ dimensions simultaneously into a tight, high-density peak or lower-dimensional manifold without causing some other dimension to bulge outward by the reciprocal factor ($\lambda_1 \lambda_2 \cdots \lambda_D = 1$).

#### 3. Relaxing $\det = 1$: Affine Coupling Layers (RealNVP & Glow)
To allow the network to expand and compress volume ($\det \neq 1$) while keeping the Jacobian triangular and $\mathcal{O}(D)$ to compute, **RealNVP** introduces a learnable scale network $s_\theta(z_1)$ alongside the translation network $m_\theta(z_1)$:
$$\begin{aligned}
x_1 &= z_1 \\
x_2 &= z_2 \odot \exp\big(s_\theta(z_1)\big) + m_\theta(z_1)
\end{aligned}$$
Now the Jacobian is still lower-triangular, and its determinant is simply the product of the scaling factors on the diagonal:
$$\frac{df(z)}{dz} = \begin{bmatrix} I & 0 \\ \frac{\partial x_2}{\partial z_1} & \text{diag}\big(\exp(s_\theta(z_1))\big) \end{bmatrix} \implies \log \left|\det\left(\frac{df(z)}{dz}\right)\right| = \sum_{d} s_{\theta, d}(z_1)$$
By stacking multiple affine coupling layers (alternating partitions via checkerboard masks or channel splits, and adding final global rescaling filters), RealNVP and Glow can compress wide regions of $\mathcal{N}(0, I)$ into razor-sharp, high-density data manifolds while maintaining exact, tractable log-likelihoods!

---

## 3. Lecture 19: Generative Adversarial Networks (GANs)

### 3.1 Implicit Density Models: Likelihood-Free Training via a Two-Player Game
In Lecture 19, Professor Levine asks: *Why do maximum likelihood models (VAEs, Autoregressive models, Normalizing Flows) often struggle or require restrictive architectures?*
* Maximum likelihood ($\max_\theta \frac{1}{N}\sum_i \log p_\theta(x_i)$) forces the model to place probability mass on **every** training point $x_i$. To avoid assigning zero probability to any point, pixel-space likelihood models tend to "hedge their bets" and spread probability mass across empty space between modes, resulting in blurry averages.
* Furthermore, representing $p_\theta(x)$ explicitly requires either variational approximations (VAEs), sequential pixel-by-pixel factorization (Autoregressive), or strict invertibility with $\dim(z)=\dim(x)$ (Flows).

**Generative Adversarial Networks (GANs)** abandon explicit likelihood $p_\theta(x)$ entirely (**Implicit Density Models**). Instead of evaluating $\log p_\theta(x)$, we train two networks in a competitive minimax game:
1. **Generator $G_\theta(z)$:** Takes random noise $z \sim p(z) = \mathcal{N}(0, I)$ and maps it directly to a synthetic sample $x = G_\theta(z)$ (inducing an implicit distribution $p_G(x)$).
2. **Discriminator $D_\phi(x) = p_\phi(y = \text{True} \mid x) \in (0, 1)$:** A binary classifier trained to output the probability that an input image $x$ came from the **True dataset $\mathcal{D}_T$** ($y=1$) rather than the **False generated dataset $\mathcal{D}_F$** ($y=0$).

---

### 3.2 The 5-Step GAN Training Algorithm
From the Lecture 19 slide we analyzed:
1. **Get a "True" dataset:** $\mathcal{D}_T = \{x_i\}_{i=1}^N \sim p_{\text{data}}(x)$.
2. **Initialize Generator $G_\theta(z)$:** Random weight initialization.
3. **Sample a "False" dataset $\mathcal{D}_F$:** Draw $z \sim p(z)$ and generate fake samples $x = G_\theta(z)$.
4. **Update Discriminator $D_\phi(x) = p_\phi(y|x)$ using $\mathcal{D}_T$ and $\mathcal{D}_F$ (1 SGD step):**
   Maximize binary cross-entropy classification accuracy:
   $$\max_\phi \Big( \mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D_\phi(G_\theta(z)))] \Big)$$
5. **Update Generator $G_\theta(z)$ (1 SGD step):**
   Use $-\log D_\phi(G_\theta(z))$ (or $\log(1 - D_\phi(G_\theta(z)))$) as the loss to update $\theta$, then loop back to Step 3.

---

### 3.3 Why $G(z)$ Wants $D(x) = 0.5$ Globally (Nash Equilibrium) Despite Minimizing $-\log D(G(z))$ Locally
On the Lecture 19 slide, Professor Levine writes at the bottom:
> **What does $G(z)$ want to do?**  
> **Make $D(x) = 0.5$ for all generated $x$.**

Why is the global target $D(x) = 0.5$ instead of $D(x) = 1.0$ ("as high as possible")?
We must distinguish between **a single gradient step against a frozen discriminator (Step 5)** versus **the global equilibrium of the alternating two-player game (Step 4 + Step 5)**:

| Perspective | Target Value of $D(G(z))$ | Mathematical & Intuitive Explanation |
| :--- | :--- | :--- |
| **Local Step 5 Loss Function**<br>($\min_\theta -\log D_\phi(G_\theta(z))$) | Pushes toward **$1.0$** | If the discriminator weights $\phi$ were frozen forever, minimizing $-\log D(G(z))$ pushes $G(z)$ into regions where the current discriminator outputs $D(x) \to 1$. |
| **Global Game Equilibrium**<br>(Step 4 updates $D_\phi$ right after $G_\theta$) | Converges to **$0.5$** | Because $D_\phi$ adapts at every step using equal batches of $\mathcal{D}_T$ and $\mathcal{D}_F$, any image $x$ that $G(z)$ produces shows up in the "False" pile $\mathcal{D}_F$. Even when $G(z)$ is **100% indistinguishable** from real data ($p_G(x) = p_{\text{data}}(x)$), half of the identical images come from $\mathcal{D}_T$ and half come from $\mathcal{D}_F$, forcing an optimal detective $D^*(x)$ to guess **$50/50$ ($D^*(x) = 0.5$)** everywhere! |

---

### 3.4 Mathematical Proof: Optimal Discriminator $D^*(x)$ & Jensen-Shannon Divergence
In Lecture 19 Part 1, Professor Levine proves why the minimax GAN objective:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{x \sim p_G}[\log(1 - D(x))]$$
has a unique global optimum at $p_G = p_{\text{data}}$ where $D^*(x) = 0.5$:

#### Step 1: Find the Optimal Discriminator $D^*(x)$ for Any Fixed $G$
Write the expectations as integrals over $x$:
$$V(D, G) = \int \Big( p_{\text{data}}(x) \log D(x) + p_G(x) \log(1 - D(x)) \Big) dx$$
To maximize this integrand point-wise with respect to $D(x)$ for any $x$, differentiate $a \log y + b \log(1 - y)$ with respect to $y = D(x)$ and set to $0$:
$$\frac{p_{\text{data}}(x)}{D(x)} - \frac{p_G(x)}{1 - D(x)} = 0 \implies p_{\text{data}}(x)(1 - D(x)) = p_G(x) D(x) \implies \mathbf{D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}}$$
* Check 1: Can $D^*(x) = 1$ for any generated sample where $p_G(x) > 0$? **No**, because $\frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)} = 1 \implies p_G(x) = 0$!
* Check 2: When $p_G(x) = p_{\text{data}}(x)$ everywhere, what is $D^*(x)$?
  $$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{\text{data}}(x)} = \frac{1}{2} = 0.5$$

#### Step 2: Substitute $D^*(x)$ Back into $V(D^*, G)$
Plugging $D^*(x)$ into $V(D, G)$ shows what the generator $G$ is actually minimizing:
$$\begin{aligned}
V(D^*, G) &= \mathbb{E}_{x \sim p_{\text{data}}} \left[ \log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)} \right] + \mathbb{E}_{x \sim p_G} \left[ \log \frac{p_G(x)}{p_{\text{data}}(x) + p_G(x)} \right] \\
&= -\log(4) + D_{\text{KL}}\left( p_{\text{data}} \,\Big\|\, \frac{p_{\text{data}} + p_G}{2} \right) + D_{\text{KL}}\left( p_G \,\Big\|\, \frac{p_{\text{data}} + p_G}{2} \right) \\
&= -\log(4) + 2 \cdot D_{\text{JS}}\big( p_{\text{data}} \,\|\, p_G \big)
\end{aligned}$$
where $D_{\text{JS}}$ is the symmetric **Jensen-Shannon Divergence**. Since $D_{\text{JS}}(p_{\text{data}} \| p_G) \ge 0$ with equality if and only if $p_G = p_{\text{data}}$, the global minimum of the generator's objective is achieved when $p_G(x) = p_{\text{data}}(x)$ and $D^*(x) = 0.5$!

---

### 3.5 Vanishing Gradients, Non-Saturating Generator Loss & Mode Collapse
In Lecture 19 Part 2, Professor Levine explains why training GANs in practice is notoriously difficult:

1. **The Vanishing Gradient Problem Early in Training:**
   * At the start of training, $G_\theta(z)$ generates terrible random noise, so $D_\phi(x)$ easily tells fake from real and outputs $D_\phi(G_\theta(z)) \approx 0$.
   * If $G$ uses the theoretical minimax loss $\min_\theta \log(1 - D_\phi(G_\theta(z)))$, look at the slope of $\log(1 - D)$ when $D \approx 0$: $\frac{d}{dD}\log(1-D)\big|_{D=0} = -1$, and after passing through the sigmoid saturation of $D(x) = \sigma(\text{logit}(x))$ where $\sigma'(z) \approx 0$, the gradient $\nabla_\theta \log(1 - D_\phi(G_\theta(z))) \to 0$! The discriminator is *too good*, so the generator gets zero gradient signal and stops learning.
   * **The Fix (Non-Saturating Heuristic Loss, Step 5 on the Slide):** Instead of minimizing $\log(1 - D(G(z)))$, we train $G$ to **minimize $-\log D(G(z))$** (maximize $\log D(G(z))$). Near $D \approx 0$, $-\log(D)$ goes to $+\infty$ with a very steep slope $-\frac{1}{D}$, providing huge gradients when $G$ is performing poorly!
2. **Mode Collapse:**
   * Instead of covering all modes of $p_{\text{data}}(x)$ (e.g., all 10 digits of MNIST), $G(z)$ discovers one single realistic image $x^*$ that currently fools $D(x^*) \approx 1$ and maps every $z$ to that single mode $x^*$.
   * Then in Step 4, $D$ notices that $x^*$ is flooding $\mathcal{D}_F$ and sets $D(x^*) \to 0$, causing $G$ to jump all its mass to a different single mode, playing an endless game of cat-and-mouse (oscillating without converging).
   * **Why VAEs don't suffer from Mode Collapse, while GANs do:** VAEs maximize $\sum_{i=1}^N \log p_\theta(x_i)$ over *every* training example $x_i$ (mode-covering $D_{\text{KL}}(p_{\text{data}} \| p_\theta)$), so ignoring even one training image incurs an infinite penalty. GANs with the $-\log D(G(z))$ loss largely optimize reverse-KL-like behavior (mode-seeking), preferring to generate a few ultra-sharp realistic samples rather than blurry averages across modes.

---

### 3.6 Architectural & Representation Advances (DCGAN, WGAN, CycleGAN, StyleGAN)
In Lecture 19 Parts 2 & 3, Professor Levine surveys the landmark stabilization techniques and architectures that made GANs practical:
* **DCGAN (Deep Convolutional GAN):** Replaced pooling layers with strided convolutions (discriminator) and fractional-strided/transposed convolutions (generator), used BatchNorm in both $G$ and $D$, removed fully connected hidden layers, and used ReLU in $G$ (with $\tanh$ output) and LeakyReLU in $D$. Also demonstrated **latent space vector arithmetic** (e.g., $z_{\text{smiling woman}} - z_{\text{neutral woman}} + z_{\text{neutral man}} = z_{\text{smiling man}}$).
* **Wasserstein GAN (WGAN) & Gradient Penalty (WGAN-GP):** Replaces the Jensen-Shannon divergence (which has zero gradient when $p_{\text{data}}$ and $p_G$ reside on non-overlapping low-dimensional manifolds) with the **Earth Mover's (Wasserstein-1) Distance**:
  $$\min_G \max_{\|D\|_L \le 1} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p(z)}[D(G(z))]$$
  where the critic $D(x)$ is unconstrained (no sigmoid at the output) and forced to be 1-Lipschitz continuous ($\|\nabla_x D(x)\| \le 1$) via weight clipping or a gradient penalty $(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2$. This provides smooth, non-vanishing gradients everywhere!
* **Conditional GANs & CycleGAN (Unpaired Image-to-Image Translation):** Conditions $G(x, z)$ on an input image $x$ from domain $A$ (e.g., horses) to translate it into domain $B$ (e.g., zebras) using a domain discriminator $D_B(y)$ plus a **Cycle-Consistency Loss** $\|F(G(x)) - x\|_1$ so the translation preserves the specific pose and background without needing paired training images.
* **Progressive GAN & StyleGAN:** Grows resolution progressively or injects the latent vector $z \to w$ at every spatial resolution scale via **Adaptive Instance Normalization (AdaIN)**, disentangling coarse style (pose, face shape), middle style (facial features), and fine style (hair/skin color).

---

## 4. Master Synthesis: Connecting VAEs, Normalizing Flows, GANs, DDPMs, and Latent Diffusion

To tie together Lectures 17, 18, and 19 with modern Diffusion and Latent Diffusion Models, this comparison matrix contrasts every generative paradigm on its mathematical objective, latent structure, and empirical behavior:

| Model Family | Exact or Approximate $\log p_\theta(x)$? | Latent Space Structure | Training Objective | Generation Speed | Sample Sharpness & Mode Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Autoregressive**<br>*(PixelCNN, GPT)* | **Exact** via chain rule $\prod_t p(x^{(t)} | x^{(<t)})$ | None (or implicit hidden states) | Exact MLE: $\max_\theta \sum_t \log p_\theta(x^{(t)} | x^{(<t)})$ | **Very Slow** ($\mathcal{O}(n)$ sequential pixel steps) | Sharp samples, full mode coverage; no global latent vector $z$. |
| **Variational Autoencoder (VAE)** | **Lower Bound (ELBO)** via encoder $q_\phi(z|x)$ | Compressed vector or 3D spatial tensor ($\dim(z) \ll \dim(x)$) | $\max_{\theta,\phi} \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))$ | **Very Fast** ($1$ forward pass through Decoder) | Covers all modes, smooth latent space, but tends to be **blurry** due to single-step Gaussian assumption & $L_2$ loss. |
| **Normalizing Flows**<br>*(NICE, RealNVP, Glow)* | **Exact** via Change of Variables | Exact same dimension as data ($\dim(z) = \dim(x)$); strictly invertible $f_\theta$ | $\max_\theta \log p(f_\theta^{-1}(x)) - \log \left|\det \frac{df_\theta(z)}{dz}\right|$ | **Fast** ($1$ forward pass through $f_\theta$) | Covers all modes with exact likelihood, but requires massive networks because $\dim(z)=\dim(x)$ and layers must be invertible. |
| **GANs**<br>*(DCGAN, WGAN, StyleGAN)* | **Implicit** (No likelihood $p_\theta(x)$ available) | Compressed vector $z \in \mathbb{R}^D$ ($\dim(z) \ll \dim(x)$) | Minimax game: $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ | **Very Fast** ($1$ forward pass through $G_\theta$) | **Extremely sharp** samples, but unstable training and prone to **mode collapse** (dropping modes of $p_{\text{data}}$). |
| **Diffusion Models (DDPM)** | **Lower Bound (ELBO)** over $T$-step Markov chain | Fixed $T$-step Markov chain $\mathbf{x}_{1:T}$ with $\dim(\mathbf{x}_t) = \dim(\mathbf{x}_0)$ | Simplified ELBO: $\min_\theta \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\big[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\big]$ | **Slow** ($T$ sequential U-Net passes in pixel space) | **State-of-the-art sharpness AND full mode coverage** (no mode collapse, stable likelihood-based objective). |
| **Latent Diffusion (LDM / Stable Diffusion)** | **Two-Stage ELBO** (VAE ELBO + Diffusion ELBO in $z$-space) | Fully Convolutional VAE 3D tensor $z_0 \in \mathbb{R}^{H' \times W' \times C}$ + $T$-step chain $z_{1:T}$ | Stage 1: Train Conv-VAE.<br>Stage 2: $\min_\theta \mathbb{E}\big[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)\|^2\big]$ | **Fast / Practical** ($T$ steps run in compressed $64 \times 64 \times 4$ latent space + $1$ VAE decode) | Combines VAE spatial compression, DDPM multi-step sharpness, and stable mode-covering training. |