# LDM Section Presentation Script (Slides 8–13)

**Target duration:** Under 3 minutes (~2:35–2:45 at 145 wpm, ~385 words)

---

## Slide 8 — Latent Diffusion Models *(0:00 – 0:40)*

Moving to Latent Diffusion Models gives us speed and a smoother, compact distribution for stable denoising. Because the latent space is compact, the U-Net estimates the score function much more easily.

In Stage 1, we train an autoencoder using reconstruction, patch-adversarial, and weak KL losses—gently pulling latents toward a standard normal while preserving perceptual detail. In Stage 2, the autoencoder is frozen and the U-Net trains jointly with a conditioning encoder. The main drawback is sensitivity to the autoencoder: if the manifold isn't smooth, a small latent error gets amplified by the decoder.

---

## Slide 9 — Conditioning: Cross-Attention vs. Concatenation *(0:40 – 1:10)*

At every reverse step, conditioning is fed into the U-Net to steer generation. For semantic inputs like text prompts, style transfer, or layouts, an encoder maps conditions into key-and-value tokens for cross-attention, handling prompts like the street sign or squirrel. Conversely, for spatially aligned tasks needing exact location and detail preservation—like object removal, four-times super-resolution, or semantic synthesis—we concatenate the input directly onto the noisy latent.

---

## Slide 10 — Operations in Latent Space *(1:10 – 1:55)*

Next, several useful operations happen directly in latent space. First, latent rescaling: weak KL leaves latent variance uncalibrated, so we divide by the batch standard deviation for unit-variance diffusion, then rescale back before decoding. Second, because the U-Net and decoder are convolutional, larger 2D latent grids scale 256-squared models up to 512-by-1024 or 1024-squared without retraining when spatial conditioning guides layout. Third, classifier-free guidance extrapolates conditional minus unconditional noise predictions. Below right, on our latent U-Net, moderate scales like 3.0 and 7.5 yield crisp, accurate digits, whereas 15.0 clips and oversaturates pixels. Above right, linear latent interpolation smoothly morphs a 3 into an 8 with no dead zones, avoiding the double-exposure ghosting of pixel blending.

---

## Slide 11 — Pixel-Space vs. Latent-Space Diffusion *(1:55 – 2:20)*

Comparing identical U-Nets on MNIST on a single T4 GPU highlights the speedup. Shifting from 32-by-32-by-1 pixels to an 8-by-8-by-4 latent cuts training time from 51.4 to 6.8 seconds per epoch—7.6 times faster—and drops peak memory from 1,152 to 83 megabytes, 13.8 times less, while enabling flexible cross-attention, with quality capped only by the autoencoder.

---

## Slides 12 & 13 — Limitations, Conclusion & Q&A *(2:20 – 2:45)*

Still, limitations remain: sequential sampling is slow and faster ODE solvers lose some detail; strong guidance oversaturates and complex prompts stay challenging; and large models risk memorization and social bias, while our experiments tested MNIST on one GPU. In conclusion, combining DDPM denoising with a compact latent space makes high-resolution, controllable generation practical. Thank you—any questions?
