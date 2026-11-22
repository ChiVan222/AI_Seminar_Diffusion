"""Simulation engines for Unconditioned vs Conditioned LDM, Latent Forward/Backward DDPM,
and Pixel-Space Forward/Backward DDPM.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import build_scheduler


ProgressCB = Callable[[float, str], None] | None


@dataclass
class LatentStepSnapshot:
    step_idx: int
    timestep: int
    alpha_bar: float
    sqrt_alpha_bar: float
    sqrt_one_minus_alpha_bar: float
    log_snr: float
    z_t: np.ndarray  # (4, H, W) float32
    z0_hat: np.ndarray | None  # (4, H, W) float32
    eps_norm_map: np.ndarray | None  # (H, W) float32
    decoded_zt: np.ndarray  # (ImgH, ImgW, 3) uint8
    decoded_z0_hat: np.ndarray | None  # (ImgH, ImgW, 3) uint8
    guidance_map: np.ndarray | None  # (H, W) float32
    mean: float
    std: float
    l2_norm: float


@dataclass
class BranchTrajectory:
    name: str
    label: str
    prompt: str
    guidance_scale: float
    steps: list[LatentStepSnapshot] = field(default_factory=list)


@dataclass
class UncondVsCondResult:
    prompt: str
    negative_prompt: str
    seed: int
    sampler: str
    num_steps: int
    resolution: int
    branches: list[BranchTrajectory]
    divergence_l2: dict[str, list[float]]
    cosine_sim: dict[str, list[float]]
    guidance_delta_norm: dict[str, list[float]]
    elapsed_sec: float
    peak_vram_mb: float
    ms_per_step: float


@dataclass
class LatentForwardBackwardResult:
    source_rgb: np.ndarray
    source_recon_rgb: np.ndarray
    z0_clean: np.ndarray
    forward_mode: str
    reverse_mode: str
    prompt: str
    guidance_scale: float
    sampler: str
    t_max: int
    num_steps: int
    forward_steps: list[LatentStepSnapshot]
    backward_steps: list[LatentStepSnapshot]
    recon_psnr_db: float
    elapsed_sec: float
    peak_vram_mb: float


@dataclass
class PixelStepSnapshot:
    step_idx: int
    timestep: int
    alpha_bar: float
    sqrt_alpha_bar: float
    sqrt_one_minus_alpha_bar: float
    x_t_rgb: np.ndarray  # (H, W, 3) uint8
    x0_hat_rgb: np.ndarray | None  # (H, W, 3) uint8
    x_t_mean: float
    x_t_std: float
    x_t_flat_sample: np.ndarray  # 1D float32 sample for histogram plotting


@dataclass
class PixelDDPMResult:
    mode: str
    sampler: str
    seed: int
    num_steps: int
    t_max: int
    source_rgb: np.ndarray | None
    forward_steps: list[PixelStepSnapshot]
    backward_steps: list[PixelStepSnapshot]
    elapsed_sec: float
    peak_vram_mb: float
    ms_per_step: float
    state_dims: int


def _get_pipe_device_dtype(pipe: Any) -> tuple[torch.device, torch.dtype]:
    unet = getattr(pipe, "unet", None)
    if unet is not None and hasattr(unet, "parameters"):
        try:
            p = next(unet.parameters())
            return p.device, p.dtype
        except StopIteration:
            pass
    dev = getattr(pipe, "device", torch.device("cpu"))
    dt = getattr(pipe, "dtype", torch.float32)
    return torch.device(dev), dt


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    gen_dev = device.type if device.type == "cuda" else "cpu"
    return torch.Generator(device=gen_dev).manual_seed(int(seed))


def _reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _read_peak_vram_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    return 0.0


def _schedule_stats(alphas_cumprod: torch.Tensor, timestep: int, is_pure_prior: bool = False) -> tuple[float, float, float, float]:
    if is_pure_prior:
        return 0.0, 0.0, 1.0, -10.0
    idx = min(max(int(timestep), 0), int(alphas_cumprod.numel()) - 1)
    ab = float(alphas_cumprod[idx].item())
    ab = min(max(ab, 0.0), 1.0)
    sqrt_ab = math.sqrt(ab)
    sqrt_1mab = math.sqrt(max(0.0, 1.0 - ab))
    log_snr = math.log(max(ab, 1e-6) / max(1.0 - ab, 1e-6))
    return ab, sqrt_ab, sqrt_1mab, log_snr


def _tensor_stats(t: torch.Tensor) -> tuple[float, float, float]:
    tf = t.detach().float()
    mean = float(tf.mean().item())
    std = float(tf.std(unbiased=False).item())
    l2 = float(torch.linalg.vector_norm(tf).item())
    return mean, std, l2


def _subsample_flat(t: torch.Tensor, max_elems: int = 8192, seed: int = 0) -> np.ndarray:
    flat = t.detach().float().reshape(-1).cpu().numpy()
    if flat.size <= max_elems:
        return flat.astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.size, size=max_elems, replace=False)
    return flat[idx].astype(np.float32)


def _tensor_to_uint8_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert `(B, 3, H, W)` tensor in `[-1, 1]` to `(H, W, 3)` uint8 numpy array."""
    x = img_tensor[0].detach().float().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()


def _prepare_rgb_array(image: Image.Image | np.ndarray, size: int) -> np.ndarray:
    """Center-crop and resize an RGB PIL image or ndarray to `(size, size, 3)` uint8."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).round().astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        pil = Image.fromarray(arr, mode="RGB")
    else:
        pil = image.convert("RGB")

    w, h = pil.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        pil = pil.crop((left, top, left + side, top + side))
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(pil, dtype=np.uint8, copy=True)


def make_procedural_sample_image(kind: str = "geometric", size: int = 512) -> np.ndarray:
    """Create a crisp procedural RGB image for zero-setup Forward-vs-Backward tests."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    if kind == "landscape":
        sky_r = 0.95 * (1.0 - yy) + 0.18 * yy
        sky_g = 0.48 * (1.0 - yy) + 0.28 * yy + 0.25 * np.sin(math.pi * yy)
        sky_b = 0.30 * (1.0 - yy) + 0.55 * yy
        img = np.stack([sky_r, sky_g, sky_b], axis=-1)
        # Sun disc
        sun = ((xx - 0.68) ** 2 + (yy - 0.32) ** 2) < 0.012
        img[sun] = np.array([1.0, 0.92, 0.55], dtype=np.float32)
        # Mountain ridges
        ridge1 = 0.58 + 0.09 * np.sin(xx * 9.0 + 0.4) - 0.06 * np.abs(xx - 0.35)
        ridge2 = 0.72 + 0.06 * np.cos(xx * 13.0)
        img[yy > ridge1] = np.array([0.16, 0.24, 0.38], dtype=np.float32)
        img[yy > ridge2] = np.array([0.07, 0.14, 0.22], dtype=np.float32)
        return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    # Default: Bauhaus-style high-contrast geometric composition
    img = np.ones((size, size, 3), dtype=np.float32) * np.array([0.94, 0.93, 0.89], dtype=np.float32)
    circle = ((xx - 0.36) ** 2 + (yy - 0.42) ** 2) < 0.055
    img[circle] = np.array([0.10, 0.27, 0.55], dtype=np.float32)
    square = (xx > 0.52) & (xx < 0.85) & (yy > 0.18) & (yy < 0.51)
    img[square] = np.array([0.78, 0.20, 0.18], dtype=np.float32)
    tri = (yy > 0.55) & (yy < 0.88) & (xx > 0.22 + (0.88 - yy) * 0.65) & (xx < 0.78 - (0.88 - yy) * 0.65)
    img[tri] = np.array([0.0, 0.47, 0.50], dtype=np.float32)
    bar = (yy > 0.82) & (yy < 0.87) & (xx > 0.12) & (xx < 0.88)
    img[bar] = np.array([0.12, 0.14, 0.18], dtype=np.float32)
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


@torch.inference_mode()
def encode_prompt_sd(pipe: Any, prompt: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode a text prompt (or empty string ``""`` for null token) with CLIP."""
    max_len = getattr(pipe.tokenizer, "model_max_length", 77)
    tokens = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    out = pipe.text_encoder(input_ids)
    emb = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
    return emb.to(device=device, dtype=dtype)


@torch.inference_mode()
def decode_latent_to_rgb(pipe: Any, z: torch.Tensor) -> np.ndarray:
    """Decode `(1, 4, H, W)` SD latent tensor to `(ImgH, ImgW, 3)` uint8 RGB."""
    scaling = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
    vae_dtype = next(pipe.vae.parameters()).dtype if hasattr(pipe.vae, "parameters") else z.dtype
    vae_dev = next(pipe.vae.parameters()).device if hasattr(pipe.vae, "parameters") else z.device
    decoded = pipe.vae.decode((z.to(device=vae_dev, dtype=vae_dtype) / scaling)).sample
    return _tensor_to_uint8_rgb(decoded)


@torch.inference_mode()
def encode_image_to_latent(pipe: Any, image: Image.Image | np.ndarray, size: int = 512) -> tuple[torch.Tensor, np.ndarray]:
    """Encode an RGB image to `(1, 4, size//8, size//8)` SD latent and return `(z0, source_rgb)`."""
    src_rgb = _prepare_rgb_array(image, size)
    dev, dt = _get_pipe_device_dtype(pipe)
    vae_dtype = next(pipe.vae.parameters()).dtype if hasattr(pipe.vae, "parameters") else dt
    vae_dev = next(pipe.vae.parameters()).device if hasattr(pipe.vae, "parameters") else dev

    x = torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    x = x.to(device=vae_dev, dtype=vae_dtype)
    latent_dist = pipe.vae.encode(x).latent_dist
    scaling = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
    z0 = latent_dist.mean * scaling
    return z0.to(device=dev, dtype=dt), src_rgb


def _extract_z0_hat(step_out: Any, z_t: torch.Tensor, eps_pred: torch.Tensor, ab: float) -> torch.Tensor:
    z0_hat = getattr(step_out, "pred_original_sample", None)
    if z0_hat is not None:
        return z0_hat
    sqrt_ab = math.sqrt(max(ab, 1e-6))
    sqrt_1mab = math.sqrt(max(1.0 - ab, 0.0))
    return (z_t - sqrt_1mab * eps_pred) / sqrt_ab


def _compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-8:
        return 99.0
    return float(10.0 * math.log10((255.0 ** 2) / mse))


@torch.inference_mode()
def simulate_uncond_vs_cond_ldm(
    sd_pipe: Any,
    prompt: str,
    seed: int = 42,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    extra_guidance_scale: float | None = None,
    sampler: str = "DDIM",
    resolution: int = 512,
    negative_prompt: str = "",
    progress_cb: ProgressCB = None,
) -> UncondVsCondResult:
    """Simulate parallel Unconditioned (empty prompt) vs Conditioned LDM from identical initial noise z_T."""
    dev, dt = _get_pipe_device_dtype(sd_pipe)
    _reset_peak_vram(dev)
    t_start = time.perf_counter()

    scheduler = build_scheduler(sd_pipe.scheduler.config, sampler=sampler, is_latent=True)
    scheduler.set_timesteps(int(num_steps), device=dev)
    timesteps = scheduler.timesteps
    alphas_cumprod = scheduler.alphas_cumprod.detach().cpu()

    lat_h = resolution // 8
    lat_w = resolution // 8
    in_ch = int(getattr(sd_pipe.unet.config, "in_channels", 4))

    # Fixed CPU generator for reproducible initial z_T across CPU & GPU
    cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    z_init = torch.randn((1, in_ch, lat_h, lat_w), generator=cpu_gen, dtype=torch.float32)
    init_sigma = float(getattr(scheduler, "init_noise_sigma", 1.0))
    z_init = (z_init * init_sigma).to(device=dev, dtype=dt)

    emb_null = encode_prompt_sd(sd_pipe, negative_prompt or "", dev, dt)
    emb_cond = encode_prompt_sd(sd_pipe, prompt, dev, dt)

    branch_specs: list[tuple[str, str, str, float]] = [
        ("uncond", "Unconditioned (∅)", negative_prompt or "∅ (unconditional)", 0.0),
        ("cond", f"Conditioned (α={guidance_scale:g})", prompt, float(guidance_scale)),
    ]
    if extra_guidance_scale is not None:
        branch_specs.append(
            (
                "cond_extra",
                f"Conditioned (α={extra_guidance_scale:g})",
                prompt,
                float(extra_guidance_scale),
            )
        )

    branches = [
        BranchTrajectory(name=n, label=lbl, prompt=pr, guidance_scale=gs)
        for n, lbl, pr, gs in branch_specs
    ]
    z_states = [z_init.clone() for _ in branches]

    # Step 0: shared pure noise z_T
    t_init_label = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    ab0, sab0, s1mab0, lsnr0 = _schedule_stats(alphas_cumprod, t_init_label, is_pure_prior=True)
    z0_np = z_init[0].detach().float().cpu().numpy()
    decoded_init = decode_latent_to_rgb(sd_pipe, z_init)
    m0, s0, l0 = _tensor_stats(z_init)

    for b in branches:
        b.steps.append(
            LatentStepSnapshot(
                step_idx=0,
                timestep=t_init_label,
                alpha_bar=ab0,
                sqrt_alpha_bar=sab0,
                sqrt_one_minus_alpha_bar=s1mab0,
                log_snr=lsnr0,
                z_t=z0_np.copy(),
                z0_hat=None,
                eps_norm_map=None,
                decoded_zt=decoded_init.copy(),
                decoded_z0_hat=None,
                guidance_map=None,
                mean=m0,
                std=s0,
                l2_norm=l0,
            )
        )

    divergence_l2: dict[str, list[float]] = {b.name: [0.0] for b in branches[1:]}
    cosine_sim: dict[str, list[float]] = {b.name: [1.0] for b in branches[1:]}
    guidance_delta_norm: dict[str, list[float]] = {b.name: [0.0] for b in branches[1:]}

    total_steps = len(timesteps)
    for step_i, t in enumerate(timesteps):
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        if progress_cb is not None:
            progress_cb(step_i / max(total_steps, 1), f"Denoising step {step_i + 1}/{total_steps} (t={t_int})")

        # Batch UNet evaluation:
        # slot 0: uncond branch
        # slots 1,2: branch 1 (null, cond)
        # slots 3,4: optional branch 2 (null, cond)
        latent_inputs = [scheduler.scale_model_input(z_states[0], t)]
        prompt_inputs = [emb_null]
        for idx in range(1, len(branches)):
            scaled_z = scheduler.scale_model_input(z_states[idx], t)
            latent_inputs.extend([scaled_z, scaled_z])
            prompt_inputs.extend([emb_null, emb_cond])

        batch_latents = torch.cat(latent_inputs, dim=0)
        batch_emb = torch.cat(prompt_inputs, dim=0)
        eps_all = sd_pipe.unet(batch_latents, t, encoder_hidden_states=batch_emb).sample

        eps_per_branch: list[torch.Tensor] = [eps_all[0:1]]
        guidance_maps: list[np.ndarray | None] = [None]
        cursor = 1
        for idx in range(1, len(branches)):
            eps_u = eps_all[cursor : cursor + 1]
            eps_c = eps_all[cursor + 1 : cursor + 2]
            cursor += 2
            alpha = branches[idx].guidance_scale
            diff = eps_c - eps_u
            eps_guided = eps_u + alpha * diff
            eps_per_branch.append(eps_guided)
            gmap = torch.linalg.vector_norm(diff.float(), dim=1)[0].cpu().numpy().astype(np.float32)
            guidance_maps.append(gmap)
            guidance_delta_norm[branches[idx].name].append(float(torch.linalg.vector_norm(diff.float()).item()))

        ab, sab, s1mab, lsnr = _schedule_stats(alphas_cumprod, t_int, is_pure_prior=False)

        for idx, branch in enumerate(branches):
            # Identical stochastic DDPM variance seed across branches so divergence = pure conditioning effect
            step_gen = _make_generator(seed + 10000 + step_i, dev)
            step_out = scheduler.step(eps_per_branch[idx], t, z_states[idx], generator=step_gen)
            z_next = step_out.prev_sample
            z0_hat = _extract_z0_hat(step_out, z_states[idx], eps_per_branch[idx], ab)
            z_states[idx] = z_next

            dec_zt = decode_latent_to_rgb(sd_pipe, z_next)
            dec_z0 = decode_latent_to_rgb(sd_pipe, z0_hat)
            eps_map = torch.linalg.vector_norm(eps_per_branch[idx].float(), dim=1)[0].cpu().numpy().astype(np.float32)
            mean_v, std_v, l2_v = _tensor_stats(z_next)

            branch.steps.append(
                LatentStepSnapshot(
                    step_idx=step_i + 1,
                    timestep=t_int,
                    alpha_bar=ab,
                    sqrt_alpha_bar=sab,
                    sqrt_one_minus_alpha_bar=s1mab,
                    log_snr=lsnr,
                    z_t=z_next[0].detach().float().cpu().numpy(),
                    z0_hat=z0_hat[0].detach().float().cpu().numpy(),
                    eps_norm_map=eps_map,
                    decoded_zt=dec_zt,
                    decoded_z0_hat=dec_z0,
                    guidance_map=guidance_maps[idx],
                    mean=mean_v,
                    std=std_v,
                    l2_norm=l2_v,
                )
            )

        z_ref = z_states[0].detach().float().reshape(1, -1)
        for idx in range(1, len(branches)):
            bname = branches[idx].name
            z_cur = z_states[idx].detach().float().reshape(1, -1)
            div_val = float(torch.linalg.vector_norm(z_cur - z_ref).item())
            cos_val = float(F.cosine_similarity(z_cur, z_ref, dim=-1).item())
            divergence_l2[bname].append(div_val)
            cosine_sim[bname].append(cos_val)

    if progress_cb is not None:
        progress_cb(1.0, "Completed Unconditioned vs Conditioned simulation")

    elapsed = time.perf_counter() - t_start
    peak_vram = _read_peak_vram_mb(dev)
    ms_per_step = (elapsed * 1000.0) / max(total_steps, 1)

    return UncondVsCondResult(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        sampler=sampler,
        num_steps=total_steps,
        resolution=resolution,
        branches=branches,
        divergence_l2=divergence_l2,
        cosine_sim=cosine_sim,
        guidance_delta_norm=guidance_delta_norm,
        elapsed_sec=elapsed,
        peak_vram_mb=peak_vram,
        ms_per_step=ms_per_step,
    )


@torch.inference_mode()
def simulate_latent_forward_backward(
    sd_pipe: Any,
    source_rgb: Image.Image | np.ndarray,
    seed: int = 42,
    t_max_frac: float = 0.8,
    num_steps: int = 20,
    forward_mode: str = "closed_form",
    reverse_mode: str = "conditioned",
    prompt: str = "",
    guidance_scale: float = 7.5,
    sampler: str = "DDIM",
    resolution: int = 512,
    progress_cb: ProgressCB = None,
) -> LatentForwardBackwardResult:
    """Simulate Forward noising (0 -> t_max) and Backward denoising (t_max -> 0) in SD latent space."""
    if forward_mode not in ("closed_form", "markov"):
        raise ValueError(f"Unsupported forward_mode '{forward_mode}'")
    if reverse_mode not in ("unconditioned", "conditioned"):
        raise ValueError(f"Unsupported reverse_mode '{reverse_mode}'")

    dev, dt = _get_pipe_device_dtype(sd_pipe)
    _reset_peak_vram(dev)
    t_start = time.perf_counter()

    z0, src_rgb = encode_image_to_latent(sd_pipe, source_rgb, size=resolution)
    source_recon_rgb = decode_latent_to_rgb(sd_pipe, z0)

    scheduler = build_scheduler(sd_pipe.scheduler.config, sampler=sampler, is_latent=True)
    scheduler.set_timesteps(int(num_steps), device=dev)
    all_ts = [int(t.item()) if torch.is_tensor(t) else int(t) for t in scheduler.timesteps]
    alphas_cumprod = scheduler.alphas_cumprod.detach().cpu()
    betas = scheduler.betas.detach().cpu()

    num_train_ts = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    t_cut = int(round(min(max(t_max_frac, 0.05), 1.0) * (num_train_ts - 1)))
    if t_max_frac >= 0.999:
        rev_ts = list(all_ts)
    else:
        rev_ts = [t for t in all_ts if t <= t_cut]
        if len(rev_ts) < 2:
            rev_ts = list(all_ts[-2:])

    t_max = rev_ts[0]
    fwd_nonzero_ts = sorted({t for t in rev_ts if t > 0})
    if not fwd_nonzero_ts:
        fwd_nonzero_ts = [t_max]
    elif t_max not in fwd_nonzero_ts:
        fwd_nonzero_ts.append(t_max)

    # Forward step 0: clean encoded z0
    m0, s0, l0 = _tensor_stats(z0)
    forward_steps: list[LatentStepSnapshot] = [
        LatentStepSnapshot(
            step_idx=0,
            timestep=0,
            alpha_bar=1.0,
            sqrt_alpha_bar=1.0,
            sqrt_one_minus_alpha_bar=0.0,
            log_snr=10.0,
            z_t=z0[0].detach().float().cpu().numpy(),
            z0_hat=z0[0].detach().float().cpu().numpy(),
            eps_norm_map=None,
            decoded_zt=source_recon_rgb.copy(),
            decoded_z0_hat=source_recon_rgb.copy(),
            guidance_map=None,
            mean=m0,
            std=s0,
            l2_norm=l0,
        )
    ]

    total_work = len(fwd_nonzero_ts) + len(rev_ts)
    done_work = 0
    z_at_t_max = z0.clone()

    if forward_mode == "closed_form":
        eps_gen = torch.Generator(device="cpu").manual_seed(int(seed))
        eps_fixed = torch.randn(z0.shape, generator=eps_gen, dtype=torch.float32).to(device=dev, dtype=dt)
        for idx, t_val in enumerate(fwd_nonzero_ts, start=1):
            if progress_cb is not None:
                progress_cb(done_work / max(total_work, 1), f"Forward closed-form jump to t={t_val}")
            t_tensor = torch.tensor([t_val], device=dev, dtype=torch.long)
            z_noisy = scheduler.add_noise(z0, eps_fixed, t_tensor)
            ab, sab, s1mab, lsnr = _schedule_stats(alphas_cumprod, t_val, is_pure_prior=False)
            dec = decode_latent_to_rgb(sd_pipe, z_noisy)
            mv, sv, lv = _tensor_stats(z_noisy)
            forward_steps.append(
                LatentStepSnapshot(
                    step_idx=idx,
                    timestep=t_val,
                    alpha_bar=ab,
                    sqrt_alpha_bar=sab,
                    sqrt_one_minus_alpha_bar=s1mab,
                    log_snr=lsnr,
                    z_t=z_noisy[0].detach().float().cpu().numpy(),
                    z0_hat=None,
                    eps_norm_map=None,
                    decoded_zt=dec,
                    decoded_z0_hat=None,
                    guidance_map=None,
                    mean=mv,
                    std=sv,
                    l2_norm=lv,
                )
            )
            if t_val == t_max:
                z_at_t_max = z_noisy.clone()
            done_work += 1
    else:
        # True Markov chain: z_s = sqrt(1 - beta_s) * z_{s-1} + sqrt(beta_s) * eps_s
        markov_gen = torch.Generator(device="cpu").manual_seed(int(seed))
        z_cur = z0.float().clone()
        record_map = {t_val: idx for idx, t_val in enumerate(fwd_nonzero_ts, start=1)}
        for s in range(0, t_max + 1):
            beta_s = float(betas[min(s, int(betas.numel()) - 1)].item())
            eps_s = torch.randn(z_cur.shape, generator=markov_gen, dtype=torch.float32).to(dev)
            z_cur = math.sqrt(max(0.0, 1.0 - beta_s)) * z_cur + math.sqrt(max(0.0, beta_s)) * eps_s
            if s in record_map:
                if progress_cb is not None:
                    progress_cb(done_work / max(total_work, 1), f"Forward Markov chain at t={s}")
                z_cur_dt = z_cur.to(dtype=dt)
                ab, sab, s1mab, lsnr = _schedule_stats(alphas_cumprod, s, is_pure_prior=False)
                dec = decode_latent_to_rgb(sd_pipe, z_cur_dt)
                mv, sv, lv = _tensor_stats(z_cur_dt)
                forward_steps.append(
                    LatentStepSnapshot(
                        step_idx=record_map[s],
                        timestep=s,
                        alpha_bar=ab,
                        sqrt_alpha_bar=sab,
                        sqrt_one_minus_alpha_bar=s1mab,
                        log_snr=lsnr,
                        z_t=z_cur_dt[0].detach().float().cpu().numpy(),
                        z0_hat=None,
                        eps_norm_map=None,
                        decoded_zt=dec,
                        decoded_z0_hat=None,
                        guidance_map=None,
                        mean=mv,
                        std=sv,
                        l2_norm=lv,
                    )
                )
                if s == t_max:
                    z_at_t_max = z_cur_dt.clone()
                done_work += 1

    # Backward denoising starting from exact terminal forward latent z_{t_max}
    emb_null = encode_prompt_sd(sd_pipe, "", dev, dt)
    use_cond = reverse_mode == "conditioned" and bool(prompt.strip())
    emb_cond = encode_prompt_sd(sd_pipe, prompt, dev, dt) if use_cond else None

    end_fwd = forward_steps[-1]
    backward_steps: list[LatentStepSnapshot] = [
        LatentStepSnapshot(
            step_idx=0,
            timestep=end_fwd.timestep,
            alpha_bar=end_fwd.alpha_bar,
            sqrt_alpha_bar=end_fwd.sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar=end_fwd.sqrt_one_minus_alpha_bar,
            log_snr=end_fwd.log_snr,
            z_t=end_fwd.z_t.copy(),
            z0_hat=None,
            eps_norm_map=None,
            decoded_zt=end_fwd.decoded_zt.copy(),
            decoded_z0_hat=None,
            guidance_map=None,
            mean=end_fwd.mean,
            std=end_fwd.std,
            l2_norm=end_fwd.l2_norm,
        )
    ]

    z_rev = z_at_t_max.clone()
    for step_i, t_val in enumerate(rev_ts):
        if progress_cb is not None:
            progress_cb(done_work / max(total_work, 1), f"Backward denoising step {step_i + 1}/{len(rev_ts)} (t={t_val})")
        t_tensor = torch.tensor(t_val, device=dev, dtype=torch.long)
        scaled_z = scheduler.scale_model_input(z_rev, t_tensor)

        if use_cond and emb_cond is not None:
            eps_both = sd_pipe.unet(
                torch.cat([scaled_z, scaled_z], dim=0),
                t_tensor,
                encoder_hidden_states=torch.cat([emb_null, emb_cond], dim=0),
            ).sample
            eps_u, eps_c = eps_both[0:1], eps_both[1:2]
            diff = eps_c - eps_u
            eps_pred = eps_u + float(guidance_scale) * diff
            gmap = torch.linalg.vector_norm(diff.float(), dim=1)[0].cpu().numpy().astype(np.float32)
        else:
            eps_pred = sd_pipe.unet(scaled_z, t_tensor, encoder_hidden_states=emb_null).sample
            gmap = None

        ab, sab, s1mab, lsnr = _schedule_stats(alphas_cumprod, t_val, is_pure_prior=False)
        step_gen = _make_generator(seed + 20000 + step_i, dev)
        step_out = scheduler.step(eps_pred, t_tensor, z_rev, generator=step_gen)
        z_next = step_out.prev_sample
        z0_hat = _extract_z0_hat(step_out, z_rev, eps_pred, ab)
        z_rev = z_next

        dec_zt = decode_latent_to_rgb(sd_pipe, z_next)
        dec_z0 = decode_latent_to_rgb(sd_pipe, z0_hat)
        eps_map = torch.linalg.vector_norm(eps_pred.float(), dim=1)[0].cpu().numpy().astype(np.float32)
        mv, sv, lv = _tensor_stats(z_next)

        backward_steps.append(
            LatentStepSnapshot(
                step_idx=step_i + 1,
                timestep=t_val,
                alpha_bar=ab,
                sqrt_alpha_bar=sab,
                sqrt_one_minus_alpha_bar=s1mab,
                log_snr=lsnr,
                z_t=z_next[0].detach().float().cpu().numpy(),
                z0_hat=z0_hat[0].detach().float().cpu().numpy(),
                eps_norm_map=eps_map,
                decoded_zt=dec_zt,
                decoded_z0_hat=dec_z0,
                guidance_map=gmap,
                mean=mv,
                std=sv,
                l2_norm=lv,
            )
        )
        done_work += 1

    if progress_cb is not None:
        progress_cb(1.0, "Completed Latent Forward vs Backward simulation")

    psnr_db = _compute_psnr(src_rgb, backward_steps[-1].decoded_zt)
    elapsed = time.perf_counter() - t_start
    peak_vram = _read_peak_vram_mb(dev)

    return LatentForwardBackwardResult(
        source_rgb=src_rgb,
        source_recon_rgb=source_recon_rgb,
        z0_clean=z0[0].detach().float().cpu().numpy(),
        forward_mode=forward_mode,
        reverse_mode=reverse_mode,
        prompt=prompt if use_cond else "",
        guidance_scale=float(guidance_scale) if use_cond else 0.0,
        sampler=sampler,
        t_max=t_max,
        num_steps=len(rev_ts),
        forward_steps=forward_steps,
        backward_steps=backward_steps,
        recon_psnr_db=psnr_db,
        elapsed_sec=elapsed,
        peak_vram_mb=peak_vram,
    )


@torch.inference_mode()
def simulate_pixel_ddpm(
    ddpm_pipe: Any,
    mode: str = "reverse_only",
    source_rgb: Image.Image | np.ndarray | None = None,
    seed: int = 42,
    num_steps: int = 30,
    t_max_frac: float = 0.75,
    sampler: str = "DDIM",
    progress_cb: ProgressCB = None,
) -> PixelDDPMResult:
    """Simulate Pixel-Space DDPM (Reverse-only generation or Forward->Backward round-trip)."""
    if mode not in ("reverse_only", "forward_backward"):
        raise ValueError(f"Unsupported pixel DDPM mode '{mode}'")

    dev, dt = _get_pipe_device_dtype(ddpm_pipe)
    _reset_peak_vram(dev)
    t_start = time.perf_counter()

    sample_size = getattr(ddpm_pipe.unet.config, "sample_size", 256)
    img_size = int(sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size)
    in_ch = int(getattr(ddpm_pipe.unet.config, "in_channels", 3))
    state_dims = in_ch * img_size * img_size

    scheduler = build_scheduler(ddpm_pipe.scheduler.config, sampler=sampler, is_latent=False)
    scheduler.set_timesteps(int(num_steps), device=dev)
    all_ts = [int(t.item()) if torch.is_tensor(t) else int(t) for t in scheduler.timesteps]
    alphas_cumprod = scheduler.alphas_cumprod.detach().cpu()

    forward_steps: list[PixelStepSnapshot] = []
    backward_steps: list[PixelStepSnapshot] = []
    src_rgb_arr: np.ndarray | None = None

    if mode == "reverse_only":
        cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
        x_t = torch.randn((1, in_ch, img_size, img_size), generator=cpu_gen, dtype=torch.float32).to(device=dev, dtype=dt)
        t_init_label = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        ab0, sab0, s1mab0, _ = _schedule_stats(alphas_cumprod, t_init_label, is_pure_prior=True)
        m0, s0, _ = _tensor_stats(x_t)

        backward_steps.append(
            PixelStepSnapshot(
                step_idx=0,
                timestep=t_init_label,
                alpha_bar=ab0,
                sqrt_alpha_bar=sab0,
                sqrt_one_minus_alpha_bar=s1mab0,
                x_t_rgb=_tensor_to_uint8_rgb(x_t),
                x0_hat_rgb=None,
                x_t_mean=m0,
                x_t_std=s0,
                x_t_flat_sample=_subsample_flat(x_t, seed=seed),
            )
        )

        total = len(all_ts)
        for step_i, t_val in enumerate(all_ts):
            if progress_cb is not None:
                progress_cb(step_i / max(total, 1), f"Pixel reverse step {step_i + 1}/{total} (t={t_val})")
            t_tensor = torch.tensor(t_val, device=dev, dtype=torch.long)
            eps = ddpm_pipe.unet(x_t, t_tensor).sample
            ab, sab, s1mab, _ = _schedule_stats(alphas_cumprod, t_val, is_pure_prior=False)
            step_gen = _make_generator(seed + 30000 + step_i, dev)
            step_out = scheduler.step(eps, t_tensor, x_t, generator=step_gen)
            x_next = step_out.prev_sample
            x0_hat = _extract_z0_hat(step_out, x_t, eps, ab)
            x_t = x_next
            mv, sv, _ = _tensor_stats(x_t)
            backward_steps.append(
                PixelStepSnapshot(
                    step_idx=step_i + 1,
                    timestep=t_val,
                    alpha_bar=ab,
                    sqrt_alpha_bar=sab,
                    sqrt_one_minus_alpha_bar=s1mab,
                    x_t_rgb=_tensor_to_uint8_rgb(x_t),
                    x0_hat_rgb=_tensor_to_uint8_rgb(x0_hat),
                    x_t_mean=mv,
                    x_t_std=sv,
                    x_t_flat_sample=_subsample_flat(x_t, seed=seed + step_i + 1),
                )
            )
        t_max = t_init_label
        eff_steps = total
    else:
        # Prepare or generate an in-distribution clean pixel image x0
        if source_rgb is None:
            if progress_cb is not None:
                progress_cb(0.05, "Generating clean Pixel DDPM sample as x_0...")
            gen_sched = build_scheduler(ddpm_pipe.scheduler.config, sampler="DDIM", is_latent=False)
            gen_steps = min(max(int(num_steps), 10), 20)
            gen_sched.set_timesteps(gen_steps, device=dev)
            g_cpu = torch.Generator(device="cpu").manual_seed(int(seed) + 777)
            x0 = torch.randn((1, in_ch, img_size, img_size), generator=g_cpu, dtype=torch.float32).to(device=dev, dtype=dt)
            for t_gen in gen_sched.timesteps:
                eps_g = ddpm_pipe.unet(x0, t_gen).sample
                x0 = gen_sched.step(eps_g, t_gen, x0).prev_sample
            x0 = x0.clamp(-1.0, 1.0)
            src_rgb_arr = _tensor_to_uint8_rgb(x0)
        else:
            src_rgb_arr = _prepare_rgb_array(source_rgb, img_size)
            x0 = (torch.from_numpy(src_rgb_arr).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0).to(device=dev, dtype=dt)

        num_train_ts = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        t_cut = int(round(min(max(t_max_frac, 0.05), 1.0) * (num_train_ts - 1)))
        if t_max_frac >= 0.999:
            rev_ts = list(all_ts)
        else:
            rev_ts = [t for t in all_ts if t <= t_cut]
            if len(rev_ts) < 2:
                rev_ts = list(all_ts[-2:])

        t_max = rev_ts[0]
        fwd_nonzero_ts = sorted({t for t in rev_ts if t > 0})
        if not fwd_nonzero_ts:
            fwd_nonzero_ts = [t_max]
        elif t_max not in fwd_nonzero_ts:
            fwd_nonzero_ts.append(t_max)

        m0, s0, _ = _tensor_stats(x0)
        forward_steps.append(
            PixelStepSnapshot(
                step_idx=0,
                timestep=0,
                alpha_bar=1.0,
                sqrt_alpha_bar=1.0,
                sqrt_one_minus_alpha_bar=0.0,
                x_t_rgb=src_rgb_arr.copy(),
                x0_hat_rgb=src_rgb_arr.copy(),
                x_t_mean=m0,
                x_t_std=s0,
                x_t_flat_sample=_subsample_flat(x0, seed=seed),
            )
        )

        eps_cpu = torch.Generator(device="cpu").manual_seed(int(seed))
        eps_fixed = torch.randn(x0.shape, generator=eps_cpu, dtype=torch.float32).to(device=dev, dtype=dt)
        total_work = len(fwd_nonzero_ts) + len(rev_ts)
        done = 0
        x_at_t_max = x0.clone()

        for idx, t_val in enumerate(fwd_nonzero_ts, start=1):
            if progress_cb is not None:
                progress_cb(0.15 + 0.85 * (done / max(total_work, 1)), f"Pixel forward noising t={t_val}")
            t_tensor = torch.tensor([t_val], device=dev, dtype=torch.long)
            x_noisy = scheduler.add_noise(x0, eps_fixed, t_tensor)
            ab, sab, s1mab, _ = _schedule_stats(alphas_cumprod, t_val, is_pure_prior=False)
            mv, sv, _ = _tensor_stats(x_noisy)
            forward_steps.append(
                PixelStepSnapshot(
                    step_idx=idx,
                    timestep=t_val,
                    alpha_bar=ab,
                    sqrt_alpha_bar=sab,
                    sqrt_one_minus_alpha_bar=s1mab,
                    x_t_rgb=_tensor_to_uint8_rgb(x_noisy),
                    x0_hat_rgb=None,
                    x_t_mean=mv,
                    x_t_std=sv,
                    x_t_flat_sample=_subsample_flat(x_noisy, seed=seed + idx),
                )
            )
            if t_val == t_max:
                x_at_t_max = x_noisy.clone()
            done += 1

        end_fwd = forward_steps[-1]
        backward_steps.append(
            PixelStepSnapshot(
                step_idx=0,
                timestep=end_fwd.timestep,
                alpha_bar=end_fwd.alpha_bar,
                sqrt_alpha_bar=end_fwd.sqrt_alpha_bar,
                sqrt_one_minus_alpha_bar=end_fwd.sqrt_one_minus_alpha_bar,
                x_t_rgb=end_fwd.x_t_rgb.copy(),
                x0_hat_rgb=None,
                x_t_mean=end_fwd.x_t_mean,
                x_t_std=end_fwd.x_t_std,
                x_t_flat_sample=end_fwd.x_t_flat_sample.copy(),
            )
        )

        x_rev = x_at_t_max.clone()
        for step_i, t_val in enumerate(rev_ts):
            if progress_cb is not None:
                progress_cb(0.15 + 0.85 * (done / max(total_work, 1)), f"Pixel backward denoising t={t_val}")
            t_tensor = torch.tensor(t_val, device=dev, dtype=torch.long)
            eps = ddpm_pipe.unet(x_rev, t_tensor).sample
            ab, sab, s1mab, _ = _schedule_stats(alphas_cumprod, t_val, is_pure_prior=False)
            step_gen = _make_generator(seed + 40000 + step_i, dev)
            step_out = scheduler.step(eps, t_tensor, x_rev, generator=step_gen)
            x_next = step_out.prev_sample
            x0_hat = _extract_z0_hat(step_out, x_rev, eps, ab)
            x_rev = x_next
            mv, sv, _ = _tensor_stats(x_rev)
            backward_steps.append(
                PixelStepSnapshot(
                    step_idx=step_i + 1,
                    timestep=t_val,
                    alpha_bar=ab,
                    sqrt_alpha_bar=sab,
                    sqrt_one_minus_alpha_bar=s1mab,
                    x_t_rgb=_tensor_to_uint8_rgb(x_rev),
                    x0_hat_rgb=_tensor_to_uint8_rgb(x0_hat),
                    x_t_mean=mv,
                    x_t_std=sv,
                    x_t_flat_sample=_subsample_flat(x_rev, seed=seed + 100 + step_i),
                )
            )
            done += 1
        eff_steps = len(rev_ts)

    if progress_cb is not None:
        progress_cb(1.0, "Completed Pixel-Space DDPM simulation")

    elapsed = time.perf_counter() - t_start
    peak_vram = _read_peak_vram_mb(dev)
    ms_per_step = (elapsed * 1000.0) / max(eff_steps, 1)

    return PixelDDPMResult(
        mode=mode,
        sampler=sampler,
        seed=seed,
        num_steps=eff_steps,
        t_max=t_max,
        source_rgb=src_rgb_arr,
        forward_steps=forward_steps,
        backward_steps=backward_steps,
        elapsed_sec=elapsed,
        peak_vram_mb=peak_vram,
        ms_per_step=ms_per_step,
        state_dims=state_dims,
    )
