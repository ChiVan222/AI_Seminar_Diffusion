"""Model loading, hardware detection, and scheduler builders for the Diffusion Simulator."""

from __future__ import annotations

import gc
import os
from typing import Any

import torch
from diffusers import DDIMScheduler, DDPMPipeline, DDPMScheduler, StableDiffusionPipeline

SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DDPM_PIXEL_MODEL_ID = "google/ddpm-celebahq-256"


def get_device_info() -> dict[str, Any]:
    """Return hardware, VRAM, dtype, and Colab environment metadata."""
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32
    gpu_name = "CPU"
    vram_total_gb = 0.0
    vram_allocated_mb = 0.0
    vram_reserved_mb = 0.0
    compute_capability = None

    if has_cuda:
        props = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = float(props.total_memory / (1024 ** 3))
        vram_allocated_mb = float(torch.cuda.memory_allocated(0) / (1024 ** 2))
        vram_reserved_mb = float(torch.cuda.memory_reserved(0) / (1024 ** 2))
        compute_capability = f"{props.major}.{props.minor}"

    in_colab = bool(
        os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("COLAB_GPU")
        or os.path.exists("/content")
    )
    return {
        "device": device,
        "has_cuda": has_cuda,
        "dtype": dtype,
        "dtype_name": "float16" if dtype == torch.float16 else "float32",
        "gpu_name": gpu_name,
        "vram_total_gb": round(vram_total_gb, 2),
        "vram_allocated_mb": round(vram_allocated_mb, 1),
        "vram_reserved_mb": round(vram_reserved_mb, 1),
        "compute_capability": compute_capability,
        "is_t4": "T4" in gpu_name.upper(),
        "in_colab": in_colab,
    }


def free_gpu_memory() -> None:
    """Release Python cyclic references and empty PyTorch CUDA allocator cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_sd15_pipeline(cache_dir: str | None = None) -> StableDiffusionPipeline:
    """Load Stable Diffusion v1.5 optimized for Google Colab Tesla T4 (FP16 + slicing).

    Uses ``token=False`` so expired/stale local Hugging Face credentials never cause
    401 errors when pulling public repositories.
    """
    info = get_device_info()
    device = info["device"]
    dtype = info["dtype"]

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
        "token": False,
    }
    if info["has_cuda"]:
        load_kwargs["variant"] = "fp16"
        load_kwargs["use_safetensors"] = True
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir

    pipe = StableDiffusionPipeline.from_pretrained(SD15_MODEL_ID, **load_kwargs)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    for comp_name in ("unet", "vae", "text_encoder"):
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "eval"):
            comp.eval()
    return pipe


def load_pixel_ddpm_pipeline(cache_dir: str | None = None) -> DDPMPipeline:
    """Load Pixel-Space DDPM CelebA-HQ 256 (``google/ddpm-celebahq-256``).

    Note: Does NOT pass ``variant="fp16"`` or ``use_safetensors=True`` because this
    repository distributes ``diffusion_pytorch_model.bin``.
    """
    info = get_device_info()
    device = info["device"]
    dtype = info["dtype"]

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "token": False,
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir

    pipe = DDPMPipeline.from_pretrained(DDPM_PIXEL_MODEL_ID, **load_kwargs)
    pipe = pipe.to(device)
    if hasattr(pipe, "unet") and hasattr(pipe.unet, "eval"):
        pipe.unet.eval()
    return pipe


def build_scheduler(base_config: Any, sampler: str, is_latent: bool):
    """Build a fresh DDIM or DDPM scheduler from a pipeline config.

    For latent diffusion (``is_latent=True``), ``clip_sample`` MUST be disabled because
    SD latents are unbounded Gaussians rather than ``[-1, 1]`` RGB pixels.
    For pixel diffusion (``is_latent=False``), ``clip_sample=True`` keeps pixels in ``[-1, 1]``.
    """
    sampler_key = sampler.strip().upper()
    if sampler_key == "DDIM":
        sched = DDIMScheduler.from_config(base_config)
        if is_latent:
            sched.register_to_config(clip_sample=False, set_alpha_to_one=False)
        else:
            sched.register_to_config(clip_sample=True)
        return sched
    if sampler_key == "DDPM":
        sched = DDPMScheduler.from_config(base_config)
        sched.register_to_config(clip_sample=not is_latent)
        return sched
    raise ValueError(f"Unsupported sampler '{sampler}'. Expected 'DDIM' or 'DDPM'.")
