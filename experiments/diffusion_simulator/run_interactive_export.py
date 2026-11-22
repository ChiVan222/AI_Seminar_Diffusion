"""Run Unconditioned vs. Conditioned SD v1.5 simulation with Latent Region Atlas,
per-step CFG direction vectors, and self-contained interactive HTML explorer export.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.dirname(_THIS_DIR)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from diffusion_simulator.engine import decode_latent_to_rgb, encode_prompt_sd
from diffusion_simulator.models import build_scheduler, free_gpu_memory, get_device_info, load_sd15_pipeline
from diffusion_simulator.visuals import _get_cmap, guidance_map_to_rgb, latent_to_rgb


def _np_to_jpeg_data_url(arr: np.ndarray, resize_to: int | None = None, quality: int = 86) -> str:
    pil = Image.fromarray(arr, mode="RGB")
    if resize_to is not None and pil.size != (resize_to, resize_to):
        pil = pil.resize((resize_to, resize_to), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _channels_strip_rgb(z: np.ndarray, tile_size: int = 64) -> np.ndarray:
    """Render 4 latent channels `z[0..3]` side by side into one `(tile_size, 4*tile_size, 3)` RGB array."""
    rdbu = _get_cmap("RdBu_r")
    tiles: list[np.ndarray] = []
    for ch in range(min(4, z.shape[0])):
        norm = np.clip((z[ch] + 3.5) / 7.0, 0.0, 1.0)
        rgb = (rdbu(norm)[..., :3] * 255.0).round().astype(np.uint8)
        if rgb.shape[0] != tile_size or rgb.shape[1] != tile_size:
            rgb = np.asarray(
                Image.fromarray(rgb, mode="RGB").resize((tile_size, tile_size), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        tiles.append(rgb)
    return np.concatenate(tiles, axis=1)


def _decode_batch_to_rgb(sd_pipe: Any, latents: torch.Tensor) -> list[np.ndarray]:
    """Decode `(B, 4, H, W)` SD latents to a list of `(ImgH, ImgW, 3)` uint8 RGB arrays."""
    scaling = float(getattr(sd_pipe.vae.config, "scaling_factor", 0.18215))
    vae_dtype = next(sd_pipe.vae.parameters()).dtype
    vae_dev = next(sd_pipe.vae.parameters()).device
    imgs = sd_pipe.vae.decode(latents.to(device=vae_dev, dtype=vae_dtype) / scaling).sample
    imgs = ((imgs.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()
    return [imgs_np[i] for i in range(imgs_np.shape[0])]


@torch.inference_mode()
def run_seed_preview(
    prompt: str,
    seeds: list[int],
    num_steps: int,
    cfg: float,
    resolution: int,
    out_dir: str,
    sampler: str = "DDIM",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sampler_clean = sampler.strip().upper()
    sd_pipe = load_sd15_pipeline()
    dev = sd_pipe.unet.device
    dt = sd_pipe.unet.dtype
    sched = build_scheduler(sd_pipe.scheduler.config, sampler=sampler_clean, is_latent=True)
    emb_null = encode_prompt_sd(sd_pipe, "", dev, dt)
    emb_cond = encode_prompt_sd(sd_pipe, prompt, dev, dt)
    emb_both = torch.cat([emb_null, emb_cond], dim=0)

    imgs: list[np.ndarray] = []
    for seed in seeds:
        sched.set_timesteps(num_steps, device=dev)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn((1, 4, resolution // 8, resolution // 8), generator=gen, dtype=torch.float32).to(dev, dt)
        for step_i, t in enumerate(sched.timesteps):
            eps_both = sd_pipe.unet(torch.cat([z, z], dim=0), t, encoder_hidden_states=emb_both).sample
            eps_u, eps_c = eps_both[0:1], eps_both[1:2]
            eps = eps_u + cfg * (eps_c - eps_u)
            step_gen = torch.Generator(device=dev.type if dev.type == "cuda" else "cpu").manual_seed(seed + 10000 + step_i)
            z = sched.step(eps, t, z, generator=step_gen).prev_sample
        imgs.append(decode_latent_to_rgb(sd_pipe, z))

    del sd_pipe
    free_gpu_memory()

    tile = 384
    pad = 16
    banner = 42
    w = pad + len(seeds) * (tile + pad)
    h = banner + tile + pad
    canvas = Image.new("RGB", (w, h), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), f"Seed Preview | Prompt: {prompt} | {sampler_clean} {num_steps} steps | CFG={cfg}", fill=(25, 70, 140))
    for i, (seed, arr) in enumerate(zip(seeds, imgs)):
        x0 = pad + i * (tile + pad)
        draw.text((x0, 26), f"Seed = {seed}", fill=(15, 23, 42))
        canvas.paste(Image.fromarray(arr, mode="RGB").resize((tile, tile), Image.Resampling.BILINEAR), (x0, banner))

    out_path = os.path.join(out_dir, "seed_preview.png")
    canvas.save(out_path)
    print(f"Saved seed preview strip to {out_path}")
    return out_path


def _build_semantic_basis(z_start: np.ndarray, z_uncond_end: np.ndarray, z_cond_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct orthonormal 2D plane passing exactly through z_T, z_0^uncond, and z_0^cond."""
    a = z_start.reshape(-1).astype(np.float64)
    b = z_uncond_end.reshape(-1).astype(np.float64)
    c = z_cond_end.reshape(-1).astype(np.float64)

    # e1: denoising direction from shared noise z_T toward midpoint of endpoints
    mid_dir = 0.5 * (b + c) - a
    n1 = float(np.linalg.norm(mid_dir))
    e1 = mid_dir / max(n1, 1e-8)

    # e2: pure conditioning axis (z_cond - z_uncond) orthogonalized against e1
    cond_dir = c - b
    cond_orth = cond_dir - np.dot(cond_dir, e1) * e1
    n2 = float(np.linalg.norm(cond_orth))
    if n2 < 1e-8:
        # Fallback orthogonal vector if endpoints collapsed
        alt = np.zeros_like(e1)
        alt[0] = 1.0
        cond_orth = alt - np.dot(alt, e1) * e1
        n2 = float(np.linalg.norm(cond_orth))
    e2 = cond_orth / max(n2, 1e-8)

    basis = np.stack([e1, e2], axis=0).astype(np.float32)  # (2, D)
    return a.astype(np.float32), basis


def _build_pca_basis(latents_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    mat = np.stack([z.reshape(-1).astype(np.float32) for z in latents_list], axis=0)
    pca = PCA(n_components=2)
    pca.fit(mat)
    origin = pca.mean_.astype(np.float32)
    basis = pca.components_.astype(np.float32)  # (2, D) orthonormal rows
    var_pct = [round(float(v * 100.0), 2) for v in pca.explained_variance_ratio_]
    return origin, basis, var_pct


def _project_point(z: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> list[float]:
    coords = (z.reshape(-1).astype(np.float32) - origin) @ basis.T
    return [round(float(coords[0]), 4), round(float(coords[1]), 4)]


def _project_vec(dz: np.ndarray, basis: np.ndarray) -> list[float]:
    coords = dz.reshape(-1).astype(np.float32) @ basis.T
    return [round(float(coords[0]), 4), round(float(coords[1]), 4)]


@torch.inference_mode()
def _build_plane_data(
    name: str,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    origin: np.ndarray,
    basis: np.ndarray,
    uncond_latents: list[np.ndarray],
    cond_latents: list[np.ndarray],
    cond_vec_uncond: list[np.ndarray | None],
    cond_vec_gcfg: list[np.ndarray | None],
    cond_vec_cfg: list[np.ndarray | None],
    uncond_vec_actual: list[np.ndarray | None],
    uncond_vec_gcfg_cf: list[np.ndarray | None],
    sd_pipe: Any,
    latent_shape: tuple[int, int, int],
    grid_cols: int,
    grid_rows: int,
    atlas_quality: int = 82,
) -> dict[str, Any]:
    uncond_pts = [_project_point(z, origin, basis) for z in uncond_latents]
    cond_pts = [_project_point(z, origin, basis) for z in cond_latents]

    all_u = [p[0] for p in uncond_pts] + [p[0] for p in cond_pts]
    all_v = [p[1] for p in uncond_pts] + [p[1] for p in cond_pts]
    u_lo, u_hi = min(all_u), max(all_u)
    v_lo, v_hi = min(all_v), max(all_v)
    du_span = max(u_hi - u_lo, 1.0)
    dv_span = max(v_hi - v_lo, 1.0)
    pad_u = 0.18 * du_span
    pad_v = 0.18 * dv_span
    u_min, u_max = u_lo - pad_u, u_hi + pad_u
    v_min, v_max = v_lo - pad_v, v_hi + pad_v

    # Project step vectors (exact linearity: p_vcfg == p_vuncond + p_gcfg)
    cond_vecs_2d = []
    uncond_vecs_2d = []
    for k in range(len(uncond_latents)):
        if cond_vec_uncond[k] is None:
            cond_vecs_2d.append(None)
            uncond_vecs_2d.append(None)
        else:
            cond_vecs_2d.append(
                {
                    "v_uncond": _project_vec(cond_vec_uncond[k], basis),
                    "g_cfg": _project_vec(cond_vec_gcfg[k], basis),
                    "v_cfg": _project_vec(cond_vec_cfg[k], basis),
                }
            )
            uncond_vecs_2d.append(
                {
                    "v_actual": _project_vec(uncond_vec_actual[k], basis),
                    "g_cfg_cf": _project_vec(uncond_vec_gcfg_cf[k], basis),
                }
            )

    # Decode 8x6 Latent Region Atlas showing what image lives in every 2D region
    u_centers = np.linspace(u_min, u_max, grid_cols, endpoint=False) + (u_max - u_min) / (2.0 * grid_cols)
    v_centers = np.linspace(v_min, v_max, grid_rows, endpoint=False) + (v_max - v_min) / (2.0 * grid_rows)

    tile_latents_np: list[np.ndarray] = []
    tile_meta: list[tuple[int, int, float, float]] = []
    for r_idx, vc in enumerate(v_centers):
        for c_idx, uc in enumerate(u_centers):
            z_flat = origin + np.float32(uc) * basis[0] + np.float32(vc) * basis[1]
            tile_latents_np.append(z_flat.reshape(latent_shape))
            tile_meta.append((c_idx, r_idx, float(uc), float(vc)))

    tiles: list[dict[str, Any]] = []
    batch_size = 4
    dev = sd_pipe.unet.device
    dt = sd_pipe.unet.dtype
    for start in range(0, len(tile_latents_np), batch_size):
        batch_np = np.stack(tile_latents_np[start : start + batch_size], axis=0)
        batch_t = torch.from_numpy(batch_np).to(device=dev, dtype=dt)
        decoded_list = _decode_batch_to_rgb(sd_pipe, batch_t)
        for offset, dec_rgb in enumerate(decoded_list):
            c_idx, r_idx, uc, vc = tile_meta[start + offset]
            z_tile = tile_latents_np[start + offset]
            tiles.append(
                {
                    "col": c_idx,
                    "row": r_idx,
                    "u": round(uc, 3),
                    "v": round(vc, 3),
                    "decoded_url": _np_to_jpeg_data_url(dec_rgb, resize_to=144, quality=atlas_quality),
                    "latent_url": _np_to_jpeg_data_url(latent_to_rgb(z_tile, upscale=2), resize_to=96, quality=atlas_quality),
                }
            )

    return {
        "name": name,
        "title": title,
        "subtitle": subtitle,
        "x_label": x_label,
        "y_label": y_label,
        "bounds": {
            "u_min": round(float(u_min), 4),
            "u_max": round(float(u_max), 4),
            "v_min": round(float(v_min), 4),
            "v_max": round(float(v_max), 4),
        },
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "uncond_points": uncond_pts,
        "cond_points": cond_pts,
        "cond_vectors": cond_vecs_2d,
        "uncond_vectors": uncond_vecs_2d,
        "atlas_tiles": tiles,
    }


def _make_filmstrip_and_gif(
    prompt: str,
    seed: int,
    cfg: float,
    uncond_decoded_full: list[np.ndarray],
    cond_decoded_full: list[np.ndarray],
    uncond_latents: list[np.ndarray],
    cond_latents: list[np.ndarray],
    cfg_heatmaps: list[np.ndarray | None],
    timesteps_lbl: list[int],
    out_dir: str,
    sampler: str = "DDIM",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_total = len(timesteps_lbl)
    idxs = np.linspace(0, n_total - 1, 7).round().astype(int).tolist()
    seen: set[int] = set()
    cols = [i for i in idxs if not (i in seen or seen.add(i))]

    fig, axes = plt.subplots(5, len(cols), figsize=(2.15 * len(cols), 10.6))
    for col_i, s_idx in enumerate(cols):
        t_v = timesteps_lbl[s_idx]
        axes[0, col_i].imshow(uncond_decoded_full[s_idx])
        axes[0, col_i].set_title(f"Step {s_idx}\nt = {t_v}", fontsize=9, fontweight="bold")
        axes[1, col_i].imshow(latent_to_rgb(uncond_latents[s_idx], upscale=4))
        axes[2, col_i].imshow(cond_decoded_full[s_idx])
        axes[3, col_i].imshow(latent_to_rgb(cond_latents[s_idx], upscale=4))
        hm = cfg_heatmaps[s_idx]
        if hm is not None:
            axes[4, col_i].imshow(guidance_map_to_rgb(hm, upscale=4))
        else:
            axes[4, col_i].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
        for r in range(5):
            axes[r, col_i].set_xticks([])
            axes[r, col_i].set_yticks([])

    axes[0, 0].set_ylabel("Uncond (∅)\nDecoded D(z_t)", fontsize=8.8, fontweight="bold")
    axes[1, 0].set_ylabel("Uncond (∅)\nLatent z_t RGB", fontsize=8.5)
    axes[2, 0].set_ylabel(f"Cond (CFG={cfg})\nDecoded D(z_t)", fontsize=8.8, fontweight="bold")
    axes[3, 0].set_ylabel(f"Cond (CFG={cfg})\nLatent z_t RGB", fontsize=8.5)
    axes[4, 0].set_ylabel("CFG Steering\n||ε_c - ε_∅||₂", fontsize=8.5, fontweight="bold")
    fig.suptitle(
        f'Unconditioned vs. Conditioned LDM ({sampler}, seed={seed}, CFG={cfg})\nPrompt: "{prompt}"',
        fontsize=11.5,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "filmstrip_uncond_vs_cond.png"), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Animated GIF
    frames: list[Image.Image] = []
    panel_sz = 230
    pad = 12
    title_h = 38
    lbl_h = 20
    w = pad + 5 * (panel_sz + pad)
    h = title_h + lbl_h + panel_sz + pad
    for s_idx in range(n_total):
        canvas = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (pad, 10),
            f'Step {s_idx}/{n_total - 1} (t={timesteps_lbl[s_idx]}) | CFG={cfg} | "{prompt}"',
            fill=(25, 70, 140),
        )
        hm_arr = (
            guidance_map_to_rgb(cfg_heatmaps[s_idx], upscale=4)
            if cfg_heatmaps[s_idx] is not None
            else np.zeros((256, 256, 3), dtype=np.uint8)
        )
        panels = [
            ("Uncond D(z_t)", uncond_decoded_full[s_idx]),
            ("Uncond Latent RGB", latent_to_rgb(uncond_latents[s_idx], upscale=4)),
            (f"Cond D(z_t) α={cfg}", cond_decoded_full[s_idx]),
            ("Cond Latent RGB", latent_to_rgb(cond_latents[s_idx], upscale=4)),
            ("CFG ||ε_c - ε_∅||₂", hm_arr),
        ]
        for p_i, (lbl, arr) in enumerate(panels):
            x0 = pad + p_i * (panel_sz + pad)
            draw.text((x0, title_h), lbl, fill=(30, 41, 59))
            canvas.paste(Image.fromarray(arr, mode="RGB").resize((panel_sz, panel_sz), Image.Resampling.BILINEAR), (x0, title_h + lbl_h))
        frames.append(canvas)

    durations = [260] * len(frames)
    durations[-1] = 1200
    frames[0].save(
        os.path.join(out_dir, "uncond_vs_cond.gif"),
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


def build_interactive_html(payload: dict[str, Any]) -> str:
    """Build self-contained single-file interactive HTML explorer."""
    json_str = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    prompt_esc = payload["meta"]["prompt"].replace('"', "&quot;")
    cfg_val = payload["meta"]["cfg_scale"]
    n_steps = payload["meta"]["num_steps"]
    seed_val = payload["meta"]["seed"]
    sampler_name = payload["meta"]["sampler"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Interactive Latent Diffusion Explorer — "{prompt_esc}"</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #111c34;
    --card: #172544;
    --border: #26395e;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --blue: #3b82f6;
    --red: #ef4444;
    --orange: #f59e0b;
    --teal: #14b8a6;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.4;
  }}
  .wrap {{ max-width: 1540px; margin: 0 auto; padding: 14px 18px 28px; }}
  .header {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: linear-gradient(135deg, #162447 0%, #101b33 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 12px;
  }}
  .header h1 {{ margin: 0; font-size: 1.2rem; color: #f8fafc; }}
  .header .sub {{ font-size: 0.85rem; color: var(--muted); margin-top: 3px; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .pill {{
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #bae6fd;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
  }}
  .controls-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 12px;
  }}
  button {{
    background: var(--card);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 0.84rem;
    font-weight: 600;
    transition: background 0.15s, border-color 0.15s;
  }}
  button:hover {{ background: #1e325c; border-color: var(--accent); }}
  button.primary {{ background: #1d4ed8; border-color: #3b82f6; color: #fff; }}
  button.primary:hover {{ background: #2563eb; }}
  input[type="range"] {{ accent-color: var(--accent); cursor: pointer; }}
  .slider-wrap {{ flex: 1 1 260px; display: flex; align-items: center; gap: 10px; }}
  .slider-wrap input {{ flex: 1; }}
  .step-badge {{
    font-variant-numeric: tabular-nums;
    background: #0b1326;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 0.84rem;
    min-width: 175px;
    text-align: center;
  }}
  .main-grid {{
    display: grid;
    grid-template-columns: minmax(430px, 1.08fr) minmax(480px, 1.25fr);
    gap: 12px;
    margin-bottom: 12px;
  }}
  @media (max-width: 1100px) {{
    .main-grid {{ grid-template-columns: 1fr; }}
  }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
  }}
  .panel-title {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }}
  .panel-title h2 {{ margin: 0; font-size: 1.0rem; color: #f1f5f9; }}
  .subcontrols {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  select {{
    background: #0b1326;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 0.8rem;
  }}
  .canvas-box {{
    position: relative;
    background: #080f20;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}
  canvas#latentCanvas {{
    display: block;
    width: 100%;
    height: auto;
    cursor: crosshair;
  }}
  .hover-lens {{
    position: absolute;
    pointer-events: none;
    background: rgba(10, 18, 36, 0.94);
    border: 1px solid var(--accent);
    border-radius: 10px;
    padding: 8px;
    display: none;
    z-index: 20;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.55);
    width: 158px;
  }}
  .hover-lens img {{ width: 140px; height: 140px; border-radius: 6px; display: block; }}
  .hover-lens .meta {{ font-size: 0.72rem; color: #cbd5e1; margin-top: 4px; text-align: center; }}
  .legend-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 8px;
    font-size: 0.77rem;
    color: #cbd5e1;
  }}
  .leg {{ display: inline-flex; align-items: center; gap: 5px; }}
  .swatch {{ width: 12px; height: 4px; border-radius: 2px; display: inline-block; }}
  .branch-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }}
  @media (max-width: 700px) {{
    .branch-grid {{ grid-template-columns: 1fr; }}
  }}
  .branch-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px;
  }}
  .branch-card.uncond {{ border-top: 3px solid #64748b; }}
  .branch-card.cond {{ border-top: 3px solid var(--blue); }}
  .branch-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-weight: 700;
    font-size: 0.9rem;
  }}
  .main-img {{
    width: 100%;
    aspect-ratio: 1/1;
    object-fit: cover;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: #080f20;
  }}
  .mini-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 6px;
  }}
  .mini-tile {{
    background: #0b1326;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
    text-align: center;
  }}
  .mini-tile img {{
    width: 100%;
    aspect-ratio: 1/1;
    object-fit: contain;
    image-rendering: pixelated;
    border-radius: 4px;
  }}
  .mini-tile span {{ display: block; font-size: 0.69rem; color: var(--muted); margin-top: 2px; }}
  .ch-strip {{
    margin-top: 6px;
    background: #0b1326;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
    text-align: center;
  }}
  .ch-strip img {{
    width: 100%;
    height: auto;
    image-rendering: pixelated;
    border-radius: 4px;
    display: block;
  }}
  .ch-strip span {{ display: block; font-size: 0.69rem; color: var(--muted); margin-top: 2px; }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(110px, 1fr));
    gap: 8px;
    margin-top: 10px;
  }}
  @media (max-width: 700px) {{
    .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  .stat-box {{
    background: #0b1326;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 9px;
  }}
  .stat-box .k {{ font-size: 0.7rem; color: var(--muted); }}
  .stat-box .v {{ font-size: 0.96rem; font-weight: 700; font-variant-numeric: tabular-nums; margin-top: 2px; }}
  .charts-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  @media (max-width: 1000px) {{
    .charts-row {{ grid-template-columns: 1fr; }}
  }}
  .chart-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
  }}
  .chart-card h3 {{ margin: 0 0 6px 0; font-size: 0.86rem; color: #e2e8f0; }}
  svg.chart {{ width: 100%; height: 170px; display: block; background: #080f20; border-radius: 8px; border: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <h1>Latent Diffusion Step-by-Step Explorer: Unconditioned (&empty;) vs. Conditioned + CFG Direction</h1>
      <div class="sub">Prompt: <b>"{prompt_esc}"</b> &nbsp;|&nbsp; Both branches start from identical Gaussian noise <code>z_T</code></div>
    </div>
    <div class="pills">
      <span class="pill">Stable Diffusion v1.5 FP16</span>
      <span class="pill">{sampler_name} {n_steps} Steps</span>
      <span class="pill">CFG &alpha; = {cfg_val}</span>
      <span class="pill">Seed {seed_val}</span>
      <span class="pill">8&times;6 Region Atlas (48 decoded tiles)</span>
    </div>
  </div>

  <div class="controls-bar">
    <button id="btnFirst" title="First step (pure noise)">&laquo; Start</button>
    <button id="btnPrev" title="Previous step (Left arrow)">&lsaquo; Prev</button>
    <button id="btnPlay" class="primary" title="Play / Pause (Space)">&#9654; Play</button>
    <button id="btnNext" title="Next step (Right arrow)">Next &rsaquo;</button>
    <button id="btnLast" title="Final step (clean output)">Final &raquo;</button>
    <div class="slider-wrap">
      <input type="range" id="stepSlider" min="0" max="{n_steps}" value="0" />
    </div>
    <div class="step-badge" id="stepBadge">Step 0 / {n_steps} (t = 1000)</div>
    <label style="font-size:0.8rem;color:var(--muted);">Speed
      <select id="speedSel">
        <option value="850">0.5&times;</option>
        <option value="450" selected>1&times;</option>
        <option value="220">2&times;</option>
      </select>
    </label>
  </div>

  <div class="main-grid">
    <div class="panel">
      <div class="panel-title">
        <h2>2D Latent Space Map &amp; Per-Step CFG Steering Vector</h2>
        <span id="planeTitle" style="font-size:0.78rem;color:var(--accent);"></span>
      </div>
      <div class="subcontrols">
        <label>2D Projection:
          <select id="planeSel">
            <option value="semantic" selected>Semantic Plane (Exact z_T, z_0^∅, z_0^y)</option>
            <option value="pca">2D PCA Plane</option>
          </select>
        </label>
        <label>Region Atlas:
          <select id="atlasSel">
            <option value="decoded" selected>Decoded VAE Images</option>
            <option value="latent">Raw 64&times;64 Latent RGB</option>
          </select>
        </label>
        <label>Atlas Opacity:
          <input type="range" id="atlasAlpha" min="0" max="100" value="62" style="width:72px;vertical-align:middle;" />
        </label>
        <label>CFG Arrow Scale:
          <select id="arrowScaleSel">
            <option value="1">1&times; Exact</option>
            <option value="2" selected>2&times; Clear</option>
            <option value="3.5">3.5&times; Boosted</option>
          </select>
        </label>
        <label><input type="checkbox" id="chkCounterfactual" checked /> Show CFG pull on &empty; path</label>
      </div>
      <div class="canvas-box">
        <canvas id="latentCanvas" width="680" height="540"></canvas>
        <div class="hover-lens" id="hoverLens">
          <img id="lensImg" alt="Region image" />
          <div class="meta" id="lensMeta"></div>
        </div>
      </div>
      <div class="legend-row">
        <span class="leg"><span class="swatch" style="background:#94a3b8;"></span>Uncond (&empty;) Trajectory</span>
        <span class="leg"><span class="swatch" style="background:#3b82f6;"></span>Cond (CFG=&alpha;) Trajectory</span>
        <span class="leg"><span class="swatch" style="background:#94a3b8;border-bottom:2px dashed #cbd5e1;height:0;"></span>Uncond Drift v<sub>&empty;</sub></span>
        <span class="leg"><span class="swatch" style="background:#ef4444;height:5px;"></span><b>CFG Direction g<sub>CFG</sub> = &alpha;(v<sub>y</sub>&minus;v<sub>&empty;</sub>)</b></span>
        <span class="leg"><span class="swatch" style="background:#38bdf8;height:5px;"></span>Actual Step v<sub>CFG</sub></span>
        <span class="leg"><span class="swatch" style="background:#f59e0b;"></span>Counterfactual CFG pull on &empty;</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">
        <h2>Step-by-Step Image &amp; Latent Channel Comparison</h2>
        <div>
          <label style="font-size:0.8rem;color:var(--muted);">Main View:
            <select id="imgLayerSel">
              <option value="decoded_zt" selected>Decoded State D(z_t)</option>
              <option value="decoded_z0_hat">Predicted Clean D(&#7825;&#8320;)</option>
            </select>
          </label>
        </div>
      </div>

      <div class="branch-grid">
        <div class="branch-card uncond">
          <div class="branch-hdr">
            <span>Unconditioned (&empty;)</span>
            <span id="uStdTag" style="font-size:0.75rem;color:var(--muted);"></span>
          </div>
          <img id="uMainImg" class="main-img" alt="Unconditioned step" />
          <div class="mini-row">
            <div class="mini-tile">
              <img id="uLatImg" alt="Unconditioned latent RGB" />
              <span>Latent z_t RGB (64&times;64)</span>
            </div>
            <div class="mini-tile">
              <img id="uZ0Img" alt="Unconditioned z0_hat" />
              <span>Pred. Clean D(&#7825;&#8320;)</span>
            </div>
          </div>
          <div class="ch-strip">
            <img id="uChImg" alt="Unconditioned 4 channels" />
            <span>4 Latent Channels z[0..3] (&minus;3.5 &harr; +3.5)</span>
          </div>
        </div>

        <div class="branch-card cond">
          <div class="branch-hdr">
            <span>Conditioned (CFG &alpha;={cfg_val})</span>
            <span id="cStdTag" style="font-size:0.75rem;color:#93c5fd;"></span>
          </div>
          <img id="cMainImg" class="main-img" alt="Conditioned step" />
          <div class="mini-row">
            <div class="mini-tile">
              <img id="cLatImg" alt="Conditioned latent RGB" />
              <span>Latent z_t RGB (64&times;64)</span>
            </div>
            <div class="mini-tile">
              <img id="cCfgMapImg" alt="CFG spatial heatmap" />
              <span>CFG Push ||&epsilon;<sub>y</sub> &minus; &epsilon;<sub>&empty;</sub>||&#8322;</span>
            </div>
          </div>
          <div class="ch-strip">
            <img id="cChImg" alt="Conditioned 4 channels" />
            <span>4 Latent Channels z[0..3] (&minus;3.5 &harr; +3.5)</span>
          </div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-box"><div class="k">CFG Push ||g<sub>CFG</sub>||&#8322;</div><div class="v" id="stGcfg">-</div></div>
        <div class="stat-box"><div class="k">Uncond Drift ||v<sub>&empty;</sub>||&#8322;</div><div class="v" id="stVunc">-</div></div>
        <div class="stat-box"><div class="k">Angle &ang;(v<sub>&empty;</sub>, g<sub>CFG</sub>)</div><div class="v" id="stAng">-</div></div>
        <div class="stat-box"><div class="k">Latent L2 Div vs &empty;</div><div class="v" id="stDiv">-</div></div>
        <div class="stat-box"><div class="k">Cosine Sim vs &empty;</div><div class="v" id="stCos">-</div></div>
        <div class="stat-box"><div class="k">Signal &radic;&#945;&#772;<sub>t</sub></div><div class="v" id="stSig">-</div></div>
        <div class="stat-box"><div class="k">Noise &radic;(1&minus;&#945;&#772;<sub>t</sub>)</div><div class="v" id="stNoise">-</div></div>
        <div class="stat-box"><div class="k">Cond Noise Shift ||&Delta;&epsilon;||&#8322;</div><div class="v" id="stDeps">-</div></div>
      </div>
    </div>
  </div>

  <div class="charts-row">
    <div class="chart-card">
      <h3>CFG Steering Magnitude ||g<sub>CFG</sub>||&#8322; &amp; Noise Shift ||&epsilon;<sub>y</sub>&minus;&epsilon;<sub>&empty;</sub>||&#8322;</h3>
      <svg id="chartCfg" class="chart" viewBox="0 0 420 170"></svg>
    </div>
    <div class="chart-card">
      <h3>Trajectory Divergence ||z<sub>t</sub><sup>y</sup> &minus; z<sub>t</sub><sup>&empty;</sup>||&#8322; &amp; Cosine Similarity</h3>
      <svg id="chartDiv" class="chart" viewBox="0 0 420 170"></svg>
    </div>
    <div class="chart-card">
      <h3>Noise Schedule (&radic;&#945;&#772;<sub>t</sub> signal vs &radic;(1&minus;&#945;&#772;<sub>t</sub>) noise)</h3>
      <svg id="chartSched" class="chart" viewBox="0 0 420 170"></svg>
    </div>
  </div>
</div>

<script id="sim-payload" type="application/json">{json_str}</script>
<script>
(() => {{
  const DATA = JSON.parse(document.getElementById("sim-payload").textContent);
  const N = DATA.steps.length - 1;
  let curStep = 0;
  let timer = null;

  const canvas = document.getElementById("latentCanvas");
  const ctx = canvas.getContext("2d");
  const stepSlider = document.getElementById("stepSlider");
  const stepBadge = document.getElementById("stepBadge");
  const planeSel = document.getElementById("planeSel");
  const atlasSel = document.getElementById("atlasSel");
  const atlasAlpha = document.getElementById("atlasAlpha");
  const arrowScaleSel = document.getElementById("arrowScaleSel");
  const chkCF = document.getElementById("chkCounterfactual");
  const imgLayerSel = document.getElementById("imgLayerSel");
  const speedSel = document.getElementById("speedSel");
  const hoverLens = document.getElementById("hoverLens");
  const lensImg = document.getElementById("lensImg");
  const lensMeta = document.getElementById("lensMeta");

  // Preload atlas tile Image objects for smooth 60fps canvas rendering
  const tileCache = {{}};
  for (const planeName of Object.keys(DATA.planes)) {{
    const p = DATA.planes[planeName];
    tileCache[planeName] = p.atlas_tiles.map(t => {{
      const imgDec = new Image();
      imgDec.src = t.decoded_url;
      imgDec.onload = () => {{ if (planeSel.value === planeName) drawCanvas(); }};
      const imgLat = new Image();
      imgLat.src = t.latent_url;
      imgLat.onload = () => {{ if (planeSel.value === planeName) drawCanvas(); }};
      return {{ meta: t, decoded: imgDec, latent: imgLat }};
    }});
  }}

  const PAD = {{ l: 54, r: 22, t: 30, b: 40 }};
  function worldToPx(u, v, b) {{
    const pw = canvas.width - PAD.l - PAD.r;
    const ph = canvas.height - PAD.t - PAD.b;
    const x = PAD.l + ((u - b.u_min) / (b.u_max - b.u_min)) * pw;
    const y = PAD.t + (1.0 - (v - b.v_min) / (b.v_max - b.v_min)) * ph;
    return [x, y];
  }}
  function pxToWorld(x, y, b) {{
    const pw = canvas.width - PAD.l - PAD.r;
    const ph = canvas.height - PAD.t - PAD.b;
    const u = b.u_min + ((x - PAD.l) / pw) * (b.u_max - b.u_min);
    const v = b.v_min + (1.0 - (y - PAD.t) / ph) * (b.v_max - b.v_min);
    return [u, v];
  }}

  function drawArrow(x1, y1, x2, y2, color, width, dashed = false, label = "") {{
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 2.0) return;
    const ang = Math.atan2(dy, dx);
    const head = Math.min(11, Math.max(6, len * 0.25));

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    if (dashed) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(ang - Math.PI / 6), y2 - head * Math.sin(ang - Math.PI / 6));
    ctx.lineTo(x2 - head * Math.cos(ang + Math.PI / 6), y2 - head * Math.sin(ang + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    if (label && len > 16) {{
      const mx = 0.5 * (x1 + x2);
      const my = 0.5 * (y1 + y2);
      ctx.font = "bold 11px sans-serif";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(8,15,32,0.9)";
      ctx.strokeText(label, mx + 6, my - 6);
      ctx.fillText(label, mx + 6, my - 6);
    }}
    ctx.restore();
  }}

  function drawStar(cx, cy, r, fill) {{
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {{
      const rad = i % 2 === 0 ? r : r * 0.45;
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      const px = cx + rad * Math.cos(a);
      const py = cy + rad * Math.sin(a);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }}
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }}

  function drawCanvas() {{
    const planeName = planeSel.value;
    const p = DATA.planes[planeName];
    const b = p.bounds;
    document.getElementById("planeTitle").textContent = p.subtitle;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#080f20";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const pw = canvas.width - PAD.l - PAD.r;
    const ph = canvas.height - PAD.t - PAD.b;
    const cellW = pw / p.grid_cols;
    const cellH = ph / p.grid_rows;
    const alpha = Number(atlasAlpha.value) / 100.0;
    const useDecoded = atlasSel.value === "decoded";

    // Draw 8x6 Latent Region Atlas tiles
    if (alpha > 0.01) {{
      ctx.save();
      ctx.globalAlpha = alpha;
      for (const item of tileCache[planeName]) {{
        const img = useDecoded ? item.decoded : item.latent;
        const x = PAD.l + item.meta.col * cellW;
        const y = PAD.t + (p.grid_rows - 1 - item.meta.row) * cellH;
        if (img.complete && img.naturalWidth > 0) {{
          ctx.drawImage(img, x, y, cellW + 0.5, cellH + 0.5);
        }}
      }}
      ctx.restore();
    }}

    // Subtle tile grid lines
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.20)";
    ctx.lineWidth = 1;
    for (let c = 0; c <= p.grid_cols; c++) {{
      const x = PAD.l + c * cellW;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + ph); ctx.stroke();
    }}
    for (let r = 0; r <= p.grid_rows; r++) {{
      const y = PAD.t + r * cellH;
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + pw, y); ctx.stroke();
    }}
    ctx.strokeStyle = "#475569";
    ctx.strokeRect(PAD.l, PAD.t, pw, ph);
    ctx.restore();

    // Axis labels
    ctx.save();
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(p.x_label, PAD.l + pw / 2, canvas.height - 10);
    ctx.translate(16, PAD.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(p.y_label, 0, 0);
    ctx.restore();

    // Helper to draw full path (faint) + traversed path up to curStep
    const drawPath = (pts, color, dash = false) => {{
      ctx.save();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.25;
      ctx.lineWidth = 2;
      if (dash) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      pts.forEach((pt, idx) => {{
        const [x, y] = worldToPx(pt[0], pt[1], b);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();

      ctx.globalAlpha = 0.95;
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      ctx.beginPath();
      for (let i = 0; i <= curStep; i++) {{
        const [x, y] = worldToPx(pts[i][0], pts[i][1], b);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();

      // Step dots
      for (let i = 0; i < pts.length; i++) {{
        const [x, y] = worldToPx(pts[i][0], pts[i][1], b);
        ctx.beginPath();
        ctx.arc(x, y, i === curStep ? 6.5 : 3.2, 0, Math.PI * 2);
        ctx.fillStyle = i <= curStep ? color : "rgba(148,163,184,0.45)";
        ctx.fill();
        if (i === curStep) {{
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
        }}
      }}
      ctx.restore();
    }};

    drawPath(p.uncond_points, "#cbd5e1", true);
    drawPath(p.cond_points, "#3b82f6", false);

    // Draw CFG Direction Vector Triangle at curStep (for k < N)
    const scale = Number(arrowScaleSel.value);
    const cv = p.cond_vectors[curStep];
    if (cv) {{
      const [u0, v0] = p.cond_points[curStep];
      const [x0, y0] = worldToPx(u0, v0, b);
      const uMid = u0 + scale * cv.v_uncond[0];
      const vMid = v0 + scale * cv.v_uncond[1];
      const [xMid, yMid] = worldToPx(uMid, vMid, b);
      const uEnd = u0 + scale * cv.v_cfg[0];
      const vEnd = v0 + scale * cv.v_cfg[1];
      const [xEnd, yEnd] = worldToPx(uEnd, vEnd, b);

      drawArrow(x0, y0, xMid, yMid, "#94a3b8", 2.2, true, "v_∅");
      drawArrow(xMid, yMid, xEnd, yEnd, "#ef4444", 3.2, false, "CFG g_CFG");
      drawArrow(x0, y0, xEnd, yEnd, "#38bdf8", 3.0, false, "v_CFG");
    }}
    if (chkCF.checked && p.uncond_vectors[curStep]) {{
      const uv = p.uncond_vectors[curStep];
      const [uu, vu] = p.uncond_points[curStep];
      const [xu0, yu0] = worldToPx(uu, vu, b);
      const [xum, yum] = worldToPx(uu + scale * uv.v_actual[0], vu + scale * uv.v_actual[1], b);
      const [xuc, yuc] = worldToPx(
        uu + scale * (uv.v_actual[0] + uv.g_cfg_cf[0]),
        vu + scale * (uv.v_actual[1] + uv.g_cfg_cf[1]),
        b
      );
      drawArrow(xum, yum, xuc, yuc, "#f59e0b", 2.4, false, "CFG pull on ∅");
    }}

    // Landmark annotations: Start z_T, Uncond End, Cond End
    const [sx, sy] = worldToPx(p.cond_points[0][0], p.cond_points[0][1], b);
    drawStar(sx, sy, 9, "#facc15");
    ctx.save();
    ctx.font = "bold 11px sans-serif";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(8,15,32,0.92)";
    ctx.fillStyle = "#fef08a";
    ctx.strokeText("Shared Noise z_T", sx + 10, sy - 6);
    ctx.fillText("Shared Noise z_T", sx + 10, sy - 6);

    const [ux, uy] = worldToPx(p.uncond_points[N][0], p.uncond_points[N][1], b);
    ctx.fillStyle = "#e2e8f0";
    ctx.strokeText("z₀^∅ (Uncond)", ux - 30, uy + 16);
    ctx.fillText("z₀^∅ (Uncond)", ux - 30, uy + 16);

    const [cx, cy] = worldToPx(p.cond_points[N][0], p.cond_points[N][1], b);
    ctx.fillStyle = "#60a5fa";
    ctx.strokeText("z₀^y (Dog + Hat)", cx - 35, cy - 12);
    ctx.fillText("z₀^y (Dog + Hat)", cx - 35, cy - 12);
    ctx.restore();
  }}

  function renderSvgCharts() {{
    const steps = DATA.steps;
    const W = 420, H = 170, PL = 42, PR = 14, PT = 14, PB = 26;
    const pw = W - PL - PR, ph = H - PT - PB;
    const xOf = idx => PL + (idx / Math.max(N, 1)) * pw;

    const makeSeriesPath = (vals, maxV, minV = 0) => {{
      const span = Math.max(maxV - minV, 1e-6);
      return vals.map((v, i) => `${{i === 0 ? "M" : "L"}}${{xOf(i).toFixed(1)}},${{(PT + (1 - (v - minV) / span) * ph).toFixed(1)}}`).join(" ");
    }};

    const paintChart = (svgId, series) => {{
      const svg = document.getElementById(svgId);
      const curX = xOf(curStep).toFixed(1);
      let inner = `<line x1="${{PL}}" y1="${{PT+ph}}" x2="${{W-PR}}" y2="${{PT+ph}}" stroke="#334155" stroke-width="1"/>`;
      inner += `<line x1="${{PL}}" y1="${{PT}}" x2="${{PL}}" y2="${{PT+ph}}" stroke="#334155" stroke-width="1"/>`;
      for (const s of series) {{
        const d = makeSeriesPath(s.vals, s.max, s.min || 0);
        inner += `<path d="${{d}}" fill="none" stroke="${{s.color}}" stroke-width="2.2"/>`;
      }}
      inner += `<line x1="${{curX}}" y1="${{PT}}" x2="${{curX}}" y2="${{PT+ph}}" stroke="#facc15" stroke-width="1.8" stroke-dasharray="4,3"/>`;
      // Legend
      let lx = PL + 6;
      series.forEach(s => {{
        inner += `<rect x="${{lx}}" y="4" width="10" height="4" fill="${{s.color}}"/><text x="${{lx+14}}" y="9" fill="#cbd5e1" font-size="9.5">${{s.label}}</text>`;
        lx += 135;
      }});
      inner += `<text x="${{PL}}" y="${{H-6}}" fill="#94a3b8" font-size="9.5">Step 0 (t=1000)</text>`;
      inner += `<text x="${{W-PR-52}}" y="${{H-6}}" fill="#94a3b8" font-size="9.5">Step ${{N}}</text>`;
      svg.innerHTML = inner;
    }};

    const gcfgVals = steps.map(s => s.vec_norms ? s.vec_norms.g_cfg_norm : 0);
    const depsVals = steps.map(s => s.delta_eps_norm || 0);
    paintChart("chartCfg", [
      {{ label: "||g_CFG||₂", vals: gcfgVals, max: Math.max(...gcfgVals, 1), color: "#ef4444" }},
      {{ label: "||ε_y - ε_∅||₂", vals: depsVals, max: Math.max(...depsVals, 1), color: "#f59e0b" }},
    ]);

    const divVals = steps.map(s => s.divergence_l2);
    const cosScaled = steps.map(s => Math.max(0, s.cosine_sim));
    paintChart("chartDiv", [
      {{ label: "L2 Divergence", vals: divVals, max: Math.max(...divVals, 1), color: "#3b82f6" }},
      {{ label: "Cosine Sim (0..1)", vals: cosScaled, max: 1.0, min: 0.0, color: "#14b8a6" }},
    ]);

    const sigVals = steps.map(s => s.sqrt_alpha_bar);
    const noiseVals = steps.map(s => s.sqrt_one_minus_alpha_bar);
    paintChart("chartSched", [
      {{ label: "Signal √ᾱ_t", vals: sigVals, max: 1.0, min: 0.0, color: "#14b8a6" }},
      {{ label: "Noise √(1-ᾱ_t)", vals: noiseVals, max: 1.0, min: 0.0, color: "#ef4444" }},
    ]);
  }}

  function updateUI() {{
    const s = DATA.steps[curStep];
    stepSlider.value = curStep;
    stepBadge.textContent = `Step ${{curStep}} / ${{N}} (t = ${{s.timestep}})`;

    const layer = imgLayerSel.value;
    document.getElementById("uMainImg").src = s.uncond[layer] || s.uncond.decoded_zt;
    document.getElementById("uLatImg").src = s.uncond.latent_rgb;
    document.getElementById("uZ0Img").src = s.uncond.decoded_z0_hat;
    document.getElementById("uChImg").src = s.uncond.channels_strip;
    document.getElementById("uStdTag").textContent = `μ=${{s.uncond.mean.toFixed(2)}} σ=${{s.uncond.std.toFixed(2)}}`;

    document.getElementById("cMainImg").src = s.cond[layer] || s.cond.decoded_zt;
    document.getElementById("cLatImg").src = s.cond.latent_rgb;
    document.getElementById("cCfgMapImg").src = s.cond.cfg_heatmap;
    document.getElementById("cChImg").src = s.cond.channels_strip;
    document.getElementById("cStdTag").textContent = `μ=${{s.cond.mean.toFixed(2)}} σ=${{s.cond.std.toFixed(2)}}`;

    const vn = s.vec_norms;
    document.getElementById("stGcfg").textContent = vn ? vn.g_cfg_norm.toFixed(2) : "End (0.00)";
    document.getElementById("stVunc").textContent = vn ? vn.v_uncond_norm.toFixed(2) : "End (0.00)";
    document.getElementById("stAng").textContent = vn ? `${{vn.angle_deg.toFixed(1)}}°` : "-";
    document.getElementById("stDiv").textContent = s.divergence_l2.toFixed(2);
    document.getElementById("stCos").textContent = s.cosine_sim.toFixed(3);
    document.getElementById("stSig").textContent = s.sqrt_alpha_bar.toFixed(3);
    document.getElementById("stNoise").textContent = s.sqrt_one_minus_alpha_bar.toFixed(3);
    document.getElementById("stDeps").textContent = s.delta_eps_norm.toFixed(2);

    drawCanvas();
    renderSvgCharts();
  }}

  function setStep(idx) {{
    curStep = Math.max(0, Math.min(N, idx));
    updateUI();
  }}

  function stopPlay() {{
    if (timer) {{ clearInterval(timer); timer = null; }}
    document.getElementById("btnPlay").innerHTML = "&#9654; Play";
  }}

  function togglePlay() {{
    if (timer) {{ stopPlay(); return; }}
    if (curStep >= N) curStep = 0;
    document.getElementById("btnPlay").innerHTML = "&#9646;&#9646; Pause";
    timer = setInterval(() => {{
      if (curStep >= N) {{ stopPlay(); return; }}
      setStep(curStep + 1);
    }}, Number(speedSel.value));
  }}

  // Canvas hover -> show decoded image of the latent region under cursor
  canvas.addEventListener("mousemove", e => {{
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const pw = canvas.width - PAD.l - PAD.r;
    const ph = canvas.height - PAD.t - PAD.b;
    if (x < PAD.l || x > PAD.l + pw || y < PAD.t || y > PAD.t + ph) {{
      hoverLens.style.display = "none";
      return;
    }}
    const p = DATA.planes[planeSel.value];
    const [u, v] = pxToWorld(x, y, p.bounds);
    const col = Math.min(p.grid_cols - 1, Math.max(0, Math.floor(((u - p.bounds.u_min) / (p.bounds.u_max - p.bounds.u_min)) * p.grid_cols)));
    const row = Math.min(p.grid_rows - 1, Math.max(0, Math.floor(((v - p.bounds.v_min) / (p.bounds.v_max - p.bounds.v_min)) * p.grid_rows)));
    const tile = p.atlas_tiles[row * p.grid_cols + col];
    if (!tile) return;

    lensImg.src = atlasSel.value === "decoded" ? tile.decoded_url : tile.latent_url;
    lensMeta.textContent = `Region (${{tile.u}}, ${{tile.v}}) [tile ${{col+1}}×${{row+1}}]`;
    hoverLens.style.display = "block";

    const boxRect = canvas.parentElement.getBoundingClientRect();
    let lx = e.clientX - boxRect.left + 14;
    let ly = e.clientY - boxRect.top - 10;
    if (lx + 168 > boxRect.width) lx = e.clientX - boxRect.left - 172;
    if (ly + 185 > boxRect.height) ly = boxRect.height - 190;
    hoverLens.style.left = `${{Math.max(4, lx)}}px`;
    hoverLens.style.top = `${{Math.max(4, ly)}}px`;
  }});
  canvas.addEventListener("mouseleave", () => {{ hoverLens.style.display = "none"; }});

  // Click nearest step node on canvas to jump directly to that step
  canvas.addEventListener("click", e => {{
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const p = DATA.planes[planeSel.value];
    let bestIdx = -1, bestDist = 20;
    for (const pts of [p.cond_points, p.uncond_points]) {{
      pts.forEach((pt, idx) => {{
        const [px, py] = worldToPx(pt[0], pt[1], p.bounds);
        const d = Math.hypot(px - x, py - y);
        if (d < bestDist) {{ bestDist = d; bestIdx = idx; }}
      }});
    }}
    if (bestIdx >= 0) {{ stopPlay(); setStep(bestIdx); }}
  }});

  document.getElementById("btnFirst").onclick = () => {{ stopPlay(); setStep(0); }};
  document.getElementById("btnPrev").onclick = () => {{ stopPlay(); setStep(curStep - 1); }};
  document.getElementById("btnNext").onclick = () => {{ stopPlay(); setStep(curStep + 1); }};
  document.getElementById("btnLast").onclick = () => {{ stopPlay(); setStep(N); }};
  document.getElementById("btnPlay").onclick = togglePlay;
  stepSlider.oninput = e => {{ stopPlay(); setStep(Number(e.target.value)); }};
  speedSel.onchange = () => {{ if (timer) {{ stopPlay(); togglePlay(); }} }};
  [planeSel, atlasSel, atlasAlpha, arrowScaleSel, chkCF, imgLayerSel].forEach(el => el.addEventListener("change", updateUI));
  atlasAlpha.addEventListener("input", drawCanvas);

  window.addEventListener("keydown", e => {{
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;
    if (e.key === "ArrowLeft") {{ e.preventDefault(); stopPlay(); setStep(curStep - 1); }}
    else if (e.key === "ArrowRight") {{ e.preventDefault(); stopPlay(); setStep(curStep + 1); }}
    else if (e.key === " ") {{ e.preventDefault(); togglePlay(); }}
  }});

  updateUI();
}})();
</script>
</body>
</html>
"""


@torch.inference_mode()
def run_full_export(
    prompt: str,
    seed: int,
    num_steps: int,
    cfg: float,
    resolution: int,
    grid_cols: int,
    grid_rows: int,
    out_dir: str,
    sampler: str = "DDIM",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sampler_clean = sampler.strip().upper()
    thumb_sz = 224 if num_steps >= 40 else 256
    thumb_q = 78 if num_steps >= 40 else 86
    atlas_q = 78 if num_steps >= 40 else 82

    t_all0 = time.perf_counter()
    sd_pipe = load_sd15_pipeline()
    dev = sd_pipe.unet.device
    dt = sd_pipe.unet.dtype

    scheduler = build_scheduler(sd_pipe.scheduler.config, sampler=sampler_clean, is_latent=True)
    scheduler.set_timesteps(num_steps, device=dev)
    timesteps = scheduler.timesteps
    alphas_cumprod = scheduler.alphas_cumprod.detach().cpu()

    lat_h = resolution // 8
    lat_w = resolution // 8
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z_init = torch.randn((1, 4, lat_h, lat_w), generator=gen, dtype=torch.float32).to(dev, dt)

    emb_null = encode_prompt_sd(sd_pipe, "", dev, dt)
    emb_cond = encode_prompt_sd(sd_pipe, prompt, dev, dt)
    batch_emb = torch.cat([emb_null, emb_cond, emb_null, emb_cond], dim=0)

    z_u = z_init.clone()
    z_c = z_init.clone()

    uncond_latents: list[np.ndarray] = [z_u[0].detach().float().cpu().numpy()]
    cond_latents: list[np.ndarray] = [z_c[0].detach().float().cpu().numpy()]
    uncond_decoded_full: list[np.ndarray] = []
    cond_decoded_full: list[np.ndarray] = []
    cfg_heatmaps: list[np.ndarray | None] = []
    timesteps_lbl: list[int] = [int(getattr(scheduler.config, "num_train_timesteps", 1000))]

    dec0 = decode_latent_to_rgb(sd_pipe, z_init)
    uncond_decoded_full.append(dec0.copy())
    cond_decoded_full.append(dec0.copy())

    # Outgoing vector fields defined at indices 0..num_steps-1 (final index = None)
    cond_vec_uncond: list[np.ndarray | None] = []
    cond_vec_gcfg: list[np.ndarray | None] = []
    cond_vec_cfg: list[np.ndarray | None] = []
    uncond_vec_actual: list[np.ndarray | None] = []
    uncond_vec_gcfg_cf: list[np.ndarray | None] = []

    step_records: list[dict[str, Any]] = []

    def _stats(arr: np.ndarray) -> tuple[float, float]:
        return float(arr.mean()), float(arr.std())

    # Temporary placeholder for step 0; we will populate z0_hat & heatmap from first UNet step
    u0_np = uncond_latents[0]
    m0, s0 = _stats(u0_np)
    step_records.append(
        {
            "step_idx": 0,
            "timestep": timesteps_lbl[0],
            "alpha_bar": 0.0,
            "sqrt_alpha_bar": 0.0,
            "sqrt_one_minus_alpha_bar": 1.0,
            "divergence_l2": 0.0,
            "cosine_sim": 1.0,
            "delta_eps_norm": 0.0,
            "vec_norms": None,
            "uncond": {
                "mean": m0,
                "std": s0,
                "decoded_zt": _np_to_jpeg_data_url(dec0, resize_to=thumb_sz, quality=thumb_q),
                "decoded_z0_hat": _np_to_jpeg_data_url(dec0, resize_to=thumb_sz, quality=thumb_q),
                "latent_rgb": _np_to_jpeg_data_url(latent_to_rgb(u0_np, upscale=2), resize_to=128, quality=thumb_q),
                "channels_strip": _np_to_jpeg_data_url(_channels_strip_rgb(u0_np, tile_size=64), quality=thumb_q),
            },
            "cond": {
                "mean": m0,
                "std": s0,
                "decoded_zt": _np_to_jpeg_data_url(dec0, resize_to=thumb_sz, quality=thumb_q),
                "decoded_z0_hat": _np_to_jpeg_data_url(dec0, resize_to=thumb_sz, quality=thumb_q),
                "latent_rgb": _np_to_jpeg_data_url(latent_to_rgb(u0_np, upscale=2), resize_to=128, quality=thumb_q),
                "channels_strip": _np_to_jpeg_data_url(_channels_strip_rgb(u0_np, tile_size=64), quality=thumb_q),
                "cfg_heatmap": None,
            },
        }
    )
    cfg_heatmaps.append(None)

    for step_i, t in enumerate(timesteps):
        t_int = int(t.item())
        scaled_u = scheduler.scale_model_input(z_u, t)
        scaled_c = scheduler.scale_model_input(z_c, t)
        batch_z = torch.cat([scaled_u, scaled_u, scaled_c, scaled_c], dim=0)

        eps_all = sd_pipe.unet(batch_z, t, encoder_hidden_states=batch_emb).sample
        eps_uu, eps_uc = eps_all[0:1], eps_all[1:2]
        eps_cu, eps_cc = eps_all[2:3], eps_all[3:4]

        eps_u_cf = eps_uu + cfg * (eps_uc - eps_uu)
        eps_c_cfg = eps_cu + cfg * (eps_cc - eps_cu)

        def _step_gen() -> torch.Generator:
            gen_dev = dev.type if dev.type == "cuda" else "cpu"
            return torch.Generator(device=gen_dev).manual_seed(seed + 10000 + step_i)

        out_u = scheduler.step(eps_uu, t, z_u, generator=_step_gen())
        out_u_cf = scheduler.step(eps_u_cf, t, z_u, generator=_step_gen())
        out_c_if_u = scheduler.step(eps_cu, t, z_c, generator=_step_gen())
        out_c_cfg = scheduler.step(eps_c_cfg, t, z_c, generator=_step_gen())

        z_u_next = out_u.prev_sample
        z_c_next = out_c_cfg.prev_sample

        # Outgoing displacement vectors at snapshot index `step_i` (moving to `step_i + 1`)
        z_u_cur_np = uncond_latents[-1]
        z_c_cur_np = cond_latents[-1]
        z_u_next_np = z_u_next[0].detach().float().cpu().numpy()
        z_c_next_np = z_c_next[0].detach().float().cpu().numpy()
        z_c_if_u_np = out_c_if_u.prev_sample[0].detach().float().cpu().numpy()
        z_u_if_cfg_np = out_u_cf.prev_sample[0].detach().float().cpu().numpy()

        v_unc = z_c_if_u_np - z_c_cur_np
        g_cfg = z_c_next_np - z_c_if_u_np
        v_cfg = z_c_next_np - z_c_cur_np
        cond_vec_uncond.append(v_unc)
        cond_vec_gcfg.append(g_cfg)
        cond_vec_cfg.append(v_cfg)

        v_u_act = z_u_next_np - z_u_cur_np
        g_u_cf = z_u_if_cfg_np - z_u_next_np
        uncond_vec_actual.append(v_u_act)
        uncond_vec_gcfg_cf.append(g_u_cf)

        n_vu = float(np.linalg.norm(v_unc))
        n_gc = float(np.linalg.norm(g_cfg))
        n_vc = float(np.linalg.norm(v_cfg))
        cos_vg = float(np.dot(v_unc.reshape(-1), g_cfg.reshape(-1)) / max(n_vu * n_gc, 1e-8))
        ang_deg = float(math.degrees(math.acos(min(1.0, max(-1.0, cos_vg)))))

        diff_eps = (eps_cc - eps_cu).detach().float()
        deps_norm = float(torch.linalg.vector_norm(diff_eps).item())
        hmap = torch.linalg.vector_norm(diff_eps, dim=1)[0].cpu().numpy().astype(np.float32)

        # Populate step 0's initial CFG sensitivity heatmap & pred clean preview from t=timesteps[0]
        if step_i == 0:
            hmap_url0 = _np_to_jpeg_data_url(guidance_map_to_rgb(hmap, upscale=2), resize_to=128, quality=thumb_q)
            step_records[0]["cond"]["cfg_heatmap"] = hmap_url0
            step_records[0]["delta_eps_norm"] = deps_norm
            step_records[0]["vec_norms"] = {
                "v_uncond_norm": n_vu,
                "g_cfg_norm": n_gc,
                "v_cfg_norm": n_vc,
                "angle_deg": ang_deg,
            }
            cfg_heatmaps[0] = hmap
        else:
            step_records[step_i]["vec_norms"] = {
                "v_uncond_norm": n_vu,
                "g_cfg_norm": n_gc,
                "v_cfg_norm": n_vc,
                "angle_deg": ang_deg,
            }

        # Advance state to `step_i + 1`
        z_u = z_u_next
        z_c = z_c_next
        uncond_latents.append(z_u_next_np)
        cond_latents.append(z_c_next_np)
        timesteps_lbl.append(t_int)
        cfg_heatmaps.append(hmap)

        # Decode z_u_next, z_c_next, z0_hat_u, z0_hat_c
        dec_batch = _decode_batch_to_rgb(
            sd_pipe,
            torch.cat([z_u_next, out_u.pred_original_sample, z_c_next, out_c_cfg.pred_original_sample], dim=0),
        )
        dec_u, dec_u0, dec_c, dec_c0 = dec_batch
        uncond_decoded_full.append(dec_u)
        cond_decoded_full.append(dec_c)

        ab = float(alphas_cumprod[min(max(t_int, 0), alphas_cumprod.numel() - 1)].item())
        div_l2 = float(np.linalg.norm(z_c_next_np - z_u_next_np))
        cos_sim = float(
            F.cosine_similarity(
                torch.from_numpy(z_c_next_np.reshape(1, -1)),
                torch.from_numpy(z_u_next_np.reshape(1, -1)),
                dim=-1,
            ).item()
        )
        mu_u, sd_u = _stats(z_u_next_np)
        mu_c, sd_c = _stats(z_c_next_np)

        step_records.append(
            {
                "step_idx": step_i + 1,
                "timestep": t_int,
                "alpha_bar": round(ab, 6),
                "sqrt_alpha_bar": round(math.sqrt(max(ab, 0.0)), 6),
                "sqrt_one_minus_alpha_bar": round(math.sqrt(max(1.0 - ab, 0.0)), 6),
                "divergence_l2": round(div_l2, 4),
                "cosine_sim": round(cos_sim, 5),
                "delta_eps_norm": round(deps_norm, 4),
                "vec_norms": None,  # filled on next step if step_i + 1 < num_steps
                "uncond": {
                    "mean": round(mu_u, 4),
                    "std": round(sd_u, 4),
                    "decoded_zt": _np_to_jpeg_data_url(dec_u, resize_to=thumb_sz, quality=thumb_q),
                    "decoded_z0_hat": _np_to_jpeg_data_url(dec_u0, resize_to=thumb_sz, quality=thumb_q),
                    "latent_rgb": _np_to_jpeg_data_url(latent_to_rgb(z_u_next_np, upscale=2), resize_to=128, quality=thumb_q),
                    "channels_strip": _np_to_jpeg_data_url(_channels_strip_rgb(z_u_next_np, tile_size=64), quality=thumb_q),
                },
                "cond": {
                    "mean": round(mu_c, 4),
                    "std": round(sd_c, 4),
                    "decoded_zt": _np_to_jpeg_data_url(dec_c, resize_to=thumb_sz, quality=thumb_q),
                    "decoded_z0_hat": _np_to_jpeg_data_url(dec_c0, resize_to=thumb_sz, quality=thumb_q),
                    "latent_rgb": _np_to_jpeg_data_url(latent_to_rgb(z_c_next_np, upscale=2), resize_to=128, quality=thumb_q),
                    "channels_strip": _np_to_jpeg_data_url(_channels_strip_rgb(z_c_next_np, tile_size=64), quality=thumb_q),
                    "cfg_heatmap": _np_to_jpeg_data_url(guidance_map_to_rgb(hmap, upscale=2), resize_to=128, quality=thumb_q),
                },
            }
        )
        print(f"  Step {step_i + 1:2d}/{num_steps} (t={t_int:3d}) | div_L2={div_l2:6.2f} | ||g_cfg||={n_gc:5.2f} | ang={ang_deg:5.1f}deg")

    # Final step has no outgoing transition vector
    cond_vec_uncond.append(None)
    cond_vec_gcfg.append(None)
    cond_vec_cfg.append(None)
    uncond_vec_actual.append(None)
    uncond_vec_gcfg_cf.append(None)

    # Save 512x512 final PNGs
    Image.fromarray(uncond_decoded_full[-1], mode="RGB").save(os.path.join(out_dir, "uncond_final.png"))
    Image.fromarray(cond_decoded_full[-1], mode="RGB").save(os.path.join(out_dir, "cond_final.png"))

    # Save publication filmstrip + GIF
    _make_filmstrip_and_gif(
        prompt=prompt,
        seed=seed,
        cfg=cfg,
        uncond_decoded_full=uncond_decoded_full,
        cond_decoded_full=cond_decoded_full,
        uncond_latents=uncond_latents,
        cond_latents=cond_latents,
        cfg_heatmaps=cfg_heatmaps,
        timesteps_lbl=timesteps_lbl,
        out_dir=out_dir,
        sampler=sampler_clean,
    )

    # Build both 2D projections + 8x6 Latent Region Atlases
    latent_shape = (4, lat_h, lat_w)
    sem_origin, sem_basis = _build_semantic_basis(uncond_latents[0], uncond_latents[-1], cond_latents[-1])
    plane_semantic = _build_plane_data(
        name="semantic",
        title="Semantic Plane",
        subtitle="Exact plane through z_T (origin), z₀^∅, and z₀^y",
        x_label="Denoising Progress (z_T → endpoint midpoint)",
        y_label="Conditioning Axis (Uncond ∅ → Prompt y)",
        origin=sem_origin,
        basis=sem_basis,
        uncond_latents=uncond_latents,
        cond_latents=cond_latents,
        cond_vec_uncond=cond_vec_uncond,
        cond_vec_gcfg=cond_vec_gcfg,
        cond_vec_cfg=cond_vec_cfg,
        uncond_vec_actual=uncond_vec_actual,
        uncond_vec_gcfg_cf=uncond_vec_gcfg_cf,
        sd_pipe=sd_pipe,
        latent_shape=latent_shape,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        atlas_quality=atlas_q,
    )

    pca_origin, pca_basis, pca_var = _build_pca_basis(uncond_latents + cond_latents)
    plane_pca = _build_plane_data(
        name="pca",
        title="2D PCA Plane",
        subtitle=f"Joint PCA across both trajectories ({pca_var[0]:.1f}% + {pca_var[1]:.1f}% var.)",
        x_label=f"PC1 ({pca_var[0]:.1f}% variance)",
        y_label=f"PC2 ({pca_var[1]:.1f}% variance)",
        origin=pca_origin,
        basis=pca_basis,
        uncond_latents=uncond_latents,
        cond_latents=cond_latents,
        cond_vec_uncond=cond_vec_uncond,
        cond_vec_gcfg=cond_vec_gcfg,
        cond_vec_cfg=cond_vec_cfg,
        uncond_vec_actual=uncond_vec_actual,
        uncond_vec_gcfg_cf=uncond_vec_gcfg_cf,
        sd_pipe=sd_pipe,
        latent_shape=latent_shape,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        atlas_quality=atlas_q,
    )

    dev_info = get_device_info()
    del sd_pipe
    free_gpu_memory()

    elapsed = round(time.perf_counter() - t_all0, 2)
    payload = {
        "meta": {
            "prompt": prompt,
            "seed": seed,
            "num_steps": num_steps,
            "cfg_scale": cfg,
            "resolution": resolution,
            "sampler": sampler_clean,
            "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "gpu_name": dev_info["gpu_name"],
            "elapsed_sec": elapsed,
            "pca_var_pct": pca_var,
        },
        "steps": step_records,
        "planes": {
            "semantic": plane_semantic,
            "pca": plane_pca,
        },
    }

    html_text = build_interactive_html(payload)
    html_path = os.path.join(out_dir, "interactive_latent_diffusion.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    # Save compact telemetry JSON without heavy base64 strings
    compact_steps = []
    for rec in step_records:
        compact_steps.append(
            {
                "step_idx": rec["step_idx"],
                "timestep": rec["timestep"],
                "alpha_bar": rec["alpha_bar"],
                "sqrt_alpha_bar": rec["sqrt_alpha_bar"],
                "sqrt_one_minus_alpha_bar": rec["sqrt_one_minus_alpha_bar"],
                "divergence_l2": rec["divergence_l2"],
                "cosine_sim": rec["cosine_sim"],
                "delta_eps_norm": rec["delta_eps_norm"],
                "vec_norms": rec["vec_norms"],
                "uncond_mean": rec["uncond"]["mean"],
                "uncond_std": rec["uncond"]["std"],
                "cond_mean": rec["cond"]["mean"],
                "cond_std": rec["cond"]["std"],
            }
        )
    compact_planes = {}
    for pname, pdata in payload["planes"].items():
        compact_planes[pname] = {
            "title": pdata["title"],
            "subtitle": pdata["subtitle"],
            "bounds": pdata["bounds"],
            "grid_cols": pdata["grid_cols"],
            "grid_rows": pdata["grid_rows"],
            "uncond_points": pdata["uncond_points"],
            "cond_points": pdata["cond_points"],
            "cond_vectors": pdata["cond_vectors"],
            "uncond_vectors": pdata["uncond_vectors"],
            "atlas_coords": [{"col": t["col"], "row": t["row"], "u": t["u"], "v": t["v"]} for t in pdata["atlas_tiles"]],
        }
    json_path = os.path.join(out_dir, "simulation_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": payload["meta"], "steps": compact_steps, "planes": compact_planes}, f, indent=2)

    print(f"Export complete in {elapsed}s -> {out_dir}")
    print(f"HTML size: {os.path.getsize(html_path) / (1024 * 1024):.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export step-by-step Latent Diffusion + CFG + Region Atlas HTML.")
    parser.add_argument("--prompt", type=str, default="a dog with birthday party hat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default="/content/dog_party_hat_simulation")
    parser.add_argument("--sampler", type=str, default="DDIM", choices=["DDIM", "DDPM"])
    parser.add_argument("--preview-seeds", type=str, default="")
    args = parser.parse_args()

    if args.preview_seeds.strip():
        seeds = [int(s.strip()) for s in args.preview_seeds.split(",") if s.strip()]
        run_seed_preview(
            prompt=args.prompt,
            seeds=seeds,
            num_steps=20,
            cfg=args.cfg,
            resolution=args.resolution,
            out_dir=args.out_dir,
            sampler=args.sampler,
        )
    else:
        run_full_export(
            prompt=args.prompt,
            seed=args.seed,
            num_steps=args.steps,
            cfg=args.cfg,
            resolution=args.resolution,
            grid_cols=args.grid_cols,
            grid_rows=args.grid_rows,
            out_dir=args.out_dir,
            sampler=args.sampler,
        )


if __name__ == "__main__":
    main()
