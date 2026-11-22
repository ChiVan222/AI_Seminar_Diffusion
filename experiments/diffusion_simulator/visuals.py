"""Visualization, latent projection, diagnostic figures, and PNG/GIF exporters."""

from __future__ import annotations

import io
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA


def _get_cmap(name: str):
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps.get_cmap(name)
    return plt.get_cmap(name)

from .engine import (
    LatentForwardBackwardResult,
    LatentStepSnapshot,
    PixelDDPMResult,
    PixelStepSnapshot,
    UncondVsCondResult,
    BranchTrajectory,
)


plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# Standard Stable Diffusion 1.5 linear projection from 4 latent channels -> sRGB
SD15_LATENT_RGB_FACTORS = np.array(
    [
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473],
    ],
    dtype=np.float32,
)

PALETTE_COLORS = ["#64748B", "#19468C", "#007880", "#B92D2D"]


def _evenly_spaced_indices(total: int, num_cols: int) -> list[int]:
    if total <= 0:
        return []
    if total <= num_cols:
        return list(range(total))
    idx = np.linspace(0, total - 1, num_cols).round().astype(int).tolist()
    # Deduplicate preserving order
    seen: set[int] = set()
    out: list[int] = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 160) -> bytes:
    """Serialize a matplotlib figure to PNG bytes and close it to prevent leaks."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _upscale_nearest(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr
    h, w = arr.shape[:2]
    pil = Image.fromarray(arr, mode="RGB").resize((w * scale, h * scale), Image.Resampling.NEAREST)
    return np.asarray(pil, dtype=np.uint8)


def latent_to_rgb(z: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Project a `(4, H, W)` latent tensor into crisp RGB via SD 1.5 basis + nearest upscale."""
    if z.shape[0] != 4:
        # Generic fallback: use first 3 channels or repeat grayscale
        c = min(3, z.shape[0])
        rgb = np.transpose(z[:c], (1, 2, 0))
        if c < 3:
            rgb = np.repeat(rgb[..., :1], 3, axis=-1)
    else:
        # (4, H, W) -> (H, W, 4) @ (4, 3) -> (H, W, 3)
        rgb = np.einsum("chw,cd->hwd", z.astype(np.float32), SD15_LATENT_RGB_FACTORS)
    rgb = np.clip((rgb * 0.5 + 0.5) * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return _upscale_nearest(rgb, upscale)


def guidance_map_to_rgb(gmap: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Render `(H, W)` spatial guidance delta magnitude as an inferno RGB heatmap."""
    arr = np.nan_to_num(gmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    hi = float(np.percentile(arr, 99.5)) if arr.size > 0 else 1.0
    lo = float(arr.min()) if arr.size > 0 else 0.0
    norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    colored = (_get_cmap("inferno")(norm)[..., :3] * 255.0).round().astype(np.uint8)
    return _upscale_nearest(colored, upscale)


def render_uncond_vs_cond_filmstrip(res: UncondVsCondResult, num_cols: int = 7) -> bytes:
    """Render publication-quality comparison strip across branches (Decoded, z0_hat, Latent RGB)."""
    n_steps = len(res.branches[0].steps)
    idxs = _evenly_spaced_indices(n_steps, num_cols)
    ncols = len(idxs)
    nrows = len(res.branches) * 2 + 1  # each branch: Decoded + Latent RGB, plus Conditioned z0_hat row

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols, 2.05 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    if ncols == 1:
        axes = axes[:, None]

    row_cursor = 0
    for b_idx, branch in enumerate(res.branches):
        for col, s_idx in enumerate(idxs):
            snap = branch.steps[s_idx]
            ax_img = axes[row_cursor, col]
            ax_img.imshow(snap.decoded_zt)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if row_cursor == 0:
                ax_img.set_title(f"Step {snap.step_idx}\nt = {snap.timestep}", fontsize=9, fontweight="bold")
            if col == 0:
                ax_img.set_ylabel(f"{branch.label}\nDecoded D(z_t)", fontsize=8.5, fontweight="bold")

            ax_lat = axes[row_cursor + 1, col]
            ax_lat.imshow(latent_to_rgb(snap.z_t, upscale=4))
            ax_lat.set_xticks([])
            ax_lat.set_yticks([])
            if col == 0:
                ax_lat.set_ylabel(f"{branch.label}\nLatent z_t RGB", fontsize=8.5)
        row_cursor += 2

    # Final row: Predicted clean D(z0_hat) of primary conditioned branch
    cond_branch = res.branches[1] if len(res.branches) > 1 else res.branches[0]
    for col, s_idx in enumerate(idxs):
        snap = cond_branch.steps[s_idx]
        ax = axes[row_cursor, col]
        img = snap.decoded_z0_hat if snap.decoded_z0_hat is not None else snap.decoded_zt
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(f"{cond_branch.label}\nPred. Clean D(ẑ₀)", fontsize=8.5, fontweight="bold")

    fig.suptitle(
        f"Unconditioned vs Conditioned LDM ({res.sampler}, {res.num_steps} steps, seed={res.seed})\n"
        f"Prompt: \"{res.prompt[:90]}\"",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return fig_to_png_bytes(fig, dpi=160)


def render_pca_trajectory_fig(
    trajectories: Sequence[tuple[str, Sequence[np.ndarray], str]],
    current_idx: int | None = None,
    title: str = "2D PCA Trajectory in Latent Space",
    timesteps: Sequence[int] | None = None,
) -> tuple[plt.Figure, bytes]:
    """Fit a joint 2D PCA across trajectories of `(4, H, W)` latents and plot paths."""
    flat_all: list[np.ndarray] = []
    lengths: list[int] = []
    for _, zs, _ in trajectories:
        arrs = [z.reshape(-1).astype(np.float32) for z in zs]
        flat_all.extend(arrs)
        lengths.append(len(arrs))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if len(flat_all) < 2:
        ax.text(0.5, 0.5, "Not enough trajectory steps for PCA", ha="center", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        return fig, buf.getvalue()

    mat = np.stack(flat_all, axis=0)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(mat)
    var_pct = pca.explained_variance_ratio_ * 100.0

    offset = 0
    for idx, (label, _, color) in enumerate(trajectories):
        n = lengths[idx]
        pts = proj[offset : offset + n]
        offset += n
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=2.2, label=label, alpha=0.9)
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=22, alpha=0.65)
        ax.scatter(pts[0, 0], pts[0, 1], marker="*", s=150, color=color, edgecolors="black", linewidths=0.7, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], marker="o", s=85, color=color, edgecolors="black", linewidths=1.0, zorder=5)
        if n >= 3:
            mid = n // 2
            ax.annotate(
                "",
                xy=(pts[mid + 1, 0], pts[mid + 1, 1]),
                xytext=(pts[mid, 0], pts[mid, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
            )
        if current_idx is not None and 0 <= current_idx < n:
            ax.scatter(
                pts[current_idx, 0],
                pts[current_idx, 1],
                marker="D",
                s=115,
                facecolors="none",
                edgecolors="#F59E0B",
                linewidths=2.4,
                zorder=6,
            )

    if timesteps is not None and len(timesteps) == lengths[0]:
        pts0 = proj[: lengths[0]]
        ax.annotate(f"start (t={timesteps[0]})", (pts0[0, 0], pts0[0, 1]), xytext=(6, 6), textcoords="offset points", fontsize=8)

    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% var.)", fontsize=9.5)
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% var.)", fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, loc="best", framealpha=0.9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return fig, buf.getvalue()


def render_divergence_curves_fig(
    res: UncondVsCondResult,
    current_idx: int | None = None,
) -> tuple[plt.Figure, bytes]:
    """Plot L2 divergence, cosine similarity, and CFG norm vs step index."""
    steps_x = [s.step_idx for s in res.branches[0].steps]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))

    for b_idx, branch in enumerate(res.branches[1:], start=1):
        color = PALETTE_COLORS[b_idx % len(PALETTE_COLORS)]
        bname = branch.name
        axes[0].plot(steps_x, res.divergence_l2[bname], lw=2.0, color=color, label=branch.label)
        axes[1].plot(steps_x, res.cosine_sim[bname], lw=2.0, color=color, label=branch.label)
        axes[2].plot(steps_x, res.guidance_delta_norm[bname], lw=2.0, color=color, label=branch.label)

    titles = [
        "L2 Latent Divergence vs ∅\n$\\|\\mathbf{z}_t^{\\mathrm{cond}} - \\mathbf{z}_t^{\\varnothing}\\|_2$",
        "Cosine Similarity vs ∅\n$\\cos(\\mathbf{z}_t^{\\mathrm{cond}}, \\mathbf{z}_t^{\\varnothing})$",
        "CFG Guidance Shift Norm\n$\\|\\epsilon_\\theta(z_t, y) - \\epsilon_\\theta(z_t, \\varnothing)\\|_2$",
    ]
    ylabels = ["L2 Distance", "Cosine Sim.", "Delta Noise Norm"]
    for ax, ttl, yl in zip(axes, titles, ylabels):
        ax.set_title(ttl, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Denoising Step", fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, loc="best")
        if current_idx is not None:
            ax.axvline(current_idx, color="#F59E0B", linestyle="--", lw=1.6)

    axes[1].set_ylim(-0.2, 1.05)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return fig, buf.getvalue()


def render_latent_inspector_fig(
    snap: LatentStepSnapshot,
    all_snaps: Sequence[LatentStepSnapshot],
    branch_label: str,
) -> tuple[plt.Figure, bytes]:
    """Render detailed 3-row inspector figure for a single latent step snapshot."""
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 0.95, 0.95], hspace=0.35, wspace=0.25)

    # Row 1: Latent RGB | Decoded D(z_t) | Predicted D(z0_hat) | Guidance or Eps Norm Map
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(latent_to_rgb(snap.z_t, upscale=4))
    ax0.set_title(f"Latent z_t RGB ({snap.z_t.shape[1]}x{snap.z_t.shape[2]}x4)", fontsize=9.5, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(snap.decoded_zt)
    ax1.set_title("VAE Decoded D(z_t)", fontsize=9.5, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(snap.decoded_z0_hat if snap.decoded_z0_hat is not None else snap.decoded_zt)
    ax2.set_title("Predicted Clean D(ẑ₀)" if snap.decoded_z0_hat is not None else "No ẑ₀ at prior", fontsize=9.5, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    if snap.guidance_map is not None:
        ax3.imshow(guidance_map_to_rgb(snap.guidance_map, upscale=4))
        ax3.set_title("CFG Guidance ||ε_c - ε_∅||₂", fontsize=9.5, fontweight="bold")
    elif snap.eps_norm_map is not None:
        ax3.imshow(guidance_map_to_rgb(snap.eps_norm_map, upscale=4))
        ax3.set_title("Predicted Noise ||ε_θ||₂", fontsize=9.5, fontweight="bold")
    else:
        ax3.imshow(latent_to_rgb(snap.z_t, upscale=4))
        ax3.set_title("Initial Prior Noise", fontsize=9.5, fontweight="bold")
    ax3.axis("off")

    # Row 2: 4 individual latent channels z[0..3]
    for ch in range(min(4, snap.z_t.shape[0])):
        ax_c = fig.add_subplot(gs[1, ch])
        ch_arr = snap.z_t[ch]
        ax_c.imshow(ch_arr, cmap="RdBu_r", vmin=-3.5, vmax=3.5, interpolation="nearest")
        ax_c.set_title(f"Channel z[{ch}] (μ={ch_arr.mean():.2f}, σ={ch_arr.std():.2f})", fontsize=8.8)
        ax_c.set_xticks([])
        ax_c.set_yticks([])

    # Row 3 left (2 cols): Latent value histogram vs Standard Normal N(0, 1)
    ax_hist = fig.add_subplot(gs[2, :2])
    vals = snap.z_t.reshape(-1)
    ax_hist.hist(vals, bins=50, range=(-4.5, 4.5), density=True, color="#19468C", alpha=0.72, label="Empirical z_t")
    xs = np.linspace(-4.5, 4.5, 250)
    gauss = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xs ** 2)
    ax_hist.plot(xs, gauss, "r--", lw=1.8, label="Prior N(0, 1)")
    ax_hist.set_title(f"Latent Distribution (μ={snap.mean:.3f}, σ={snap.std:.3f})", fontsize=9.5, fontweight="bold")
    ax_hist.set_xlabel("Latent value")
    ax_hist.set_ylabel("Density")
    ax_hist.grid(True, linestyle="--", alpha=0.35)
    ax_hist.legend(fontsize=8.5)

    # Row 3 right (2 cols): Schedule & empirical std trajectory
    ax_sch = fig.add_subplot(gs[2, 2:])
    sx = [s.step_idx for s in all_snaps]
    ax_sch.plot(sx, [s.sqrt_alpha_bar for s in all_snaps], color="#007880", lw=2.0, label="Signal √ᾱ_t")
    ax_sch.plot(sx, [s.sqrt_one_minus_alpha_bar for s in all_snaps], color="#B92D2D", lw=2.0, label="Noise √(1-ᾱ_t)")
    ax_sch.plot(sx, [s.std for s in all_snaps], color="#19468C", lw=1.8, linestyle=":", label="Empirical std(z_t)")
    ax_sch.axvline(snap.step_idx, color="#F59E0B", lw=1.8, linestyle="--", label=f"Current step {snap.step_idx}")
    ax_sch.set_title(f"Noise Schedule & Std (t={snap.timestep}, log-SNR={snap.log_snr:.2f})", fontsize=9.5, fontweight="bold")
    ax_sch.set_xlabel("Step Index")
    ax_sch.set_ylabel("Scale")
    ax_sch.grid(True, linestyle="--", alpha=0.35)
    ax_sch.legend(fontsize=8, ncol=2)

    fig.suptitle(f"Latent Space Inspector — {branch_label} | Step {snap.step_idx} (t={snap.timestep})", fontsize=12, fontweight="bold", y=0.98)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return fig, buf.getvalue()


def render_forward_backward_filmstrip(res: LatentForwardBackwardResult, num_cols: int = 7) -> bytes:
    """Render 4-row Forward (0->t_max) vs Backward (t_max->0) filmstrip in image + latent space."""
    fwd_idxs = _evenly_spaced_indices(len(res.forward_steps), num_cols)
    bwd_idxs = _evenly_spaced_indices(len(res.backward_steps), num_cols)
    ncols = max(len(fwd_idxs), len(bwd_idxs))

    fig, axes = plt.subplots(4, ncols, figsize=(2.15 * ncols, 8.3))
    if ncols == 1:
        axes = axes[:, None]

    for col in range(ncols):
        if col < len(fwd_idxs):
            fs = res.forward_steps[fwd_idxs[col]]
            axes[0, col].imshow(fs.decoded_zt)
            axes[0, col].set_title(f"Fwd t={fs.timestep}\n√ᾱ={fs.sqrt_alpha_bar:.2f}", fontsize=8.8, fontweight="bold", color="#B92D2D")
            axes[1, col].imshow(latent_to_rgb(fs.z_t, upscale=4))
        else:
            axes[0, col].axis("off")
            axes[1, col].axis("off")

        if col < len(bwd_idxs):
            bs = res.backward_steps[bwd_idxs[col]]
            axes[2, col].imshow(bs.decoded_zt)
            axes[2, col].set_title(f"Bwd t={bs.timestep}\n√ᾱ={bs.sqrt_alpha_bar:.2f}", fontsize=8.8, fontweight="bold", color="#007880")
            axes[3, col].imshow(latent_to_rgb(bs.z_t, upscale=4))
        else:
            axes[2, col].axis("off")
            axes[3, col].axis("off")

        for r in range(4):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    axes[0, 0].set_ylabel("Forward\nDecoded D(z_t)", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Forward\nLatent z_t RGB", fontsize=9)
    axes[2, 0].set_ylabel("Backward\nDecoded D(z_t)", fontsize=9, fontweight="bold")
    axes[3, 0].set_ylabel("Backward\nLatent z_t RGB", fontsize=9)

    mode_str = f"Fwd={res.forward_mode} -> Rev={res.reverse_mode}"
    fig.suptitle(
        f"Latent Forward vs Backward DDPM ({mode_str}, {res.sampler}, t_max={res.t_max}, PSNR={res.recon_psnr_db:.1f} dB)",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return fig_to_png_bytes(fig, dpi=160)


def render_forward_backward_metrics_fig(res: LatentForwardBackwardResult) -> tuple[plt.Figure, bytes]:
    """Plot Forward vs Backward cosine similarity to clean z_0, empirical std, and 2D PCA loop."""
    z0_flat = res.z0_clean.reshape(-1).astype(np.float32)
    z0_norm = max(float(np.linalg.norm(z0_flat)), 1e-8)

    def _cos_to_z0(snaps: Sequence[LatentStepSnapshot]) -> list[float]:
        out = []
        for s in snaps:
            zf = s.z_t.reshape(-1).astype(np.float32)
            out.append(float(np.dot(zf, z0_flat) / (max(float(np.linalg.norm(zf)), 1e-8) * z0_norm)))
        return out

    fwd_t = [s.timestep for s in res.forward_steps]
    bwd_t = [s.timestep for s in res.backward_steps]
    fwd_cos = _cos_to_z0(res.forward_steps)
    bwd_cos = _cos_to_z0(res.backward_steps)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axes[0].plot(fwd_t, fwd_cos, "o-", color="#B92D2D", lw=2.0, markersize=4, label="Forward (0 → t_max)")
    axes[0].plot(bwd_t, bwd_cos, "s-", color="#007880", lw=2.0, markersize=4, label="Backward (t_max → 0)")
    axes[0].invert_xaxis()
    axes[0].set_title("Cosine Similarity to Clean z₀ vs t", fontsize=9.8, fontweight="bold")
    axes[0].set_xlabel("Timestep t (← noising | denoising →)")
    axes[0].set_ylabel("cos(z_t, z₀)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8.5)

    axes[1].plot(fwd_t, [s.std for s in res.forward_steps], "o-", color="#B92D2D", lw=2.0, markersize=4, label="Forward std(z_t)")
    axes[1].plot(bwd_t, [s.std for s in res.backward_steps], "s-", color="#007880", lw=2.0, markersize=4, label="Backward std(z_t)")
    axes[1].invert_xaxis()
    axes[1].set_title("Latent Standard Deviation Trajectory", fontsize=9.8, fontweight="bold")
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel("std(z_t)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8.5)

    # Joint 2D PCA loop
    all_zs = [s.z_t.reshape(-1).astype(np.float32) for s in res.forward_steps] + [
        s.z_t.reshape(-1).astype(np.float32) for s in res.backward_steps
    ]
    if len(all_zs) >= 2:
        mat = np.stack(all_zs, axis=0)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(mat)
        nf = len(res.forward_steps)
        pf, pb = proj[:nf], proj[nf:]
        axes[2].plot(pf[:, 0], pf[:, 1], "o-", color="#B92D2D", lw=2.0, markersize=4, label="Forward")
        axes[2].plot(pb[:, 0], pb[:, 1], "s-", color="#007880", lw=2.0, markersize=4, label="Backward")
        axes[2].scatter(pf[0, 0], pf[0, 1], marker="*", s=160, color="#19468C", edgecolors="black", zorder=5, label="Clean z₀")
        axes[2].scatter(pf[-1, 0], pf[-1, 1], marker="X", s=110, color="#F59E0B", edgecolors="black", zorder=5, label="Turnaround z_{t_max}")
        var = pca.explained_variance_ratio_ * 100.0
        axes[2].set_xlabel(f"PC1 ({var[0]:.1f}% var.)")
        axes[2].set_ylabel(f"PC2 ({var[1]:.1f}% var.)")
        axes[2].legend(fontsize=8, loc="best")
    axes[2].set_title("2D PCA Forward-Backward Loop", fontsize=9.8, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return fig, buf.getvalue()


def render_pixel_ddpm_filmstrip(res: PixelDDPMResult, num_cols: int = 7) -> bytes:
    """Render Pixel-Space DDPM filmstrip (Reverse x_t + x0_hat, plus Forward row if present)."""
    has_fwd = len(res.forward_steps) > 0
    bwd_idxs = _evenly_spaced_indices(len(res.backward_steps), num_cols)
    fwd_idxs = _evenly_spaced_indices(len(res.forward_steps), num_cols) if has_fwd else []
    ncols = max(len(bwd_idxs), len(fwd_idxs))
    nrows = 3 if has_fwd else 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols, 2.25 * nrows))
    if ncols == 1:
        axes = axes[:, None]

    row_offset = 0
    if has_fwd:
        for col in range(ncols):
            ax = axes[0, col]
            if col < len(fwd_idxs):
                fs = res.forward_steps[fwd_idxs[col]]
                ax.imshow(fs.x_t_rgb)
                ax.set_title(f"Fwd t={fs.timestep}\nσ={fs.x_t_std:.2f}", fontsize=8.8, color="#B92D2D", fontweight="bold")
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, 0].set_ylabel("Forward\nPixel x_t", fontsize=9, fontweight="bold")
        row_offset = 1

    for col in range(ncols):
        ax_xt = axes[row_offset, col]
        ax_x0 = axes[row_offset + 1, col]
        if col < len(bwd_idxs):
            bs = res.backward_steps[bwd_idxs[col]]
            ax_xt.imshow(bs.x_t_rgb)
            ax_xt.set_title(f"Rev t={bs.timestep}\nσ={bs.x_t_std:.2f}", fontsize=8.8, color="#007880", fontweight="bold")
            ax_x0.imshow(bs.x0_hat_rgb if bs.x0_hat_rgb is not None else bs.x_t_rgb)
        else:
            ax_xt.axis("off")
            ax_x0.axis("off")
        ax_xt.set_xticks([])
        ax_xt.set_yticks([])
        ax_x0.set_xticks([])
        ax_x0.set_yticks([])

    axes[row_offset, 0].set_ylabel("Backward\nPixel x_t", fontsize=9, fontweight="bold")
    axes[row_offset + 1, 0].set_ylabel("Predicted\nClean x̂₀(x_t)", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Pixel-Space DDPM ({res.mode}, {res.sampler}, {res.num_steps} steps, dims={res.state_dims:,})",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return fig_to_png_bytes(fig, dpi=160)


def render_pixel_hist_fig(
    snap: PixelStepSnapshot,
    all_snaps: Sequence[PixelStepSnapshot],
    phase_label: str,
) -> tuple[plt.Figure, bytes]:
    """Render pixel intensity histogram vs N(0, 1) and schedule std progression."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    axes[0].hist(snap.x_t_flat_sample, bins=50, range=(-3.5, 3.5), density=True, color="#007880", alpha=0.72, label="Empirical x_t")
    xs = np.linspace(-3.5, 3.5, 220)
    gauss = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xs ** 2)
    axes[0].plot(xs, gauss, "r--", lw=1.7, label="Standard N(0, 1)")
    axes[0].set_title(f"{phase_label} Distribution (t={snap.timestep}, μ={snap.x_t_mean:.2f}, σ={snap.x_t_std:.2f})", fontsize=9.5, fontweight="bold")
    axes[0].set_xlabel("Pixel space value")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8.5)

    sx = [s.step_idx for s in all_snaps]
    axes[1].plot(sx, [s.sqrt_alpha_bar for s in all_snaps], color="#007880", lw=2.0, label="Signal √ᾱ_t")
    axes[1].plot(sx, [s.sqrt_one_minus_alpha_bar for s in all_snaps], color="#B92D2D", lw=2.0, label="Noise √(1-ᾱ_t)")
    axes[1].plot(sx, [s.x_t_std for s in all_snaps], color="#19468C", lw=1.8, linestyle=":", label="Empirical std(x_t)")
    axes[1].axvline(snap.step_idx, color="#F59E0B", lw=1.8, linestyle="--")
    axes[1].set_title("Pixel Schedule & Standard Deviation", fontsize=9.5, fontweight="bold")
    axes[1].set_xlabel("Step Index")
    axes[1].set_ylabel("Scale")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return fig, buf.getvalue()


def _encode_gif_from_frames(frames: list[Image.Image], fps: int = 4) -> bytes:
    if not frames:
        raise ValueError("Cannot build GIF from empty frame list")
    # Hold final frame slightly longer for readability
    durations = [int(round(1000 / max(fps, 1)))] * len(frames)
    durations[-1] = durations[-1] * 3
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    buf.seek(0)
    return buf.getvalue()


def _tile_frame(panels: list[tuple[str, np.ndarray]], header: str, panel_size: int = 256) -> Image.Image:
    pad = 12
    title_h = 34
    label_h = 22
    n = len(panels)
    w = pad + n * (panel_size + pad)
    h = title_h + label_h + panel_size + pad
    canvas = Image.new("RGB", (w, h), color=(250, 251, 253))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 9), header, fill=(25, 70, 140))

    for i, (lbl, arr) in enumerate(panels):
        x0 = pad + i * (panel_size + pad)
        y0 = title_h + label_h
        draw.text((x0, title_h + 2), lbl, fill=(35, 40, 48))
        pil_p = Image.fromarray(arr, mode="RGB").resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        canvas.paste(pil_p, (x0, y0))
    return canvas


def build_uncond_vs_cond_gif(res: UncondVsCondResult, fps: int = 4, include_latent: bool = True) -> bytes:
    """Build animated side-by-side GIF comparing Unconditioned vs Conditioned denoising."""
    n_steps = len(res.branches[0].steps)
    frames: list[Image.Image] = []
    for s_idx in range(n_steps):
        panels: list[tuple[str, np.ndarray]] = []
        t_val = res.branches[0].steps[s_idx].timestep
        for b in res.branches:
            snap = b.steps[s_idx]
            panels.append((b.label[:28], snap.decoded_zt))
            if include_latent:
                panels.append((f"{b.name} latent z_t", latent_to_rgb(snap.z_t, upscale=4)))
        if len(res.branches) > 1 and res.branches[1].steps[s_idx].guidance_map is not None:
            panels.append(("CFG ||Δε||₂", guidance_map_to_rgb(res.branches[1].steps[s_idx].guidance_map, upscale=4)))
        header = f"{res.sampler} Step {s_idx}/{n_steps - 1} (t={t_val}) | Prompt: {res.prompt[:55]}"
        frames.append(_tile_frame(panels, header=header, panel_size=224))
    return _encode_gif_from_frames(frames, fps=fps)


def build_latent_inspector_gif(branch: BranchTrajectory, fps: int = 3) -> bytes:
    """Build animated GIF showing Decoded D(z_t), Predicted D(z0_hat), Latent RGB, and 4 channels."""
    frames: list[Image.Image] = []
    n_steps = len(branch.steps)
    for snap in branch.steps:
        panels: list[tuple[str, np.ndarray]] = [
            ("Latent z_t RGB", latent_to_rgb(snap.z_t, upscale=4)),
            ("Decoded D(z_t)", snap.decoded_zt),
            ("Pred. D(z0_hat)", snap.decoded_z0_hat if snap.decoded_z0_hat is not None else snap.decoded_zt),
        ]
        # Render channels 0..3 into RdBu colormap tiles
        rdbu = _get_cmap("RdBu_r")
        for ch in range(min(4, snap.z_t.shape[0])):
            norm = np.clip((snap.z_t[ch] + 3.5) / 7.0, 0.0, 1.0)
            ch_rgb = (rdbu(norm)[..., :3] * 255.0).round().astype(np.uint8)
            panels.append((f"Channel z[{ch}]", _upscale_nearest(ch_rgb, 4)))
        header = f"Latent Inspector ({branch.label}) | Step {snap.step_idx}/{n_steps - 1} | t={snap.timestep} | σ={snap.std:.2f}"
        frames.append(_tile_frame(panels, header=header, panel_size=180))
    return _encode_gif_from_frames(frames, fps=fps)


def build_forward_backward_gif(res: LatentForwardBackwardResult, fps: int = 4) -> bytes:
    """Build round-trip GIF animating Forward noising (0->t_max) followed by Backward denoising (t_max->0)."""
    frames: list[Image.Image] = []
    for snap in res.forward_steps:
        panels = [
            ("Clean Source x", res.source_rgb),
            ("Forward D(z_t)", snap.decoded_zt),
            ("Forward Latent z_t", latent_to_rgb(snap.z_t, upscale=4)),
        ]
        header = f"FORWARD ({res.forward_mode}) | Step {snap.step_idx} | t={snap.timestep} | √ᾱ={snap.sqrt_alpha_bar:.2f}"
        frames.append(_tile_frame(panels, header=header, panel_size=240))

    for snap in res.backward_steps:
        panels = [
            ("Clean Source x", res.source_rgb),
            ("Backward D(z_t)", snap.decoded_zt),
            ("Backward Latent z_t", latent_to_rgb(snap.z_t, upscale=4)),
        ]
        header = f"BACKWARD ({res.reverse_mode}) | Step {snap.step_idx} | t={snap.timestep} | √ᾱ={snap.sqrt_alpha_bar:.2f}"
        frames.append(_tile_frame(panels, header=header, panel_size=240))
    return _encode_gif_from_frames(frames, fps=fps)


def build_pixel_ddpm_gif(res: PixelDDPMResult, fps: int = 4) -> bytes:
    """Build animated GIF for Pixel-Space DDPM (Forward if present, then Backward + x0_hat)."""
    frames: list[Image.Image] = []
    for snap in res.forward_steps:
        panels = [("Forward Pixel x_t", snap.x_t_rgb)]
        if res.source_rgb is not None:
            panels.insert(0, ("Source x_0", res.source_rgb))
        header = f"PIXEL FORWARD | Step {snap.step_idx} | t={snap.timestep} | σ={snap.x_t_std:.2f}"
        frames.append(_tile_frame(panels, header=header, panel_size=256))

    for snap in res.backward_steps:
        panels = [
            ("Backward Pixel x_t", snap.x_t_rgb),
            ("Predicted Clean x0_hat", snap.x0_hat_rgb if snap.x0_hat_rgb is not None else snap.x_t_rgb),
        ]
        header = f"PIXEL BACKWARD ({res.sampler}) | Step {snap.step_idx} | t={snap.timestep} | σ={snap.x_t_std:.2f}"
        frames.append(_tile_frame(panels, header=header, panel_size=256))
    return _encode_gif_from_frames(frames, fps=fps)
