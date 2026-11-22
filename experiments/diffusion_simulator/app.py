"""Interactive Streamlit Diffusion & Latent Space Simulator (Colab T4 / Local GPU)."""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# Ensure `experiments/` is importable regardless of working directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.dirname(_THIS_DIR)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from diffusion_simulator.engine import (
    LatentForwardBackwardResult,
    PixelDDPMResult,
    UncondVsCondResult,
    make_procedural_sample_image,
    simulate_latent_forward_backward,
    simulate_pixel_ddpm,
    simulate_uncond_vs_cond_ldm,
)
from diffusion_simulator.models import (
    DDPM_PIXEL_MODEL_ID,
    SD15_MODEL_ID,
    free_gpu_memory,
    get_device_info,
    load_pixel_ddpm_pipeline,
    load_sd15_pipeline,
)
from diffusion_simulator.visuals import (
    build_forward_backward_gif,
    build_latent_inspector_gif,
    build_pixel_ddpm_gif,
    build_uncond_vs_cond_gif,
    guidance_map_to_rgb,
    latent_to_rgb,
    render_divergence_curves_fig,
    render_forward_backward_filmstrip,
    render_forward_backward_metrics_fig,
    render_latent_inspector_fig,
    render_pca_trajectory_fig,
    render_pixel_ddpm_filmstrip,
    render_pixel_hist_fig,
    render_uncond_vs_cond_filmstrip,
)


st.set_page_config(
    page_title="Diffusion & Latent Space Simulator — CS420 [S06]",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .sim-header {
    padding: 0.75rem 1.1rem;
    border-left: 5px solid #19468C;
    background: linear-gradient(90deg, #ECF2FA 0%, #F8FAFC 100%);
    border-radius: 0.4rem;
    margin-bottom: 0.9rem;
  }
  .sim-header h1 {
    margin: 0;
    font-size: 1.55rem;
    color: #19468C;
  }
  .sim-header p {
    margin: 0.2rem 0 0 0;
    color: #334155;
    font-size: 0.92rem;
  }
  .badge-pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.35rem;
  }
  .badge-blue { background: #DBEAFE; color: #1E40AF; }
  .badge-teal { background: #CCFBF1; color: #0F766E; }
  .badge-red  { background: #FEE2E2; color: #991B1B; }
</style>
""",
    unsafe_allow_html=True,
)


def _stretch_kw(widget_fn: Any) -> dict[str, Any]:
    """Return compatible full-width kwargs across old and new Streamlit versions."""
    try:
        sig = inspect.signature(widget_fn)
        if "width" in sig.parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


@st.cache_resource(show_spinner=False)
def _cached_sd15_pipeline():
    return load_sd15_pipeline()


@st.cache_resource(show_spinner=False)
def _cached_pixel_ddpm_pipeline():
    return load_pixel_ddpm_pipeline()


def _clamp_slider_state(key: str, max_val: int) -> int:
    cur = int(st.session_state.get(key, 0))
    cur = min(max(cur, 0), max(max_val, 0))
    st.session_state[key] = cur
    return cur


def _step_delta_cb(key: str, delta: int, max_val: int) -> None:
    cur = int(st.session_state.get(key, 0))
    st.session_state[key] = min(max(cur + delta, 0), max(max_val, 0))


def _render_step_controls(slider_key: str, max_idx: int, label: str = "Step") -> int:
    _clamp_slider_state(slider_key, max_idx)
    c_prev, c_slide, c_next = st.columns([1, 8, 1])
    with c_prev:
        st.button(
            "◀ Prev",
            key=f"{slider_key}_prev",
            on_click=_step_delta_cb,
            args=(slider_key, -1, max_idx),
            **_stretch_kw(st.button),
        )
    with c_slide:
        idx = st.slider(
            label,
            min_value=0,
            max_value=max(max_idx, 1),
            key=slider_key,
            disabled=(max_idx == 0),
        )
    with c_next:
        st.button(
            "Next ▶",
            key=f"{slider_key}_next",
            on_click=_step_delta_cb,
            args=(slider_key, 1, max_idx),
            **_stretch_kw(st.button),
        )
    return min(int(idx), max_idx)


def _render_header() -> None:
    st.markdown(
        """
<div class="sim-header">
  <h1>🌀 Diffusion & Latent Space Interactive Simulator</h1>
  <p>Compare <b>Unconditioned vs. Conditioned LDM</b> (Stable Diffusion v1.5 FP16), trace
     <b>Forward vs. Backward DDPM</b> in Latent &amp; Pixel space (<code>google/ddpm-celebahq-256</code>),
     and inspect the 4-channel latent tensor step by step.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_sidebar() -> tuple[str, int]:
    info = get_device_info()
    st.sidebar.header("⚙️ Runtime & Global Controls")

    if info["has_cuda"]:
        tag = "Colab Tesla T4" if info["is_t4"] else "CUDA GPU"
        st.sidebar.success(
            f"**{tag}:** `{info['gpu_name']}`\n\n"
            f"VRAM: `{info['vram_total_gb']:.1f} GB` | Used: `{info['vram_allocated_mb']:.0f} MB` | Precision: `{info['dtype_name']}`"
        )
    else:
        st.sidebar.warning(
            "**Device:** CPU (`float32`)\n\n"
            "Pretrained SD v1.5 & DDPM-256 run best on a Google Colab **Tesla T4 GPU**."
        )

    if st.sidebar.button("🧹 Free GPU VRAM Cache", key="btn_free_vram", **_stretch_kw(st.button)):
        free_gpu_memory()
        st.sidebar.toast("Cleared Python & CUDA cache.")

    st.sidebar.divider()
    sampler = st.sidebar.radio(
        "Sampler / Integrator",
        options=["DDIM", "DDPM"],
        index=0,
        key="global_sampler",
        help="DDIM allows crisp deterministic trajectories in 15-30 steps; DDPM adds Markov stochastic variance at every step.",
    )
    seed = int(
        st.sidebar.number_input(
            "Shared Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            key="global_seed",
            help="Locks identical starting Gaussian noise z_T / x_T across unconditioned & conditioned branches.",
        )
    )

    with st.sidebar.expander("📐 Pretrained Models & Shapes", expanded=False):
        st.markdown(
            f"- **Latent Diffusion:** `{SD15_MODEL_ID}` (FP16)\n"
            f"  - Pixel: `512×512×3` (`786,432` dims)\n"
            f"  - Latent: `64×64×4` (`16,384` dims, `48×` smaller)\n"
            f"- **Pixel DDPM:** `{DDPM_PIXEL_MODEL_ID}`\n"
            f"  - Pixel: `256×256×3` (`196,608` dims)\n"
            f"- **Lazy Loading:** Weights download only when you run a simulation tab."
        )

    return sampler, seed


def _render_tab1(sampler: str, seed: int) -> None:
    st.subheader("1. Unconditioned (∅) vs. Conditioned Latent Diffusion (CFG)")
    st.caption(
        "Both branches start from the **exact same initial noise** $z_T \\sim \\mathcal{N}(0, I)$ and receive identical per-step variance noise. "
        "Any trajectory difference comes purely from text cross-attention conditioning and Classifier-Free Guidance (CFG)."
    )

    presets = {
        "Custom Prompt": "",
        "Astronaut riding a horse on Mars": "A cinematic photograph of an astronaut riding a horse on Mars, golden hour lighting, sharp detail",
        "Japanese garden in autumn rain": "A serene traditional Japanese garden with a wooden bridge over a koi pond in autumn rain, soft studio lighting",
        "Glowing cyberpunk alleyway": "A futuristic cyberpunk street market at night with neon reflections on wet pavement, ultra detailed",
        "Minimalist geometric sculpture": "A museum photograph of a minimalist blue and terracotta geometric sculpture on a white pedestal",
    }

    c_pre, c_prompt = st.columns([1.3, 2.7])
    with c_pre:
        chosen_preset = st.selectbox("Prompt Preset", options=list(presets.keys()), index=1, key="t1_preset")
    default_prompt = presets[chosen_preset] if presets[chosen_preset] else st.session_state.get(
        "t1_prompt_val",
        "A cinematic photograph of an astronaut riding a horse on Mars, golden hour lighting",
    )
    with c_prompt:
        prompt = st.text_input("Conditioning Prompt y", value=default_prompt, key="t1_prompt_input")
        st.session_state["t1_prompt_val"] = prompt

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        guidance_scale = st.slider("Primary CFG Scale α", min_value=1.0, max_value=15.0, value=7.5, step=0.5, key="t1_cfg")
    with c2:
        compare_3way = st.checkbox("Add 3rd Branch (Compare 2nd α)", value=False, key="t1_3way")
        extra_cfg = (
            st.slider("Secondary CFG α₂", min_value=1.0, max_value=20.0, value=15.0, step=0.5, key="t1_cfg2")
            if compare_3way
            else None
        )
    with c3:
        num_steps = st.slider("Denoising Steps", min_value=8, max_value=40, value=20, step=2, key="t1_steps")
    with c4:
        resolution = st.selectbox("Image Resolution", options=[512, 384], index=0, key="t1_res")

    neg_prompt = st.text_input("Optional Negative / Baseline Prompt (leave empty for true ∅)", value="", key="t1_neg")

    run_clicked = st.button("🚀 Run Unconditioned vs. Conditioned LDM", type="primary", key="t1_run", **_stretch_kw(st.button))
    if run_clicked:
        if not prompt.strip():
            st.error("Please enter a non-empty conditioning prompt.")
        else:
            pbar = st.progress(0.0, text="Loading Stable Diffusion v1.5 FP16...")
            try:
                sd_pipe = _cached_sd15_pipeline()
                res = simulate_uncond_vs_cond_ldm(
                    sd_pipe,
                    prompt=prompt.strip(),
                    seed=seed,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    extra_guidance_scale=extra_cfg,
                    sampler=sampler,
                    resolution=resolution,
                    negative_prompt=neg_prompt.strip(),
                    progress_cb=lambda f, msg: pbar.progress(min(max(float(f), 0.0), 1.0), text=msg),
                )
                st.session_state["uncond_cond_result"] = res
                st.session_state["t1_step_slider"] = len(res.branches[0].steps) - 1
                st.session_state.pop("t1_filmstrip_png", None)
                st.session_state.pop("t1_gif_bytes", None)
                pbar.empty()
            except Exception as exc:
                pbar.empty()
                st.error(f"Simulation failed: {exc}")

    res: UncondVsCondResult | None = st.session_state.get("uncond_cond_result")
    if res is None:
        st.info("Configure a prompt and click **Run Unconditioned vs. Conditioned LDM** to simulate step-by-step trajectories.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runtime", f"{res.elapsed_sec:.2f} s")
    m2.metric("Latency / Step", f"{res.ms_per_step:.0f} ms")
    m3.metric("Peak VRAM", f"{res.peak_vram_mb:.0f} MB" if res.peak_vram_mb > 0 else "CPU")
    primary_name = res.branches[1].name
    m4.metric("Final L2 Divergence", f"{res.divergence_l2[primary_name][-1]:.1f}")

    max_idx = len(res.branches[0].steps) - 1
    step_idx = _render_step_controls("t1_step_slider", max_idx, label="Scrub Denoising Step (0 = Pure Noise z_T → Final = Clean z_0)")

    cur_t = res.branches[0].steps[step_idx].timestep
    cur_ab = res.branches[0].steps[step_idx].alpha_bar
    st.markdown(
        f"**Step `{step_idx}/{max_idx}`** | Timestep `t = {cur_t}` | "
        f"Signal retention `ᾱ_t = {cur_ab:.4f}` (`√ᾱ_t = {res.branches[0].steps[step_idx].sqrt_alpha_bar:.3f}`)"
    )

    view_mode = st.radio(
        "Visual Layer",
        options=["Decoded D(z_t) + Latent z_t RGB + CFG Map", "Predicted Clean D(ẑ₀) + Latent z_t RGB"],
        horizontal=True,
        key="t1_view_mode",
    )
    show_z0_hat = view_mode.startswith("Predicted")

    cols = st.columns(len(res.branches))
    for col, branch in zip(cols, res.branches):
        snap = branch.steps[step_idx]
        with col:
            st.markdown(f"**{branch.label}**")
            main_img = (snap.decoded_z0_hat if (show_z0_hat and snap.decoded_z0_hat is not None) else snap.decoded_zt)
            cap_main = "Predicted Clean D(ẑ₀)" if show_z0_hat else f"Decoded D(z_t) | std={snap.std:.2f}"
            st.image(main_img, caption=cap_main, **_stretch_kw(st.image))

            sub_l, sub_r = st.columns(2)
            with sub_l:
                st.image(latent_to_rgb(snap.z_t, upscale=4), caption="Raw Latent z_t RGB (64×64)", **_stretch_kw(st.image))
            with sub_r:
                if snap.guidance_map is not None:
                    st.image(guidance_map_to_rgb(snap.guidance_map, upscale=4), caption="CFG ||ε_c - ε_∅||₂", **_stretch_kw(st.image))
                elif snap.decoded_z0_hat is not None:
                    st.image(snap.decoded_z0_hat, caption="One-step D(ẑ₀)", **_stretch_kw(st.image))
                else:
                    st.caption("Prior step (no UNet pass)")

    st.divider()
    c_pca, c_div = st.columns([1.0, 1.45])
    with c_pca:
        palette = ["#64748B", "#19468C", "#007880"]
        pca_inputs = [
            (b.label, [s.z_t for s in b.steps], palette[i % len(palette)])
            for i, b in enumerate(res.branches)
        ]
        fig_pca, pca_png = render_pca_trajectory_fig(
            pca_inputs,
            current_idx=step_idx,
            title="2D PCA Latent Trajectory (Uncond vs Cond)",
            timesteps=[s.timestep for s in res.branches[0].steps],
        )
        st.pyplot(fig_pca, **_stretch_kw(st.pyplot))
        plt.close(fig_pca)
    with c_div:
        fig_div, div_png = render_divergence_curves_fig(res, current_idx=step_idx)
        st.pyplot(fig_div, **_stretch_kw(st.pyplot))
        plt.close(fig_div)

    st.subheader("📤 Export & Animated Playback")
    e1, e2, e3 = st.columns(3)
    with e1:
        if "t1_filmstrip_png" not in st.session_state:
            st.session_state["t1_filmstrip_png"] = render_uncond_vs_cond_filmstrip(res)
        st.download_button(
            "⬇️ Download Filmstrip PNG",
            data=st.session_state["t1_filmstrip_png"],
            file_name=f"uncond_vs_cond_{res.sampler}_seed{res.seed}.png",
            mime="image/png",
            key="dl_t1_filmstrip",
            **_stretch_kw(st.download_button),
        )
    with e2:
        st.download_button(
            "⬇️ Download 2D PCA PNG",
            data=pca_png,
            file_name=f"uncond_vs_cond_pca_seed{res.seed}.png",
            mime="image/png",
            key="dl_t1_pca",
            **_stretch_kw(st.download_button),
        )
    with e3:
        if st.button("🎞️ Render Animated GIF", key="btn_t1_gif", **_stretch_kw(st.button)):
            with st.spinner("Encoding comparison GIF..."):
                st.session_state["t1_gif_bytes"] = build_uncond_vs_cond_gif(res, fps=4, include_latent=True)
        if "t1_gif_bytes" in st.session_state:
            st.download_button(
                "⬇️ Download Comparison GIF",
                data=st.session_state["t1_gif_bytes"],
                file_name=f"uncond_vs_cond_{res.sampler}_seed{res.seed}.gif",
                mime="image/gif",
                key="dl_t1_gif",
                **_stretch_kw(st.download_button),
            )
    if "t1_gif_bytes" in st.session_state and st.checkbox("Preview Animated GIF Inline", value=True, key="chk_t1_gif"):
        st.image(st.session_state["t1_gif_bytes"], caption="Unconditioned vs. Conditioned Trajectory GIF", **_stretch_kw(st.image))


def _collect_available_latent_branches() -> dict[str, tuple[str, Any]]:
    options: dict[str, tuple[str, Any]] = {}
    res1: UncondVsCondResult | None = st.session_state.get("uncond_cond_result")
    if res1 is not None:
        for b in res1.branches:
            options[f"Tab 1 | {b.label}"] = (b.label, b.steps)
    res3: LatentForwardBackwardResult | None = st.session_state.get("latent_fb_result")
    if res3 is not None:
        options[f"Tab 3 | Forward ({res3.forward_mode})"] = (f"Forward ({res3.forward_mode})", res3.forward_steps)
        options[f"Tab 3 | Backward ({res3.reverse_mode})"] = (f"Backward ({res3.reverse_mode})", res3.backward_steps)
    return options


def _render_tab2() -> None:
    st.subheader("2. Step-by-Step Latent Space Inspector (4 Channels + Histogram + Schedule)")
    st.caption(
        "Inspect the exact $4 \\times 64 \\times 64$ Stable Diffusion latent tensor at any denoising or noising step: "
        "individual channels $z[0..3]$, linear RGB projection, VAE decoded $\\mathcal{D}(z_t)$, predicted clean $\\mathcal{D}(\\hat{z}_0)$, "
        "and empirical Gaussianity vs. $\\mathcal{N}(0, I)$."
    )

    branch_map = _collect_available_latent_branches()
    if not branch_map:
        st.info("Run **Tab 1 (Unconditioned vs Conditioned LDM)** or **Tab 3 (Latent Forward vs Backward DDPM)** first to populate latent trajectories.")
        return

    selected_key = st.selectbox("Select Simulated Latent Trajectory", options=list(branch_map.keys()), key="t2_traj_select")
    branch_label, snaps = branch_map[selected_key]

    max_idx = len(snaps) - 1
    step_idx = _render_step_controls("t2_step_slider", max_idx, label="Inspect Step")
    snap = snaps[step_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Timestep t", f"{snap.timestep}")
    c2.metric("Signal √ᾱ_t", f"{snap.sqrt_alpha_bar:.4f}")
    c3.metric("Noise √(1-ᾱ_t)", f"{snap.sqrt_one_minus_alpha_bar:.4f}")
    c4.metric("Latent μ / σ", f"{snap.mean:.3f} / {snap.std:.3f}")

    fig_insp, insp_png = render_latent_inspector_fig(snap, snaps, branch_label=branch_label)
    st.pyplot(fig_insp, **_stretch_kw(st.pyplot))
    plt.close(fig_insp)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download Step Snapshot PNG",
            data=insp_png,
            file_name=f"latent_inspector_step{snap.step_idx}_t{snap.timestep}.png",
            mime="image/png",
            key="dl_t2_png",
            **_stretch_kw(st.download_button),
        )
    with dl2:
        if st.button("🎞️ Render Full Latent Inspector GIF", key="btn_t2_gif", **_stretch_kw(st.button)):
            with st.spinner("Building multi-channel inspector GIF..."):
                from diffusion_simulator.engine import BranchTrajectory
                tmp_branch = BranchTrajectory(name="insp", label=branch_label, prompt="", guidance_scale=0.0, steps=list(snaps))
                st.session_state["t2_gif_bytes"] = build_latent_inspector_gif(tmp_branch, fps=3)
                st.session_state["t2_gif_key"] = selected_key
        if st.session_state.get("t2_gif_bytes") and st.session_state.get("t2_gif_key") == selected_key:
            st.download_button(
                "⬇️ Download Latent Inspector GIF",
                data=st.session_state["t2_gif_bytes"],
                file_name="latent_inspector_trajectory.gif",
                mime="image/gif",
                key="dl_t2_gif",
                **_stretch_kw(st.download_button),
            )
    if st.session_state.get("t2_gif_bytes") and st.session_state.get("t2_gif_key") == selected_key:
        if st.checkbox("Preview Latent Inspector GIF Inline", value=True, key="chk_t2_gif"):
            st.image(st.session_state["t2_gif_bytes"], caption=f"Step-by-step Latent Inspector ({branch_label})", **_stretch_kw(st.image))


def _render_tab3(sampler: str, seed: int) -> None:
    st.subheader("3. Forward vs. Backward DDPM in Latent Space (SD v1.5 VAE + UNet)")
    st.caption(
        "Encode an image $\\mathbf{x} \\to \\mathbf{z}_0 = \\mathcal{E}(\\mathbf{x})$, corrupt it via **Forward Diffusion** "
        "(either **Closed-Form** $z_t = \\sqrt{\\bar{\\alpha}_t}z_0 + \\sqrt{1-\\bar{\\alpha}_t}\\epsilon$ or true **Step-by-Step Markov Chain**), "
        "and run **Backward Denoising** from $z_{t_{\\max}}$ back to $z_0$."
    )

    c_src, c_preview = st.columns([2.2, 1.0])
    res1: UncondVsCondResult | None = st.session_state.get("uncond_cond_result")
    src_choices = ["Procedural: Bauhaus Geometric", "Procedural: Sunset Landscape", "Upload Custom Image"]
    if res1 is not None and len(res1.branches) > 1:
        src_choices.insert(0, "Use Tab 1 Conditioned Output")

    with c_src:
        src_choice = st.selectbox("Source Image x", options=src_choices, key="t3_src_choice")
        uploaded = None
        if src_choice == "Upload Custom Image":
            uploaded = st.file_uploader("Upload PNG / JPG image", type=["png", "jpg", "jpeg", "webp"], key="t3_upload")

    resolution = 512
    if src_choice == "Use Tab 1 Conditioned Output" and res1 is not None:
        source_arr = res1.branches[1].steps[-1].decoded_zt
        resolution = res1.resolution
    elif src_choice == "Upload Custom Image" and uploaded is not None:
        source_arr = np.asarray(Image.open(uploaded).convert("RGB"), dtype=np.uint8)
    elif src_choice == "Procedural: Sunset Landscape":
        source_arr = make_procedural_sample_image("landscape", size=resolution)
    else:
        source_arr = make_procedural_sample_image("geometric", size=resolution)

    with c_preview:
        st.image(source_arr, caption="Selected Source x", width=180)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        fwd_label = st.radio(
            "Forward Process Mode",
            options=["Closed-Form Direct Jump", "Step-by-Step Markov Chain"],
            index=0,
            key="t3_fwd_mode",
        )
        fwd_mode = "closed_form" if fwd_label.startswith("Closed") else "markov"
    with p2:
        t_max_frac = st.slider("Max Noise Fraction (t_max / T)", min_value=0.20, max_value=1.00, value=0.75, step=0.05, key="t3_tmax")
    with p3:
        rev_label = st.radio(
            "Backward Reverse Mode",
            options=["Conditioned (Prompt + CFG)", "Unconditioned (∅)"],
            index=0,
            key="t3_rev_mode",
        )
        rev_mode = "conditioned" if rev_label.startswith("Conditioned") else "unconditioned"
    with p4:
        num_steps = st.slider("Schedule Steps", min_value=10, max_value=40, value=20, step=2, key="t3_steps")

    prompt_t3 = ""
    cfg_t3 = 7.5
    if rev_mode == "conditioned":
        cp1, cp2 = st.columns([3, 1])
        with cp1:
            default_p3 = res1.prompt if (res1 is not None and res1.prompt) else "A crisp high-resolution photograph, sharp studio lighting"
            prompt_t3 = st.text_input("Reverse Conditioning Prompt y", value=default_p3, key="t3_prompt")
        with cp2:
            cfg_t3 = st.slider("CFG Scale α", min_value=1.0, max_value=15.0, value=7.5, step=0.5, key="t3_cfg")

    if st.button("🔄 Run Latent Forward vs. Backward DDPM", type="primary", key="t3_run", **_stretch_kw(st.button)):
        pbar = st.progress(0.0, text="Loading Stable Diffusion v1.5 FP16...")
        try:
            sd_pipe = _cached_sd15_pipeline()
            fb_res = simulate_latent_forward_backward(
                sd_pipe,
                source_rgb=source_arr,
                seed=seed,
                t_max_frac=t_max_frac,
                num_steps=num_steps,
                forward_mode=fwd_mode,
                reverse_mode=rev_mode,
                prompt=prompt_t3.strip(),
                guidance_scale=cfg_t3,
                sampler=sampler,
                resolution=resolution,
                progress_cb=lambda f, msg: pbar.progress(min(max(float(f), 0.0), 1.0), text=msg),
            )
            st.session_state["latent_fb_result"] = fb_res
            st.session_state["t3_fwd_slider"] = len(fb_res.forward_steps) - 1
            st.session_state["t3_bwd_slider"] = len(fb_res.backward_steps) - 1
            st.session_state.pop("t3_filmstrip_png", None)
            st.session_state.pop("t3_gif_bytes", None)
            pbar.empty()
        except Exception as exc:
            pbar.empty()
            st.error(f"Latent Forward/Backward failed: {exc}")

    fb_res: LatentForwardBackwardResult | None = st.session_state.get("latent_fb_result")
    if fb_res is None:
        st.info("Choose a source image and click **Run Latent Forward vs. Backward DDPM** to trace the forward-backward loop.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Turnaround t_max", f"{fb_res.t_max}")
    m2.metric("Round-Trip PSNR", f"{fb_res.recon_psnr_db:.2f} dB")
    m3.metric("Runtime", f"{fb_res.elapsed_sec:.2f} s")
    m4.metric("Peak VRAM", f"{fb_res.peak_vram_mb:.0f} MB" if fb_res.peak_vram_mb > 0 else "CPU")

    col_f, col_b = st.columns(2)
    with col_f:
        st.markdown(f"#### ➡️ Forward Noising (`{fb_res.forward_mode}`: `0 → {fb_res.t_max}`)")
        f_idx = _render_step_controls("t3_fwd_slider", len(fb_res.forward_steps) - 1, label="Forward Step")
        fs = fb_res.forward_steps[f_idx]
        fi1, fi2 = st.columns(2)
        with fi1:
            st.image(fs.decoded_zt, caption=f"Decoded D(z_t) | t={fs.timestep}", **_stretch_kw(st.image))
        with fi2:
            st.image(latent_to_rgb(fs.z_t, upscale=4), caption=f"Latent z_t RGB | σ={fs.std:.2f}", **_stretch_kw(st.image))

    with col_b:
        st.markdown(f"#### ⬅️ Backward Denoising (`{fb_res.reverse_mode}`: `{fb_res.t_max} → 0`)")
        b_idx = _render_step_controls("t3_bwd_slider", len(fb_res.backward_steps) - 1, label="Backward Step")
        bs = fb_res.backward_steps[b_idx]
        bi1, bi2 = st.columns(2)
        with bi1:
            st.image(bs.decoded_zt, caption=f"Decoded D(z_t) | t={bs.timestep}", **_stretch_kw(st.image))
        with bi2:
            st.image(latent_to_rgb(bs.z_t, upscale=4), caption=f"Latent z_t RGB | σ={bs.std:.2f}", **_stretch_kw(st.image))

    fig_fb, fb_metrics_png = render_forward_backward_metrics_fig(fb_res)
    st.pyplot(fig_fb, **_stretch_kw(st.pyplot))
    plt.close(fig_fb)

    e1, e2, e3 = st.columns(3)
    with e1:
        if "t3_filmstrip_png" not in st.session_state:
            st.session_state["t3_filmstrip_png"] = render_forward_backward_filmstrip(fb_res)
        st.download_button(
            "⬇️ Download Fwd/Bwd Filmstrip PNG",
            data=st.session_state["t3_filmstrip_png"],
            file_name=f"latent_fwd_bwd_{fb_res.forward_mode}_{fb_res.reverse_mode}.png",
            mime="image/png",
            key="dl_t3_filmstrip",
            **_stretch_kw(st.download_button),
        )
    with e2:
        st.download_button(
            "⬇️ Download PCA Loop & Metrics PNG",
            data=fb_metrics_png,
            file_name="latent_fwd_bwd_metrics.png",
            mime="image/png",
            key="dl_t3_metrics",
            **_stretch_kw(st.download_button),
        )
    with e3:
        if st.button("🎞️ Render Round-Trip GIF", key="btn_t3_gif", **_stretch_kw(st.button)):
            with st.spinner("Encoding Forward-Backward GIF..."):
                st.session_state["t3_gif_bytes"] = build_forward_backward_gif(fb_res, fps=4)
        if "t3_gif_bytes" in st.session_state:
            st.download_button(
                "⬇️ Download Round-Trip GIF",
                data=st.session_state["t3_gif_bytes"],
                file_name="latent_forward_backward.gif",
                mime="image/gif",
                key="dl_t3_gif",
                **_stretch_kw(st.download_button),
            )
    if "t3_gif_bytes" in st.session_state and st.checkbox("Preview Round-Trip GIF Inline", value=True, key="chk_t3_gif"):
        st.image(st.session_state["t3_gif_bytes"], caption="Latent Forward → Backward Round-Trip GIF", **_stretch_kw(st.image))


def _render_tab4(sampler: str, seed: int) -> None:
    st.subheader("4. Pixel-Space Forward vs. Backward DDPM (CelebA-HQ 256) & Pixel vs. Latent Benchmark")
    st.caption(
        f"Runs Ho et al.'s pixel-space DDPM (`{DDPM_PIXEL_MODEL_ID}`, `256×256×3 = 196,608` dims) directly on RGB pixels "
        "without any VAE encoder/decoder — letting you contrast pixel-space trajectories and compute cost against SD v1.5 latent diffusion."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mode_lbl = st.radio(
            "Pixel DDPM Mode",
            options=["Pure Reverse Generation (x_T → x_0)", "Forward → Backward Round-Trip"],
            index=0,
            key="t4_mode",
        )
        mode = "reverse_only" if mode_lbl.startswith("Pure") else "forward_backward"
    with c2:
        num_steps = st.slider("Denoising Steps", min_value=10, max_value=50, value=25, step=5, key="t4_steps")
    with c3:
        t_max_frac = (
            st.slider("Forward Max Noise Fraction", min_value=0.2, max_value=1.0, value=0.70, step=0.05, key="t4_tmax")
            if mode == "forward_backward"
            else 1.0
        )

    src_pix = None
    if mode == "forward_backward":
        pix_src_opt = st.radio(
            "Clean x_0 Source for Pixel Forward Pass",
            options=["Auto-generate CelebA-HQ Face Sample", "Upload Custom Image (256×256)"],
            horizontal=True,
            key="t4_src_opt",
        )
        if pix_src_opt.startswith("Upload"):
            up_pix = st.file_uploader("Upload PNG / JPG for Pixel Forward->Backward", type=["png", "jpg", "jpeg", "webp"], key="t4_upload")
            if up_pix is not None:
                src_pix = np.asarray(Image.open(up_pix).convert("RGB"), dtype=np.uint8)

    if st.button("🖼️ Run Pixel-Space DDPM", type="primary", key="t4_run", **_stretch_kw(st.button)):
        pbar = st.progress(0.0, text=f"Loading {DDPM_PIXEL_MODEL_ID}...")
        try:
            ddpm_pipe = _cached_pixel_ddpm_pipeline()
            pix_res = simulate_pixel_ddpm(
                ddpm_pipe,
                mode=mode,
                source_rgb=src_pix,
                seed=seed,
                num_steps=num_steps,
                t_max_frac=t_max_frac,
                sampler=sampler,
                progress_cb=lambda f, msg: pbar.progress(min(max(float(f), 0.0), 1.0), text=msg),
            )
            st.session_state["pixel_ddpm_result"] = pix_res
            st.session_state["t4_bwd_slider"] = len(pix_res.backward_steps) - 1
            if pix_res.forward_steps:
                st.session_state["t4_fwd_slider"] = len(pix_res.forward_steps) - 1
            st.session_state.pop("t4_filmstrip_png", None)
            st.session_state.pop("t4_gif_bytes", None)
            pbar.empty()
        except Exception as exc:
            pbar.empty()
            st.error(f"Pixel DDPM simulation failed: {exc}")

    pix_res: PixelDDPMResult | None = st.session_state.get("pixel_ddpm_result")
    if pix_res is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("State Dimensions", f"{pix_res.state_dims:,}")
        m2.metric("Runtime", f"{pix_res.elapsed_sec:.2f} s")
        m3.metric("Latency / Step", f"{pix_res.ms_per_step:.0f} ms")
        m4.metric("Peak VRAM", f"{pix_res.peak_vram_mb:.0f} MB" if pix_res.peak_vram_mb > 0 else "CPU")

        if pix_res.forward_steps:
            st.markdown("#### ➡️ Pixel Forward Noising (0 → t_max)")
            pf_idx = _render_step_controls("t4_fwd_slider", len(pix_res.forward_steps) - 1, label="Pixel Forward Step")
            pfs = pix_res.forward_steps[pf_idx]
            cf1, cf2 = st.columns([1, 2])
            with cf1:
                st.image(pfs.x_t_rgb, caption=f"Pixel x_t | t={pfs.timestep} | σ={pfs.x_t_std:.2f}", **_stretch_kw(st.image))
            with cf2:
                fig_h, _ = render_pixel_hist_fig(pfs, pix_res.forward_steps, "Pixel Forward")
                st.pyplot(fig_h, **_stretch_kw(st.pyplot))
                plt.close(fig_h)

        st.markdown("#### ⬅️ Pixel Backward Denoising (x_t vs. Predicted Clean x̂₀)")
        pb_idx = _render_step_controls("t4_bwd_slider", len(pix_res.backward_steps) - 1, label="Pixel Backward Step")
        pbs = pix_res.backward_steps[pb_idx]
        cb1, cb2, cb3 = st.columns([1, 1, 2])
        with cb1:
            st.image(pbs.x_t_rgb, caption=f"Noisy Pixel x_t | t={pbs.timestep}", **_stretch_kw(st.image))
        with cb2:
            st.image(pbs.x0_hat_rgb if pbs.x0_hat_rgb is not None else pbs.x_t_rgb, caption="Predicted Clean x̂₀(x_t)", **_stretch_kw(st.image))
        with cb3:
            fig_hb, _ = render_pixel_hist_fig(pbs, pix_res.backward_steps, "Pixel Backward")
            st.pyplot(fig_hb, **_stretch_kw(st.pyplot))
            plt.close(fig_hb)

        e1, e2 = st.columns(2)
        with e1:
            if "t4_filmstrip_png" not in st.session_state:
                st.session_state["t4_filmstrip_png"] = render_pixel_ddpm_filmstrip(pix_res)
            st.download_button(
                "⬇️ Download Pixel DDPM Filmstrip PNG",
                data=st.session_state["t4_filmstrip_png"],
                file_name=f"pixel_ddpm_{pix_res.mode}_{pix_res.sampler}.png",
                mime="image/png",
                key="dl_t4_filmstrip",
                **_stretch_kw(st.download_button),
            )
        with e2:
            if st.button("🎞️ Render Pixel DDPM GIF", key="btn_t4_gif", **_stretch_kw(st.button)):
                with st.spinner("Encoding Pixel DDPM GIF..."):
                    st.session_state["t4_gif_bytes"] = build_pixel_ddpm_gif(pix_res, fps=4)
            if "t4_gif_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download Pixel DDPM GIF",
                    data=st.session_state["t4_gif_bytes"],
                    file_name=f"pixel_ddpm_{pix_res.mode}_{pix_res.sampler}.gif",
                    mime="image/gif",
                    key="dl_t4_gif",
                    **_stretch_kw(st.download_button),
                )
        if "t4_gif_bytes" in st.session_state and st.checkbox("Preview Pixel DDPM GIF Inline", value=True, key="chk_t4_gif"):
            st.image(st.session_state["t4_gif_bytes"], caption="Pixel-Space DDPM Trajectory GIF", **_stretch_kw(st.image))
    else:
        st.info("Click **Run Pixel-Space DDPM** to simulate pixel-space forward/backward diffusion on `google/ddpm-celebahq-256`.")

    st.divider()
    st.markdown("### 📊 Live Pixel-Space vs. Latent-Space Diffusion Comparison")
    res1: UncondVsCondResult | None = st.session_state.get("uncond_cond_result")
    lat_dims = 4 * ((res1.resolution // 8) ** 2) if res1 is not None else 4 * 64 * 64
    lat_res_str = f"{res1.resolution}×{res1.resolution}" if res1 is not None else "512×512"
    lat_ms = f"{res1.ms_per_step:.0f} ms/step (2-3 branches)" if res1 is not None else "Run Tab 1 to measure"
    lat_vram = f"{res1.peak_vram_mb:.0f} MB" if (res1 is not None and res1.peak_vram_mb > 0) else "Run Tab 1 to measure"

    pix_dims = pix_res.state_dims if pix_res is not None else 3 * 256 * 256
    pix_ms = f"{pix_res.ms_per_step:.0f} ms/step (1 branch)" if pix_res is not None else "Run Tab 4 to measure"
    pix_vram = f"{pix_res.peak_vram_mb:.0f} MB" if (pix_res is not None and pix_res.peak_vram_mb > 0) else "Run Tab 4 to measure"

    rows = [
        {
            "Metric": "Model",
            "Latent Diffusion (SD v1.5)": SD15_MODEL_ID,
            "Pixel Diffusion (CelebA-HQ)": DDPM_PIXEL_MODEL_ID,
        },
        {
            "Metric": "Output Image Resolution",
            "Latent Diffusion (SD v1.5)": f"{lat_res_str} RGB",
            "Pixel Diffusion (CelebA-HQ)": "256×256 RGB",
        },
        {
            "Metric": "Diffusion State Tensor",
            "Latent Diffusion (SD v1.5)": f"4 × {res1.resolution // 8 if res1 else 64} × {res1.resolution // 8 if res1 else 64} ({lat_dims:,} floats)",
            "Pixel Diffusion (CelebA-HQ)": f"3 × 256 × 256 ({pix_dims:,} floats)",
        },
        {
            "Metric": "Text / CFG Conditioning",
            "Latent Diffusion (SD v1.5)": "CLIP Cross-Attention + CFG α",
            "Pixel Diffusion (CelebA-HQ)": "Unconditional only",
        },
        {
            "Metric": "Measured Step Latency",
            "Latent Diffusion (SD v1.5)": lat_ms,
            "Pixel Diffusion (CelebA-HQ)": pix_ms,
        },
        {
            "Metric": "Measured Peak VRAM",
            "Latent Diffusion (SD v1.5)": lat_vram,
            "Pixel Diffusion (CelebA-HQ)": pix_vram,
        },
    ]
    st.dataframe(rows, hide_index=True, **_stretch_kw(st.dataframe))


def main() -> None:
    _render_header()
    sampler, seed = _render_sidebar()
    t1, t2, t3, t4 = st.tabs(
        [
            "1️⃣ Uncond vs. Cond LDM",
            "2️⃣ Latent Space Inspector",
            "3️⃣ Latent Fwd vs. Bwd DDPM",
            "4️⃣ Pixel DDPM & Benchmark",
        ]
    )
    with t1:
        _render_tab1(sampler, seed)
    with t2:
        _render_tab2()
    with t3:
        _render_tab3(sampler, seed)
    with t4:
        _render_tab4(sampler, seed)


if __name__ == "__main__":
    main()
