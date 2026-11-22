"""End-to-end automated tests for diffusion_simulator (runs locally with zero HF downloads)."""

from __future__ import annotations

import io
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from PIL import Image
from streamlit.testing.v1 import AppTest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(THIS_DIR)
REPO_DIR = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from diffusion_simulator.engine import (
    make_procedural_sample_image,
    simulate_latent_forward_backward,
    simulate_pixel_ddpm,
    simulate_uncond_vs_cond_ldm,
)
from diffusion_simulator.models import build_scheduler, get_device_info
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


class _DummyTokenizer:
    model_max_length = 77

    def __call__(self, text: str, padding="max_length", max_length=77, truncation=True, return_tensors="pt"):
        ids = torch.zeros((1, max_length), dtype=torch.long)
        if text:
            vals = [max(1, ord(c) % 1000) for c in text[:max_length]]
            ids[0, : len(vals)] = torch.tensor(vals, dtype=torch.long)
        return SimpleNamespace(input_ids=ids)


class _DummyTextEncoder:
    def __call__(self, input_ids: torch.Tensor):
        b, l = input_ids.shape
        out = torch.zeros((b, l, 32), dtype=torch.float32, device=input_ids.device)
        # Non-empty prompts produce non-zero directional embedding
        scale = (input_ids.float().mean(dim=-1, keepdim=True) / 500.0).unsqueeze(-1)
        out = out + scale
        return (out,)


class _DummyVAE:
    def __init__(self, scale_factor: int = 8):
        self.config = SimpleNamespace(scaling_factor=0.18215)
        self.scale_factor = scale_factor
        self._p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def parameters(self):
        return iter([self._p])

    def encode(self, x: torch.Tensor):
        # Downsample (B, 3, H, W) -> (B, 4, H//8, W//8)
        b, _, h, w = x.shape
        ds = torch.nn.functional.avg_pool2d(x, kernel_size=self.scale_factor, stride=self.scale_factor)
        mean = torch.cat([ds, ds[:, :1]], dim=1) / self.config.scaling_factor
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))

    def decode(self, z: torch.Tensor):
        # Upsample (B, 4, H, W) -> (B, 3, H*8, W*8)
        rgb_lat = torch.tanh(z[:, :3])
        up = torch.nn.functional.interpolate(rgb_lat, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return SimpleNamespace(sample=up)


class _DummySDUNet:
    def __init__(self):
        self.config = SimpleNamespace(in_channels=4)
        self._p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def parameters(self):
        return iter([self._p])

    def __call__(self, x: torch.Tensor, t, encoder_hidden_states: torch.Tensor | None = None):
        # Deterministic noise estimate depending on x, timestep, and conditioning strength
        t_val = float(t.item()) if torch.is_tensor(t) else float(t)
        cond_bias = 0.0
        if encoder_hidden_states is not None:
            cond_bias = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1) * 0.45
        eps = 0.55 * x + (0.05 * (t_val / 1000.0)) + cond_bias
        return SimpleNamespace(sample=eps)


class _DummyPixelUNet:
    def __init__(self, img_size: int = 64):
        self.config = SimpleNamespace(sample_size=img_size, in_channels=3)
        self._p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def parameters(self):
        return iter([self._p])

    def __call__(self, x: torch.Tensor, t):
        t_val = float(t.item()) if torch.is_tensor(t) else float(t)
        eps = 0.6 * x + 0.03 * (t_val / 1000.0)
        return SimpleNamespace(sample=eps)


def _make_mock_sd_pipe():
    sched = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )
    return SimpleNamespace(
        tokenizer=_DummyTokenizer(),
        text_encoder=_DummyTextEncoder(),
        vae=_DummyVAE(scale_factor=8),
        unet=_DummySDUNet(),
        scheduler=sched,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _make_mock_pixel_pipe(img_size: int = 64):
    sched = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
    )
    return SimpleNamespace(
        unet=_DummyPixelUNet(img_size=img_size),
        scheduler=sched,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _assert_png(data: bytes, name: str) -> None:
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), f"{name} missing PNG header"
    assert len(data) > 500, f"{name} unexpectedly small ({len(data)} bytes)"


def _assert_gif(data: bytes, name: str) -> None:
    assert data[:6] in (b"GIF89a", b"GIF87a"), f"{name} missing GIF header"
    im = Image.open(io.BytesIO(data))
    assert getattr(im, "n_frames", 1) > 1, f"{name} must contain multiple frames"


def test_schedulers_and_device_info() -> None:
    info = get_device_info()
    assert info["device"] in ("cuda", "cpu")
    assert info["dtype"] in (torch.float16, torch.float32)

    sd_pipe = _make_mock_sd_pipe()
    px_pipe = _make_mock_pixel_pipe()

    for sampler in ("DDIM", "DDPM"):
        s_lat = build_scheduler(sd_pipe.scheduler.config, sampler=sampler, is_latent=True)
        assert s_lat.config.clip_sample is False, f"{sampler} latent must keep clip_sample=False"
        s_pix = build_scheduler(px_pipe.scheduler.config, sampler=sampler, is_latent=False)
        assert s_pix.config.clip_sample is True, f"{sampler} pixel must keep clip_sample=True"


def test_uncond_vs_cond_ldm() -> None:
    sd_pipe = _make_mock_sd_pipe()
    for sampler in ("DDIM", "DDPM"):
        res = simulate_uncond_vs_cond_ldm(
            sd_pipe,
            prompt="A cinematic robot painting a sunset",
            seed=42,
            num_steps=6,
            guidance_scale=7.5,
            extra_guidance_scale=15.0 if sampler == "DDIM" else None,
            sampler=sampler,
            resolution=128,
        )
        expected_branches = 3 if sampler == "DDIM" else 2
        assert len(res.branches) == expected_branches
        assert len(res.branches[0].steps) == 7
        assert np.allclose(res.branches[0].steps[0].z_t, res.branches[1].steps[0].z_t), "Initial z_T must match"
        assert res.divergence_l2["cond"][-1] > 0.0, "Conditioned trajectory must diverge from unconditioned"
        assert res.branches[1].steps[-1].guidance_map.shape == (16, 16)
        assert res.branches[1].steps[-1].decoded_zt.shape == (128, 128, 3)
        assert res.branches[1].steps[-1].decoded_zt.dtype == np.uint8

        if sampler == "DDIM":
            assert res.divergence_l2["cond_extra"][-1] > res.divergence_l2["cond"][-1]
            # Visual exports
            _assert_png(render_uncond_vs_cond_filmstrip(res, num_cols=5), "uncond_vs_cond_filmstrip")
            fig_pca, png_pca = render_pca_trajectory_fig(
                [(b.label, [s.z_t for s in b.steps], "#19468C") for b in res.branches],
                current_idx=3,
            )
            _assert_png(png_pca, "pca_trajectory")
            fig_div, png_div = render_divergence_curves_fig(res, current_idx=3)
            _assert_png(png_div, "divergence_curves")
            fig_insp, png_insp = render_latent_inspector_fig(
                res.branches[1].steps[3], res.branches[1].steps, res.branches[1].label
            )
            _assert_png(png_insp, "latent_inspector")
            import matplotlib.pyplot as plt
            plt.close(fig_pca)
            plt.close(fig_div)
            plt.close(fig_insp)

            _assert_gif(build_uncond_vs_cond_gif(res, fps=3), "uncond_vs_cond_gif")
            _assert_gif(build_latent_inspector_gif(res.branches[1], fps=3), "latent_inspector_gif")

            lat_rgb = latent_to_rgb(res.branches[1].steps[-1].z_t, upscale=4)
            gmap_rgb = guidance_map_to_rgb(res.branches[1].steps[-1].guidance_map, upscale=4)
            assert lat_rgb.shape == (64, 64, 3) and gmap_rgb.shape == (64, 64, 3)


def test_latent_forward_backward() -> None:
    sd_pipe = _make_mock_sd_pipe()
    src = make_procedural_sample_image("geometric", size=128)
    assert src.shape == (128, 128, 3) and src.dtype == np.uint8
    src_land = make_procedural_sample_image("landscape", size=128)
    assert src_land.shape == (128, 128, 3)

    for fwd_mode, rev_mode, sampler in [
        ("closed_form", "conditioned", "DDIM"),
        ("markov", "unconditioned", "DDPM"),
    ]:
        res = simulate_latent_forward_backward(
            sd_pipe,
            source_rgb=src,
            seed=7,
            t_max_frac=0.7,
            num_steps=8,
            forward_mode=fwd_mode,
            reverse_mode=rev_mode,
            prompt="geometric artwork",
            guidance_scale=5.0,
            sampler=sampler,
            resolution=128,
        )
        assert res.forward_steps[0].timestep == 0
        assert res.forward_steps[-1].timestep == res.t_max
        assert res.backward_steps[0].timestep == res.t_max
        assert res.backward_steps[-1].timestep == 0
        assert np.allclose(res.forward_steps[-1].z_t, res.backward_steps[0].z_t)

        _assert_png(render_forward_backward_filmstrip(res, num_cols=5), f"fb_filmstrip_{fwd_mode}")
        fig_m, png_m = render_forward_backward_metrics_fig(res)
        _assert_png(png_m, f"fb_metrics_{fwd_mode}")
        import matplotlib.pyplot as plt
        plt.close(fig_m)
        _assert_gif(build_forward_backward_gif(res, fps=3), f"fb_gif_{fwd_mode}")


def test_pixel_ddpm() -> None:
    px_pipe = _make_mock_pixel_pipe(img_size=64)
    src = make_procedural_sample_image("landscape", size=64)

    res_rev = simulate_pixel_ddpm(px_pipe, mode="reverse_only", seed=11, num_steps=6, sampler="DDIM")
    assert res_rev.state_dims == 3 * 64 * 64
    assert len(res_rev.forward_steps) == 0
    assert len(res_rev.backward_steps) == 7
    assert res_rev.backward_steps[-1].x_t_rgb.shape == (64, 64, 3)

    res_fb = simulate_pixel_ddpm(
        px_pipe, mode="forward_backward", source_rgb=src, seed=11, num_steps=8, t_max_frac=0.65, sampler="DDPM"
    )
    assert len(res_fb.forward_steps) >= 2
    assert res_fb.backward_steps[0].timestep == res_fb.forward_steps[-1].timestep
    _assert_png(render_pixel_ddpm_filmstrip(res_fb, num_cols=5), "pixel_filmstrip")
    fig_h, png_h = render_pixel_hist_fig(res_fb.backward_steps[-1], res_fb.backward_steps, "Pixel Backward")
    _assert_png(png_h, "pixel_hist")
    import matplotlib.pyplot as plt
    plt.close(fig_h)
    _assert_gif(build_pixel_ddpm_gif(res_fb, fps=3), "pixel_gif")


def test_streamlit_app_renders() -> None:
    app_path = os.path.join(THIS_DIR, "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert len(at.exception) == 0, f"Streamlit AppTest raised exceptions: {at.exception}"
    assert len(at.tabs) == 4, f"Expected 4 tabs, got {len(at.tabs)}"
    assert len(at.sidebar.radio) >= 1, "Expected global sampler radio in sidebar"


def test_colab_notebook_valid() -> None:
    nb_path = os.path.join(EXP_DIR, "diffusion_simulator_colab.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) >= 6
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code_cells) == 5
    for idx, c in enumerate(code_cells):
        src_lines = [ln for ln in c["source"] if not ln.lstrip().startswith(("!", "%"))]
        compile("".join(src_lines), f"colab_cell_{idx}", "exec")


def main() -> None:
    tests = [
        ("schedulers_and_device_info", test_schedulers_and_device_info),
        ("uncond_vs_cond_ldm", test_uncond_vs_cond_ldm),
        ("latent_forward_backward", test_latent_forward_backward),
        ("pixel_ddpm", test_pixel_ddpm),
        ("streamlit_app_renders", test_streamlit_app_renders),
        ("colab_notebook_valid", test_colab_notebook_valid),
    ]
    for name, fn in tests:
        fn()
        print(f"[PASS] {name}")
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
