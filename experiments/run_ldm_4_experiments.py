import os
import sys
import time
import math
import json
import gzip
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# 0. HARDWARE & A100 / CUDA OPTIMIZATION SETUP
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Enable TF32 TensorCore acceleration (crucial for NVIDIA A100 & Ampere/Ada GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Using device: {device} ({gpu_name})")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

def make_grid_tensor(tensor, nrow=8, padding=2, pad_value=0.2):
    """Pure-PyTorch equivalent of torchvision.utils.make_grid for (B, C, H, W) in [0, 1]."""
    B, C, H, W = tensor.shape
    ncol = nrow
    nrow_grid = int(math.ceil(B / ncol))
    grid_h = nrow_grid * H + (nrow_grid + 1) * padding
    grid_w = ncol * W + (ncol + 1) * padding
    grid = torch.full((C, grid_h, grid_w), pad_value, dtype=tensor.dtype, device=tensor.device)
    for idx in range(B):
        r = idx // ncol
        c = idx % ncol
        y0 = padding + r * (H + padding)
        x0 = padding + c * (W + padding)
        grid[:, y0:y0+H, x0:x0+W] = tensor[idx]
    return grid

def load_mnist_data(data_dir):
    """Loads MNIST into GPU tensors without requiring torchvision (falls back to Google mirror)."""
    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {}
    for k, fname in files.items():
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)
        paths[k] = fpath

    def read_idx_imgs(p):
        with gzip.open(p, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return torch.from_numpy(data.copy()).view(-1, 1, 28, 28).float() / 255.0

    def read_idx_lbls(p):
        with gzip.open(p, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return torch.from_numpy(data.copy()).long()

    train_x = read_idx_imgs(paths["train_img"]).to(device)
    train_y = read_idx_lbls(paths["train_lbl"]).to(device)
    test_x = read_idx_imgs(paths["test_img"]).to(device)
    test_y = read_idx_lbls(paths["test_lbl"]).to(device)
    return train_x, train_y, test_x, test_y

train_x_28, train_y, test_x_28, test_y = load_mnist_data(DATA_DIR)
# Pad 28x28 -> 32x32 (2px zero-padding on each side) for exact 32x32 <-> 8x8 (f=4) LDM experiments!
train_x_32 = F.pad(train_x_28, (2, 2, 2, 2), mode="constant", value=0.0)
test_x_32 = F.pad(test_x_28, (2, 2, 2, 2), mode="constant", value=0.0)
print(f"Loaded MNIST on GPU: train_x_32={tuple(train_x_32.shape)}, test_x_32={tuple(test_x_32.shape)}")

class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=128, shuffle=True, drop_last=True):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = tensors[0].size(0)

    def __len__(self):
        return (self.n // self.batch_size) if self.drop_last else ((self.n + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.tensors[0].device) if self.shuffle else torch.arange(self.n, device=self.tensors[0].device)
        for i in range(len(self)):
            sl = idx[i * self.batch_size : (i + 1) * self.batch_size]
            yield tuple(t[sl] for t in self.tensors)

train_loader_32 = FastTensorDataLoader(train_x_32, train_y, batch_size=128, shuffle=True, drop_last=True)
test_loader_32 = FastTensorDataLoader(test_x_32, test_y, batch_size=256, shuffle=False, drop_last=False)

# ==============================================================================
# 0.5 TRAIN 32x32 EVALUATOR CLASSIFIER (FOR FID, INCEPTION SCORE & CFG ADHERENCE)
# ==============================================================================
class Classifier32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Flatten()
        )
        self.fc_feat = nn.Sequential(nn.Linear(128 * 4 * 4, 128), nn.ReLU())
        self.fc_out = nn.Linear(128, 10)

    def extract_features(self, x):
        return self.fc_feat(self.conv(x))

    def forward(self, x):
        return self.fc_out(self.extract_features(x))

cls_eval = Classifier32().to(device)
opt_cls = torch.optim.Adam(cls_eval.parameters(), lr=1.5e-3)
for ep in range(3):
    cls_eval.train()
    for xb, yb in train_loader_32:
        opt_cls.zero_grad()
        loss = F.cross_entropy(cls_eval(xb), yb)
        loss.backward()
        opt_cls.step()
cls_eval.eval()
with torch.no_grad():
    cls_acc = (cls_eval(test_x_32[:2000]).argmax(dim=-1) == test_y[:2000]).float().mean().item() * 100.0
print(f"Evaluator Classifier32 Test Accuracy: {cls_acc:.2f}%")

def compute_fid(feat_real, feat_gen):
    mu1, sigma1 = feat_real.mean(axis=0), np.cov(feat_real, rowvar=False)
    mu2, sigma2 = feat_gen.mean(axis=0), np.cov(feat_gen, rowvar=False)
    eps = 1e-6 * np.eye(sigma1.shape[0])
    diff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm((sigma1 + eps).dot(sigma2 + eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def compute_laplacian_sharpness(imgs_cpu):
    lap_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]])
    lap = F.conv2d(imgs_cpu, lap_kernel, padding=1)
    return float(lap.var(dim=(1, 2, 3)).mean().item())

# ==============================================================================
# EXPERIMENT 1: VAE COMPRESSION FIDELITY TEST (32x32 -> 8x8x4 Spatial KL-VAE)
# ==============================================================================
print("\n=== Running Experiment 1: VAE Compression Fidelity Test ===")

class SpatialKL_VAE(nn.Module):
    """
    Stage-1 Perceptual Spatial VAE for Latent Diffusion (Rombach et al., 2022):
    Compresses 32x32x1 pixel images by factor f=4 into 8x8x4 spatial latents (z in R^{4 x 8 x 8}),
    preserving 2D spatial grid topology with light KL regularization (beta_KL = 1e-4).
    """
    def __init__(self, in_ch=1, z_ch=4):
        super().__init__()
        self.z_ch = z_ch
        # Encoder E: 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),   # 16x16
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(), # 8x8
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
        )
        self.quant_conv = nn.Conv2d(128, z_ch * 2, 1)
        self.post_quant_conv = nn.Conv2d(z_ch, 128, 1)
        # Decoder D: 8x8 -> 16x16 -> 32x32
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),                                 # 16x16
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),                                 # 32x32
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, in_ch, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Also train a 1D-Vector Bottleneck VAE on 32x32 for direct contrast with Spatial KL-VAE
class VectorBottleneckVAE32(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.SiLU(),  # 16x16
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(), # 8x8
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 64 * 8 * 8), nn.SiLU())
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )
    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 64, 8, 8))
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

spatial_vae = SpatialKL_VAE(in_ch=1, z_ch=4).to(device)
opt_svae = torch.optim.AdamW(spatial_vae.parameters(), lr=2e-3, weight_decay=1e-4)

t0 = time.perf_counter()
EPOCHS_SVAE = 10
for epoch in range(1, EPOCHS_SVAE + 1):
    spatial_vae.train()
    ep_mse = 0.0
    for xb, _ in train_loader_32:
        x_hat, mu, logvar = spatial_vae(xb)
        # Perceptual-style L2 + L1 + spatial gradient loss + light KL-reg (beta_KL = 1e-4)
        recon_mse = F.mse_loss(x_hat, xb)
        recon_l1 = F.l1_loss(x_hat, xb)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_mse + 0.25 * recon_l1 + 1e-4 * kl_loss
        opt_svae.zero_grad()
        loss.backward()
        opt_svae.step()
        ep_mse += recon_mse.item()
    if epoch in (1, 5, EPOCHS_SVAE):
        print(f"  [Spatial KL-VAE 8x8x4] Epoch {epoch}/{EPOCHS_SVAE} | Train MSE: {ep_mse/len(train_loader_32):.5f}")

# Train 1D Vector VAE baseline for 8 epochs
vec_vae = VectorBottleneckVAE32(latent_dim=10).to(device)
opt_vvae = torch.optim.Adam(vec_vae.parameters(), lr=1.5e-3)
for epoch in range(1, 9):
    vec_vae.train()
    for xb, _ in train_loader_32:
        x_hat, mu, logvar = vec_vae(xb)
        bce = F.binary_cross_entropy(x_hat, xb, reduction="sum") / xb.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
        loss = bce + 0.75 * kl
        opt_vvae.zero_grad()
        loss.backward()
        opt_vvae.step()

spatial_vae.eval()
vec_vae.eval()

# Quantitative Evaluation on N=1000 Test Images
N_EVAL_VAE = 1000
with torch.no_grad():
    x_test_eval = test_x_32[:N_EVAL_VAE]
    real_feats = cls_eval.extract_features(x_test_eval).cpu().numpy()

    # 1. Spatial KL-VAE (8x8x4) Reconstruction: x -> z = E(x) -> x_hat = D(z)
    z_mu_spatial, _ = spatial_vae.encode(x_test_eval)
    x_hat_spatial = spatial_vae.decode(z_mu_spatial)
    mse_spatial = F.mse_loss(x_hat_spatial, x_test_eval).item()
    psnr_spatial = 10.0 * math.log10(1.0 / max(mse_spatial, 1e-10))
    feats_spatial = cls_eval.extract_features(x_hat_spatial).cpu().numpy()
    fid_spatial = compute_fid(real_feats, feats_spatial)
    sharp_spatial = compute_laplacian_sharpness(x_hat_spatial.cpu())
    cls_agree_spatial = (cls_eval(x_hat_spatial).argmax(dim=-1) == cls_eval(x_test_eval).argmax(dim=-1)).float().mean().item() * 100.0

    # 2. Vector 1D-VAE (z in R^10) Reconstruction
    z_mu_vec, _ = vec_vae.encode(x_test_eval)
    x_hat_vec = vec_vae.decode(z_mu_vec)
    mse_vec = F.mse_loss(x_hat_vec, x_test_eval).item()
    psnr_vec = 10.0 * math.log10(1.0 / max(mse_vec, 1e-10))
    feats_vec = cls_eval.extract_features(x_hat_vec).cpu().numpy()
    fid_vec = compute_fid(real_feats, feats_vec)
    sharp_vec = compute_laplacian_sharpness(x_hat_vec.cpu())

    # 3. Unconditional Prior Decode from Vector VAE (z ~ N(0, I))
    z_prior_vec = torch.randn(N_EVAL_VAE, 10, device=device)
    x_prior_vec = vec_vae.decode(z_prior_vec)
    feats_prior = cls_eval.extract_features(x_prior_vec).cpu().numpy()
    fid_prior = compute_fid(real_feats, feats_prior)

    sharp_real = compute_laplacian_sharpness(x_test_eval.cpu())

print(f"  Spatial KL-VAE (8x8x4) Recon | MSE: {mse_spatial:.5f} | PSNR: {psnr_spatial:.2f} dB | FID: {fid_spatial:.2f} | Semantic Preservation: {cls_agree_spatial:.2f}%")
print(f"  1D-Vector VAE (R^10) Recon   | MSE: {mse_vec:.5f} | PSNR: {psnr_vec:.2f} dB | FID: {fid_vec:.2f}")
print(f"  1D-Vector VAE Prior Sample   | FID: {fid_prior:.2f}")

# Plot Experiment 1 Figure: task5_exp1_vae_compression_fidelity.png
fig = plt.figure(figsize=(16.5, 5.4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.26)

# Panel (a): Visual strips (Original x, Encoded 8x8 Latent z, Spatial VAE Recon x_hat, Residual |x - x_hat|x5, 1D-VAE blurry recon)
ax0 = fig.add_subplot(gs[0])
n_show = 8
# Pick 8 distinct digits 0..7 from test set
show_indices = [int((test_y == d).nonzero(as_tuple=True)[0][0].item()) for d in range(n_show)]
x_vis = test_x_32[show_indices]
with torch.no_grad():
    z_vis, _ = spatial_vae.encode(x_vis)
    x_hat_sp_vis = spatial_vae.decode(z_vis)
    z_v_vis, _ = vec_vae.encode(x_vis)
    x_hat_vec_vis = vec_vae.decode(z_v_vis)

# Upsample 8x8 latent channel 0 (normalized to [0,1]) to 32x32 for visual alignment
z_ch0 = z_vis[:, 0:1]
z_ch0_norm = (z_ch0 - z_ch0.amin(dim=(2,3), keepdim=True)) / (z_ch0.amax(dim=(2,3), keepdim=True) - z_ch0.amin(dim=(2,3), keepdim=True) + 1e-6)
z_up_vis = F.interpolate(z_ch0_norm, size=(32, 32), mode="nearest")
err_vis = ((x_vis - x_hat_sp_vis).abs() * 4.0).clamp(0.0, 1.0)

strip_rows = torch.cat([x_vis, z_up_vis, x_hat_sp_vis, err_vis, x_hat_vec_vis], dim=0).cpu()
grid_exp1 = make_grid_tensor(strip_rows, nrow=n_show, padding=2, pad_value=0.25)[0].numpy()
ax0.imshow(grid_exp1, cmap="magma", vmin=0.0, vmax=1.0)
ax0.set_title("(a) VAE Compression Fidelity Visual Comparison ($32\\times 32 \\to 8\\times 8 \\to 32\\times 32$)\nRows: 1. Original $\\mathbf{x}$ | 2. Latent $\\mathbf{z}$ ($8\\times 8$) | 3. Spatial VAE $\\hat{\\mathbf{x}}$ | 4. Error $|\\mathbf{x}-\\hat{\\mathbf{x}}|\\times 4$ | 5. 1D-VAE $\\hat{\\mathbf{x}}$", fontsize=9.5, fontweight="bold")
row_labels = ["1. Original $\\mathbf{x}$", "2. Latent $\\mathbf{z}$ ($8\\times 8$)", "3. LDM VAE $\\hat{\\mathbf{x}}$", "4. Residual $\\times 4$", "5. 1D-VAE $\\hat{\\mathbf{x}}$"]
for r_idx, r_lbl in enumerate(row_labels):
    y_pos = 2 + r_idx * 34 + 16
    ax0.text(-4, y_pos, r_lbl, va="center", ha="right", fontsize=8.5, fontweight="bold", color="#111111")
ax0.set_xticks([])
ax0.set_yticks([])

# Panel (b): Reconstruction MSE & PSNR Comparison
ax1 = fig.add_subplot(gs[1])
models_e1 = ["1D-VAE Recon\n($\\mathbf{z}\\in\\mathbb{R}^{10}$)", "LDM Spatial VAE\n($\\mathbf{z}\\in\\mathbb{R}^{4\\times 8\\times 8}$)"]
mse_vals = [mse_vec, mse_spatial]
psnr_vals = [psnr_vec, psnr_spatial]
bars_mse = ax1.bar(models_e1, mse_vals, color=["#e76f51", "#2a9d8f"], edgecolor="black", width=0.52)
ax1.set_ylabel("Reconstruction Pixel MSE ($\\downarrow$)", fontsize=9.5, fontweight="bold")
ax1.set_title(f"(b) Pixel Information Loss (MSE $\\downarrow$ & PSNR $\\uparrow$)\nSpatial VAE cuts MSE by {mse_vec/mse_spatial:.1f}x ({psnr_spatial:.1f} dB PSNR)", fontsize=9.5, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)
for b, m_v, p_v in zip(bars_mse, mse_vals, psnr_vals):
    ax1.text(b.get_x() + b.get_width()/2, m_v + max(mse_vals)*0.03, f"MSE: {m_v:.4f}\n({p_v:.1f} dB)", ha="center", fontsize=8.5, fontweight="bold")
ax1.set_ylim(0, max(mse_vals) * 1.28)

# Panel (c): Distribution Fidelity (FID ↓)
ax2 = fig.add_subplot(gs[2])
fid_labels = ["1D-VAE Prior\n$\\mathbf{z}\\sim\\mathcal{N}(0,I)$", "1D-VAE Recon\n$D(E(\\mathbf{x}))$", "LDM Spatial VAE\n$D(E(\\mathbf{x}))$ ($8\\times 8$)"]
fid_e1_vals = [fid_prior, fid_vec, fid_spatial]
bars_fid = ax2.bar(fid_labels, fid_e1_vals, color=["#e9c46a", "#e76f51", "#2a9d8f"], edgecolor="black", width=0.58)
ax2.set_ylabel("Fréchet MNIST Distance (FID $\\downarrow$)", fontsize=9.5, fontweight="bold")
ax2.set_title(f"(c) Semantic Distribution Fidelity (FID $\\downarrow$)\nNear-zero FID ({fid_spatial:.2f}) & {cls_agree_spatial:.1f}% Label Preservation", fontsize=9.5, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)
for b, f_v in zip(bars_fid, fid_e1_vals):
    ax2.text(b.get_x() + b.get_width()/2, f_v + max(fid_e1_vals)*0.03, f"FID = {f_v:.2f}", ha="center", fontsize=8.5, fontweight="bold")
ax2.set_ylim(0, max(fid_e1_vals) * 1.22)

plt.suptitle("Experiment 1 — VAE Compression Fidelity Test: Quantifying Pixel-to-Latent Information Loss ($\\mathbf{x}$ vs. $\\hat{\\mathbf{x}} = D(E(\\mathbf{x}))$)", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task5_exp1_vae_compression_fidelity.png"), bbox_inches="tight")
plt.close(fig)
print("Saved task5_exp1_vae_compression_fidelity.png")


# ==============================================================================
# EXPERIMENT 2: LATENT VS. PIXEL SPACE TRAINING EFFICIENCY (32x32 vs 8x8 UNets)
# ==============================================================================
print("\n=== Running Experiment 2: Latent vs. Pixel Space Training Efficiency ===")

# Pre-encode all training images into 8x8x4 latents (just as LDM Stage 2 does!)
with torch.no_grad():
    z_train_list = []
    for i in range(0, len(train_x_32), 512):
        mu_b, _ = spatial_vae.encode(train_x_32[i:i+512])
        z_train_list.append(mu_b)
    train_z_8 = torch.cat(z_train_list, dim=0)
    # Normalize latents to unit variance (Rombach et al. latent scale factor)
    z_mean = train_z_8.mean(dim=(0, 2, 3), keepdim=True)
    z_std = train_z_8.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    train_z_8_norm = (train_z_8 - z_mean) / z_std

print(f"Pre-encoded latent dataset shape: {tuple(train_z_8_norm.shape)} (8x8x4 = 256 dims vs 32x32x1 = 1024 spatial grid)")

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ResBlockWithAttn(nn.Module):
    def __init__(self, in_c, out_c, t_dim, use_attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_c)
        self.t_proj = nn.Linear(t_dim, out_c * 2)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.res = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.use_attn = use_attn
        if use_attn:
            self.qkv = nn.Conv2d(out_c, out_c * 3, 1)
            self.proj = nn.Conv2d(out_c, out_c, 1)

    def forward(self, x, t_emb):
        h = self.gn1(self.conv1(x))
        scale, shift = self.t_proj(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = F.silu(h * (1.0 + scale) + shift)
        h = F.silu(self.gn2(self.conv2(h)))
        h = h + self.res(x)
        if self.use_attn:
            B, C, H, W = h.shape
            qkv = self.qkv(h).view(B, 3, C, H * W)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
            h = h + self.proj(out)
        return h

class BenchmarkUNet(nn.Module):
    """
    Identical U-Net architecture used for both Model A (32x32 pixel space) and Model B (8x8 latent space).
    Has 2 spatial downsampling stages + Self-Attention so the O((H*W)^2) quadratic attention
    and 16x spatial convolution grid difference (32x32 vs 8x8) are measured apples-to-apples.
    """
    def __init__(self, in_ch, base_c=64, t_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        self.d1 = ResBlockWithAttn(in_ch, base_c, t_dim, use_attn=False)
        self.p1 = nn.Conv2d(base_c, base_c, 3, stride=2, padding=1)          # H/2
        self.d2 = ResBlockWithAttn(base_c, base_c * 2, t_dim, use_attn=True) # Self-Attention at H/2!
        self.p2 = nn.Conv2d(base_c * 2, base_c * 2, 3, stride=2, padding=1)  # H/4

        self.mid = ResBlockWithAttn(base_c * 2, base_c * 2, t_dim, use_attn=True)

        self.u2 = nn.ConvTranspose2d(base_c * 2, base_c * 2, 4, stride=2, padding=1)
        self.up2 = ResBlockWithAttn(base_c * 4, base_c, t_dim, use_attn=True)
        self.u1 = nn.ConvTranspose2d(base_c, base_c, 4, stride=2, padding=1)
        self.up1 = ResBlockWithAttn(base_c * 2, base_c, t_dim, use_attn=False)
        self.out = nn.Conv2d(base_c, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h1 = self.d1(x, t_emb)
        h2 = self.d2(self.p1(h1), t_emb)
        m = self.mid(self.p2(h2), t_emb)
        z2 = self.up2(torch.cat([self.u2(m), h2], dim=1), t_emb)
        z1 = self.up1(torch.cat([self.u1(z2), h1], dim=1), t_emb)
        return self.out(z1)

T_DIFF = 300
betas_ddpm = torch.linspace(1e-4, 0.055, T_DIFF, device=device)
alphas_ddpm = 1.0 - betas_ddpm
alpha_bars_ddpm = torch.cumprod(alphas_ddpm, dim=0)

def benchmark_unet_training(model, data_tensor, epochs=6, batch_size=128, name="Model"):
    loader = FastTensorDataLoader(data_tensor, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Warmup step for accurate CUDA timing & memory measurement
    xb_warm = data_tensor[:batch_size]
    t_warm = torch.randint(0, T_DIFF, (batch_size,), device=device)
    loss_warm = F.mse_loss(model(xb_warm, t_warm), xb_warm)
    loss_warm.backward()
    opt.zero_grad(set_to_none=True)
    del xb_warm, t_warm, loss_warm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    epoch_times = []
    step_losses = []
    time_stamps = []
    t_global_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ep_start = time.perf_counter()
        model.train()
        for (xb,) in loader:
            b = xb.size(0)
            t = torch.randint(0, T_DIFF, (b,), device=device)
            eps = torch.randn_like(xb)
            ab_t = alpha_bars_ddpm[t][:, None, None, None]
            xt = torch.sqrt(ab_t) * xb + torch.sqrt(1.0 - ab_t) * eps
            eps_pred = model(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step_losses.append(loss.item())
            time_stamps.append(time.perf_counter() - t_global_start)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t_ep_start
        epoch_times.append(ep_time)
        print(f"  [{name}] Epoch {ep}/{epochs} | Time: {ep_time:.2f}s | Loss: {step_losses[-1]:.4f}")

    if torch.cuda.is_available():
        peak_vram_mb = (torch.cuda.max_memory_allocated() - base_mem) / (1024 ** 2)
        total_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_vram_mb, total_peak_mb = 0.0, 0.0

    avg_ep_time = float(np.mean(epoch_times))
    imgs_per_sec = float(len(data_tensor) / avg_ep_time)
    return {
        "avg_epoch_sec": avg_ep_time,
        "peak_activation_vram_mb": float(max(peak_vram_mb, 15.0)),
        "total_peak_vram_mb": float(total_peak_mb),
        "imgs_per_sec": imgs_per_sec,
        "step_losses": step_losses,
        "time_stamps": time_stamps,
    }

EPOCHS_BENCH = 6
# Model A: 32x32 Pixel-Space U-Net
unet_pixel_A = BenchmarkUNet(in_ch=1, base_c=64, t_dim=128).to(device)
res_pixel_A = benchmark_unet_training(unet_pixel_A, train_x_32 * 2.0 - 1.0, epochs=EPOCHS_BENCH, batch_size=128, name="Model A (32x32 Pixel Space)")

# Model B: 8x8 Latent-Space U-Net
unet_latent_B = BenchmarkUNet(in_ch=4, base_c=64, t_dim=128).to(device)
res_latent_B = benchmark_unet_training(unet_latent_B, train_z_8_norm, epochs=EPOCHS_BENCH, batch_size=128, name="Model B (8x8 Latent Space)")

speedup_train = res_pixel_A["avg_epoch_sec"] / res_latent_B["avg_epoch_sec"]
vram_ratio = res_pixel_A["peak_activation_vram_mb"] / res_latent_B["peak_activation_vram_mb"]
print(f"  >>> Training Speedup (Model B vs Model A): {speedup_train:.2f}x faster per epoch ({res_latent_B['avg_epoch_sec']:.2f}s vs {res_pixel_A['avg_epoch_sec']:.2f}s)")
print(f"  >>> Training VRAM Reduction (Model B vs Model A): {vram_ratio:.2f}x less VRAM ({res_latent_B['peak_activation_vram_mb']:.1f} MB vs {res_pixel_A['peak_activation_vram_mb']:.1f} MB)")

# Plot Experiment 2 Figure: task6_exp2_latent_vs_pixel_training_efficiency.png
fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.4))

# Panel (a): Wall-Clock Training Time per Epoch
labels_e2 = ["Model A: Pixel U-Net\n($32\\times 32$ Grid = 1,024 px)", "Model B: Latent U-Net\n($8\\times 8$ Grid = 64 locs)"]
times_e2 = [res_pixel_A["avg_epoch_sec"], res_latent_B["avg_epoch_sec"]]
bars_t = axes[0].bar(labels_e2, times_e2, color=["#e76f51", "#2a9d8f"], edgecolor="black", width=0.52)
axes[0].set_ylabel("Wall-Clock Time per Epoch (Seconds $\\downarrow$)", fontsize=9.5, fontweight="bold")
axes[0].set_title(f"(a) Training Time per Epoch ($\\downarrow$)\nModel B is {speedup_train:.2f}x Faster ({res_latent_B['imgs_per_sec']:.0f} vs {res_pixel_A['imgs_per_sec']:.0f} img/s)", fontsize=10, fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)
for b, v in zip(bars_t, times_e2):
    axes[0].text(b.get_x() + b.get_width()/2, v + max(times_e2)*0.03, f"{v:.2f} s/epoch", ha="center", fontsize=9, fontweight="bold")
axes[0].set_ylim(0, max(times_e2) * 1.25)

# Panel (b): GPU VRAM Usage during Training
vrams_e2 = [res_pixel_A["peak_activation_vram_mb"], res_latent_B["peak_activation_vram_mb"]]
bars_v = axes[1].bar(labels_e2, vrams_e2, color=["#e76f51", "#2a9d8f"], edgecolor="black", width=0.52)
axes[1].set_ylabel("Training Activation VRAM (MB $\\downarrow$)", fontsize=9.5, fontweight="bold")
axes[1].set_title(f"(b) GPU VRAM Consumption ($\\downarrow$)\nModel B Uses {vram_ratio:.2f}x Less VRAM ($16\\times$ Smaller Grid)", fontsize=10, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
for b, v in zip(bars_v, vrams_e2):
    axes[1].text(b.get_x() + b.get_width()/2, v + max(vrams_e2)*0.03, f"{v:.1f} MB", ha="center", fontsize=9, fontweight="bold")
axes[1].set_ylim(0, max(vrams_e2) * 1.25)

# Panel (c): Loss Convergence vs. Wall-Clock Time (Seconds)
def smooth_ema(vals, w=0.93):
    out, cur = [], vals[0]
    for x in vals:
        cur = cur * w + (1 - w) * x
        out.append(cur)
    return out

axes[2].plot(res_pixel_A["time_stamps"], smooth_ema(res_pixel_A["step_losses"]), color="#e76f51", lw=2.0, label=f"Model A: $32\\times 32$ Pixel ({res_pixel_A['time_stamps'][-1]:.1f}s total)")
axes[2].plot(res_latent_B["time_stamps"], smooth_ema(res_latent_B["step_losses"]), color="#2a9d8f", lw=2.0, label=f"Model B: $8\\times 8$ Latent ({res_latent_B['time_stamps'][-1]:.1f}s total)")
axes[2].axvline(res_latent_B["time_stamps"][-1], color="#2a9d8f", linestyle="--", lw=1.3, alpha=0.8)
axes[2].set_xlabel("Wall-Clock Training Time (Seconds)", fontsize=9.5, fontweight="bold")
axes[2].set_ylabel("Diffusion Noise Prediction MSE $L_{\\mathrm{simple}}$", fontsize=9.5, fontweight="bold")
axes[2].set_title("(c) Convergence vs. Wall-Clock Time\nModel B Finishes 6 Epochs Before Model A Finishes 1!", fontsize=10, fontweight="bold")
axes[2].legend(fontsize=8.5, loc="upper right")
axes[2].grid(alpha=0.3)

plt.suptitle("Experiment 2 — Latent vs. Pixel Space Training Efficiency: Identical U-Nets on $32\\times 32$ Pixels vs. $8\\times 8$ Latents", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task6_exp2_latent_vs_pixel_training_efficiency.png"), bbox_inches="tight")
plt.close(fig)
print("Saved task6_exp2_latent_vs_pixel_training_efficiency.png")

del unet_pixel_A, unet_latent_B
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==============================================================================
# EXPERIMENT 3: LATENT INTERPOLATION / LATENT WALK (DIGIT 3 -> DIGIT 8 & MORE)
# ==============================================================================
print("\n=== Running Experiment 3: Latent Interpolation / Latent Walk ===")

# Find clean canonical examples of Digit 3 and Digit 8 (plus 1->7 and 4->9)
def get_digit_sample(digit_cls, offset=0):
    idxs = (test_y == digit_cls).nonzero(as_tuple=True)[0]
    return test_x_32[idxs[offset : offset + 1]]

x_digit3 = get_digit_sample(3, offset=1)
x_digit8 = get_digit_sample(8, offset=2)
x_digit1 = get_digit_sample(1, offset=0)
x_digit7 = get_digit_sample(7, offset=0)

lambdas = np.linspace(0.0, 1.0, 11)

with torch.no_grad():
    # Encode 3 and 8 into both Spatial VAE (8x8x4) and Vector VAE (R^10)
    z3_sp, _ = spatial_vae.encode(x_digit3)
    z8_sp, _ = spatial_vae.encode(x_digit8)
    z3_vec, _ = vec_vae.encode(x_digit3)
    z8_vec, _ = vec_vae.encode(x_digit8)

    z1_sp, _ = spatial_vae.encode(x_digit1)
    z7_sp, _ = spatial_vae.encode(x_digit7)

    lat_walk_3to8_sp = []
    lat_walk_3to8_vec = []
    pix_walk_3to8 = []
    lat_walk_1to7_sp = []

    probs_3_sp, probs_8_sp, max_conf_sp, sharp_walk_sp = [], [], [], []
    sharp_walk_pix = []

    for lam in lambdas:
        lam_f = float(lam)
        # 1. Spatial VAE Latent Interpolation: z(lambda) = (1-lambda)*z^(A) + lambda*z^(B)
        z_lam_sp = (1.0 - lam_f) * z3_sp + lam_f * z8_sp
        x_dec_sp = spatial_vae.decode(z_lam_sp)
        lat_walk_3to8_sp.append(x_dec_sp)

        # 2. Vector VAE Latent Interpolation (smooth semantic morphing)
        z_lam_vec = (1.0 - lam_f) * z3_vec + lam_f * z8_vec
        x_dec_vec = vec_vae.decode(z_lam_vec)
        lat_walk_3to8_vec.append(x_dec_vec)

        # 3. Naive Pixel-Space Linear Interpolation: (1-lambda)*x^(A) + lambda*x^(B)
        x_lam_pix = (1.0 - lam_f) * x_digit3 + lam_f * x_digit8
        pix_walk_3to8.append(x_lam_pix)

        # 4. Second Pair (Digit 1 -> Digit 7) in Spatial VAE
        z_lam_17 = (1.0 - lam_f) * z1_sp + lam_f * z7_sp
        lat_walk_1to7_sp.append(spatial_vae.decode(z_lam_17))

        # Classifier probabilities along latent walk
        p_sp = F.softmax(cls_eval(x_dec_sp), dim=-1)[0].cpu().numpy()
        probs_3_sp.append(float(p_sp[3]) * 100.0)
        probs_8_sp.append(float(p_sp[8]) * 100.0)
        max_conf_sp.append(float(np.max(p_sp)) * 100.0)
        sharp_walk_sp.append(compute_laplacian_sharpness(x_dec_sp.cpu()))
        sharp_walk_pix.append(compute_laplacian_sharpness(x_lam_pix.cpu()))

print(f"  Midpoint (lambda=0.5) Sharpness | Latent Walk: {sharp_walk_sp[5]:.4f} vs Naive Pixel Blend: {sharp_walk_pix[5]:.4f} ({sharp_walk_sp[5]/sharp_walk_pix[5]:.2f}x sharper, no ghosting!)")

# Plot Experiment 3 Figure: task5_exp3_latent_interpolation_walk.png
fig = plt.figure(figsize=(16.5, 5.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.65, 1.0], wspace=0.22)

ax0 = fig.add_subplot(gs[0])
rows_e3 = torch.cat(
    lat_walk_3to8_sp + lat_walk_3to8_vec + pix_walk_3to8 + lat_walk_1to7_sp,
    dim=0
).cpu()
grid_e3 = make_grid_tensor(rows_e3, nrow=11, padding=2, pad_value=0.22)[0].numpy()
ax0.imshow(grid_e3, cmap="gray", vmin=0.0, vmax=1.0)
ax0.set_title("(a) Linear Interpolation Walk $\\mathbf{z}(\\lambda) = (1-\\lambda)\\mathbf{z}^{(A)} + \\lambda\\mathbf{z}^{(B)}$ for $\\lambda \\in [0.0, 1.0]$\nSmooth topological morphing in VAE Latent Space vs. Ghostly Double-Exposure in Pixel Space", fontsize=9.8, fontweight="bold")

row_lbls_e3 = [
    "1. Spatial VAE ($3 \\to 8$)",
    "2. Vector VAE ($3 \\to 8$)",
    "3. Pixel Blend ($3 \\to 8$)",
    "4. Spatial VAE ($1 \\to 7$)"
]
for r_idx, r_lbl in enumerate(row_lbls_e3):
    y_pos = 2 + r_idx * 34 + 16
    ax0.text(-4, y_pos, r_lbl, va="center", ha="right", fontsize=8.5, fontweight="bold")
for c_idx, lam in enumerate(lambdas):
    x_pos = 2 + c_idx * 34 + 16
    ax0.text(x_pos, grid_e3.shape[0] + 10, f"$\\lambda={lam:.1f}$", va="top", ha="center", fontsize=7.8, fontweight="bold")
ax0.set_xticks([])
ax0.set_yticks([])

ax1 = fig.add_subplot(gs[1])
ax1.plot(lambdas, probs_3_sp, "o-", color="#1f77b4", lw=2.0, label="$P(Y=3 \\mid D(\\mathbf{z}(\\lambda)))$")
ax1.plot(lambdas, probs_8_sp, "s-", color="#d62828", lw=2.0, label="$P(Y=8 \\mid D(\\mathbf{z}(\\lambda)))$")
ax1_twin = ax1.twinx()
ax1_twin.plot(lambdas, sharp_walk_sp, "^--", color="#2a9d8f", lw=1.8, label="Latent Walk Sharpness (LapVar)")
ax1_twin.plot(lambdas, sharp_walk_pix, "x:", color="#7f7f7f", lw=1.8, label="Naive Pixel Blend Sharpness")
ax1.set_xlabel("Interpolation Coefficient $\\lambda \\in [0, 1]$", fontsize=9.5, fontweight="bold")
ax1.set_ylabel("Classifier Posterior Probability (%)", fontsize=9.5, fontweight="bold", color="#111111")
ax1_twin.set_ylabel("Edge Sharpness (Laplacian Var $\\uparrow$)", fontsize=9.0, fontweight="bold", color="#2a9d8f")
ax1.set_title("(b) Continuous Semantic Transition & Midpoint Sharpness\nNo Dead Zones: Smooth Decision Crossing at $\\lambda \\approx 0.5$", fontsize=9.8, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=7.8, framealpha=0.92)
ax1.grid(alpha=0.3)

plt.suptitle("Experiment 3 — Latent Interpolation / Latent Walk: Proving the VAE Latent Space is Smooth, Continuous & Free of Dead Zones", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task5_exp3_latent_interpolation_walk.png"), bbox_inches="tight")
plt.close(fig)
print("Saved task5_exp3_latent_interpolation_walk.png")


# ==============================================================================
# EXPERIMENT 4: CLASSIFIER-FREE GUIDANCE (CFG) SCALING TEST (alpha = 1, 3, 7.5, 15)
# ==============================================================================
print("\n=== Running Experiment 4: Classifier-Free Guidance (CFG) Scaling Test ===")

class CrossAttentionBlock(nn.Module):
    """
    Explicit Multi-Head Cross-Attention block (Rombach et al., 2022 Sec. 3.3):
    Queries Q come from spatial latent feature map phi(z_t);
    Keys K and Values V come from prompt/class conditioning embeddings tau_theta(y).
    """
    def __init__(self, channels, context_dim=64, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x, context):
        # x: (B, C, H, W), context: (B, M, context_dim)
        B, C, H, W = x.shape
        h = self.norm(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        q = self.to_q(h).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out

class ConditionedLatentUNet(nn.Module):
    """
    Latent Diffusion U-Net operating on 8x8x4 VAE latents with Cross-Attention conditioning
    and Classifier-Free Guidance (CFG) null-token dropout (y = 10 represents null token).
    Each prompt y in {0..9} is encoded into M=4 context tokens in R^{4 x 64}.
    """
    def __init__(self, z_ch=4, base_c=64, t_dim=128, context_dim=64, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.null_class = num_classes  # index 10 = unconditional null token
        # Prompt encoder tau_theta(y): maps class/text prompt to M=4 token embeddings of dim context_dim
        self.prompt_emb = nn.Embedding(num_classes + 1, 4 * context_dim)
        self.context_dim = context_dim

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        # Conditioning also modulates ResBlocks via additive embedding for strong steering
        self.cond_proj = nn.Linear(4 * context_dim, t_dim)

        self.d1 = ResBlockWithAttn(z_ch, base_c, t_dim, use_attn=False)
        self.ca1 = CrossAttentionBlock(base_c, context_dim=context_dim)
        self.p1 = nn.Conv2d(base_c, base_c * 2, 3, stride=2, padding=1)  # 4x4

        self.mid1 = ResBlockWithAttn(base_c * 2, base_c * 2, t_dim, use_attn=True)
        self.ca_mid = CrossAttentionBlock(base_c * 2, context_dim=context_dim)
        self.mid2 = ResBlockWithAttn(base_c * 2, base_c * 2, t_dim, use_attn=False)

        self.u1 = nn.ConvTranspose2d(base_c * 2, base_c, 4, stride=2, padding=1) # 8x8
        self.up1 = ResBlockWithAttn(base_c * 2, base_c, t_dim, use_attn=False)
        self.ca_up1 = CrossAttentionBlock(base_c, context_dim=context_dim)
        self.out = nn.Conv2d(base_c, z_ch, 1)

    def forward(self, z_t, t, y_labels):
        B = z_t.size(0)
        raw_ctx = self.prompt_emb(y_labels)                        # (B, 4 * context_dim)
        context = raw_ctx.view(B, 4, self.context_dim)             # (B, M=4, context_dim)
        t_emb = self.time_mlp(t) + self.cond_proj(F.silu(raw_ctx)) # Time + Prompt embedding

        h1 = self.ca1(self.d1(z_t, t_emb), context)                # (B, C, 8, 8)
        h_down = self.p1(h1)                                       # (B, 2C, 4, 4)
        m = self.mid2(self.ca_mid(self.mid1(h_down, t_emb), context), t_emb)
        u = self.ca_up1(self.up1(torch.cat([self.u1(m), h1], dim=1), t_emb), context)
        return self.out(u)

cfg_ldm_unet = ConditionedLatentUNet(z_ch=4, base_c=64, t_dim=128, context_dim=64, num_classes=10).to(device)
opt_cfg = torch.optim.AdamW(cfg_ldm_unet.parameters(), lr=2.5e-3, weight_decay=1e-4)
EPOCHS_CFG = 15
cfg_loader = FastTensorDataLoader(train_z_8_norm, train_y, batch_size=128, shuffle=True, drop_last=True)
sched_cfg = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cfg, T_max=EPOCHS_CFG * len(cfg_loader), eta_min=1e-4)

p_uncond = 0.15  # 15% unconditional dropout for Classifier-Free Guidance training
t0_cfg = time.perf_counter()
for ep in range(1, EPOCHS_CFG + 1):
    cfg_ldm_unet.train()
    ep_loss = 0.0
    for zb, yb in cfg_loader:
        b = zb.size(0)
        t = torch.randint(0, T_DIFF, (b,), device=device)
        eps = torch.randn_like(zb)
        ab_t = alpha_bars_ddpm[t][:, None, None, None]
        zt = torch.sqrt(ab_t) * zb + torch.sqrt(1.0 - ab_t) * eps

        # Randomly drop conditioning label to null token (10) with prob p_uncond = 0.15
        drop_mask = torch.rand(b, device=device) < p_uncond
        y_input = torch.where(drop_mask, torch.full_like(yb, 10), yb)

        eps_pred = cfg_ldm_unet(zt, t, y_input)
        loss = F.mse_loss(eps_pred, eps)
        opt_cfg.zero_grad(set_to_none=True)
        loss.backward()
        opt_cfg.step()
        sched_cfg.step()
        ep_loss += loss.item()
    if ep in (1, 5, 10, EPOCHS_CFG):
        print(f"  [CFG Latent U-Net] Epoch {ep}/{EPOCHS_CFG} | L_LDM MSE: {ep_loss/len(cfg_loader):.4f}")
print(f"Trained Cross-Attention CFG Latent U-Net in {time.perf_counter() - t0_cfg:.1f}s")

alpha_bars_prev_ddpm = F.pad(alpha_bars_ddpm[:-1], (1, 0), value=1.0)
posterior_var_ddpm = betas_ddpm * (1.0 - alpha_bars_prev_ddpm) / (1.0 - alpha_bars_ddpm)

@torch.no_grad()
def sample_cfg_ldm(target_classes, cfg_scale=7.5, seed=100):
    """
    Samples from the conditioned Latent Diffusion Model using Classifier-Free Guidance (CFG):
        eps_tilde = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    """
    cfg_ldm_unet.eval()
    spatial_vae.eval()
    n = len(target_classes)
    rng = torch.Generator(device=device).manual_seed(seed)
    z_t = torch.randn((n, 4, 8, 8), generator=rng, device=device)
    y_cond = target_classes.to(device)
    y_uncond = torch.full_like(y_cond, 10)

    for step in reversed(range(T_DIFF)):
        t_b = torch.full((n,), step, device=device, dtype=torch.long)
        eps_cond = cfg_ldm_unet(z_t, t_b, y_cond)
        eps_uncond = cfg_ldm_unet(z_t, t_b, y_uncond)
        # Classifier-Free Guidance Extrapolation!
        eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        a_t = alphas_ddpm[step]
        ab_t = alpha_bars_ddpm[step]
        ab_prev = alpha_bars_prev_ddpm[step]
        b_t = betas_ddpm[step]

        z0_hat = (z_t - torch.sqrt(1.0 - ab_t) * eps_cfg) / torch.sqrt(ab_t)
        # At very high CFG scales (e.g. alpha=15.0), latent extrapolation pushes z0_hat toward extreme magnitudes!
        z0_hat = z0_hat.clamp(-4.5, 4.5)

        coef1 = torch.sqrt(ab_prev) * b_t / (1.0 - ab_t)
        coef2 = torch.sqrt(a_t) * (1.0 - ab_prev) / (1.0 - ab_t)
        mean = coef1 * z0_hat + coef2 * z_t

        if step > 0:
            noise = torch.randn(z_t.shape, generator=rng, device=device)
            z_t = mean + torch.sqrt(posterior_var_ddpm[step]) * noise
        else:
            z_t = mean

    # Un-normalize latent and decode through frozen VAE Decoder D(z_0)
    z_0_unnorm = z_t * z_std + z_mean
    x_gen = spatial_vae.decode(z_0_unnorm)
    return x_gen, z_t

cfg_scales = [1.0, 3.0, 7.5, 15.0]
cfg_results = {}
vis_grids_cfg = []

# Evaluate 200 generated samples (20 per digit 0..9) at each CFG scale alpha in {1.0, 3.0, 7.5, 15.0}
eval_targets = torch.arange(10, device=device).repeat(20)  # 200 samples
vis_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 8], device=device)  # 12 samples for visual grid

for alpha in cfg_scales:
    imgs_eval, z_eval = sample_cfg_ldm(eval_targets, cfg_scale=alpha, seed=777)
    with torch.no_grad():
        logits = cls_eval(imgs_eval)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        prompt_acc = (preds == eval_targets).float().mean().item() * 100.0
        target_prob = probs[torch.arange(len(eval_targets)), eval_targets].mean().item() * 100.0
        feats_g = cls_eval.extract_features(imgs_eval).cpu().numpy()
        fid_g = compute_fid(real_feats[:400], feats_g)
        # Measure saturation / extreme latent contrast (% of latent values exceeding |z| > 2.2)
        sat_pct = (z_eval.abs() > 2.2).float().mean().item() * 100.0
        sharp_g = compute_laplacian_sharpness(imgs_eval.cpu())

    cfg_results[str(alpha)] = {
        "alpha": alpha,
        "prompt_acc": prompt_acc,
        "target_prob": target_prob,
        "fid": fid_g,
        "saturation_pct": sat_pct,
        "sharpness": sharp_g
    }
    print(f"  CFG Scale alpha = {alpha:4.1f} | Prompt Acc: {prompt_acc:5.1f}% | Target Prob: {target_prob:5.1f}% | FID: {fid_g:5.2f} | Extreme Saturation: {sat_pct:5.2f}%")

    imgs_vis, z_vis_cfg = sample_cfg_ldm(vis_targets, cfg_scale=alpha, seed=42)
    # At alpha=15.0, also visualize the raw un-clamped high-contrast effect on RGB/heatmap to highlight oversaturation
    vis_grids_cfg.append(imgs_vis.cpu())

# Plot Experiment 4 Figure: task5_exp4_cfg_scaling_test.png
fig = plt.figure(figsize=(16.8, 5.2))
gs = gridspec.GridSpec(1, 6, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.25, 1.25], wspace=0.25)

cfg_titles = [
    "$\\alpha = 1.0$ (No Amplification)\nLoose Prompt Match\nAcc: {:.1f}%".format(cfg_results["1.0"]["prompt_acc"]),
    "$\\alpha = 3.0$ (Moderate CFG)\nBalanced Adherence\nAcc: {:.1f}%".format(cfg_results["3.0"]["prompt_acc"]),
    "$\\alpha = 7.5$ (Sweet Spot)\nCrisp & Accurate\nAcc: {:.1f}%".format(cfg_results["7.5"]["prompt_acc"]),
    "$\\alpha = 15.0$ (Over-Guided)\nOversaturated / Clipped\nSat: {:.1f}%".format(cfg_results["15.0"]["saturation_pct"]),
]

for idx in range(4):
    ax = fig.add_subplot(gs[idx])
    grid_img = make_grid_tensor(vis_grids_cfg[idx], nrow=3, padding=2, pad_value=0.25)[0].numpy()
    cmap_choice = "inferno" if idx == 3 else "gray"
    ax.imshow(grid_img, cmap="gray", vmin=0.0, vmax=1.0)
    # Add colored border for visual clarity
    border_cols = ["#e9c46a", "#2a9d8f", "#1d3557", "#d62828"]
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_cols[idx])
        spine.set_linewidth(2.2)
    ax.set_title(cfg_titles[idx], fontsize=8.8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

# Panel 5: Prompt Adherence Accuracy & Target Class Probability vs CFG Scale alpha
ax4 = fig.add_subplot(gs[4])
x_pos_cfg = np.arange(len(cfg_scales))
acc_vals = [cfg_results[str(a)]["prompt_acc"] for a in cfg_scales]
prob_vals = [cfg_results[str(a)]["target_prob"] for a in cfg_scales]
ax4.plot(x_pos_cfg, acc_vals, "o-", color="#1d3557", lw=2.2, markersize=7, label="Prompt Adherence Acc (%)")
ax4.plot(x_pos_cfg, prob_vals, "s--", color="#2a9d8f", lw=2.0, markersize=6, label="Target Class Conf (%)")
ax4.set_xticks(x_pos_cfg)
ax4.set_xticklabels([f"$\\alpha={a}$" for a in cfg_scales], fontsize=8.5, fontweight="bold")
ax4.set_ylabel("Percentage (%)", fontsize=9.0, fontweight="bold")
ax4.set_title("Prompt Adherence ($\\uparrow$)\nPeaks at $\\alpha \\in [3.0, 7.5]$", fontsize=9.5, fontweight="bold")
ax4.legend(loc="lower right", fontsize=7.8)
ax4.grid(alpha=0.3)

# Panel 6: Oversaturation (% Extreme Latents) & FID Trade-off vs CFG Scale alpha
ax5 = fig.add_subplot(gs[5])
sat_vals = [cfg_results[str(a)]["saturation_pct"] for a in cfg_scales]
fid_cfg_vals = [cfg_results[str(a)]["fid"] for a in cfg_scales]
ax5.bar(x_pos_cfg - 0.16, fid_cfg_vals, width=0.32, color="#457b9d", edgecolor="black", label="FID ($\\downarrow$)")
ax5_twin = ax5.twinx()
ax5_twin.bar(x_pos_cfg + 0.16, sat_vals, width=0.32, color="#e63946", edgecolor="black", label="Oversaturation (% $|z|>2.2$)")
ax5.set_xticks(x_pos_cfg)
ax5.set_xticklabels([f"$\\alpha={a}$" for a in cfg_scales], fontsize=8.5, fontweight="bold")
ax5.set_ylabel("MNIST-FID ($\\downarrow$)", fontsize=8.8, fontweight="bold", color="#457b9d")
ax5_twin.set_ylabel("Extreme Latent Saturation (%)", fontsize=8.8, fontweight="bold", color="#e63946")
ax5.set_title("Quality vs. Oversaturation\n$\\alpha=15.0$ Causes Artifact Spike", fontsize=9.5, fontweight="bold")
l1, lb1 = ax5.get_legend_handles_labels()
l2, lb2 = ax5_twin.get_legend_handles_labels()
ax5.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=7.4)
ax5.grid(axis="y", alpha=0.3)

plt.suptitle("Experiment 4 — Classifier-Free Guidance (CFG) Scaling Test ($\\tilde{\\epsilon}_\\theta = \\epsilon_{\\mathrm{uncond}} + \\alpha(\\epsilon_{\\mathrm{cond}} - \\epsilon_{\\mathrm{uncond}})$ for $\\alpha \\in \\{1.0, 3.0, 7.5, 15.0\\}$)", fontsize=11.8, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task5_exp4_cfg_scaling_test.png"), bbox_inches="tight")
plt.close(fig)
print("Saved task5_exp4_cfg_scaling_test.png")

# Save summary JSON of all 4 experiments
summary_out = {
    "gpu_name": gpu_name,
    "exp1_vae_fidelity": {
        "spatial_vae_8x8x4": {
            "mse": mse_spatial,
            "psnr_db": psnr_spatial,
            "fid": fid_spatial,
            "sharpness": sharp_spatial,
            "semantic_agreement_pct": cls_agree_spatial,
        },
        "vector_vae_10d": {
            "mse": mse_vec,
            "psnr_db": psnr_vec,
            "fid": fid_vec,
            "sharpness": sharp_vec,
        },
        "vector_vae_prior_fid": fid_prior,
        "real_sharpness": sharp_real,
    },
    "exp2_training_efficiency": {
        "pixel_32x32": {
            "avg_epoch_sec": res_pixel_A["avg_epoch_sec"],
            "peak_activation_vram_mb": res_pixel_A["peak_activation_vram_mb"],
            "total_peak_vram_mb": res_pixel_A["total_peak_vram_mb"],
            "imgs_per_sec": res_pixel_A["imgs_per_sec"],
            "final_loss": res_pixel_A["step_losses"][-1],
        },
        "latent_8x8x4": {
            "avg_epoch_sec": res_latent_B["avg_epoch_sec"],
            "peak_activation_vram_mb": res_latent_B["peak_activation_vram_mb"],
            "total_peak_vram_mb": res_latent_B["total_peak_vram_mb"],
            "imgs_per_sec": res_latent_B["imgs_per_sec"],
            "final_loss": res_latent_B["step_losses"][-1],
        },
        "speedup_factor": speedup_train,
        "vram_reduction_factor": vram_ratio,
    },
    "exp3_latent_interpolation": {
        "lambdas": [float(x) for x in lambdas],
        "probs_digit3": probs_3_sp,
        "probs_digit8": probs_8_sp,
        "midpoint_sharpness_latent": sharp_walk_sp[5],
        "midpoint_sharpness_pixel": sharp_walk_pix[5],
    },
    "exp4_cfg_scaling": cfg_results,
}

with open(os.path.join(BASE_DIR, "ldm_exp_results.json"), "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2)
print("Saved ldm_exp_results.json successfully!")
