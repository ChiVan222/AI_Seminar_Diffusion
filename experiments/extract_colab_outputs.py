import os
import json
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TARGET_FIGS = [
    "task5_exp1_vae_compression_fidelity.png",
    "task6_exp2_latent_vs_pixel_training_efficiency.png",
    "task5_exp3_latent_interpolation_walk.png",
    "task5_exp4_cfg_scaling_test.png",
]

saved_figs = []
for nb_fname in ["task5_6_ldm_4_experiments.ipynb", "task1_generative_paradigms_comparison.ipynb"]:
    nb_path = os.path.join(BASE_DIR, nb_fname)
    if not os.path.exists(nb_path):
        continue
    with open(nb_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except Exception:
            continue

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for fig_name in TARGET_FIGS:
            if fig_name in src:
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "stream":
                        print(f"--- {nb_fname} Cell {idx} ({fig_name}) stdout ---")
                        print("".join(out.get("text", [])).strip())
                    data = out.get("data", {})
                    if "image/png" in data:
                        b64_str = data["image/png"]
                        if isinstance(b64_str, list):
                            b64_str = "".join(b64_str)
                        img_bytes = base64.b64decode(b64_str)
                        out_path = os.path.join(FIG_DIR, fig_name)
                        with open(out_path, "wb") as img_f:
                            img_f.write(img_bytes)
                        saved_figs.append((fig_name, len(img_bytes)))

print("\nExtracted Colab figures from notebook outputs:", saved_figs)
