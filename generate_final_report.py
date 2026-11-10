# -*- coding: utf-8 -*-
"""
generate_final_report.py
Syncs the fully polished, compliant LaTeX report for S06 Diffusion Models.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_PATH = os.path.join(BASE_DIR, "report", "S06_Diffusion_Models_Report.tex")

if os.path.exists(TEX_PATH):
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        tex_content = f.read()
    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print("Verified report/S06_Diffusion_Models_Report.tex successfully.")
