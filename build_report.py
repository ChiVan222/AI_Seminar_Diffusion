import os
import subprocess

with open('report/S06_Diffusion_Models_Report_backup.tex', 'r', encoding='utf-8') as f:
    orig = f.read()

# Extract abstract text
abs_start = orig.find(r'\begin{abstract}')
abs_end = orig.find(r'\end{abstract}')
abstract_content = orig[abs_start + len(r'\begin{abstract}'):abs_end].strip()

# Extract body from Section 1 Introduction to \end{document}
body_start = orig.find(r'\section{Introduction}')
body_end = orig.rfind(r'\end{document}')
body = orig[body_start:body_end].strip()

# Replace [H] with [htbp] for cleaner float placement
body = body.replace(r'\begin{figure}[H]', r'\begin{figure}[htbp]')
body = body.replace(r'\begin{table}[H]', r'\begin{table}[htbp]')

# Tune subfigure sizes in multi-panel figures for tighter, more elegant layout
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task1_and_4_diffusion_trajectories.png}{0.92\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task1_and_4_diffusion_trajectories.png}{0.88\textwidth}')
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task1_visual_samples_comparison.png}{0.96\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task1_visual_samples_comparison.png}{0.88\textwidth}')
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task1_training_stability_curves.png}{0.96\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task1_training_stability_curves.png}{0.88\textwidth}')
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task1_quantitative_trilemma_metrics.png}{0.94\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task1_quantitative_trilemma_metrics.png}{0.88\textwidth}')
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task1_pretrained_rgb_comparison.png}{\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task1_pretrained_rgb_comparison.png}{0.92\textwidth}')
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task4_pretrained_ddpm_reverse_trajectory.png}{\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task4_pretrained_ddpm_reverse_trajectory.png}{0.92\textwidth}')
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task6_pixel_ddpm_vs_latent_ldm.png}{\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task6_pixel_ddpm_vs_latent_ldm.png}{0.92\textwidth}')
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task2_score_field_training_and_reverse.png}{\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task2_score_field_training_and_reverse.png}{0.92\textwidth}')
body = body.replace(r'\IncludeExpFigure{../experiments/figures/task1_discriminative_vs_generative_2d.png}{\textwidth}',
                    r'\IncludeExpFigure{../experiments/figures/task1_discriminative_vs_generative_2d.png}{0.92\textwidth}')

# DDPM Algorithms 1 & 2 (side-by-side as in Ho et al., NeurIPS 2020)
alg_box = r'''
% ==============================================================================
% 3.2 DDPM TRAINING AND SAMPLING ALGORITHMS (HO ET AL., NEURIPS 2020)
% ==============================================================================
\subsection{DDPM Training and Sampling Algorithms}

Ho et al.\ (2020) \cite{ho2020denoising} operationalized diffusion probabilistic modeling through two complementary algorithms displayed below: \textbf{Algorithm 1 (Training)} optimizes the unweighted MSE loss $L_{\text{simple}}$ over uniformly sampled timesteps $t \sim \text{Uniform}(\{1, \dots, T\})$ using the closed-form jump $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\bm{\epsilon}$; \textbf{Algorithm 2 (Sampling)} starts from pure standard Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and progressively denoises the latent state back to a clean sample $\mathbf{x}_0$ over $T$ sequential Langevin-like steps.

\begin{figure}[htbp]
\centering
\begin{minipage}[t]{0.485\textwidth}
  \vspace{0pt}
  \hrule height 0.8pt
  \vspace{3pt}
  \noindent\textbf{Algorithm 1} Training
  \vspace{2pt}
  \hrule height 0.4pt
  \vspace{3pt}
  \footnotesize
  \begin{algorithmic}[1]
    \Repeat
      \State $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
      \State $t \sim \text{Uniform}(\{1, \dots, T\})$
      \State $\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
      \State Take gradient descent step on
      \Statex \hspace{1.2em} $\nabla_\theta \big\|\bm{\epsilon} - \bm{\epsilon}_\theta\big(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\bm{\epsilon}, \, t\big)\big\|^2$
    \Until{converged}
  \end{algorithmic}
  \vspace{2pt}
  \hrule height 0.8pt
\end{minipage}%
\hfill
\begin{minipage}[t]{0.485\textwidth}
  \vspace{0pt}
  \hrule height 0.8pt
  \vspace{3pt}
  \noindent\textbf{Algorithm 2} Sampling
  \vspace{2pt}
  \hrule height 0.4pt
  \vspace{3pt}
  \footnotesize
  \begin{algorithmic}[1]
    \State $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
    \For{$t = T, \dots, 1$}
      \State $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ if $t > 1$, else $\mathbf{z} = \mathbf{0}$
      \State $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bm{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}$
    \EndFor
    \State \textbf{return} $\mathbf{x}_0$
  \end{algorithmic}
  \vspace{2pt}
  \hrule height 0.8pt
\end{minipage}
\end{figure}

'''

target_marker = r'\subsection{Formal Equation Specifications and 1D Worked Numerical Example — Section III \& Task 3}'
if target_marker in body:
    body = body.replace(target_marker, alg_box + target_marker)
    print('Algorithm box inserted successfully.')
else:
    print('Error: Target marker not found in body!')

full_latex = r'''\UseRawInputEncoding
\documentclass[10pt,a4paper]{article}

% ==============================================================================
% PACKAGES: NEURIPS / DDPM RESEARCH PAPER STYLE (Ho et al., NeurIPS 2020)
% ==============================================================================
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}              % Times Roman text font (NeurIPS / DDPM standard)
\usepackage{amsmath, amssymb, amsfonts, amsthm, bm}
\usepackage[a4paper, left=1.0in, right=1.0in, top=1.0in, bottom=1.0in]{geometry}
\usepackage{booktabs}           % Formal tables (toprule, midrule, bottomrule)
\usepackage{array}
\usepackage{multirow}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, calc, fit, backgrounds}
\usepackage{microtype}          % Publication-grade typography and kerning
\usepackage[colorlinks=true, linkcolor=blue!85!black, citecolor=green!45!black, urlcolor=blue!85!black]{hyperref}

% NeurIPS-style section headings without external dependency
\makeatletter
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-2.0ex \@plus -0.5ex \@minus -.2ex}%
                                   {0.8ex \@plus.2ex}%
                                   {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-1.5ex\@plus -0.4ex \@minus -.2ex}%
                                     {0.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-1.2ex\@plus -0.3ex \@minus -.2ex}%
                                     {0.4ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {1.0ex \@plus 0.3ex \@minus 0.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\makeatother

% Spacing and formatting (DDPM / NeurIPS style)
\setlength{\parindent}{1.5em}
\setlength{\parskip}{0pt}
\renewcommand{\arraystretch}{1.12}

% Float spacing
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 3pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}

% Caption styling (NeurIPS: bold label, small font)
\captionsetup{font=small, labelfont=bf, labelsep=colon}
\captionsetup[table]{font=small, labelfont=bf, labelsep=colon, position=top, skip=4pt}

% Helper macro: includes experimental figure if already generated, otherwise renders a clean placeholder box
\newcommand{\IncludeExpFigure}[3]{%
    \IfFileExists{#1}{%
        \includegraphics[width=#2]{#1}%
    }{%
        \fbox{\parbox[c][3.2cm][c]{0.92\linewidth}{\centering \textbf{\small [Experimental Figure from Notebook]}\\[0.3em] {\scriptsize\nolinkurl{#1}}\\[0.3em] \footnotesize #3}}%
    }%
}

% Metadata for cover page (title.tex)
\newcommand{\coursename}{CS420 — Introduction to Artificial Intelligence}
\newcommand{\reportname}{[S06] Diffusion Models: From Adding Noise to Generating Images}
\newcommand{\reporttitle}{SEMINAR REPORT}
\newcommand{\studentname}{Vong Chi Van (24125049)\\ Le Dat Phuc An (24125052)}

\begin{document}

% ------------------------------------------------------------------------------
% PAGE 1: COVER PAGE (HCMUS SEMINAR TEMPLATE - KEPT AS REQUESTED)
% ------------------------------------------------------------------------------
\input{title.tex}
\newpage

% ------------------------------------------------------------------------------
% PAGE 2+: DDPM RESEARCH PAPER FORMAT (Ho et al., NeurIPS 2020)
% ------------------------------------------------------------------------------
\pagestyle{plain}
\setcounter{page}{1}

% NeurIPS Title Banner
\begin{center}
  \hrule height 4pt
  \vskip 0.22in
  {\LARGE\bfseries [S06] Diffusion Models: From Adding Noise to Generating Images\par}
  \vskip 0.08in
  {\large\textit{Theory, Algorithmic Foundations, and Latent Scaling}}\par
  \vskip 0.20in
  \hrule height 1pt
  \vskip 0.20in

  \begin{tabular}{cc}
    \textbf{Vong Chi Van} & \textbf{Le Dat Phuc An} \\
    Student ID: 24125049 & Student ID: 24125052 \\
    \texttt{24125049@student.hcmus.edu.vn} & \texttt{24125052@student.hcmus.edu.vn}
  \end{tabular}\\[0.65em]
  Faculty of Information Technology, University of Science --- VNU-HCM\\
  CS420: Introduction to Artificial Intelligence (Seminar Research Report)
  \vskip 0.18in
\end{center}

% Abstract (NeurIPS style)
\begin{center}
  {\bfseries\large Abstract}
\end{center}
\vspace{-0.5em}
\begin{quote}
\small
''' + abstract_content + r'''

\vspace{0.4em}
\noindent\textbf{Keywords:} Generative AI, Denoising Diffusion Probabilistic Models (DDPM), Latent Diffusion Models (LDM), Variational Autoencoders (VAEs), Cross-Attention, Bayes' Theorem.
\end{quote}
\vspace{0.8em}

''' + body + r'''

\end{document}
'''

with open('report/S06_Diffusion_Models_Report.tex', 'w', encoding='utf-8') as f:
    f.write(full_latex)

print('Generated report/S06_Diffusion_Models_Report.tex successfully.')
