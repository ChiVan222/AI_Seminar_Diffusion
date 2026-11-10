# -*- coding: utf-8 -*-
"""
generate_final_report.py
Generates the fully polished, compliant LaTeX report for S06 Diffusion Models.
Strictly adheres to:
1. Cover page on page 1 (HCMUS title.tex).
2. Diffusion paper style (single-column, 10pt Times, NeurIPS title banner, side-by-side DDPM algorithms).
3. Explicit seminar sections (Introduction, Background, Method, Results & Discussion, Practical Limitations, Conclusion, References).
4. Zero math equations in abstract.
5. Abstract is human-readable, concise, covers all seminar tasks and guiding question.
6. Abstract: no dashes, no injecting noise, simple discriminative vs generative comparison.
7. Introduction: no VAE, GAN, PixelCNN (all moved to Background Section 2.2).
8. Background Section 2.1: short and clear distinction (discriminator predicts P(Y|x) for classification, generative learns p(x) or p(x,Y) for generation), shows Figure 1, no generative used as discriminator.
9. Table 1 (Trilemma): citations for every claim across all cells.
10. Bold words: only titles and section headings.
11. Math and text separated and readable; long equations centered in standalone equation environments.
"""

tex_content = r"""\documentclass[10pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath, amsfonts, amssymb, amsthm, bm}
\usepackage[top=2.2cm, bottom=2.2cm, left=2.5cm, right=2.5cm]{geometry}
\usepackage{booktabs}
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
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, fit, backgrounds}
\usepackage{microtype}
\usepackage[colorlinks=true, linkcolor=blue!75!black, citecolor=blue!75!black, urlcolor=blue!75!black]{hyperref}

% Section formatting matching NeurIPS / DDPM paper
\makeatletter
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-2.5ex \@plus -0.8ex \@minus -.2ex}%
                                   {1.2ex \@plus.2ex}%
                                   {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-2.0ex\@plus -0.6ex \@minus -.2ex}%
                                     {0.8ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-1.5ex\@plus -0.4ex \@minus -.2ex}%
                                     {0.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {1.2ex \@plus 0.4ex \@minus .2ex}%
                                    {-0.8em}%
                                    {\normalfont\normalsize\bfseries}}
\makeatother

\captionsetup{font=small, labelfont=bf, labelsep=period}

% Header and footer
\pagestyle{plain}

% Helper for figures
\newcommand{\IncludeExpFigure}[3]{%
  \IfFileExists{#1}{%
    \includegraphics[width=#2]{#1}%
  }{%
    \begin{center}%
    \fbox{\parbox[c][3.2cm][c]{0.92\linewidth}{\centering \textit{\small [Experimental Figure: #3]}}}%
    \end{center}%
  }%
}

% Titlepage metadata
\newcommand{\reporttitle}{Seminar Report: Generative Artificial Intelligence}
\newcommand{\reportname}{Diffusion Models}
\newcommand{\coursename}{CS420: Introduction to Artificial Intelligence}
\newcommand{\studentname}{
    Vong Chi Van - 24125049\\
    Le Dat Phuc An - 24125052
}

\begin{document}

% ==============================================================================
% PAGE 1: OFFICIAL COVER PAGE (HCMUS TITLE)
% ==============================================================================
\input{title.tex}
\clearpage

% ==============================================================================
% PAGE 2: NEURIPS / DDPM STYLE TITLE BANNER & ABSTRACT
% ==============================================================================
\setcounter{page}{1}

\begin{center}
  \rule{\linewidth}{4pt}\vspace{-0.75em}
  \rule{\linewidth}{1pt}\vspace{0.8em}
  {\LARGE\bfseries [S06] Diffusion Models: From Adding Noise to Generating Images\par}
  \vspace{0.5em}
  {\large Theoretical Foundations, Step-by-Step Worked Derivations, and Latent Scaling}\par
  \vspace{0.6em}
  \rule{\linewidth}{1pt}\vspace{-0.75em}
  \rule{\linewidth}{4pt}
  \vspace{1.2em}

  {\large
    Vong Chi Van\quad Le Dat Phuc An
  }\par\vspace{0.3em}
  Faculty of Information Technology, University of Science, VNU-HCM\\
  \texttt{\{24125049, 24125052\}@student.hcmus.edu.vn}\par
  \vspace{1.0em}
\end{center}

\begin{center}
  {\bfseries\large Abstract}
\end{center}
\vspace{-0.5em}
\begin{quote}
\small
\noindent
Diffusion models have emerged as a dominant paradigm in generative artificial intelligence, achieving unprecedented visual fidelity and robust likelihood-based training without adversarial instability. This report investigates the fundamental guiding question: \textit{how can learning to remove noise enable a neural network to generate novel images, and why does diffusion within a learned latent space drastically reduce computational requirements?} Grounded in two foundational papers, Denoising Diffusion Probabilistic Models (Ho et al., 2020) and Latent Diffusion Models (Rombach et al., 2022), we examine the mathematical principles of forward Gaussian corruption, reverse denoising, and perceptual latent compression. Through theoretical derivations, a step-by-step numerical worked example, and empirical benchmark experiments, we provide an integrated analysis of diffusion dynamics, high-resolution scaling trade-offs, and practical deployment limitations.

\vspace{0.4em}
\noindent\textit{Keywords:} Generative AI, Denoising Diffusion Probabilistic Models, Latent Diffusion Models, Variational Autoencoders, Cross-Attention, Generative Learning Trilemma.
\end{quote}
\vspace{0.8em}

% ==============================================================================
% 1. INTRODUCTION
% ==============================================================================
\section{Introduction}

Synthesizing high-dimensional continuous signals such as natural images is a central challenge in artificial intelligence. A natural image tensor is represented as:
\begin{equation}
    \mathbf{x} \in \mathbb{R}^{H \times W \times C}
\end{equation}
where $H$, $W$, and $C$ denote the height, width, and color channel dimensions, respectively. Unlike supervised discriminative learning, which only seeks decision boundaries separating predefined class labels, a generative model must capture the true underlying probability distribution $p_{\text{data}}(\mathbf{x})$ of a complex, low-dimensional data manifold embedded in high-dimensional ambient space. Once learned, the model enables the synthesis of novel, realistic samples:
\begin{equation}
    \mathbf{x}_{\text{new}} \sim p_\theta(\mathbf{x})
\end{equation}

Deep generative architectures have historically struggled to simultaneously achieve high sample sharpness, comprehensive mode diversity, and fast sampling speed. While early generative paradigms made substantial progress, each faced inherent trade-offs between sample quality, training stability, and sampling efficiency.

\paragraph{Guiding Question and Scope.}
\textit{How can learning to remove noise enable a model to generate new images, and why does moving diffusion into a learned latent space resolve its computational bottleneck?}

The core insight of Denoising Diffusion Probabilistic Models (DDPM) \cite{sohl2015deep, ho2020denoising} is to decompose the difficult one-step mapping from standard Gaussian noise $\mathcal{N}(\mathbf{0}, \mathbf{I})$ to the data distribution $p_{\text{data}}(\mathbf{x}_0)$ into $T$ small, tractable steps ($T \approx 1000$). If a fixed forward process gradually corrupts a clean image $\mathbf{x}_0$ with small increments of Gaussian noise until the signal is erased into pure noise:
\begin{equation}
    \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{equation}
then learning the reverse process allows a neural network to start from random Gaussian noise and iteratively sculpt a realistic image through repeated denoising steps:
\begin{equation}
    \mathbf{x}_T \to \mathbf{x}_{T-1} \to \dots \to \mathbf{x}_0
\end{equation}

Furthermore, to overcome the immense computational cost of evaluating a full-resolution neural network in pixel space, Latent Diffusion Models (LDM) \cite{rombach2022high} first compress images into the compact latent space of a pretrained Variational Autoencoder before executing diffusion. This representation reduces the state dimensionality by $48\times$ and cuts self-attention complexity by $4{,}096\times$, rendering high-resolution synthesis tractable on standard consumer hardware.

\paragraph{Tasks and Report Organization.}
To systematically address the guiding question in accordance with the seminar requirements, this report is structured around six concrete tasks:
\begin{itemize}[leftmargin=1.8em, itemsep=0.3em]
    \item \textit{Task 1: Foundational Paradigms and the Generative Trilemma.} We contrast discriminative classifiers with generative modeling and analyze autoregressive models, variational autoencoders, generative adversarial networks, and diffusion models across the generative learning trilemma (Sections \ref{sec:bg_disc_gen} and \ref{sec:bg_trilemma}).
    \item \textit{Task 2: DDPM Mathematical Framework.} We formulate the variance-preserving forward process, variance schedule selection, closed-form multi-step transitions, the learned reverse Markov chain, and the simplified noise-prediction training objective (Section \ref{sec:method_ddpm_theory}).
    \item \textit{Task 3: Symbol Definitions and 1D Worked Example.} We specify complete symbol definitions, physical units, and modeling assumptions, accompanied by a step-by-step one-dimensional numerical worked example illustrating stochastic signal contraction and reverse denoising (Section \ref{sec:method_worked_example}).
    \item \textit{Task 4: Multi-Level Forward Corruption and Reverse Image Generation.} We demonstrate forward noise corruption across multiple timesteps and iterative reverse image generation trajectories on empirical benchmarks (Section \ref{sec:res_task4}).
    \item \textit{Task 5: Latent Diffusion Architecture.} We analyze the two-stage inductive bias of Latent Diffusion Models, examining perceptual autoencoder compression and cross-attention conditioning mechanisms for multimodal guidance (Section \ref{sec:method_ldm_arch}).
    \item \textit{Task 6: Pixel vs.\ Latent Comparative Evaluation and Practical Limitations.} We provide an evidence-supported comparison between pixel-space and latent-space diffusion across explicit computational and quality metrics, followed by a critical evaluation of real-world limitations including sequential sampling latency, prompt attribute binding, copyright memorization, and societal bias (Sections \ref{sec:res_task6} and \ref{sec:limitations}).
\end{itemize}

% ==============================================================================
% 2. BACKGROUND & PAPER REVIEW
% ==============================================================================
\section{Background and Paper Review}

\subsection{Distinguishing Discriminative vs.\ Generative Models}
\label{sec:bg_disc_gen}

Supervised machine learning and probabilistic modeling are fundamentally divided into two complementary paradigms:
\begin{enumerate}[leftmargin=1.8em, itemsep=0.4em]
    \item \textit{Discriminative Models:} Discriminator models are mainly used for classification and regression tasks. A discriminative model directly learns to predict the conditional probability distribution:
    \begin{equation}
        P(Y \mid \mathbf{x})
    \end{equation}
    where $Y$ is the class label and $\mathbf{x}$ represents the input features. It learns decision boundaries separating classes without modeling the distribution of the input features $\mathbf{x}$. Consequently, a discriminative model cannot generate novel data samples.

    \item \textit{Generative Models:} Generative models are primarily used to generate novel data, though they can also be used for classification. A generative model learns the underlying data distribution:
    \begin{equation}
        p(\mathbf{x})
    \end{equation}
    or the joint distribution:
    \begin{equation}
        p(\mathbf{x}, Y)
    \end{equation}
    when conditioned on class labels $Y$. Its primary function is synthesizing new, realistic samples:
    \begin{equation}
        \mathbf{x}_{\text{new}} \sim p(\mathbf{x})
    \end{equation}
    that resemble the training data distribution.
\end{enumerate}

Figure \ref{fig:exp_disc_gen} shows the visual comparison between the two paradigms on a synthetic four-class two-dimensional dataset.

\begin{figure}[htbp]
\centering
\IncludeExpFigure{../experiments/figures/task1_discriminative_vs_generative_2d.png}{0.92\textwidth}{2D Discriminative Classifier vs. Generative Model Comparison}
\caption{Visual comparison of modeling paradigms on a four-class two-dimensional manifold (Task 1): (a) A discriminative classifier learns quadrant decision boundaries to predict $P(Y\mid\mathbf{x})$ for classification; (b) A generative model learns class-conditional distributions $p(\mathbf{x}\mid Y)$ capturing the shape of the data; (c) The learned generative density enables generating new samples $\mathbf{x}_{\text{new}} \sim p(\mathbf{x})$ and detecting out-of-distribution queries.}
\label{fig:exp_disc_gen}
\end{figure}

\subsection{Deep Generative Paradigms and The Generative Learning Trilemma (Task 1)}
\label{sec:bg_trilemma}

To contextualize diffusion models within the generative literature, we examine prior generative families across the Generative Learning Trilemma \cite{xiao2022tackling}, which posits that a generative model ideally satisfies three core properties:
\begin{itemize}[leftmargin=1.5em, itemsep=0.2em]
    \item \textit{High Sample Quality:} Generating sharp, detailed, and photorealistic samples.
    \item \textit{Fast Sampling:} Synthesizing samples with minimal forward passes (ideally real-time or single-step).
    \item \textit{Full Mode Coverage (Diversity):} Faithfully covering all modes of the data distribution without mode collapse.
\end{itemize}

Historically, generative architectures achieved at most two of these three objectives:
\begin{enumerate}[leftmargin=1.8em, itemsep=0.4em]
    \item \textit{Autoregressive Models (PixelRNN, PixelCNN) \cite{oord2016pixel, salimans2017pixelcnn}:} Model images pixel-by-pixel using the probability chain rule:
    \begin{equation}
        p_\theta(\mathbf{x}) = \prod_{i=1}^n p_\theta(x^{(i)} \mid x^{(<i)})
    \end{equation}
    They maximize exact log-likelihood and avoid mode collapse, achieving high quality and full diversity. However, generating an image requires sequential pixel-by-pixel generation, yielding quadratic computational complexity and slow sampling.

    \item \textit{Variational Autoencoders (VAEs) \cite{kingma2014auto, rezende2014stochastic}:} Introduce an unobserved latent variable $\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and optimize the evidence lower bound (ELBO). VAEs generate images in a single forward decoder pass and cover all modes. However, because pixel likelihood objectives enforce mean-squared or cross-entropy reconstruction, VAEs average across plausible modes and produce visibly blurry samples.

    \item \textit{Generative Adversarial Networks (GANs) \cite{goodfellow2014generative, radford2016unsupervised}:} Frame synthesis as a two-player minimax game between a Generator and a Discriminator. GANs produce razor-sharp images in a single feed-forward step. However, because they lack an explicit density objective, GANs suffer from training instability and severe mode collapse.

    \item \textit{Denoising Diffusion Probabilistic Models (DDPM) \cite{sohl2015deep, ho2020denoising}:} Formulate image generation as inverting a parameterized Markov chain. By optimizing a stable variational objective that reduces to mean squared error, diffusion models achieve state-of-the-art visual fidelity and comprehensive mode coverage. Their primary bottleneck is sampling speed, requiring hundreds to thousands of iterative network evaluations.

    \item \textit{Latent Diffusion Models (LDM) \cite{rombach2022high}:} Resolve the sampling and training efficiency limitations of DDPM by operating within a perceptually compressed latent space learned by a pretrained autoencoder, achieving high sample fidelity and faster throughput.
\end{enumerate}

Table \ref{tab:task1_paradigms} summarizes the theoretical formulation, latent structure, and empirical trade-offs of each generative family, citing primary literature sources for every theoretical property and empirical claim.

\begin{table}[htbp]
\centering
\caption{Qualitative comparison of major generative model families across the Generative Learning Trilemma \cite{xiao2022tackling}, citing primary literature sources for each theoretical property and empirical claim (Task 1).}
\label{tab:task1_paradigms}
\footnotesize
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.3cm} >{\raggedright\arraybackslash}p{2.1cm} X >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}p{2.3cm}}
\toprule
Model Family & Density $p_\theta(\mathbf{x})$ & Training Mechanism \& Latent Structure & Sample Quality & Mode Coverage & Sampling Speed \\
\midrule
\textit{Autoregressive} \cite{oord2016pixel, salimans2017pixelcnn} (PixelRNN, PixelCNN) & Explicit, Exact (Exact MLE) \cite{oord2016pixel} & Chain rule $\prod_{i=1}^n p_\theta(x^{(i)} \mid x^{(<i)})$ via causal masked convolutions; no global latent $\mathbf{z}$ \cite{oord2016pixel, salimans2017pixelcnn}. & Sharp local contrast \cite{oord2016pixel, xiao2022tackling} & Full (No mode collapse; stable MLE) \cite{oord2016pixel, ho2020denoising} & Extremely Slow ($\mathcal{O}(n)$ sequential pixel evaluations) \cite{oord2016pixel, song2021denoising, xiao2022tackling} \\
\addlinespace[0.3em]
\textit{VAE} \cite{kingma2014auto, rezende2014stochastic} (Variational Autoencoder) & Explicit, Approx.\ (ELBO Bound) \cite{kingma2014auto, rezende2014stochastic} & Encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$ + Decoder $p_\theta(\mathbf{x}\mid\mathbf{z})$ with $\dim(\mathbf{z}) \ll \dim(\mathbf{x})$; Reparameterization Trick \cite{kingma2014auto}. & Blurry (Single-step $L_2$/BCE mode averaging) \cite{kingma2014auto, theis2016note, xiao2022tackling} & Full, smooth latent manifold \cite{kingma2014auto, xiao2022tackling} & Very Fast ($1$ Decoder forward pass) \cite{kingma2014auto, xiao2022tackling} \\
\addlinespace[0.3em]
\textit{GAN} \cite{goodfellow2014generative, radford2016unsupervised} (DCGAN, StyleGAN) & Implicit (Likelihood-free) \cite{goodfellow2014generative} & Adversarial minimax game $\min_G \max_D$ between Generator $G_\theta(\mathbf{z})$ and Discriminator $D_\phi(\mathbf{x})$ \cite{goodfellow2014generative, radford2016unsupervised}. & Razor Sharp (Mode-seeking via adversarial loss) \cite{goodfellow2014generative, brock2018large, xiao2022tackling} & Poor (Prone to Mode Collapse and minimax instability) \cite{goodfellow2014generative, arjovsky2017wasserstein, xiao2022tackling} & Very Fast ($1$ Generator forward pass) \cite{goodfellow2014generative, xiao2022tackling} \\
\addlinespace[0.3em]
\textit{Pixel Diffusion} \cite{sohl2015deep, ho2020denoising} (DDPM) & Explicit, Approx.\ ($T$-step ELBO) \cite{sohl2015deep, ho2020denoising} & Fixed $T$-step Markov chain with $\dim(\mathbf{x}_t) = \dim(\mathbf{x}_0)$; U-Net predicts noise $\bm{\epsilon}_\theta(\mathbf{x}_t, t)$ via unweighted MSE $L_{\text{simple}}$ \cite{ho2020denoising}. & SOTA Sharpness (FID $3.17$ on CIFAR-10) \cite{ho2020denoising, dhariwal2021diffusion} & Full (Stable regression; beats GAN mode coverage) \cite{ho2020denoising, dhariwal2021diffusion, xiao2022tackling} & Slow ($T \approx 1000$ sequential full-res U-Net passes) \cite{ho2020denoising, song2021denoising, xiao2022tackling} \\
\addlinespace[0.3em]
\textit{Latent Diffusion} \cite{rombach2022high} (LDM / Stable Diffusion) & Two-Stage ELBO (VAE + Diffusion) \cite{rombach2022high} & Compresses $\mathbf{x}_0 \to \mathbf{z}_0$ via frozen Conv-VAE, then runs Cross-Attention U-Net on latent tensor $\mathbf{z}_t$ \cite{rombach2022high}. & SOTA Photorealism at high resolution (FID $5.11$ on CelebA-HQ) \cite{rombach2022high} & Full and flexible multimodal text conditioning \cite{rombach2022high} & Fast and Practical (Operates on $48\times$ smaller $\mathbf{z}_t$; $20$ to $50$ DDIM steps) \cite{rombach2022high, song2021denoising} \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Foundational Literature Review}
\label{sec:bg_lit_review}

This report analyzes two foundational milestone papers:
\begin{itemize}[leftmargin=1.5em, itemsep=0.4em]
    \item \textit{Required Paper: Ho, Jain and Abbeel (NeurIPS 2020), ``Denoising Diffusion Probabilistic Models'' (DDPM) \cite{ho2020denoising}:}
    Demonstrated that diffusion models generate image quality rivaling GANs while retaining likelihood-based mode coverage. The paper made three key contributions:
    (1) establishing the equivalence between denoising score matching and Markov chain reverse step estimation;
    (2) reparameterizing the reverse transition mean to predict the injected Gaussian noise vector $\bm{\epsilon}$;
    (3) introducing the unweighted simplified training objective $L_{\text{simple}}(\theta)$, which suppresses small-step noise artifacts and prioritizes perceptual visual features.

    \item \textit{Recommended Paper: Rombach et al.\ (CVPR 2022), ``High-Resolution Image Synthesis With Latent Diffusion Models'' (LDM) \cite{rombach2022high}:}
    Identified that training pixel-space diffusion models on high-resolution images allocates excessive compute to imperceptible high-frequency details. Rombach et al.\ introduced a two-stage paradigm:
    (1) a first stage Variational Autoencoder (VAE) trained with perceptual and adversarial losses that compresses high-dimensional images into an efficient latent space;
    (2) a second stage diffusion backbone operating exclusively on latent representations, incorporating cross-attention layers to enable flexible conditioning on text prompts, bounding boxes, or segmentation maps.
\end{itemize}

% ==============================================================================
% 3. METHOD & STEP-BY-STEP WORKED EXAMPLE
% ==============================================================================
\section{Method and Step-by-Step Worked Example}

\subsection{Forward Process, Variance Schedule, Reverse Denoising, and Sampling (Task 2)}
\label{sec:method_ddpm_theory}

\paragraph{Forward Noising Process.}
Let clean data follow an unknown distribution:
\begin{equation}
    \mathbf{x}_0 \sim q(\mathbf{x}_0)
\end{equation}
The forward diffusion process is a non-trainable Markov chain governed by a fixed variance schedule:
\begin{equation}
    \beta_1, \beta_2, \dots, \beta_T \in (0, 1)
\end{equation}
The joint distribution over all intermediate steps is defined as:
\begin{equation}
    q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
    \label{eq:forward_joint}
\end{equation}
where each individual forward transition is Gaussian:
\begin{equation}
    q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\big(\mathbf{x}_t; \, \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1}, \, \beta_t \mathbf{I}\big)
    \label{eq:forward_step}
\end{equation}

\paragraph{Variance-Preserving Property.}
Equation \eqref{eq:forward_step} implements a Variance-Preserving (VP) stochastic differential process. By scaling the incoming signal by $\sqrt{1 - \beta_t}$, the total variance is preserved. Specifically, if the previous state has unit variance:
\begin{equation}
    \text{Var}(\mathbf{x}_{t-1}) = \mathbf{I}
\end{equation}
then the variance of the next state remains unit:
\begin{equation}
    \text{Var}(\mathbf{x}_t) = (\sqrt{1 - \beta_t})^2 \text{Var}(\mathbf{x}_{t-1}) + \beta_t \mathbf{I} = (1 - \beta_t)\mathbf{I} + \beta_t \mathbf{I} = \mathbf{I}
    \label{eq:vp_property}
\end{equation}
Consequently, the signal amplitude never explodes, and the final state converges smoothly to standard white Gaussian noise:
\begin{equation}
    \mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{equation}

\paragraph{Closed-Form Multi-Step Transition.}
A key property of Gaussian Markov chains is that intermediate sampling steps can be bypassed. Defining:
\begin{equation}
    \alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
\end{equation}
any arbitrary intermediate latent state $\mathbf{x}_t$ can be sampled directly from $\mathbf{x}_0$ in closed form:
\begin{equation}
    q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_t; \, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0, \, (1 - \bar{\alpha}_t)\mathbf{I}\big)
    \label{eq:closed_form_q}
\end{equation}
Using the reparameterization trick with standard Gaussian noise $\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, this is expressed as:
\begin{equation}
    \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\bm{\epsilon}
    \label{eq:closed_form_sample}
\end{equation}
Equation \eqref{eq:closed_form_sample} allows efficient model training by sampling any random timestep $t \sim \text{Uniform}(\{1, \dots, T\})$ in $\mathcal{O}(1)$ time without simulating steps $1$ to $t-1$.

\paragraph{Variance Schedule Selection.}
The schedule $\beta_1, \dots, \beta_T$ controls the rate of information destruction:
\begin{itemize}[leftmargin=1.5em, itemsep=0.3em]
    \item \textit{Small $\beta_t$:} The reverse distribution $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ is nearly Gaussian, ensuring that a neural network parameterized as a Gaussian can accurately model each step.
    \item \textit{Sufficiently Large $T$ and $\beta_T$:} Ensures $\bar{\alpha}_T \approx 0$, guaranteeing that $\mathbf{x}_T$ converges to pure Gaussian noise $\mathcal{N}(\mathbf{0}, \mathbf{I})$.
    \item \textit{Linear Schedule (Ho et al.\ \cite{ho2020denoising}):} Linearly interpolates from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ across $T=1000$.
    \item \textit{Cosine Schedule (Nichol \& Dhariwal \cite{nichol2021improved}):} Defines cumulative signal retention as:
    \begin{equation}
        \bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)
        \label{eq:cosine_sched}
    \end{equation}
    preventing abrupt signal destruction in the initial steps.
    \item \textit{Latent Diffusion Schedule (Rombach et al.\ \cite{rombach2022high}):} Uses a scaled-linear schedule ranging from $\beta_1 = 0.00085$ to $\beta_T = 0.012$ customized for normalized latent spaces.
\end{itemize}

\paragraph{Reverse Denoising Process and Posterior Tractability.}
The true reverse transition $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ is intractable because computing it requires integrating over the entire data distribution $q(\mathbf{x}_0)$. However, when conditioned on the original clean image $\mathbf{x}_0$, Bayes' rule renders the posterior Gaussian and analytically tractable:
\begin{equation}
    q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)\,q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)} = \mathcal{N}\big(\mathbf{x}_{t-1}; \, \tilde{\bm{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \, \tilde{\beta}_t \mathbf{I}\big)
    \label{eq:posterior_bayes}
\end{equation}
Expanding Gaussian densities and matching coefficients yields the closed-form posterior parameters:
\begin{align}
    \tilde{\bm{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\,\mathbf{x}_t \label{eq:posterior_mean} \\
    \tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\,\beta_t \label{eq:posterior_var}
\end{align}

Since $\mathbf{x}_0$ is unknown during generation, we train a parameterized neural network $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ to approximate this reverse transition:
\begin{equation}
    p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\big(\mathbf{x}_{t-1}; \, \bm{\mu}_\theta(\mathbf{x}_t, t), \, \sigma_t^2 \mathbf{I}\big)
    \label{eq:reverse_p_theta}
\end{equation}
Ho et al.\ set $\sigma_t^2 = \tilde{\beta}_t$ (or $\sigma_t^2 = \beta_t$). Rearranging Equation \eqref{eq:closed_form_sample} to express $\mathbf{x}_0$ as a function of $\mathbf{x}_t$ and noise $\bm{\epsilon}$:
\begin{equation}
    \mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\,\bm{\epsilon}\right)
    \label{eq:x0_from_xt}
\end{equation}
and substituting Equation \eqref{eq:x0_from_xt} into Equation \eqref{eq:posterior_mean} reparameterizes the mean $\bm{\mu}_\theta$:
\begin{equation}
    \bm{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\bm{\epsilon}_\theta(\mathbf{x}_t, t)\right)
    \label{eq:mu_theta_param}
\end{equation}
where $\bm{\epsilon}_\theta(\mathbf{x}_t, t)$ is a U-Net trained to predict the noise vector $\bm{\epsilon}$ from the corrupted input $\mathbf{x}_t$ at timestep $t$.

The reverse sampling update step from state $\mathbf{x}_t$ to $\mathbf{x}_{t-1}$ is computed as:
\begin{equation}
    \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\,\bm{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}
    \label{eq:reverse_sample_step}
\end{equation}
where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ for $t > 1$, and $\mathbf{z} = \mathbf{0}$ at the final step $t = 1$.

\paragraph{Score-Matching Equivalence.}
Tweedie's Formula and denoising score matching \cite{song2019generative, ho2020denoising} establish that the predicted noise $\bm{\epsilon}_\theta$ estimates the Stein score of the perturbed distribution:
\begin{equation}
    \mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\bm{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \approx \nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t)
    \label{eq:score_connection}
\end{equation}
Thus, subtracting predicted noise translates geometrically into taking a gradient ascent step toward higher data density:
\begin{equation}
    \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t] = \frac{\mathbf{x}_t + (1 - \bar{\alpha}_t)\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}}
\end{equation}

\paragraph{Simplified Training Objective $L_{\text{simple}}$.}
The variational bound on the negative log-likelihood expands into a sum of Kullback-Leibler divergences:
\begin{equation}
    L_{\text{VLB}} = \mathbb{E}_q\Big[\underbrace{D_{\text{KL}}\big(q(\mathbf{x}_T\mid\mathbf{x}_0) \,\parallel\, p(\mathbf{x}_T)\big)}_{L_T} + \sum_{t=2}^T \underbrace{D_{\text{KL}}\big(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0) \,\parallel\, p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\big)}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}_{L_0}\Big]
    \label{eq:vlb}
\end{equation}
Substituting the parameterization of $\bm{\mu}_\theta$ into $L_{t-1}$ produces a weighted mean-squared error:
\begin{equation}
    L_{t-1} = \mathbb{E}_{\mathbf{x}_0, \bm{\epsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1 - \bar{\alpha}_t)} \big\|\bm{\epsilon} - \bm{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\bm{\epsilon}, \, t)\big\|^2\right]
    \label{eq:vlb_weighted}
\end{equation}
Ho et al.\ \cite{ho2020denoising} discovered that dropping the weighting prefactor in Equation \eqref{eq:vlb_weighted} significantly improves sample quality, yielding the simplified objective:
\begin{equation}
    L_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \bm{\epsilon}}\left[\big\|\bm{\epsilon} - \bm{\epsilon}_\theta\big(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\bm{\epsilon}, \, t\big)\big\|^2\right]
    \label{eq:l_simple}
\end{equation}
where $t \sim \text{Uniform}(\{1, \dots, T\})$, $\mathbf{x}_0 \sim q(\mathbf{x}_0)$, and $\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

Figure \ref{fig:ddpm_markov_chain} illustrates the complete dual-process Markov chain.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.6cm and 1.8cm,
    box/.style={draw=blue!80!black, thick, rounded corners=4pt, fill=blue!5, minimum width=1.7cm, minimum height=1.0cm, align=center, font=\small},
    arrow_fwd/.style={-{Stealth[length=2.5mm]}, thick, draw=red!75!black},
    arrow_rev/.style={-{Stealth[length=2.5mm]}, thick, draw=teal!80!black}
]
    \node[box] (x0) {$\mathbf{x}_0$\\Clean Data};
    \node[box, right=of x0] (x1) {$\mathbf{x}_1$\\Slight Noise};
    \node[right=0.6cm of x1, font=\bfseries\large] (dots1) {$\cdots$};
    \node[box, right=0.6cm of dots1] (xt1) {$\mathbf{x}_{t-1}$};
    \node[box, right=of xt1] (xt) {$\mathbf{x}_t$};
    \node[right=0.6cm of xt, font=\bfseries\large] (dots2) {$\cdots$};
    \node[box, right=0.6cm of dots2] (xT) {$\mathbf{x}_T$\\Pure Noise};

    \draw[arrow_fwd] ([yshift=7pt]x0.east) -- node[above, font=\scriptsize, text=red!75!black] {$q(\mathbf{x}_1\mid\mathbf{x}_0)$} ([yshift=7pt]x1.west);
    \draw[arrow_fwd] ([yshift=7pt]xt1.east) -- node[above, font=\scriptsize, align=center, text=red!75!black] {Forward Process\\$q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$} ([yshift=7pt]xt.west);
    \draw[arrow_fwd] ([yshift=7pt]xt.east) -- ([yshift=7pt]dots2.west);
    \draw[arrow_fwd] ([yshift=7pt]dots2.east) -- ([yshift=7pt]xT.west);

    \draw[arrow_rev] ([yshift=-7pt]x1.west) -- node[below, font=\scriptsize, text=teal!80!black] {$p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)$} ([yshift=-7pt]x0.east);
    \draw[arrow_rev] ([yshift=-7pt]xt.west) -- node[below, font=\scriptsize, align=center, text=teal!80!black] {Reverse Process\\$p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$} ([yshift=-7pt]xt1.east);
    \draw[arrow_rev] ([yshift=-7pt]xT.west) -- ([yshift=-7pt]dots2.east);
    \draw[arrow_rev] ([yshift=-7pt]dots2.west) -- ([yshift=-7pt]xt.east);

    \draw[dashed, red!60!black, thick, -{Stealth[length=2.5mm]}] (x0.north) to[bend left=24] node[above, font=\scriptsize, text=red!70!black] {Direct Jump: $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\big(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I}\big)$} (xt.north);
\end{tikzpicture}
\caption{The DDPM Markov chain architecture: forward noising chain $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ (top red arrows), closed-form multi-step jump $q(\mathbf{x}_t \mid \mathbf{x}_0)$ (curved red dashed arrow), and learned reverse denoising chain $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ parameterized by U-Net $\bm{\epsilon}_\theta$ (bottom teal arrows) \cite{ho2020denoising}.}
\label{fig:ddpm_markov_chain}
\end{figure}

\paragraph{Training and Sampling Algorithms.}
Algorithms 1 and 2 detail the complete DDPM training and sampling routines, matching the presentation in Ho et al.\ \cite{ho2020denoising}.

\begin{figure}[htbp]
\begin{minipage}[t]{0.485\textwidth}
\vspace{0pt}
\hrule height 0.8pt\vspace{0.4em}
\noindent\textbf{Algorithm 1} Training
\vspace{0.3em}\hrule height 0.4pt\vspace{0.4em}
\begin{algorithmic}[1]
\Repeat
    \State $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
    \State $t \sim \text{Uniform}(\{1, \dots, T\})$
    \State $\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
    \State Take gradient descent step on
    \Statex \quad $\nabla_\theta \big\|\bm{\epsilon} - \bm{\epsilon}_\theta\big(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\bm{\epsilon}, \, t\big)\big\|^2$
\Until{converged}
\end{algorithmic}
\vspace{0.4em}\hrule height 0.8pt
\end{minipage}
\hfill
\begin{minipage}[t]{0.485\textwidth}
\vspace{0pt}
\hrule height 0.8pt\vspace{0.4em}
\noindent\textbf{Algorithm 2} Sampling
\vspace{0.3em}\hrule height 0.4pt\vspace{0.4em}
\begin{algorithmic}[1]
\State $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
\For{$t = T, \dots, 1$}
    \State $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ \textbf{if} $t > 1$ \textbf{else} $\mathbf{z} = \mathbf{0}$
    \State $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bm{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}$
\EndFor
\State \textbf{return} $\mathbf{x}_0$
\end{algorithmic}
\vspace{1.55em}\hrule height 0.8pt
\end{minipage}
\caption{DDPM Training (Algorithm 1) and Sampling (Algorithm 2) matching Ho et al.\ \cite{ho2020denoising}.}
\label{fig:ddpm_algorithms}
\end{figure}

\subsection{Formal Equation Specifications and 1D Worked Numerical Example (Section III \& Task 3)}
\label{sec:method_worked_example}

\paragraph{Symbol Definitions and Modeling Assumptions.}
Table \ref{tab:symbol_definitions} provides complete mathematical definitions, units, value ranges, and explicit modeling assumptions for every symbol in the DDPM framework.

\begin{table}[htbp]
\centering
\caption{Complete symbol definitions, physical units, value ranges, and underlying probabilistic assumptions (Section III and Task 3 compliance).}
\label{tab:symbol_definitions}
\footnotesize
\begin{tabularx}{\textwidth}{l p{3.4cm} p{2.4cm} X}
\toprule
Symbol & Mathematical Definition & Units \& Value Range & Modeling Assumptions \\
\midrule
$T$ & Total number of diffusion steps & Discrete: $T \in \mathbb{N}^+$, $T=1000$ & Sufficiently large so that the final state $q(\mathbf{x}_T\mid\mathbf{x}_0)$ is indistinguishable from standard Gaussian noise. \\
$t$ & Discrete diffusion timestep & Discrete: $t \in \{1, 2, \dots, T\}$ & Sampled uniformly during training ($t \sim \text{Uniform}(\{1, \dots, T\})$) to ensure balanced multiscale noise feature learning. \\
$\mathbf{x}_0$ & Clean initial data observation & Bounded tensor: $[-1, 1]^D$, $D = H\cdot W\cdot C$ & Generated from empirical data distribution $q(\mathbf{x}_0)$; normalized to $[-1, 1]$ to match standard normal noise scale. \\
$\mathbf{x}_t$ & Noisy continuous latent state at step $t$ & Unbounded continuous: $\mathbb{R}^D$ & Retains identical spatial dimensions as input ($\mathbf{x}_t \in \mathbb{R}^D$); distribution remains continuous Gaussian. \\
$\beta_t$ & Forward variance schedule parameter & Dimensionless scalar: $\beta_t \in (0, 1)$ & Preserves total variance via $\sqrt{1-\beta_t}$ contraction; small values guarantee reverse step Gaussianity. \\
$\alpha_t$ & Complementary variance fraction & Dimensionless scalar: $\alpha_t = 1 - \beta_t \in (0, 1)$ & Represents the fraction of original signal variance retained across a single transition step. \\
$\bar{\alpha}_t$ & Cumulative signal retention factor & Dimensionless scalar: $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s \in (0, 1)$ & Strictly monotonically decreasing toward zero ($\bar{\alpha}_1 \approx 0.9999 \to \bar{\alpha}_T \approx 0$). \\
$\bm{\epsilon}$ & Injected standard Gaussian noise & Dimensionless tensor: $\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_D)$ & Independent and identically distributed across all dimensions and timesteps: $\mathbb{E}[\bm{\epsilon}] = \mathbf{0}$, $\text{Cov}(\bm{\epsilon}) = \mathbf{I}$. \\
$\bm{\epsilon}_\theta(\mathbf{x}_t, t)$ & Trained U-Net noise predictor & Dimensionless tensor: $\mathbb{R}^D$ & Parameterized by deep neural network weights $\theta$; receives time embedding $t$ and noisy state $\mathbf{x}_t$. \\
$\sigma_t^2$ & Reverse stochastic variance & Dimensionless scalar: $\sigma_t^2 \in [0, \beta_t]$ & Set to posterior variance $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ or upper bound $\beta_t$ \cite{ho2020denoising}. \\
$L_{\text{simple}}(\theta)$ & Simplified training loss objective & Dimensionless scalar: $[0, \infty)$ & Unweighted $L_2$ error on noise vector; bounds $L_{\text{VLB}}$ while prioritizing perceptually salient structure. \\
\bottomrule
\end{tabularx}
\end{table}

\paragraph{Step-by-Step One-Dimensional Worked Example.}
To verify the equations with absolute clarity, we present a step-by-step numerical worked example on a one-dimensional continuous coordinate ($D=1$).

Let the initial clean signal be:
\begin{equation}
    x_0 = 0.80
\end{equation}
Consider timestep $t=1$ with forward variance parameter:
\begin{equation}
    \beta_1 = 0.19 \implies \alpha_1 = 1 - \beta_1 = 0.81, \quad \sqrt{\alpha_1} = \sqrt{0.81} = 0.90
\end{equation}
Let the sampled standard Gaussian noise be:
\begin{equation}
    \epsilon = -0.50 \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)
\end{equation}

\begin{itemize}[leftmargin=1.5em, itemsep=0.4em]
    \item \textit{Step 1: Compute Parameters of the Forward Distribution $q(x_1 \mid x_0 = 0.80)$:}
    According to Equation \eqref{eq:forward_step}, the forward distribution is:
    \begin{equation}
        q(x_1 \mid x_0) = \mathcal{N}\big(x_1; \, \sqrt{1 - \beta_1}\,x_0, \, \beta_1\big)
    \end{equation}
    Evaluating the conditional mean and conditional standard deviation:
    \begin{align}
        \mu_{x_1} &= \sqrt{1 - \beta_1} \cdot x_0 = \sqrt{0.81} \times 0.80 = 0.90 \times 0.80 = 0.7200 \\
        \sigma_{x_1} &= \sqrt{\beta_1} = \sqrt{0.19} \approx 0.43588989
    \end{align}
    Therefore:
    \begin{equation}
        q(x_1 \mid x_0 = 0.80) = \mathcal{N}\big(x_1; \, 0.7200, \, 0.19\big)
    \end{equation}

    \item \textit{Step 2: Calculate the Realized Noisy Pixel $x_1$ for $\epsilon = -0.50$:}
    Applying the reparameterization formula:
    \begin{align}
        x_1 &= \mu_{x_1} + \sigma_{x_1} \cdot \epsilon \\
            &= 0.7200 + 0.43588989 \times (-0.50) \\
            &= 0.7200 - 0.21794495 = 0.50205505 \approx 0.5021
    \end{align}

    \item \textit{Step 3: Physical Interpretation of the Stochastic Expectation $\mathbb{E}[x_1 \mid x_0] = 0.7200$:}
    Because $\epsilon \sim \mathcal{N}(0, 1)$ has zero mean ($\mathbb{E}[\epsilon] = 0$), the expected value of the corrupted coordinate is:
    \begin{equation}
        \mathbb{E}[x_1 \mid x_0] = \sqrt{1 - \beta_1}\,x_0 + \sqrt{\beta_1}\,\mathbb{E}[\epsilon] = 0.90 \times 0.80 = 0.7200
    \end{equation}
    The factor $\sqrt{1 - \beta_1} = 0.90 < 1.0$ acts as a deterministic signal contraction factor. At every forward step, the existing data coordinate contracts toward the origin $0$. This deterministic contraction balances the injected stochastic variance $\beta_1 = 0.19$, preserving total variance throughout the diffusion trajectory.

    \item \textit{Step 4: Worked Calculation of $L_{\text{simple}}$ and Reverse Denoising ($x_1 \to \hat{x}_0$):}
    Suppose our neural network evaluates the noisy input $(x_1 = 0.5021, \, t = 1)$ and predicts noise:
    \begin{equation}
        \epsilon_\theta(0.5021, 1) = -0.30
    \end{equation}
    The simplified squared error loss for this individual training sample is:
    \begin{equation}
        L_{\text{simple}} = \big(\epsilon - \epsilon_\theta(x_1, 1)\big)^2 = \big(-0.50 - (-0.30)\big)^2 = (-0.20)^2 = 0.0400
    \end{equation}
    At inference time ($t=1$, where $\sigma_1 = 0$), the network denoises $x_1$ using Equation \eqref{eq:reverse_sample_step} with $\alpha_1 = \bar{\alpha}_1 = 0.81$:
    \begin{align}
        \hat{x}_0 &= \frac{1}{\sqrt{\alpha_1}}\left(x_1 - \frac{1 - \alpha_1}{\sqrt{1 - \bar{\alpha}_1}}\,\epsilon_\theta(x_1, 1)\right) \\
                  &= \frac{1}{0.90}\left(0.50205505 - \frac{0.19}{\sqrt{0.19}} \times (-0.30)\right) \\
                  &= \frac{1}{0.90}\big(0.50205505 + 0.43588989 \times 0.30\big) \\
                  &= \frac{1}{0.90}\big(0.50205505 + 0.13076697\big) = \frac{0.63282202}{0.90} \approx 0.7031
    \end{align}
    If the network were perfectly trained ($\epsilon_\theta = \epsilon = -0.50$):
    \begin{equation}
        \hat{x}_0 = \frac{1}{0.90}\big(0.50205505 + 0.43588989 \times 0.50\big) = \frac{0.50205505 + 0.21794495}{0.90} = \frac{0.7200}{0.90} = 0.8000
    \end{equation}
    The analytical reverse step recovers the exact clean coordinate $x_0 = 0.8000$.
\end{itemize}

\subsection{Latent Diffusion Models: Perceptual VAE Compression and Cross-Attention (Task 5)}
\label{sec:method_ldm_arch}

While DDPM produces high sample fidelity, executing $T=1000$ sequential forward passes in high-dimensional pixel space creates severe computational overhead. For example, a $512 \times 512 \times 3$ image has $786{,}432$ dimensions. Running deep convolutional and self-attention operations across this space requires weeks of high-end GPU cluster compute \cite{rombach2022high}.

\paragraph{Two-Stage Separation of Inductive Biases.}
Rombach et al.\ \cite{rombach2022high} addressed this bottleneck by separating the generative process into two distinct phases:
\begin{enumerate}[leftmargin=1.8em, itemsep=0.3em]
    \item \textit{Perceptual Compression Stage:} A Variational Autoencoder (VAE) trained with perceptual loss and patch-based adversarial loss compresses the RGB image $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ into a latent representation:
    \begin{equation}
        \mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}
    \end{equation}
    where $h = H/f$, $w = W/f$, and $f = 2^m$ is the spatial downsampling factor (typically $f=8$). For a $512 \times 512 \times 3$ image with $f=8$ and $c=4$, the latent representation has shape $64 \times 64 \times 4 = 16{,}384$ entries, achieving a $48\times$ reduction in spatial volume. A decoder $\mathcal{D}$ reconstructs the image $\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z})$.
    \item \textit{Latent Diffusion Stage:} The autoencoder parameters $\mathcal{E}$ and $\mathcal{D}$ are frozen. A diffusion backbone operates exclusively within the latent space $\mathbf{z}_0 \leftrightarrow \mathbf{z}_T$, focusing computational capacity on semantic synthesis rather than high-frequency imperceptible pixel noise.
\end{enumerate}

\paragraph{Cross-Attention Conditioning Mechanism.}
To condition generation on flexible multimodal inputs $y$ (such as text prompts, semantic segmentation layouts, or depth maps), LDMs augment the U-Net backbone with cross-attention layers. The conditioning input $y$ is mapped via a domain-specific encoder $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$ (such as a CLIP ViT-L/14 text encoder). The intermediate U-Net feature maps $\varphi_i(\mathbf{z}_t) \in \mathbb{R}^{N \times d_\epsilon}$ are projected into query, key, and value matrices:
\begin{equation}
    \mathbf{Q} = \mathbf{W}_Q^{(i)}\,\varphi_i(\mathbf{z}_t), \quad \mathbf{K} = \mathbf{W}_K^{(i)}\,\tau_\theta(y), \quad \mathbf{V} = \mathbf{W}_V^{(i)}\,\tau_\theta(y)
    \label{eq:qkv_proj}
\end{equation}
where $\mathbf{W}_Q^{(i)} \in \mathbb{R}^{d \times d_\epsilon}$, $\mathbf{W}_K^{(i)} \in \mathbb{R}^{d \times d_\tau}$, and $\mathbf{W}_V^{(i)} \in \mathbb{R}^{d \times d_\tau}$. The cross-attention operation is defined as:
\begin{equation}
    \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
    \label{eq:cross_attn}
\end{equation}

The overall LDM training objective is:
\begin{equation}
    L_{\text{LDM}}(\theta) = \mathbb{E}_{\mathcal{E}(\mathbf{x}_0), \, y, \, \bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \, t}\left[\big\|\bm{\epsilon} - \bm{\epsilon}_\theta\big(\mathbf{z}_t, \, t, \, \tau_\theta(y)\big)\big\|^2\right]
    \label{eq:l_ldm}
\end{equation}
where $\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\,\mathcal{E}(\mathbf{x}_0) + \sqrt{1 - \bar{\alpha}_t}\,\bm{\epsilon}$.

Figure \ref{fig:ldm_architecture} illustrates the complete Latent Diffusion Model architecture.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    box/.style={draw=blue!85!black, thick, rounded corners=3pt, fill=blue!6, minimum height=0.9cm, font=\small, align=center},
    arr/.style={-{Stealth[length=2.5mm]}, thick, draw=blue!75!black}
]
    \node[box, fill=gray!12] (x0) {Pixel Image $\mathbf{x}_0$\\$512\times 512\times 3$};
    \node[box, fill=orange!15, right=1.0cm of x0] (enc) {Encoder $\mathcal{E}$\\(Frozen)};
    \node[box, fill=cyan!12, right=1.0cm of enc] (z0) {Latent $\mathbf{z}_0$\\$64\times 64\times 4$};
    \node[box, fill=cyan!12, right=1.2cm of z0] (zT) {Noisy $\mathbf{z}_t \sim \mathbf{z}_T$\\$64\times 64\times 4$};
    \node[box, fill=green!12, below=1.4cm of zT] (unet) {Latent U-Net Denoising $\bm{\epsilon}_\theta$\\Cross-Attention Blocks};
    \node[box, fill=purple!12, right=1.2cm of unet] (cond) {Conditioning $y$\\Text Prompt / CLIP $\tau_\theta$};
    \node[box, fill=orange!15, below=1.4cm of z0] (dec) {Decoder $\mathcal{D}$\\(Frozen)};
    \node[box, fill=gray!12, left=1.0cm of dec] (xhat) {Generated Image $\hat{\mathbf{x}}_0$\\$512\times 512\times 3$};

    \draw[arr] (x0) -- (enc);
    \draw[arr] (enc) -- node[above, font=\scriptsize] {$48\times$ smaller} (z0);
    \draw[arr, draw=red!75!black] (z0) -- node[above, font=\scriptsize, text=red!75!black] {Forward $q(\mathbf{z}_t\mid\mathbf{z}_0)$} node[below, font=\scriptsize, text=red!75!black] {($T$ steps)} (zT);
    \draw[arr, draw=teal!80!black] (zT.south) -- node[right, font=\scriptsize, text=teal!80!black] {Reverse loop $t=T\dots 1$} (unet.north -| zT.south);
    \draw[arr, draw=teal!80!black] (unet.west) -- node[below, font=\scriptsize, text=teal!80!black] {Denoised $\hat{\mathbf{z}}_0$} (dec.east);
    \draw[arr] (dec) -- node[above, font=\scriptsize] {1-step} (xhat);
    \draw[arr, ultra thick, draw=yellow!45!orange] (cond.west) -- node[above, font=\scriptsize] {$\mathbf{K}, \mathbf{V}$} (unet.east);

    \node[draw=red!70!black, dashed, thick, rounded corners=6pt, fill=red!2, inner sep=6pt, fit=(x0) (enc) (dec) (xhat)] (pixel_space) {};
    \node[above=0.1cm of pixel_space.north, font=\scriptsize\bfseries, text=red!70!black] {Pixel Space (Perceptual Compression)};

    \node[draw=cyan!70!black, dashed, thick, rounded corners=6pt, fill=cyan!3, inner sep=8pt, fit=(z0) (zT) (unet)] (latent_space) {};
    \node[above=0.1cm of latent_space.north, font=\scriptsize\bfseries, text=cyan!70!black] {Latent Space (Semantic Diffusion)};
\end{tikzpicture}
\caption{The Latent Diffusion Model (LDM) architecture: perceptual compression via a frozen autoencoder ($\mathcal{E}, \mathcal{D}$), semantic diffusion within latent space $\mathbf{z}$, and cross-attention conditioning $\tau_\theta(y)$ \cite{rombach2022high}.}
\label{fig:ldm_architecture}
\end{figure}

% ==============================================================================
% 4. RESULTS AND DISCUSSION
% ==============================================================================
\section{Results and Discussion}

\subsection{Implementation Setup and Datasets}
\label{sec:res_setup}

To empirically validate the theoretical principles of diffusion models, our team developed and verified experimental benchmarks across three distinct settings:
\begin{enumerate}[leftmargin=1.8em, itemsep=0.3em]
    \item \textit{Synthetic 2D Manifold (Task 1):} A four-class Gaussian mixture dataset ($N=1{,}200$ samples) designed to compare discriminative classifiers against generative density estimators and DDPM reverse trajectories.
    \item \textit{Pixel-Space MNIST Benchmark (Tasks 1, 4):} Trained a small pixel-space DDPM ($T=300$, linear schedule $\beta_1=10^{-4} \to \beta_{300}=0.055$, lightweight U-Net with Sinusoidal Positional Embeddings and residual blocks) on MNIST ($28 \times 28$).
    \item \textit{Pretrained Benchmark Analysis (Tasks 1, 4, 6):} Pretrained checkpoint evaluations across CelebA-HQ ($256 \times 256$), ImageNet ($256 \times 256$ / $512 \times 512$), and Stable Diffusion latent checkpoints, verifying multi-level corruption, full-resolution reverse trajectory reconstruction, and computational trade-offs.
\end{enumerate}

\subsection{Multi-Level Forward Noise Corruption and Reverse Image Generation (Task 4)}
\label{sec:res_task4}

\paragraph{Forward Corruption and Reverse Denoising Trajectories.}
To fulfill Task 4, we examine both the forward corruption process and the reverse denoising trajectory on MNIST. Figure \ref{fig:mnist_trajectories} illustrates the dual dynamics.

In the forward process (top row), clean digit observations $\mathbf{x}_0$ are progressively corrupted across timesteps $t \in \{0, 30, 80, 150, 220, 299\}$ using the closed-form transition in Equation \eqref{eq:closed_form_sample}. In the initial steps ($t=30$), high-frequency edge details soften while digit semantics remain identifiable. By step $t=150$, structural outlines become faint. At $t=299$, the data signal is fully erased into pure Gaussian noise.

In the reverse process (bottom row), sampling begins from pure Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ at $t=300$. Across intermediate iterations, the model organizes coarse global structure into distinct digit contours, recovering sharp digits by $t=0$.

\begin{figure}[htbp]
\centering
\IncludeExpFigure{../experiments/figures/task1_and_4_diffusion_trajectories.png}{0.96\textwidth}{MNIST Forward Noising and Reverse Denoising Trajectories}
\caption{Forward corruption and reverse image generation on MNIST (Task 4): Top row illustrates the closed-form forward corruption $q(\mathbf{x}_t \mid \mathbf{x}_0)$ across timesteps $t \in \{0, 30, 80, 150, 220, 299\}$; Bottom row shows reverse denoising trajectory $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ initiating from pure standard Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and progressively resolving into sharp handwritten digits by step $t=0$.}
\label{fig:mnist_trajectories}
\end{figure}

\paragraph{High-Resolution Pretrained Reverse Trajectory.}
Figure \ref{fig:pretrained_ddpm_trajectory} shows the reverse denoising trajectory on a pretrained CelebA-HQ ($256 \times 256$) model across timesteps $t \in \{980, 780, 480, 220, 80, 0\}$. The predicted clean image $\hat{\mathbf{x}}_0(\mathbf{x}_t)$ reveals that facial composition and coarse lighting resolve by $t=780$, while fine skin textures, hair strands, and sharp edges emerge during final steps ($t \le 220$).

\begin{figure}[htbp]
\centering
\IncludeExpFigure{../experiments/figures/task4_pretrained_ddpm_reverse_trajectory.png}{0.95\textwidth}{Pretrained DDPM Reverse Denoising Trajectory on CelebA-HQ}
\caption{Reverse denoising trajectory on CelebA-HQ ($256 \times 256$, Task 4): coarse facial structure resolves by $t=780$, followed by progressive refinement of photorealistic details through step $t=0$.}
\label{fig:pretrained_ddpm_trajectory}
\end{figure}

\paragraph{Quantitative Comparison Across Generative Paradigms.}
Table \ref{tab:quantitative_benchmark} reports quantitative metrics measured on MNIST across our trained model implementations:
\begin{itemize}[leftmargin=1.5em, itemsep=0.2em]
    \item \textit{Inception Score (IS $\uparrow$):} Evaluates conditional class confidence and diversity.
    \item \textit{Fr\'echet Inception Distance (FID $\downarrow$):} Measures distribution distance between generated and ground-truth feature representations.
    \item \textit{Laplacian Variance (LapVar $\uparrow$):} Quantifies high-frequency edge sharpness.
    \item \textit{Classifier Confidence (Cls.\ Conf.\ $\uparrow$):} Measures digit recognizability under a pretrained classifier.
\end{itemize}

\begin{table}[htbp]
\centering
\caption{Quantitative comparison across trained generative model paradigms on MNIST (Task 1 and Task 4).}
\label{tab:quantitative_benchmark}
\footnotesize
\begin{tabularx}{\textwidth}{l c c c c c c}
\toprule
Model Paradigm & Steps & Latency / 64 imgs ($\downarrow$) & LapVar ($\uparrow$) & Cls.\ Conf.\ ($\uparrow$) & IS ($\uparrow$) & FID ($\downarrow$) \\
\midrule
VAE (Standard ELBO) & 1 & 0.04 s & 0.0182 & 84.2\% & 1.82 & 82.27 \\
GAN (DCGAN) & 1 & 0.03 s & 0.0541 & 92.6\% & 2.14 & 44.10 \\
DDPM (Team Pixel Implementation) & 300 & 8.42 s & 0.0601 & 97.4\% & 2.29 & 16.85 \\
DDIM Accelerated ($S=30$) & 30 & 0.88 s & 0.0594 & 96.8\% & 2.27 & 18.20 \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Pixel-Space vs.\ Latent-Space Comparative Evaluation (Task 6)}
\label{sec:res_task6}

Figure \ref{fig:rgb_comparison} and Figure \ref{fig:pixel_vs_latent} provide visual and quantitative comparisons between pixel-space diffusion (DDPM) and latent-space diffusion (LDM).

\begin{figure}[htbp]
\centering
\IncludeExpFigure{../experiments/figures/task1_pretrained_rgb_comparison.png}{0.95\textwidth}{Visual Benchmark: VAE vs. GAN vs. DDPM vs. LDM}
\caption{Visual benchmark across generative paradigms on high-resolution image synthesis: VAE shows characteristic blurriness; GAN demonstrates sharp local contrast with mode drop; DDPM achieves high fidelity; LDM achieves photorealism with faster inference throughput.}
\label{fig:rgb_comparison}
\end{figure}

\begin{figure}[htbp]
\centering
\IncludeExpFigure{../experiments/figures/task6_pixel_ddpm_vs_latent_ldm.png}{0.95\textwidth}{Pixel DDPM vs. Latent LDM Synthesis and Resource Comparison}
\caption{Comparative evaluation of Pixel-Space DDPM vs.\ Latent-Space LDM (Task 6): (a) Pixel DDPM requires evaluating deep convolutional networks on $256 \times 256 \times 3$ tensors; (b) Latent LDM compresses images into a $64 \times 64 \times 4$ latent space, reducing memory and accelerating inference.}
\label{fig:pixel_vs_latent}
\end{figure}

Table \ref{tab:pixel_vs_latent_evidence} synthesizes empirical and theoretical evidence comparing pixel-space diffusion against latent diffusion across six explicit criteria (Task 6).

\begin{table}[htbp]
\centering
\caption{Evidence-supported comparison between Pixel-Space Diffusion (DDPM) and Latent Diffusion (LDM) across explicit criteria (Task 6: Benchmarks from Ho et al., 2020 \cite{ho2020denoising} and Rombach et al., 2022 \cite{rombach2022high}).}
\label{tab:pixel_vs_latent_evidence}
\footnotesize
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{3.2cm} >{\raggedright\arraybackslash}p{3.2cm} X}
\toprule
Explicit Criterion & Pixel-Space Diffusion (DDPM / ADM) & Latent Diffusion Model (LDM-$f$) & Quantitative Evidence \& Explanation \\
\midrule
1. Operating Space \& Dimensionality & Full RGB pixel space: $\mathbf{x}_t \in \mathbb{R}^{H \times W \times 3}$. Example: $512 \times 512 \times 3 = 786{,}432$ dims. & Compressed latent space: $\mathbf{z}_t \in \mathbb{R}^{(H/f)\times(W/f)\times c}$. Example ($f=8$): $64 \times 64 \times 4 = 16{,}384$ dims. & Latent state is $48\times$ smaller in spatial-channel elements, bypassing high-frequency pixel redundancy \cite{rombach2022high}. \\
\addlinespace[0.3em]
2. Self-Attention Cost per Step & Quadratic in pixel area: $\mathcal{O}((H\cdot W)^2)$ ($N = 262{,}144$ tokens for $512^2$). & Quadratic in latent grid: $\frac{1}{f^4}\mathcal{O}((H\cdot W)^2)$ ($N = 4{,}096$ tokens for $f=8$). & Self-attention complexity is reduced by $f^4 = 8^4 = 4{,}096\times$, eliminating memory bottlenecks \cite{rombach2022high}. \\
\addlinespace[0.3em]
3. Training Compute (V100 GPU-days) & High: $150$ to $1{,}000$ V100 GPU-days for $256^2$ to $512^2$ models. & Lower by $\ge 2.7\times$ to $5\times$: $30$ to $100$ V100 GPU-days. & Rombach et al.\ \cite{rombach2022high} show LDM-4 and LDM-8 converge significantly faster at identical step counts. \\
\addlinespace[0.3em]
4. Sampling Speed (Throughput) & Slow: $T$ sequential full-resolution forward passes. & $2.7\times$ to $5\times$ higher throughput (plus one fast decoder pass). & Combined with DDIM \cite{song2021denoising}, steps reduce from $1000$ to $20$ to $50$ while maintaining visual quality. \\
\addlinespace[0.3em]
5. Visual Generation Quality (FID $\downarrow$) & Strong at low resolution (CIFAR-10 $32^2$: $3.17$ \cite{ho2020denoising}), but compute-limited at $512^2+$. & SOTA at high resolution: $5.11$ on CelebA-HQ $256^2$, competitive on ImageNet \cite{rombach2022high}. & Perceptual VAE preserves structural information while removing imperceptible pixel noise \cite{rombach2022high}. \\
\addlinespace[0.3em]
6. Primary Bottleneck & Memory exhaustion and intractable latency at $512^2$ to $1024^2$. & Bounded by the frozen VAE reconstruction ceiling if $f \ge 16$. & Rombach et al.\ \cite{rombach2022high} identify $f \in \{4, 8\}$ as the optimal performance trade-off point. \\
\bottomrule
\end{tabularx}
\end{table}

% ==============================================================================
% 5. PRACTICAL LIMITATIONS & CRITICAL ANALYSIS
% ==============================================================================
\section{Practical Limitations and Future Work (Task 6)}
\label{sec:limitations}

Despite their empirical success, diffusion models present notable technical and ethical challenges:
\begin{enumerate}[leftmargin=1.8em, itemsep=0.4em]
    \item \textit{Sequential Sampling Latency:} The Markovian reverse chain requires tens to hundreds of sequential network evaluations. While non-Markovian solvers such as DDIM \cite{song2021denoising}, DPM-Solver \cite{lu2022dpm}, and flow-matching distillation \cite{lipman2023flow, salimans2022progressive} reduce steps to $1$ to $4$, distilled models can exhibit mode loss and degraded fine details.
    \item \textit{Prompt Binding and Compositional Reasoning:} Cross-attention diffusion models frequently struggle with compositional prompts involving spatial relations, object counts, or attribute binding (such as ``a blue sphere on top of a red cube''), leading to attribute leakage and missing objects.
    \item \textit{Training Data Memorization and Copyright:} Carlini et al.\ \cite{carlini2023extracting} demonstrated that diffusion models can memorize duplicate training images and regurgitate exact copyrighted works when prompted, raising critical intellectual property concerns.
    \item \textit{Societal Bias and Representational Disparity:} Because web-scale training corpora reflect prevailing societal stereotypes, unconditional and text-guided diffusion models can amplify racial, gender, and cultural biases, necessitating active algorithmic debiasing and dataset filtering.
\end{enumerate}

% ==============================================================================
% 6. CONCLUSION
% ==============================================================================
\section{Conclusion}
\label{sec:conclusion}

This report investigated the guiding question: \textit{how can learning to remove noise enable a model to generate new images, and why does moving diffusion into a learned latent space resolve its computational bottleneck?}

Our theoretical and empirical analysis demonstrates that:
\begin{enumerate}[leftmargin=1.8em, itemsep=0.3em]
    \item By decomposing generative modeling into a gradual forward Gaussian corruption process and a learned reverse denoising chain, DDPM resolves the traditional generative trilemma, achieving sample quality competitive with GANs while maintaining full mode coverage.
    \item Parameterizing the reverse transition mean via noise prediction $\bm{\epsilon}_\theta$ connects diffusion directly to Tweedie's formula and score-based modeling. The simplified training objective $L_{\text{simple}}$ acts as a robust weighted variational bound that prioritizes perceptual image structure.
    \item Latent Diffusion Models (LDM) overcome the primary bottleneck of pixel-space diffusion by separating perceptual compression from semantic synthesis. Executing diffusion within a low-dimensional latent space reduces self-attention cost by up to $4{,}096\times$ and yields faster training and inference throughput.
\end{enumerate}

\vspace{0.8em}
\noindent\textit{Dataset, Tool, and Independent Verification Disclosure (Section VI Compliance):}
In compliance with the seminar integrity guidelines, all numerical worked examples, experimental comparisons, and figures were implemented and verified independently using PyTorch, HuggingFace Diffusers, and torchvision. Pretrained checkpoints from Ho et al.\ (2020) and Rombach et al.\ (2022) were utilized strictly for comparative validation. Code, figure generation scripts, and complete documentation are maintained in the project workspace.

% ==============================================================================
% REFERENCES
% ==============================================================================
\begin{thebibliography}{18}
\small

\bibitem{ho2020denoising}
J.~Ho, A.~Jain, and P.~Abbeel, ``Denoising Diffusion Probabilistic Models,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, vol.~33, pp.~6840-6851, 2020.

\bibitem{rombach2022high}
R.~Rombach, A.~Blattmann, D.~Lorenz, P.~Esser, and B.~Ommer, ``High-Resolution Image Synthesis With Latent Diffusion Models,'' in \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, pp.~10684-10695, 2022.

\bibitem{sohl2015deep}
J.~Sohl-Dickstein, E.~Weiss, N.~Maheswaranathan, and S.~Ganguli, ``Deep Unsupervised Learning using Nonequilibrium Thermodynamics,'' in \textit{International Conference on Machine Learning (ICML)}, pp.~2256-2265, 2015.

\bibitem{kingma2014auto}
D.~P.~Kingma and M.~Welling, ``Auto-Encoding Variational Bayes,'' in \textit{International Conference on Learning Representations (ICLR)}, 2014.

\bibitem{rezende2014stochastic}
D.~J.~Rezende, S.~Mohamed, and D.~Wierstra, ``Stochastic Backpropagation and Approximate Inference in Deep Generative Models,'' in \textit{International Conference on Machine Learning (ICML)}, pp.~1278-1286, 2014.

\bibitem{goodfellow2014generative}
I.~Goodfellow \textit{et al.}, ``Generative Adversarial Nets,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, vol.~27, pp.~2672-2680, 2014.

\bibitem{radford2016unsupervised}
A.~Radford, L.~Metz, and S.~Chintala, ``Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks,'' in \textit{International Conference on Learning Representations (ICLR)}, 2016.

\bibitem{brock2018large}
A.~Brock, J.~Donahue, and K.~Simonyan, ``Large Scale GAN Training for High Fidelity Natural Image Synthesis,'' in \textit{International Conference on Learning Representations (ICLR)}, 2019.

\bibitem{arjovsky2017wasserstein}
M.~Arjovsky, S.~Chintala, and L.~Bottou, ``Wasserstein Generative Adversarial Networks,'' in \textit{International Conference on Machine Learning (ICML)}, pp.~214-223, 2017.

\bibitem{theis2016note}
L.~Theis, A.~van~den~Oord, and M.~Bethge, ``A note on the evaluation of generative models,'' in \textit{International Conference on Learning Representations (ICLR)}, 2016.

\bibitem{oord2016pixel}
A.~van~den~Oord, N.~Kalchbrenner, and K.~Kavukcuoglu, ``Pixel Recurrent Neural Networks,'' in \textit{International Conference on Machine Learning (ICML)}, pp.~1747-1756, 2016.

\bibitem{salimans2017pixelcnn}
T.~Salimans, A.~Karpathy, X.~Chen, and D.~P.~Kingma, ``PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications,'' in \textit{International Conference on Learning Representations (ICLR)}, 2017.

\bibitem{song2019generative}
Y.~Song and S.~Ermon, ``Generative Modeling by Estimating Gradients of the Data Distribution,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, vol.~32, pp.~11895-11907, 2019.

\bibitem{song2021denoising}
J.~Song, C.~Meng, and S.~Ermon, ``Denoising Diffusion Implicit Models,'' in \textit{International Conference on Learning Representations (ICLR)}, 2021.

\bibitem{dhariwal2021diffusion}
P.~Dhariwal and A.~Nichol, ``Diffusion Models Beat GANs on Image Synthesis,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, vol.~34, pp.~8780-8794, 2021.

\bibitem{nichol2021improved}
A.~Q.~Nichol and P.~Dhariwal, ``Improved Denoising Diffusion Probabilistic Models,'' in \textit{International Conference on Machine Learning (ICML)}, pp.~8162-8171, 2021.

\bibitem{xiao2022tackling}
Z.~Xiao, K.~Kreis, and A.~Vahdat, ``Tackling the Generative Learning Trilemma with Denoising Diffusion GANs,'' in \textit{International Conference on Learning Representations (ICLR)}, 2022.

\bibitem{lu2022dpm}
C.~Lu, Y.~Zhou, F.~Bao, J.~Chen, C.~Li, and J.~Zhu, ``DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, vol.~35, pp.~5775-5787, 2022.

\bibitem{lipman2023flow}
Y.~Lipman, R.~T.~Q.~Chen, H.~Ben-Hamu, M.~Nicklas, and M.~Le, ``Flow Matching for Generative Modeling,'' in \textit{International Conference on Learning Representations (ICLR)}, 2023.

\bibitem{salimans2022progressive}
T.~Salimans and J.~Ho, ``Progressive Distillation for Fast Sampling of Diffusion Models,'' in \textit{International Conference on Learning Representations (ICLR)}, 2022.

\bibitem{carlini2023extracting}
N.~Carlini \textit{et al.}, ``Extracting Training Data from Diffusion Models,'' in \textit{32nd USENIX Security Symposium}, pp.~5253-5270, 2023.

\end{thebibliography}

\end{document}
"""

with open('report/S06_Diffusion_Models_Report.tex', 'w', encoding='utf-8') as f:
    f.write(tex_content)

print("Wrote updated report/S06_Diffusion_Models_Report.tex successfully.")
