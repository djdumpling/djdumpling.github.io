---
title: "activation engineering for privacy protection in LLMs"
date: 2025-12-05
image: /public/sea_privacy/design.png
---

LLMs trained on web-scale corpora inadvertently memorize and leak personally identifiable information (PII) present in their training data. We investigate inference-time interventions to suppress this privacy leakage without model retraining. We evaluate three editing strategies: (1) activation patching with computed steering vectors (APNEAP), (2) random Gaussian noise steering, and (3) Spectral Editing of Activations (SEA). Using the Enron email corpus with `GPT-Neo-1.3B` and finetuned `Qwen3-8B-enron`, we measure targeted PII suppression via mean reciprocal rank (MRR) and exposure metrics, and utility via perplexity.

This work was jointly done with two classmates, [Coby Kassner](https://cobylk.io/) and [Yejun Yun](https://www.linkedin.com/in/yejun-yun/), as a part of Yale's Trustworthy Deep Learning class taught by [Rex Ying](https://www.cs.yale.edu/homes/ying-rex/). Our implementation is available [here](https://github.com/cobylk/CPSC-4710-privacy).

## tl;dr

- **APNEAP baseline**: Achieves **43.2%** MRR suppression and **30.6%** exposure reduction with **5.2%** perplexity degradation on `GPT-Neo-1.3B` with 759 privacy neurons
- **Random noise steering**: Performs comparably, achieving **42.9%** MRR suppression and **43.6%** exposure reduction (superior to APNEAP) with **7.9%** perplexity degradation, suggesting that identifying privacy neurons is more critical than the specific steering direction
- **SEA (Spectral Editing)**: Achieves the strongest privacy protection (**65.6%** MRR suppression, **67.4%** exposure reduction) but at substantial utility cost (**37.5%** perplexity increase) and shows limited effectiveness on finetuned models (**0.6%** exposure, **6.4%** MRR reduction on `Qwen3-8B-enron`)
- **Key insight**: Simpler undirected interventions (random noise) can be as effective as complex steering-based approaches, with privacy neuron identification being the critical component

## introduction

Large language models (LLMs) such as GPT-3, LLaMA, and others have achieved remarkable capabilities in natural language understanding and generation by training on massive web-scale datasets. However, this training paradigm creates serious privacy risks: models memorize verbatim snippets from their training data, including personally identifiable information (PII) such as email addresses, phone numbers, and social security numbers. Adversarial prompts can extract this memorized data with alarming success rates.

Traditional mitigation strategies include data preprocessing to remove PII before training, differential privacy mechanisms that add noise during training, and machine unlearning approaches that fine-tune models to forget specific data. Each approach has limitations: preprocessing cannot catch all PII patterns, differential privacy significantly degrades model quality, and unlearning requires expensive retraining. These drawbacks motivate the development of inference-time interventions that can suppress privacy leakage in already-deployed models without weight updates.

Recent work has shown that privacy leakage is mechanistically localized: specific neurons in the feedforward layers of transformers are disproportionately responsible for memorizing and recalling private information. This observation enables targeted editing: by identifying these "privacy neurons" through gradient attribution and selectively intervening on their activations at inference time, we can reduce leakage while preserving general model capabilities.

In this work, we implement and evaluate three inference-time privacy editing strategies. First, we reproduce the Augmented Privacy Neuron Editing via Activation Patching (APNEAP) framework, which computes steering vectors by comparing model activations on sensitive versus desensitized prompts and additively patches these vectors onto privacy neurons during generation. Second, we evaluate random Gaussian noise steering as a simpler alternative that requires no steering vector computation—only the privacy neuron coordinates—inspired by differential privacy principles. Third, we adapt Spectral Editing of Activations (SEA), originally designed for truthfulness and bias alignment, to the privacy domain by projecting activations into subspaces that maximize covariance with desensitized contexts while minimizing covariance with privacy-leaking contexts. We conduct experiments on `GPT-Neo-1.3B` and `Qwen3-8B-enron` using the Enron email corpus, measuring privacy suppression via MRR and exposure metrics, and utility via perplexity.

### what is different

Our implementation extends prior work in several ways. First, we adapt SEA, originally designed for truthfulness and bias, to the privacy domain by formulating PII suppression as a subspace projection problem. Second, we provide a modular, reusable codebase that separates attribution, selection, and editing into independent stages, enabling future researchers to swap out components easily. Third, we introduce the option to use layer-suffix editing and to scope spectral projections to privacy neurons only, hybridizing the targeted nature of APNEAP with the principled subspace geometry of SEA.

## problem definition

We consider the following setting: a pretrained language model $M_\theta$ has been trained on a large corpus $\mathcal{D}$ that contains personally identifiable information. Given a prompt $X$ that provides context mentioning or related to a private entity, the model may generate a continuation $Y$ that leaks the associated private information (e.g., an email address, phone number, or social security number). Formally, we define a privacy leak as a tuple $(X, Y)$ where:

$$P(Y \mid X, M_\theta) > \tau$$

for some threshold $\tau$, indicating that the model assigns high probability to generating the private sequence $Y$ given the prompt $X$.

Our goal is to develop an inference-time intervention $\mathcal{E}$ that modifies the model's internal activations such that the edited model $M_{\mathcal{E}}$ satisfies:

$$\min_{\mathcal{E}} \left\{ \mathbb{E}_{(X,Y) \in \mathcal{T}} \left[ P(Y \mid X, M_{\mathcal{E}}) \right], \quad \text{PPL}(M_{\mathcal{E}}, \mathcal{V}) - \text{PPL}(M_\theta, \mathcal{V}) \right\}$$

where $\mathcal{T}$ is a set of known privacy leaks (the targeted PII we want to suppress), $\mathcal{V}$ is a validation corpus of non-sensitive text, and PPL denotes perplexity (our measure of general model utility). This formulation seeks to reduce the probability of generating targeted private information while minimizing degradation in language modeling quality.

**Base Model:** We use `GPT-Neo-1.3B`, an open-source autoregressive transformer with 24 layers and 3072-dimensional hidden states, trained on the Pile dataset. GPT-Neo was chosen for its accessibility, moderate size that allows rapid iteration, and architectural similarity to GPT-2/GPT-3, making our findings transferable to other models in this family. We also explore `Qwen3-8B-enron`, finetuned off `Qwen3-8B` for few epochs on Enron with LoRA. This model was chosen for its larger size and its domain-specific finetuning on Enron data, which provides a more realistic evaluation scenario where the model has prior exposure to the privacy-sensitive domain.

**Trustworthy Aspect:** Our project focuses on **privacy**, specifically mitigating memorization and leakage of personally identifiable information. Privacy is a critical trustworthiness concern for deployed LLMs: models trained on web data can regurgitate sensitive information about individuals who never consented to their data being used in this way. By developing inference-time interventions that reduce privacy leakage without requiring model retraining, we contribute to making LLMs safer and more aligned with privacy regulations such as GDPR and CCPA.

## methods

Our approach follows a three-stage pipeline adapted from APNEAP: (1) attribution to identify privacy neurons, (2) selection to aggregate attributions across examples, and (3) editing to intervene on activations at inference time. We implement three editing strategies: activation patching (the APNEAP baseline), steering with random noise, and spectral editing (our proposed extensions).

![Methodology flowchart](/public/sea_privacy/design.png)

### privacy neuron attribution

We use integrated gradients to attribute each neuron's contribution to privacy leakage. For a given leak tuple $(X, Y)$, we tokenize $X$ and identify the position $t$ immediately before the first token of the secret $Y$. We then capture the intermediate activation $\mathbf{z}_{\ell}$ at the input to the MLP in layer $\ell$ at position $t$. The integrated gradient for neuron $j$ in layer $\ell$ is approximated as:

$$\widehat{\text{IG}}_{\ell,j} = \frac{\mathbf{z}_{\ell,j}}{m} \sum_{k=1}^{m} \frac{\partial \log P(y_1 \mid X, \frac{k}{m}\mathbf{z}_{\ell})}{\partial \mathbf{z}_{\ell,j}}$$

where $m=10$ interpolation steps are used, $y_1$ is the first token of the secret $Y$, and the gradient is computed via backpropagation. We normalize each gradient by the target token probability to account for varying confidence levels. This process is repeated for every layer $\ell \in \{0, 1, \ldots, 23\}$ and every neuron dimension $j \in \{0, 1, \ldots, 3071\}$, yielding an attribution vector per layer for each leak example.

### privacy neuron selection

Raw attributions are noisy and vary across examples. Following APNEAP, we use a two-threshold voting procedure. For each example $i$ and layer $\ell$:

1. Compute $\text{max}_{\ell,i} = \max_j |\widehat{\text{IG}}_{\ell,j,i}|$
2. Keep neuron $(\ell, j)$ if $|\widehat{\text{IG}}_{\ell,j,i}| > 0.1 \cdot \text{max}_{\ell,i}$ (text-level threshold)

We then count how many examples activate each neuron and retain neurons that appear in more than 50% of examples (batch-level threshold). The result is a set $\mathcal{N} = \{(\ell, j)\}$ of privacy neuron coordinates that are consistently implicated in leakage across the dataset.

### activation patching

For each leak example $(X, Y)$, we construct a harmless variant $X'$ by replacing the secret $Y$ with a placeholder token (e.g., "placeholder"). We perform forward passes on both $X$ and $X'$, capturing the MLP activations $\mathbf{h}_{\ell}$ and $\mathbf{h}'_{\ell}$ at every layer. We compute the steering vector as the mean difference:

$$\mathbf{v}_{\ell} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} (\mathbf{h}_{\ell,i}' - \mathbf{h}_{\ell,i})$$

At inference time, we register PyTorch forward hooks that modify the MLP activations:

$$\mathbf{h}_{\ell} \leftarrow \mathbf{h}_{\ell} + \alpha \cdot \mathbf{v}_{\ell} \odot \mathbf{m}_{\ell}$$

where $\alpha=10$ is a scaling hyperparameter and $\mathbf{m}_{\ell}$ is a binary mask that is 1 for privacy neurons in $\mathcal{N}$ and 0 otherwise. This additive intervention steers the model's behavior away from privacy-leaking activations toward harmless placeholder activations. This approach forms our APNEAP baseline.

As a simpler alternative, we also evaluate steering with random Gaussian noise by replacing the computed steering vector $\mathbf{v}_{\ell}$ with freshly sampled noise $\boldsymbol{\epsilon}_{\ell} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ on each forward pass. This undirected perturbation requires no paired examples or steering vector computation—only the privacy neuron coordinates—and serves as a baseline inspired by differential privacy principles.

### spectral editing of activations

We adapt SEA to privacy by constructing three variants for each leak $(X, Y)$:

- **Neutral**: Remove the secret $Y$ and its surrounding context
- **Positive (harmless)**: Replace $Y$ with a placeholder
- **Negative (leak)**: Keep the original prompt $X$ that leaks $Y$

We perform forward passes on all three variants, capturing the last-token MLP activation $\mathbf{h}_{\ell}^{(0)}$, $\mathbf{h}_{\ell}^{(+)}$, and $\mathbf{h}_{\ell}^{(-)}$ for each layer $\ell$. Stacking these across all examples yields matrices $\mathbf{H}_{\ell}^{(0)}$, $\mathbf{H}_{\ell}^{(+)}$, and $\mathbf{H}_{\ell}^{(-)}$ of shape $N \times d$. We center each matrix and compute cross-covariance matrices:

$$\mathbf{\Omega}_{\ell}^{+} = (\mathbf{H}_{\ell}^{(0)})^\top \mathbf{H}_{\ell}^{(+)} / N, \quad \mathbf{\Omega}_{\ell}^{-} = (\mathbf{H}_{\ell}^{(0)})^\top \mathbf{H}_{\ell}^{(-)} / N$$

We perform singular value decomposition on each:

$$\mathbf{\Omega}_{\ell}^{\pm} = \mathbf{U}_{\ell}^{\pm} \mathbf{\Sigma}_{\ell}^{\pm} (\mathbf{V}_{\ell}^{\pm})^\top$$

and retain the top $k^+$ (positive) and bottom $k^-$ (negative) singular vectors based on energy thresholds (99% explained variance by default). The projection matrices are:

$$\mathbf{P}_{\ell}^{+} = \mathbf{U}_{\ell}^{+} (\mathbf{U}_{\ell}^{+})^\top, \quad \mathbf{P}_{\ell}^{-} = \mathbf{I} - \mathbf{U}_{\ell}^{-} (\mathbf{U}_{\ell}^{-})^\top$$

At inference, we apply both projections and renormalize:

$$\mathbf{z}_{\ell}^{\text{sea}} = \frac{\|\mathbf{z}_{\ell}\|_2}{\|\mathbf{P}_{\ell}^{+} \mathbf{z}_{\ell} + \mathbf{P}_{\ell}^{-} \mathbf{z}_{\ell}\|_2} (\mathbf{P}_{\ell}^{+} \mathbf{z}_{\ell} + \mathbf{P}_{\ell}^{-} \mathbf{z}_{\ell})$$

Optionally, we can restrict the projection to privacy neuron dimensions only by masking the projection matrices.

## results

### dataset

We use the CMU Enron email dataset, which contains approximately 500,000 emails from 150 Enron employees. We process the raw maildir format to extract email bodies (dropping headers and normalizing whitespace), then scan for PII using regular expressions:

- **Email addresses**: `[a-zA-Z0-9.\-+\_]+@[a-zA-Z0-9.\-+\_]+\.[a-zA-Z]+`
- **Phone numbers**: `(\+?\d[\d\-\-\.\(\)\s]{9,18}\d)`

For each detected PII span, we extract a contextual prefix (up to 128 tokens before the PII) as the prompt $X$ and the PII itself as the secret $Y$. We deduplicate by keeping only the first occurrence of each unique PII value.

We then score each candidate using an exposure metric computed against a candidate set of 100,000 entries (real secrets plus synthetic decoys). Candidates are filtered by exposure thresholds: samples with exposure above 12.0 bits are labeled "collected" (high leakage), those between 9.0 and 12.0 bits are labeled "memorized" (moderate leakage), and lower-exposure samples are discarded. This yields a curated evaluation set of approximately 200 samples stored in JSONL format, where each line contains the prompt prefix, secret, PII type, source path, and precomputed metrics (exposure, MRR, greedy match status).

Perplexity is computed on the same evaluation samples (prefix + secret sequences) rather than a separate validation corpus, providing a direct measure of utility on privacy-relevant text.

### evaluation metrics

We evaluate privacy protection and model utility using the following metrics:

**Exposure.** Following APNEAP, we compute exposure for each secret as a rank-based metric that measures how much information the model leaks about the secret relative to a candidate set. For a secret $Y$ with $n$ tokens, we compute its weighted rank $r$ across token positions, then calculate:

$$\text{Exposure}(X, Y) = \log_2(|R|^n) - \log_2(r)$$

where $|R|$ is the size of the candidate token set. Higher exposure indicates the model assigns higher probability to the secret.

**Perplexity (PPL).** We measure autoregressive perplexity on the validation corpus to quantify general language modeling capability. Autoregressive perplexity on the validation corpus $\mathcal{V}$:

$$\text{PPL} = \exp\left( -\frac{1}{|\mathcal{V}|} \sum_{i=1}^{|\mathcal{V}|} \log P(x_i \mid x_{<i}) \right)$$

Lower perplexity indicates better utility. We compare perplexity before and after editing to measure utility degradation.

**Mean Reciprocal Rank (MRR).** We compute the mean reciprocal rank of the secret across all token positions in the candidate set. For a secret $Y$ with $n$ tokens, we compute the mean reciprocal rank across all token positions:

$$\text{MRR}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{r_i}$$

where $r_i$ is the rank of token $y_i$ in the candidate set at position $i$, with $r_i = \infty$ (and thus $1/r_i = 0$) if the token does not appear in the top-$k$ candidates. Higher MRR indicates the model ranks the secret higher, indicating stronger privacy leakage. Lower MRR after editing indicates better privacy protection.

**Span Log-Probability.** For qualitative analysis, we record the mean log-probability of the secret span before and after editing. Mean log-probability of the secret span:

$$\text{SpanLogProb}(X, Y) = \frac{1}{|Y|} \sum_{i=1}^{|Y|} \log P(y_i \mid X, y_{<i})$$

A decrease indicates reduced model confidence in generating the secret.

### implementation details

We run experiments on a single NVIDIA A100 GPU with 40GB VRAM, using bfloat16 precision to fit the 1.3B parameter model and activation caches in memory. All edits are applied via PyTorch forward hooks registered on the MLP modules (`transformer.h[i].mlp.c_fc`) without modifying model weights.

**Privacy Neuron Distribution.** Using the integrated gradients attribution with thresholds $\tau_{\text{text}}=0.1$ and $\tau_{\text{batch}}=0.5$, we identify 759 privacy neurons distributed across 24 of the 24 layers. The distribution is concentrated in early-to-mid layers, with layers 2--4 containing the majority of privacy neurons (312 in layer 4, 201 in layer 3, 139 in layer 2), while later layers contain fewer neurons (e.g., 25 in layer 23).

### main results

| Method | MRR ↓ | Exposure ↓ | PPL |
|--------|-------|------------|-----|
| Base | 0.26 | 114.30 | 9.61 |
| APNEAP | -43.2% | -30.6% | 10.11 |
| Random Noise | -42.9% | -43.6% | 10.37 |
| SEA | -65.6% | -67.4% | 13.21 |

*Results for each method with 759 privacy neurons on `GPT-Neo-1.3B`. Results for base are shown as absolute numbers, with the other percentages in the table being calculated from these values.*

Table above shows the performance of all three editing methods on GPT-Neo-1.3B with 759 privacy neurons. APNEAP achieves 43.2% MRR suppression and 30.6% exposure reduction with 5.2% perplexity degradation, demonstrating a favorable privacy-utility tradeoff. Random noise steering performs comparably, achieving 42.9% MRR suppression and 43.6% exposure reduction (superior to APNEAP) with 7.9% perplexity degradation. This suggests that identifying privacy neurons is more critical than the specific steering direction. SEA achieves the strongest privacy protection (65.6% MRR suppression, 67.4% exposure reduction) but at substantial utility cost (37.5% perplexity increase), which may limit its practical applicability.

### ablation study

We conduct ablations across three axes to understand the sensitivity of our methods to hyperparameter choices:

**Privacy Neuron Count.** By varying the selection threshold $\tau_{\text{text}}$ from 0.0075 to 0.20, we obtain privacy neuron sets ranging from 2,303 neurons (3.1% of MLP dimensions) down to 34 neurons (0.05%). Results show that larger neuron sets generally achieve stronger privacy suppression but with greater utility cost, while smaller sets preserve perplexity but provide weaker protection.

**Editing Strength.** For activation patching, we vary the steering coefficient $\alpha$. For random noise steering, we vary the standard deviation $\sigma$ from 0.1 to 2.0. Both show a consistent trade-off: stronger interventions yield greater exposure reduction at the cost of increased perplexity. The optimal operating point depends on the application's privacy-utility requirements.

**Steering Method.** We compare directed steering (activation patching with computed steering vectors) against undirected perturbation (random Gaussian noise). While activation patching leverages task-specific information about privacy-leaking versus harmless activations, random noise provides a simpler baseline that requires no steering vector computation. Our results suggest that random noise can achieve comparable privacy suppression at similar perplexity costs, though directed steering may offer more predictable behavior.

**Text threshold ablation.** Varying the `text_threshold` parameter reveals a clear privacy-utility trade-off in full-layer editing: lower thresholds yield larger privacy improvements but higher perplexity, while higher thresholds yield smaller privacy improvements but lower perplexity. Results show that `GPT-Neo-1.3B`-0.075-full (2303 neurons) achieves the largest reductions but the highest perplexity, while `GPT-Neo-1.3B`-0.020-full (34 neurons) achieves more modest reductions but maintains better utility. This suggests that more neurons capture broader memorization signals but affect general model behavior, while selective neuron sets provide more targeted interventions with less utility degradation.

### activation patching results

| Model-threshold | α | # Neurons | Exposure (%) | MRR (%) | PPL |
|-----------------|---|-----------|--------------|----------|-----|
| GPT-Neo-1.3B-0.0075 | 2.00 | 2303 | -24.6% | -36.3% | 9.78 |
| GPT-Neo-1.3B-0.0075 | 3.00 | 2303 | -38.3% | -53.6% | 10.40 |
| GPT-Neo-1.3B-0.01 | 2.00 | 759 | -19.4% | -26.5% | 9.74 |
| GPT-Neo-1.3B-0.01 | 3.00 | 759 | -30.6% | -43.2% | 10.11 |
| GPT-Neo-1.3B-0.0125 | 2.00 | 250 | -20.7% | -33.1% | 9.67 |
| GPT-Neo-1.3B-0.0125 | 3.00 | 250 | -28.5% | -43.0% | 9.94 |
| GPT-Neo-1.3B-0.015 | 2.00 | 99 | -14.7% | -23.0% | 9.67 |
| GPT-Neo-1.3B-0.015 | 3.00 | 99 | -27.0% | -36.2% | 9.85 |
| GPT-Neo-1.3B-0.02 | 2.00 | 34 | -14.6% | -26.0% | 9.62 |
| GPT-Neo-1.3B-0.02 | 3.00 | 34 | -22.8% | -39.5% | 9.71 |

*Activation patching targeted metrics across privacy neuron thresholds and alphas. Negative percentages indicate privacy improvement.*

### random noise steering results

| Model-threshold | Std | # Neurons | Exposure (%) | MRR (%) | PPL |
|-----------------|-----|-----------|--------------|---------|-----|
| GPT-Neo-1.3B-0.0075 | 0.10 | 2303 | -0.0% | 0.0% | 9.69 |
| GPT-Neo-1.3B-0.0075 | 0.50 | 2303 | -1.4% | 0.0% | 9.75 |
| GPT-Neo-1.3B-0.0075 | 1.25 | 2303 | -52.1% | -42.9% | 12.42 |
| GPT-Neo-1.3B-0.0075 | 2.00 | 2303 | -96.6% | -100.0% | 52501.80 |
| GPT-Neo-1.3B-0.01 | 0.10 | 759 | 0.0% | -0.0% | 9.69 |
| GPT-Neo-1.3B-0.01 | 0.50 | 759 | -17.9% | -21.4% | 9.65 |
| GPT-Neo-1.3B-0.01 | 1.25 | 759 | -43.6% | -42.9% | 10.37 |
| GPT-Neo-1.3B-0.01 | 2.00 | 759 | -72.5% | -89.3% | 31.18 |
| GPT-Neo-1.3B-0.0125 | 0.10 | 250 | -3.0% | -7.1% | 9.70 |
| GPT-Neo-1.3B-0.0125 | 0.50 | 250 | -7.7% | -7.1% | 9.72 |
| GPT-Neo-1.3B-0.0125 | 1.25 | 250 | -25.2% | -21.4% | 10.12 |
| GPT-Neo-1.3B-0.0125 | 2.00 | 250 | -56.9% | -57.1% | 11.57 |
| GPT-Neo-1.3B-0.015 | 0.10 | 99 | +0.5% | 0.0% | 9.70 |
| GPT-Neo-1.3B-0.015 | 0.50 | 99 | -14.6% | -21.4% | 9.74 |
| GPT-Neo-1.3B-0.015 | 1.25 | 99 | -22.0% | -21.4% | 10.01 |
| GPT-Neo-1.3B-0.015 | 2.00 | 99 | -44.6% | -64.3% | 10.50 |
| GPT-Neo-1.3B-0.020 | 0.10 | 34 | -0.0% | 0.0% | 9.69 |
| GPT-Neo-1.3B-0.020 | 0.50 | 34 | -11.4% | -21.4% | 9.72 |
| GPT-Neo-1.3B-0.020 | 1.25 | 34 | -8.2% | -7.1% | 9.89 |
| GPT-Neo-1.3B-0.020 | 2.00 | 34 | -32.3% | -35.7% | 10.11 |

The above table presents the complete results for random noise steering experiments on GPT-Neo-1.3B. Each row represents a configuration with a specific privacy neuron selection threshold and noise standard deviation ($\sigma$). The "Model-threshold" column indicates the threshold used for privacy neuron selection (lower thresholds yield more neurons). Exposure and MRR columns show percentage change from the unedited baseline, where negative values indicate privacy improvement. PPL shows the absolute perplexity of the edited model (baseline PPL $\approx$ 9.69).

### spectral editing results

| Model-threshold-layers | # Layers | # Neurons | Exposure (%) | MRR (%) | PPL |
|------------------------|----------|-----------|--------------|---------|-----|
| GPT-Neo-1.3B-0.075-12 | 24 | 2303 | -1.0% | 0.0% | 10.18 |
| GPT-Neo-1.3B-0.075-full | 24 | 2303 | -75.0% | -77.1% | 23.82 |
| GPT-Neo-1.3B-0.01-12 | 24 | 759 | -6.0% | -7.6% | 10.12 |
| GPT-Neo-1.3B-0.01-full | 24 | 759 | -67.4% | -65.6% | 13.14 |
| GPT-Neo-1.3B-0.0125-12 | 19 | 250 | -8.2% | -7.6% | 10.15 |
| GPT-Neo-1.3B-0.0125-full | 19 | 250 | -65.3% | -61.1% | 11.09 |
| GPT-Neo-1.3B-0.015-12 | 16 | 99 | -29.8% | -38.2% | 10.25 |
| GPT-Neo-1.3B-0.015-full | 16 | 99 | -60.8% | -65.0% | 10.76 |
| GPT-Neo-1.3B-0.020-12 | 13 | 34 | -23.7% | -34.5% | 10.21 |
| GPT-Neo-1.3B-0.020-full | 13 | 34 | -38.0% | -53.4% | 10.55 |
| Qwen3-8B-enron-0.025-full | 36 | 328 | -0.6% | -6.4% | 7.57 |
| Qwen3-8B-enron-0.025-18 | 36 | 328 | +2.2% | +6.7% | 7.32 |

*SEA results for `GPT-Neo-1.3B` and `Qwen3-8B-enron` models. For GPT-Neo-1.3B, the base perplexity is 9.64 across all configurations. Base exposure values range from 106.19 to 106.72, and base MRR values range from 0.2218 to 0.2220. For Qwen3-8B-enron, the base perplexity is 6.68 across all configurations. Base exposure values range from 111.24 to 113.60, and base MRR values range from 0.2303 to 0.2390.*

For GPT-Neo-1.3B, SEA-style editing generally reduced both targeted exposure and MRR (negative percentages in the table), but the magnitude depended strongly on how many layers were edited. Editing only the last 12 layers produced modest gains (6--30% exposure reduction, 0--38% MRR reduction) with small-to-moderate perplexity increases (~5--6%), while full-layer editing consistently produced much larger privacy gains (up to ~75% exposure and ~77% MRR reduction) at the cost of substantially higher perplexity increases (ranging from ~9% up to ~147%). For the same threshold, differences between the 12 layer and the full layer are large (upwards of 74% for difference in exposure and 77% for difference in MRR); in `GPT-Neo-1.3B`-0.020, adding an extra layer (13 total distinct layers with privacy neurons) decreased exposure by an extra 14.3% and decreased MRR by an extra 18.9%.

The `Qwen3-8B-enron`-0.025-18 run increased both exposure and MRR, while the full-layer run decreased both metrics but achieved only modest improvements (0.6% exposure, 6.4% MRR reduction) compared to GPT-Neo's 30--75% reductions. Qwen's weaker performance likely stems from finetuning on Enron data, which embeds secrets more deeply in the model's representations. This finetuning reduces SEA's effectiveness because the placeholder-based "positive" demonstrations capture different semantic relationships in a domain-specific model, making the cross-covariance projections less effective at suppressing memorization.

Comparing the 328 privacy neurons of `Qwen3-8B-enron` using text threshold 0.0025 with the 2303 and 759 privacy neurons of `GPT-Neo-1.3B` using text threshold 0.0075 and 0.01 respectively suggests that finetuned models exhibit more concentrated attribution patterns. Even considering Qwen's larger architecture (36 layers vs. GPT-Neo's 24), the finetuning process may have focused memorization signals into sparse but highly-attributed neurons rather than distributing them broadly. This concentration may contribute to SEA's reduced effectiveness on Qwen, as the memorization is encoded in a more compact, less linearly-separable representation.

## conclusion

This work implements and evaluates three inference-time privacy editing strategies for large language models. Building on the APNEAP framework, we demonstrate that gradient-based attribution can identify a small set of neurons (759 out of approximately 73,700 in GPT-Neo-1.3B) that are disproportionately responsible for privacy leakage. Our experiments on GPT-Neo-1.3B with 759 privacy neurons reveal distinct performance characteristics for each method. APNEAP achieves 43.2% MRR suppression and 30.6% exposure reduction with only 5.2% perplexity degradation, demonstrating a favorable privacy-utility tradeoff. Random noise steering performs comparably, achieving 42.9% MRR suppression and 43.6% exposure reduction (superior to APNEAP's exposure reduction) with 7.9% perplexity degradation. This finding suggests that identifying privacy neurons is the critical component, while the specific steering direction may be less important than simply perturbing these neurons. SEA achieves the strongest privacy protection on GPT-Neo-1.3B with 65.6% MRR suppression and 67.4% exposure reduction, but at a substantial utility cost (37.5% perplexity increase). However, SEA's effectiveness is highly dependent on editing many layers and shows much weaker results on finetuned models, achieving only modest reductions (0.6% exposure, 6.4% MRR) on Qwen3-8B-enron.

### implications

Our findings suggest that inference-time privacy editing is a viable strategy for deployed LLMs, particularly when retraining is infeasible or privacy leakage is discovered post-deployment. The localized nature of privacy neurons enables surgical interventions that preserve general model capabilities. A key insight is that random noise steering performs comparably to or better than directed APNEAP steering, suggesting that simpler undirected interventions may be sufficient and reducing the need for paired examples and steering vector computation. However, the modest suppression rates (30--44% for APNEAP and random noise) indicate that editing alone is insufficient for high-stakes privacy protection; it should be combined with preprocessing, differential privacy, and careful data curation. SEA's strong performance on GPT-Neo-1.3B demonstrates the potential of spectral methods, but its limited effectiveness on finetuned models suggests that domain-specific training may embed privacy information in ways that are less amenable to linear subspace projection, requiring alternative editing strategies for finetuned models.

### limitations

Our approach has several limitations. First, we focus on simple PII patterns (emails, phone numbers); more complex privacy violations (contextual personal information, proprietary data) may not localize as cleanly to specific neurons. Second, our evaluation is limited to targeted suppression on a fixed test set; we do not measure robustness to adversarial prompting strategies or evaluate generalization to held-out PII types. Third, our experiments mainly use a single model (GPT-Neo-1.3B) on a single dataset (Enron); generalization to other model families (LLaMA, Mistral) and domains (medical records, financial data) remains to be validated. Fourth, the privacy-neuron selection process assumes access to a dataset of known leaks, which may not be available in all deployment scenarios.

Concerning the spectral editing, there are also many limitations. First, four additional Qwen privacy-neuron sets were generated at different `text_threshold` values, but not all could be evaluated due to GPU/runtime constraints; as a result, we focused on the most comprehensive set available for Qwen (the 0.025 threshold file with the largest neuron count).