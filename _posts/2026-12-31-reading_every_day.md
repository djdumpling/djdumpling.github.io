---
title: "a paper a day keeps the rust away"
date: 2026-12-31
ongoing: true
---

Every day in 2026, I'll read an ML paper/blog. This is my running log of distillations, meant to be a learning archive for myself as well as a capsule of how the field evolves across 2026.

# 1/1: manifold-constrained hyper-connections

From the DeepSeek team, [this paper](https://arxiv.org/pdf/2512.24880) explores how to enforce the identity mapping property intrinsic to residual connection (which otherwise causes training instability and restricted scalability) to hyper-connections via **manifold-constrained hyper-connections**. The structure of a single layer is

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)
$$

where $\mathbf{x}_{l}, \mathbf{x}_{l+1}$ are the input/output of the $l^\text{th}$ layer, respectively, and $\mathcal{F}$ denotes the residual function. **Hyper-connections**, popularized by [Zhu et al., 2024](https://arxiv.org/abs/2409.19606), added learnable mappings that instead of carrying $\mathbf{x}_l$, carries a bundle of $n$ parallel residual streams defined by

$$
\mathbf{x}_{l+1} = \mathcal{H}_l^\text{res}\mathbf{x}_l + \left(\mathcal{H}_l^\text{post}\right)^\top\mathcal{F}(\mathcal{H}_l^\text{pre}\mathbf{x}_l, \mathcal{W}_l)
$$

where $n$ is the expansion rate. $\mathcal{H}_l^\text{res}$ represents the learnable mapping that mixes features. $\mathcal{H}_l^\text{pre}$ aggregates features from the expanded stream into the standard stream. Conversely, $\mathcal{H}_l^\text{post}$ maps the layer output back onto the stream. However, these compromise the identity mapping property, meaning that during forward/backward passes, both the activations and the gradients are amplified or accentuated, resulting in instability for large-scale training due to many hyper-connections. Specifically, they compute the coefficients by

$$
\begin{align*}
\tilde{\mathbf{x}_l} &= \text{RMSNorm}(\mathbf{x}_l) \\
\mathcal{H}_l^\text{pre} &= \alpha_l^\text{pre} \cdot \tanh(\theta_l^\text{pre}\tilde{\mathbf{x}_l} ^\top) + \mathbf{b}_l^\text{pre} \\
\mathcal{H}_l^\text{post} &= \alpha_l^\text{post} \cdot \tanh(\theta_l^\text{post}\tilde{\mathbf{x}_l} ^\top) + \mathbf{b}_l^\text{post} \\
\mathcal{H}_l^\text{res} &= \alpha_l^\text{res} \cdot \tanh(\theta_l^\text{res}\tilde{\mathbf{x}_l} ^\top) + \mathbf{b}_l^\text{res}
\end{align*}
$$

where all pre, post, and residual types of $\alpha_l$ and $\mathbf{b}_l$ are learnable. For typical expansion rates, they incur negligible computational overhead but increase memory access cost by a factor roughly proportional to the expansion rate, which degrades throughput.

![Illustrations of residual connection paradigms](/public/reading/mHC.png)

**Manifold-constrained hyper-connections** restore the identity mapping property by using the Sinkhorn-Knopp algorithm to project $\mathcal{H}_l^\text{res}$ onto the Birkhoff polytope, the set of all doubly stochastic matrices. These matrices have the property that the sum of all rows and columns equals 1, meaning that the matrix acts more like averaging rather than scaling since the spectral norm is less than 1. Due to closure of matrix multiplication for doubly stochastic matrices, then $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^\text{res}$ is still doubly stochastic, meaning that the transformer maintains the stability of identity mappings. If the dimension is 1, then the doubly stochastic matrix is exactly the identity matrix. The HC formulae can be reformulated to obtain

$$
\begin{align*}
\tilde{\mathbf{x}_l}' &= \text{RMSNorm}(\mathbf{x}_l) \\
\tilde{\mathcal{H}_l}^\text{pre} &= \alpha_l^\text{pre} \cdot (\mathbf{x}_l' \phi_l^\text{pre}) + \mathbf{b}_l^\text{pre} \\
\tilde{\mathcal{H}_l}^\text{post} &= \alpha_l^\text{post} \cdot (\mathbf{x}_l' \phi_l^\text{post}) + \mathbf{b}_l^\text{post} \\
\tilde{\mathcal{H}_l}^\text{res} &= \alpha_l^\text{res} \cdot \text{mat}(\mathbf{x}_l' \phi_l^\text{res}) + \mathbf{b}_l^\text{res}
\end{align*}
$$

where $\mathbf{x}_l$ is the flattened version of itself, $\text{mat}( \cdot)$ is the reshape function to be a square matrix, and all types of $\phi_l$ are linear projections from learnable mappings. The final constrained mappings are

$$
\begin{align*}
\mathcal{H}_l^\text{pre} &= \sigma(\tilde{\mathcal{H}_l}^\text{pre})\\
\mathcal{H}_l^\text{post} &= 2\sigma(\tilde{\mathcal{H}_l}^\text{post})\\
\mathcal{H}_l^\text{res} &= \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}_l}^\text{res})
\end{align*}
$$

where the sigmoid function gives a positive, bounded matrix that's useful for mixture weights. I'm not exactly sure why there's a doubling for $\mathcal{H}_l^\text{post}$.

---
The other principal contribution of the paper was an efficient infrastructure design. The first is to move the dividing-by-norm operation to follow the matrix multiplication, which improves efficiency because the operation commutes with linear projections. They also use mixed-precision strategies to maximize numerical accuracy, such that having most matrices and values in `float32`, $\mathbf{x}_l$ in `bfloat16` to save memory/bandwidth, and $\phi_l$ in `tfloat32` (higher numerical accuracy). Also, fuse multiple operations with the shared memory access into unified compute kernels to reduce memory bandwidth bottlenecks; this includes implementing the Sinkhorn-Knopp iteration within a single kernel, as well as a unified kernel that fuses two scans on $\mathbf{x}_l$ and a single kernel for the backward pass which contains two matrix multiplications. This effectively cuts the number of read/write elements by a third.

To cut memory, they don’t store most intermediate activations from the mHC kernels during forward. In the backward pass, they recompute those intermediates on demand, using a block strategy where only the input to the first layer of each block is saved; they pick a block size $L_r$ to balance “saved memory vs extra recompute,” with an approximate optimum. Also, mHC increases communication and can add pipeline “bubbles,” so they modify the schedule to overlap boundary communication with high-priority compute (notably the post/res kernels), avoid long “persistent” attention kernels that block scheduling, and rely on cached block-start activations so recomputation doesn’t stall on communication. [**TODO: look over this and understand more deeply**]

---
Across many benchmarks like GSM8k, HellaSwag, and MMLU, 27B with mHC outperforms both the baseline and the 27B with HC. Moreover, mHC is very effective in large-scale scenarios, since there is minimum degradation in terms of loss with regards to compute and token count. As suggested by Sinkhorn-Knopp, mHC significantly enhanced propagation stability compared to HC by three orders of magnitude, from a maximum gain of nearly 3000 to approximately 1.6.