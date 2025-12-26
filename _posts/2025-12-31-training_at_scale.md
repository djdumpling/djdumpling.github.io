---
title: "training at scale"
date: 2025-12-31
---

How do labs train a multi-billion parameter model? We look towards Hugging Face's SmolLM3, Allen Institute's Olmo 3, Prime Intellect's Intellect 3, and OpenAI's GPT-OSS-120B. This blog is an attempt towards distilling the motivations, considerations, and techniques used to train their models and is structured in more of a "notes" style.

These notes have not been thoroughly reviewed. Any errors below are my own responsibility.

# general practices

1. [HF] "**Learn to identify what's worth testing, not just how to run tests.** Perfect ablations on irrelevant choices waste as much compute as sloppy ablations on important ones." 
    - Ablations need to be **fast** (faster iteration $\rightarrow$ more hypotheses tested) and **reliable** (need strong discriminative power because otherwise, it may be noise)
    - For variables that change the parameter count (e.g. MHA to GQA), they occasionally adjust other hyperparameters to keel model sizes roughly the same.
2. [HF] **Choose an established baseline with good architecture and training setup design**. These take years of iteration, and people have discovered common failure commons and instabilities.
    - There are a plethera of modifiable components (attention mechanisms and positional encodings to name a few), but follow the principle of **derisking**: "never change anything unless you've tested that it helps."
3. [HF] **In evals, look for monotonicity** (score improvement), **low noise** (e.g. score resistance to random seeds), above-random performance (random-level peformance for extended time frames isn't useful), and ranking consistency (ranking of approaches should remain stable throughout training). 

## architecture

Model families like DeepSeek, MiniMax, Kimi, OLMo, and SmolLM have vastly different architectures (dense vs MoE), attention mechanisms (MHA vs MLA vs GQA), position encodings (RoPE, partial RoPE, NoPE), among many, many, others.

| | DeepSeek | MiniMax | Kimi | OLMo | SmolLM |
|--|----------|---------|-------|------|--------|
| Attention | X | X | X | X | GQA (4 groups)|
| Embedding Sharing | X | X | X | X | tied|
| Positional Embedding| X | X | X | X | RNoPE|
| Z-Loss | X | X | X | X | No|
| Architecture | X | X | X | X | dense|

Between choosing architecture, HF suggests following a decision tree such that if one of these is true, then to choose a dense architecture:
- memory-constrained (since MoEs must have all experts loaded)
- new to LLM training (focus on basics)
- tighter timeline (simpler training with well-documented recipes)

### attention

To address the large KV-cache (an inference bottleneck and GPU memory hoarder) associated with MHA, researchers developed multi-query attention (MQA) and grouped query attention (GQA). In MQA, KV values are shared across all heads, but this comes at a cost of leaking attention capacity because heads can't store information specialized for that head's role. GQA softens this issue by sharing KV values across a small group of heads (e.g. 4). Another alternative is multi-latent attention (MLA) which stores a latent variable that be decompressed/projected into KV values at runtime. This results in a KV-cache parameter count more comparable to GQA and performance stronger than MQA. 

When ablating, HF found that **GQA with small groups beats MHA** and that **MHA beats MQA and GPQ with 16 groups**. Across benchmarks like HellaSwag, MMLU, and ARC, GQA with 2/4/8 groups do best.

[TODO: add paper links to associated mechanism]

### document masking

When pre-training, a common consideration is **fixed sequence lengths** since training uses tensors of the form [batch, sequence length, hidden], so with regards to batching and distributed training, GPUs are most happy when every example has the same sequence length. But due to variable document length and wanting to avoid padding which wastes compute, **packing** enables shuffling and concatenating documents within the same sequence to achieve the sequence length. 

Casual masking means that for unrelated files $A$ and $B$ in the same batch, the tokens in $B$ can attend to the tokens in $A$, which degrades performance. With **intra-document masking**, the attention mask is modified so tokens can only attend to previous tokens within the same document. Many papers have found benefits relating to [long-context extension](https://arxiv.org/abs/2407.21783) and [some short context benchmarks](https://arxiv.org/abs/2410.02660) as well as [shortening the average context length](https://arxiv.org/abs/2503.15450).

When implementing docuemnt masking, HF saw small improvements on PIQA but otherwise no noticable impact on short context tasks. But in line with aforementioned research, they observed that it became crucial for scaling from 4k to 64k tokens.

### embedding sharing

Input embeddings (token-to-vector lookup) and output embeddings (hidden states to vocab logits) are typically representated as separate metrics, so the total embedding parameters are $2 \times \text{vocab size} \times \text{hidden dim}$. In small language models can account up to 20% of total parameters, as is the case with `Llama 3.2 1B` (in larger models, the embeddings represent a much smaller fraction of the parameter count, only 3% in `Llama 3.1 70B`). The issue with tying them is that input/output embeddings still represent different geometries, and frequent tokens like "the" can dominate representation learning due to getting gradients from both the input stream and the predicted output. 

HF found that on a 1.2B model, tied embeddings did comparably well despite having 18% fewer parameters (down from 1.46B), and that compared to an untied model also with 12.B models (fewer layers), untied showed higher loss and lower downstream eval scores.

### positional encodings

Without positional encoding, transformers have no sense of word order, akin to the bag of words idea. Initially, [absolute position embeddings](https://arxiv.org/abs/1706.03762) were used by learning a lookup table that mapped the position index to a vector added to token embeddings, but the max input sequence length was limited by the max input sequence length of what it was trained on. **Relative position encodings** followed since capturing distance between tokens matters more than capturing their absolute positions.

The most commonly used technique is [rotary position embedding (RoPE)](https://arxiv.org/abs/2104.09864), which encodes relative position as rotation angles. Based on the dimensionality of the query/key vector, RoPE splits it into pairs (since they rotate in 2D space) and rotates depending on the absolute position of a token and a base frequency. During attention, the dot product between their rotated positions directly encodes their relative distance via the phase difference in their rotation angles, where tokens $x$ positions part always maintain the same angular relationship. 

During pretraining, models are trained on shorter context lengths (similar ideas to document masking, and quadratic attention is expensive) to learn short range correlation between words. But as sequence length grows, the rotation angles grows via $\theta= \text{position} \times \frac1{\text{base}^{\frac{k}{\text{dim}/2}}}$. This can fixed by increasing the base frequency as the sequence length increases using methods like [ABF](https://arxiv.org/abs/2309.16039) or [YaRN](https://arxiv.org/abs/2309.00071), which applies a more granular interpolation of frequencies on different components and includes other techniques like dynamic attention scaling and temperature adjustment. For extremely long contexts, YaRN does best.

More recently, with the emphasis on long contexts, [NoPE](https://arxiv.org/abs/2305.19466) (no position embedding) and [RNoPE](https://arxiv.org/abs/2501.18795), a hybrid method, have emerged. NoPE uses only casual masking and attention patterns, so it doesn't bump into the issue of extrapolating beyond training lengths but shows weaker performance on short context reasoning and knowledge-based tasks. RNoPE alternates applying RoPE and NoPE on attention blocks, where RoPE handles local context and NoPE helps with longer-range information retrieval. Another idea is Partial RoPE, which applies RoPE/NoPE within the same layer. [TODO: consider adding partial ROPE with MLA]

HF ran ablations using RoPE, RNoPE (removing positional encoding every 4th layer), and RNoPE with document masking. They found that all achieve similar performance on short-context tasks, so they adopt RNoPE + document masking because it provides the foundation for long-context handling.

### attention for long contexts

An alternative to adjusting positional encodings for long contexts is specifying the strength of which tokens can attend to one another. 
- **Chunked Attention**: divides the sequence into fix-sized chunks where tokens can only attend within their chunk. [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) paired with RNoPE (specifically the RoPE layers) which also reduces the KV cache size per layer, but it's performance on long context tasks would degrade.
- **Sliding Window Attention (SWA)**: every token can see up to $p$ positions back, creating a sliding window that maintains local context. Gemma 3 combined SWA with full attention every other layer.
- **Dual Chunk Attention (DCA)**: $K$ tokens are chunked into $M$ groups. Within each group (like chunked attention), tokens attend normally. Between successive chunks have their own local window to preserve locality, and more broadly, inter chunk attention allows queries to attend to previous chunks with a capped relative position capped. Qwen-2.5 used DCA to support context windows of up to 1 million tokens.

[TODO: add image from HF of attention patterns]

### MoE

MoEs (mixture of experts), analgous to our brain activating different parts of our brain, provide an alternative to dense models due to only certain "experts" being used at inference time, saving lots of compute. The MoE works by replacing the feed forward layer with multiple MLPs (experts) and add a learnable router before the MLPs to select the experts.

In general, for fixed number and size of active experts, increasing the total number of experts improves loss, and [high sparsity improves performance](https://arxiv.org/abs/2507.20534) and [benefits more from increasing compute](https://arxiv.org/abs/2507.17702). Recent models are much more sparse, with over 100 experts and around 10 activate per token. [TODO: add image]

To determine how large each expert should be, a common metric is granularity, defined by $G = 2 \cdot \frac{d_\text{model}}{d_\text{expert}}$, where a higher granularity corresponds to more experts with a smaller dimension; this can be intermediate as a number proportional to the experts needed to match the dense MLP width. Recent models have granularity anywhere from 2 (`gpt-oss-120b`) to 8 (`qwen3-next-80b-a3b`). [Ant Group](https://arxiv.org/pdf/2507.17702) showed that granularity doesn't significantly change loss but does drive **efficiency leverage** (the ratio of flops needed for an MoE to achieve the same loss as a dense model). And overall, MoEs present a good alternative to dense models in terms of compute for training and inference.

**Shared experts** are always-on experts, which absorb the basic, recurring patterns so that other experts can more aggressively specialize; one is often enough (`deepseek-v2` uses two, which adds a bit of complexity).

**Load balancing** is crucial in that if it fails, not only do training and inference efficiency plummet, but so do effective learning capacity. This can be addressed by adding a **loss-based load balancer** (LBL) given by $\mathcal{L} = \alpha \sum_{i=1}^{N_r} f_i P_i$ where $\alpha$ determines the strength, $f_i$ is the fraction of tokens going through expert $i$, and $P_i$ is the probability mass that sums the probability of tokens going through an expert;, so in perfect load balancing, $f_i=P_i=\frac1{N_r}$. Also, $\alpha$ should not be so large that routing uniformity overwhelms the primary training objective. These should be monitored using *global statistcs*, not local statistics which may suffer from a local batch being narrow, biasing the routing statistics. 

`deepseek-v3` does loss free load balancing differently, by adding a bias term that is added to affinity scores going into the routing softmax.

### hybrid models

Because transformers dn't deal efficiently with long context while RNNs can, one idea is to combine both to get the best of both worlds. By dropping the softmax from the output for token $t$:
$$\mathbf{o}_t = \sum_{j=1}^t \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)\mathbf{v}_j}{\sum_{l=1}^t \exp(\mathbf{q}_t^\top \mathbf{k}_l)} \Longrightarrow \mathbf{o}_t = \sum_{j=1}^t (\mathbf{q}_t^\top \mathbf{k}_j)\mathbf{v}_j = \left(\sum_{j=1}^t \mathbf{v}_j \mathbf{k}_jk^\top\right)\mathbf{q}_t$$
And by defining $S_t :=\sum_{j=1}^t \mathbf{k}_j \mathbf{v}_j^\top$, then we geta recurrent relation where $S_t$ summarizes all past $(k_j, v_j)$. 
$$S_t=S_{t-1}+\mathbf{k}_j \mathbf{v}_j^\top \Longrightarrow \mathbf{o}_t = S_t \mathbf{q}_t = S_{t-1}\mathbf{q}_t+\mathbf{v}_k\left(\mathbf{k}_j \mathbf{v}_j^\top\right)$$

While this gets us closer to an RNN-esque structure, in practice, softmax stabilizes training, and the linear form can cause instability without normalization. With RNNs, it is sometimes helpful to forget the past, via introducing a gate $\mathbf{G}_t$ for the previous state 
$$\mathbf{S}_t=\mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{v}_t\mathbf{k}_t^\top$$
[Mamba-2](https://arxiv.org/abs/2405.21060) is among the most popular, being used in hybrid models like [Nemotron-H](https://arxiv.org/abs/2504.03624) and [Falcon H1](https://arxiv.org/abs/2507.22448). Hybrid models are becoming increasing popular, notably in [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) with a gated DeltaNet update and Kimi's next model, likely using their ["kimi delta attention.]"(https://github.com/fla-org/flash-linear-attention/pull/621)

## stability

### $z$-loss

$z$-loss is an regularization term added to the standard cross entropy loss that keeps logits from drifting to large magnitudes. By adding $\mathcal{L}_{\text{z-loss}} = \lambda \cdot \log^2(Z) = \lambda \sum_{i=1}^V e^{z_i}$, representing the denominator in the softmax the loss now penalized based on $\log(Z)$ which represents the overall logit scale. 

On their 1B model, HF found that adding $Z$-loss didn't impact training loss or downstream performance, so they chose not to include it due to training overhead.

### removing weight decay from embeddings

Despite being a regularization technique, weight decay being removed from embeddings can improve training stability. Weight decay causes embedding norm to decrease, but this can lead to larger gradients in earlier layers since the LayerNorm Jacobian has a $\frac1{\sigma}$ term (coming from normalization) which is inversely proportional to the input norm $\sigma$.

HF tested this using a weight decay baseline, a no wieght decay baseline, and another combining all previous adopted changes and found no significant loss or eval results, so they included no weight decay.

### qk norm

Similar to $z$-loss, QK-norm helps prevent attention logits from becoming too large by applying LayerNorm to both the query and key vectors before computing attention. However, [the same paper which proposed RNoPE](https://arxiv.org/abs/2501.18795) found that it hurts long-context tasks because the normalization demphasizes relevant tokens and emphasizes irrelevant tokens by stripping the query-key dot product of its magnitude.

### other considerations

1. **Parameter initialization**: either normalization initation ($\mu=0$, $\sigma=0.02, 0.006$) with clipping (often with $\pm 2-3 \sigma$) or a scheme like $\mu\text{P}$ ([maximal update parametrization](https://arxiv.org/abs/2011.14522)) which dictates how weights and learning rates should scale with width so that training dynamics stay comparable.
2. **Activation Function**: SwiGLU is what most modern LLMs use, not ReLU or GeLU. Some exceptions are Gemma2 using GeGLU and nvidia using $\text{relu}^2$. 
3. **Width vs Height**: deeper models tend to outperform outperform equally sized wider ones on language modeling and compositional tasks. In smaller models, this is more pronounced, but larger models make use wider models for faster inference due modern architectures supporting better parallelism. 
