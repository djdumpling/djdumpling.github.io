---
title: "training at scale"
date: 2025-12-31
---

How do labs train a multi-billion parameter model? We look towards Hugging Face's SmolLM3, Allen Institute's Olmo 3, Prime Intellect's Intellect 3, and OpenAI's GPT-OSS-120B. This blog is an attempt towards distilling the motivations, considerations, and techniques used to train their models and is structured in more of a "notes" style.

These notes have not been thoroughly reviewed. Any errors below are my own responsibility.

# General Practices

1. [HF] "**Learn to identify what's worth testing, not just how to run tests.** Perfect ablations on irrelevant choices waste as much compute as sloppy ablations on important ones." 
    - Ablations need to be **fast** (faster iteration $\rightarrow$ more hypotheses tested) and **reliable** (need strong discriminative power because otherwise, it may be noise)
    - For variables that change the parameter count (e.g. MHA to GQA), they occasionally adjust other hyperparameters to keel model sizes roughly the same.
2. [HF] **Choose an established baseline with good architecture and training setup design**. These take years of iteration, and people have discovered common failure commons and instabilities.
    - There are a plethera of modifiable components (attention mechanisms and positional encodings to name a few), but follow the principle of **derisking**: "never change anything unless you've tested that it helps."
3. [HF] **In evals, look for monotonicity** (score improvement), **low noise** (e.g. score resistance to random seeds), above-random performance (random-level peformance for extended time frames isn't useful), and ranking consistency (ranking of approaches should remain stable throughout training). 

# huggingface's smolLM3

## architecture

Model families like DeepSeek, MiniMax, Kimi, OLMo, and SmolLM have vastly different architectures (dense vs MoE), attention mechanisms (MHA vs MLA vs GQA), position encodings (RoPE, partial RoPE, NoPE), among many, many, others.

| | DeepSeek | MiniMax | Kimi | OLMo | SmolLM |
|--|----------|---------|-------|------|--------|
| Attention | X | X | X | X | GQA (4 groups)|
| Embedding Sharing | X | X | X | X | tied|
| Positional Embedding| X | X | X | X | RNoPE|

### attention

To address the large KV-cache (an inference bottleneck and GPU memory hoarder) associated with MHA, researchers developed multi-query attention (MQA) and grouped query attention (GQA). In MQA, KV values are shared across all heads, but this comes at a cost of leaking attention capacity because heads can't store information specialized for that head's role. GQA softens this issue by sharing KV values across a small group of heads (e.g. 4). Another alternative is multi-latent attention (MLA) which stores a latent variable that be decompressed/projected into KV values at runtime. This results in a KV-cache parameter count more comparable to GQA and performance stronger than MQA. 

When ablating, HF found that **GQA with small groups beats MHA** and that **MHA beats MQA and GPQ with 16 groups**. Across benchmarks like HellaSwag, MMLU, and ARC, GQA with 2/4/8 groups do best.

[TODO: add paper links to associated mechanism]

### document masking

When pre-training, a common consideration is **fixed sequence lengths** since training uses tensors of the form [batch, sequence length, hidden], so with regards to batching and distributed training, GPUs are most happy when every example has the same sequence length. But due to variable document length and wanting to avoid padding which wastes compute, **packing** enables shuffling and concatenating documents within the same sequence to achieve the sequence length. 

Casual masking means that for unrelated files $A$ and $B$ in the same batch, the tokens in $B$ can attend to the tokens in $A$, which degrades performance. With **intra-document masking**, the attention mask is modified so tokens can only attend to previous tokens within the same document. Many papers have found benefits relating to [long-context extension](https://arxiv.org/abs/2407.21783) and [some short context benchmarks](https://arxiv.org/abs/2410.02660) as well as [shortening the average context length](https://arxiv.org/abs/2503.15450).

When implementing docuemnt masking, HF saw small improvements on PIQA but otherwise no noticable impact on short context tasks. But in line with aforementioned research, they observed that it became crucial for scaling from 4k to 64k tokens.

# embedding sharing

Input embeddings (token-to-vector lookup) and output embeddings (hidden states to vocab logits) are typically representated as separate metrics, so the total embedding parameters are $2 \times \text{vocab size} \times \text{hidden dim}$. In small language models can account up to 20% of total parameters, as is the case with `Llama 3.2 1B` (in larger models, the embeddings represent a much smaller fraction of the parameter count, only 3% in `Llama 3.1 70B`). The issue with tying them is that input/output embeddings still represent different geometries, and frequent tokens like "the" can dominate representation learning due to getting gradients from both the input stream and the predicted output. 

HF found that on a 1.2B model, tied embeddings did comparably well despite having 18% fewer parameters (down from 1.46B), and that compared to an untied model also with 12.B models (fewer layers), untied showed higher loss and lower downstream eval scores.

### positional encodings

Without positional encoding, transformers have no sense of word order, akin to the bag of words idea. Initially, [absolute position embeddings](https://arxiv.org/abs/1706.03762) were used by learning a lookup table that mapped the position index to a vector added to token embeddings, but the max input sequence length was limited by the max input sequence length of what it was trained on. **Relative position encodings** followed since capturing distance between tokens matters more than capturing their absolute positions.

The most commonly used technique is [rotary position embedding (RoPE)](https://arxiv.org/abs/2104.09864), which encodes relative position as rotation angles. Based on the dimensionality of the query/key vector, RoPE splits it into pairs (since they rotate in 2D space) and rotates depending on the absolute position of a token and a base frequency. During attention, the dot product between their rotated positions directly encodes their relative distance via the phase difference in their rotation angles, where tokens $x$ positions part always maintain the same angular relationship. 

During pretraining, models are trained on shorter context lengths (similar ideas to document masking, and quadratic attention is expensive) to learn short range correlation between words. But as sequence length grows, the rotation angles grows via $\theta= \text{position} \times \frac1{\text{base}^{\frac{k}{\text{dim}/2}}}$. This can fixed by increasing the base frequency as the sequence length increases using methods like [ABF](https://arxiv.org/abs/2309.16039) or [YaRN](https://arxiv.org/abs/2309.00071), which applies a more granular interpolation of frequencies on different components and includes other techniques like dynamic attention scaling and temperature adjustment. For extremely long contexts, YaRN does best.

More recently, with the emphasis on long contexts, [NoPE](https://arxiv.org/abs/2305.19466) (no position embedding) and [RNoPE](https://arxiv.org/abs/2501.18795), a hybrid method, have emerged. NoPE uses only casual masking and attention patterns, so it doesn't bump into the issue of extrapolating beyond training lengths but shows weaker performance on short context reasoning and knowledge-based tasks. RNoPE alternates applying RoPE and NoPE on attention blocks, where RoPE handles local context and NoPE helps with longer-range information retrieval. Another idea is Partial RoPE, which applies RoPE/NoPE within the same layer. [TODO: consider adding partial ROPE with MLA]

HF ran ablations using RoPE, RNoPE (removing positional encoding every 4th layer), and RNoPE with document masking. They found that all achieve similar performance on short-context tasks, so they adopt RNoPE + document masking because it provides the foundation for long-context handling.