---
title: "training at scale"
date: 2025-12-31
---

How do labs train a multi-billion parameter model? We look towards Hugging Face's SmolLM3, Allen Institute's Olmo 3, Prime Intellect's Intellect 3, and OpenAI's GPT-OSS-120B. This blog is an attempt towards distilling the motivations, considerations, and techniques used to train their models and is structured in more of a "notes" style.

These notes are largely structured off of Hugging Face's [SmolLM3 report](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#math-data) due to it's extensiveness, and it is supplemented with notes from other reports. Also, these notes have not been thoroughly reviewed. Any errors below are my own responsibility.

## (extremely broad) general practices

1. [HF] "**Learn to identify what's worth testing, not just how to run tests.** Perfect ablations on irrelevant choices waste as much compute as sloppy ablations on important ones." 
    - Ablations need to be **fast** (faster iteration $\rightarrow$ more hypotheses tested) and **reliable** (need strong discriminative power because otherwise, it may be noise)
    - "“The real value of a solid ablation setup goes beyond just building a good model. When things inevitably go wrong during our main training run (and they will, no matter how much we prepare), we want to be confident in every decision we made and quickly identify which components weren’t properly tested and could be causing the issues. This preparation saves debugging time and keeps our sanity intact. There’s nothing worse than staring at a mysterious training failure with no idea where the bug could be hiding.”
2. [HF] **Choose an established baseline with good architecture and training setup design**. These take years of iteration, and people have discovered common failure modes and instabilities.
    - There are a plethora of modifiable components (attention mechanisms and positional encodings to name a few), but follow the principle of **derisking**: "never change anything unless you've tested that it helps."
3. [HF] **In evals, look for monotonicity** (score improvement), **low noise** (e.g. score resistance to random seeds), above-random performance (random-level performance for extended time frames isn't useful), and ranking consistency (ranking of approaches should remain stable throughout training). 
4. [HF] **Balance exploration and execution.** For methods, choose flexibility and stability over peak performance, set a deadline for exploration.

# architecture and set-up

Model families like DeepSeek, MiniMax, Kimi, OLMo, and SmolLM have vastly different architectures (dense vs MoE), attention mechanisms (MHA vs MLA vs GQA), position encodings (RoPE, partial RoPE, NoPE), among many, many, others.

| | DeepSeek | MiniMax | Kimi | OLMo | SmolLM |
|--|----------|---------|-------|------|--------|
| Attention | X | X | X | X | GQA (4 groups)|
| Embedding Sharing | X | X | X | X | tied|
| Positional Embedding| X | X | X | X | RNoPE|
| Z-Loss | X | X | X | X | No|
| Architecture | X | X | X | X | dense|
| Tokenizer | X | X | X | X | Llama3|
| Optimizer | X | X | X | X | AdamW|
| Scheduler| X | X | X | X | WSD (10%)|
| Learning Rate| X | X | X | X | 2e-4|

Between choosing architecture, HF suggests following a decision tree such that if one of these is true, then to choose a dense architecture:
- memory-constrained (since MoEs must have all experts loaded)
- new to LLM training (focus on basics)
- tighter timeline (simpler training with well-documented recipes)

## attention

To address the large KV-cache (an inference bottleneck and GPU memory hoarder) associated with MHA, researchers developed multi-query attention (MQA) and grouped query attention (GQA). In MQA, KV values are shared across all heads, but this comes at a cost of leaking attention capacity because heads can't store information specialized for that head's role. GQA softens this issue by sharing KV values across a small group of heads (e.g. 4). Another alternative is multi-latent attention (MLA) which stores a latent variable that can be decompressed/projected into KV values at runtime. This results in a KV-cache parameter count more comparable to GQA and performance stronger than MQA. 

When ablating (for variables that change the parameter count such as changing MHA to GQA, they occasionally adjust other hyperparameters to keep model sizes roughly the same), HF found that **GQA with small groups beats MHA** and that **MHA beats MQA and GPQ with 16 groups**. Across benchmarks like HellaSwag, MMLU, and ARC, GQA with 2/4/8 groups do best.

[TODO: add paper links to associated mechanism]

## document masking

When pre-training, a common consideration is **fixed sequence lengths** since training uses tensors of the form [batch, sequence length, hidden], so with regards to batching and distributed training, GPUs are most happy when every example has the same sequence length. But due to variable document length and wanting to avoid padding which wastes compute, **packing** enables shuffling and concatenating documents within the same sequence to achieve the sequence length. 

Causal masking means that for unrelated files $A$ and $B$ in the same batch, the tokens in $B$ can attend to the tokens in $A$, which degrades performance. With **intra-document masking**, the attention mask is modified so tokens can only attend to previous tokens within the same document. Many papers have found benefits relating to [long-context extension](https://arxiv.org/abs/2407.21783) and [some short context benchmarks](https://arxiv.org/abs/2410.02660) as well as [shortening the average context length](https://arxiv.org/abs/2503.15450).

When implementing document masking, HF saw small improvements on PIQA but otherwise no noticeable impact on short context tasks. But in line with aforementioned research, they observed that it became crucial for scaling from 4k to 64k tokens.

## embedding sharing

Input embeddings (token-to-vector lookup) and output embeddings (hidden states to vocab logits) are typically represented as separate matrices, so the total embedding parameters are $2 \times \text{vocab size} \times \text{hidden dim}$. In small language models can account up to 20% of total parameters, as is the case with `Llama 3.2 1B` (in larger models, the embeddings represent a much smaller fraction of the parameter count, only 3% in `Llama 3.1 70B`). The issue with tying them is that input/output embeddings still represent different geometries, and frequent tokens like "the" can dominate representation learning due to getting gradients from both the input stream and the predicted output. 

HF found that on a 1.2B model, tied embeddings did comparably well despite having 18% fewer parameters (down from 1.46B), and that compared to an untied model also with 1.2B models (fewer layers), untied showed higher loss and lower downstream eval scores.

## positional encodings

Without positional encoding, transformers have no sense of word order, akin to the bag of words idea. Initially, [absolute position embeddings](https://arxiv.org/abs/1706.03762) were used by learning a lookup table that mapped the position index to a vector added to token embeddings, but the max input sequence length was limited by the max input sequence length of what it was trained on. **Relative position encodings** followed since capturing distance between tokens matters more than capturing their absolute positions.

The most commonly used technique is [rotary position embedding (RoPE)](https://arxiv.org/abs/2104.09864), which encodes relative position as rotation angles. Based on the dimensionality of the query/key vector, RoPE splits it into pairs (since they rotate in 2D space) and rotates depending on the absolute position of a token and a base frequency. During attention, the dot product between their rotated positions directly encodes their relative distance via the phase difference in their rotation angles, where tokens $x$ positions apart always maintain the same angular relationship. 

During pretraining, models are trained on shorter context lengths (similar ideas to document masking, and quadratic attention is expensive) to learn short range correlation between words. But as sequence length grows, the rotation angles grows via $\theta= \text{position} \times \frac1{\text{base}^{\frac{k}{\text{dim}/2}}}$. This can be fixed by increasing the base frequency as the sequence length increases using methods like [ABF](https://arxiv.org/abs/2309.16039) or [YaRN](https://arxiv.org/abs/2309.00071), which applies a more granular interpolation of frequencies on different components and includes other techniques like dynamic attention scaling and temperature adjustment. For extremely long contexts, YaRN does best.

More recently, with the emphasis on long contexts, [NoPE](https://arxiv.org/abs/2305.19466) (no position embedding) and [RNoPE](https://arxiv.org/abs/2501.18795), a hybrid method, have emerged. NoPE uses only causal masking and attention patterns, so it doesn't bump into the issue of extrapolating beyond training lengths but shows weaker performance on short context reasoning and knowledge-based tasks. RNoPE alternates applying RoPE and NoPE on attention blocks, where RoPE handles local context and NoPE helps with longer-range information retrieval. Another idea is Partial RoPE, which applies RoPE/NoPE within the same layer. [TODO: consider adding partial ROPE with MLA]

HF ran ablations using RoPE, RNoPE (removing positional encoding every 4th layer), and RNoPE with document masking. They found that all achieve similar performance on short-context tasks, so they adopt RNoPE + document masking because it provides the foundation for long-context handling.

## attention for long contexts

An alternative to adjusting positional encodings for long contexts is specifying the strength of which tokens can attend to one another. 
- **Chunked Attention**: divides the sequence into fixed-sized chunks where tokens can only attend within their chunk. [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) paired with RNoPE (specifically the RoPE layers) which also reduces the KV cache size per layer, but it's performance on long context tasks would degrade.
- **Sliding Window Attention (SWA)**: every token can see up to $p$ positions back, creating a sliding window that maintains local context. Gemma 3 combined SWA with full attention every other layer.
- **Dual Chunk Attention (DCA)**: $K$ tokens are chunked into $M$ groups. Within each group (like chunked attention), tokens attend normally. Between successive chunks have their own local window to preserve locality, and more broadly, inter-chunk attention allows queries to attend to previous chunks with a capped relative position cap. Qwen-2.5 used DCA to support context windows of up to 1 million tokens.

[TODO: add image from HF of attention patterns]

## MoE

MoEs (mixture of experts), analogous to our brain activating different parts of our brain, provide an alternative to dense models due to only certain "experts" being used at inference time, saving lots of compute. The MoE works by replacing the feed forward layer with multiple MLPs (experts) and add a learnable router before the MLPs to select the experts.

In general, for fixed number and size of active experts, increasing the total number of experts improves loss, and [high sparsity improves performance](https://arxiv.org/abs/2507.20534) and [benefits more from increasing compute](https://arxiv.org/abs/2507.17702). Recent models are much more sparse, with over 100 experts and around 10 active per token. [TODO: add image]

To determine how large each expert should be, a common metric is granularity, defined by $G = 2 \cdot \frac{d_\text{model}}{d_\text{expert}}$, where a higher granularity corresponds to more experts with a smaller dimension; this can be intermediate as a number proportional to the experts needed to match the dense MLP width. Recent models have granularity anywhere from 2 (`gpt-oss-120b`) to 8 (`qwen3-next-80b-a3b`). [Ant Group](https://arxiv.org/pdf/2507.17702) showed that granularity doesn't significantly change loss but does drive **efficiency leverage** (the ratio of flops needed for an MoE to achieve the same loss as a dense model). And overall, MoEs present a good alternative to dense models in terms of compute for training and inference.

**Shared experts** are always-on experts, which absorb the basic, recurring patterns so that other experts can more aggressively specialize; one is often enough (`deepseek-v2` uses two, which adds a bit of complexity).

**Load balancing** is crucial in that if it fails, not only do training and inference efficiency plummet, but so do effective learning capacity. This can be addressed by adding a **loss-based load balancer** (LBL) given by $\mathcal{L} = \alpha \sum_{i=1}^{N_r} f_i P_i$ where $\alpha$ determines the strength, $f_i$ is the fraction of tokens going through expert $i$, and $P_i$ is the probability mass that sums the probability of tokens going through an expert; so in perfect load balancing, $f_i=P_i=\frac1{N_r}$. Also, $\alpha$ should not be so large that routing uniformity overwhelms the primary training objective. These should be monitored using *global statistics*, not local statistics which may suffer from a local batch being narrow, biasing the routing statistics. 

`deepseek-v3` does loss free load balancing differently, by adding a bias term that is added to affinity scores going into the routing softmax.

## hybrid models

Because transformers don't deal efficiently with long context while RNNs can, one idea is to combine both to get the best of both worlds. By dropping the softmax from the output for token $t$:
$$\mathbf{o}_t = \sum_{j=1}^t \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)\mathbf{v}_j}{\sum_{l=1}^t \exp(\mathbf{q}_t^\top \mathbf{k}_l)} \Longrightarrow \mathbf{o}_t = \sum_{j=1}^t (\mathbf{q}_t^\top \mathbf{k}_j)\mathbf{v}_j = \left(\sum_{j=1}^t \mathbf{v}_j \mathbf{k}_jk^\top\right)\mathbf{q}_t$$
And by defining $S_t :=\sum_{j=1}^t \mathbf{k}_j \mathbf{v}_j^\top$, then we get a recurrent relation where $S_t$ summarizes all past $(k_j, v_j)$. 
$$S_t=S_{t-1}+\mathbf{k}_j \mathbf{v}_j^\top \Longrightarrow \mathbf{o}_t = S_t \mathbf{q}_t = S_{t-1}\mathbf{q}_t+\mathbf{v}_k\left(\mathbf{k}_j \mathbf{v}_j^\top\right)$$

While this gets us closer to an RNN-esque structure, in practice, softmax stabilizes training, and the linear form can cause instability without normalization. With RNNs, it is sometimes helpful to forget the past, via introducing a gate $\mathbf{G}_t$ for the previous state 
$$\mathbf{S}_t=\mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{v}_t\mathbf{k}_t^\top$$
[Mamba-2](https://arxiv.org/abs/2405.21060) is among the most popular, being used in hybrid models like [Nemotron-H](https://arxiv.org/abs/2504.03624) and [Falcon H1](https://arxiv.org/abs/2507.22448). Hybrid models are becoming increasingly popular, notably in [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) with a gated DeltaNet update and Kimi's next model, likely using their ["kimi delta attention.]"(https://github.com/fla-org/flash-linear-attention/pull/621)

# stability

## $z$-loss

$z$-loss is a regularization term added to the standard cross entropy loss that keeps logits from drifting to large magnitudes. By adding $\mathcal{L}_{\text{z-loss}} = \lambda \cdot \log^2(Z) = \lambda \sum_{i=1}^V e^{z_i}$, representing the denominator in the softmax the loss now penalizes based on $\log(Z)$ which represents the overall logit scale. 

On their 1B model, HF found that adding $Z$-loss didn't impact training loss or downstream performance, so they chose not to include it due to training overhead.

## removing weight decay from embeddings

Despite being a regularization technique, weight decay being removed from embeddings can improve training stability. Weight decay causes embedding norm to decrease, but this can lead to larger gradients in earlier layers since the LayerNorm Jacobian has a $\frac1{\sigma}$ term (coming from normalization) which is inversely proportional to the input norm $\sigma$.

HF tested this using a weight decay baseline, a no weight decay baseline, and another combining all previous adopted changes and found no significant loss or eval results, so they included no weight decay.

## qk norm

Similar to $z$-loss, QK-norm helps prevent attention logits from becoming too large by applying LayerNorm to both the query and key vectors before computing attention. However, [the same paper which proposed RNoPE](https://arxiv.org/abs/2501.18795) found that it hurts long-context tasks because the normalization demphasizes relevant tokens and emphasizes irrelevant tokens by stripping the query-key dot product of its magnitude.

## other design considerations

1. **Parameter initialization**: either normalization initialization ($\mu=0$, $\sigma=0.02, 0.006$) with clipping (often with $\pm 2-3 \sigma$) or a scheme like $\mu\text{P}$ ([maximal update parametrization](https://arxiv.org/abs/2011.14522)) which dictates how weights and learning rates should scale with width so that training dynamics stay comparable.
2. **Activation Function**: SwiGLU is what most modern LLMs use, not ReLU or GeLU. Some exceptions are Gemma2 using GeGLU and nvidia using $\text{relu}^2$. 
3. **Width vs Height**: deeper models tend to outperform equally sized wider ones on language modeling and compositional tasks. In smaller models, this is more pronounced, but larger models make use of wider models for faster inference due to modern architectures supporting better parallelism. 

# tokenizer

There are a few considerations that typically guide tokenizer design:
1. **domains**: in domains like math and code, digits and other special characters require careful treatment. Most tokenizers do single-digit splitting, which helps with arithmetic patterns more effectively and prevents memorization of numbers. Some tokenizers like [Llama3](https://arxiv.org/abs/2407.21783) further encode numbers 1 to 999 as unique tokens.
2. **supported languages**: a tokenizer trained on english text would be extremely inefficient if it encountered another language, say mandarin or farsi. 
3. **target data mixture**: when training a tokenizer from scratch, we should train on samples that mirror our final training mixture.

Larger vocabularies can compress text more efficiently, but they come at the cost of a larger embedding matrix, which as mentioned in the embeddings section, can take up a sizable portion of the parameter count. For english-only models, 50k is often enough, while multilingual models need over 100k. There is an optimal size that exists since [compression gains from larger vocabularies decrease exponentially](https://arxiv.org/abs/2402.01035).

Large models benefit from large vocabularies since the extra compression saves more on the forward pass (project to QKV, attention, and MLP) than the additional embedding tokens during softmax. For memory, larger vocab means fewer tokens, so a smaller KV cache.

**BPE** ([byte-pair encoding](https://arxiv.org/abs/1508.07909)) still remains the de facto choice. Starting with tiny units (e.g. characters or bytes), the BPE algorithm repeatedly merge the most common adjacent pair into a new token. To evaluate a tokenizer's performance, **fertility** is a common metric, measuring the average number of tokens needed to encode a word (alternatively, characters-to-tokens ratio or bytes-to-tokens ratio, but these have limitations due to word length variability and byte representations). Another is **proportion of continued words**, describing what percentage of words get split into multiple pieces. For both, smaller metrics indicate more efficient tokenizers.

There are many strong existing tokenizers, like [GPT4's tokenizer](https://arxiv.org/abs/2303.08774) and Gemma3's tokenizer. Often, using existing tokenizers is enough; only when we want to train for low-resource languages or have a different data mixture should we continue training our own tokenizer.

# optimizers and training hyperparameters

Choosing optimizers and tuning hyperparameters is notoriously time-consuming. While we may be tempted to distill those from models of larger labs (albeit a useful prior), it may not fit the use case.

## adamW

Despite being invented over 10 years agao, AdamW still stands the test of time. Adam (adaptive momentum estimation) updates weights individually based on an exponential weighted average of gradients $g_t$ and an exponential weighted average of squared gradients $g_t^2$, along with weight decay (the "W"): 
$$\begin{align*}
\theta &\leftarrow (1-\alpha \lambda)\theta - \alpha \frac{\hat{m}_t}{\sqrt{v_t}+\epsilon} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t}, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\end{align*}$$
Even for modern LLMs, the hyperparameters remain largely unchanged: weight decay factor $\lambda=0.1$ or $\lambda=0.01$, $\beta_1=0.9$, and $\beta_2=0.95$. 

## muon

Unlike adamW which updates per-parameter, muon treats the weight matrix as a singular object and updates based on:
$$
\begin{align*}
g_t &\leftarrow \nabla_\theta \mathcal{L}_t(\theta_{t-1}) \\
B_t &\leftarrow \mu B_{t-1} + G_t \\
O_t &\leftarrow \text{NewtonSchulz5}(B_t) \\
\theta_t &\leftarrow \theta_{t-1} - \eta O_t
\end{align*}
$$
where $B_0=0$, and NewtonSchulz5 describes the odd function $f(x)=3.4445x-4.7750x^3+2.0315x^5$. [This blog](https://docs.modula.systems/algorithms/newton-schulz/) describes the algebra of it in more detail, but we can estimate the SVD decompositions of $G=U \Sigma V^\top$ by $UV^\top$, and $f(x)$ essentially replaces $\Sigma$ because $f \circ f \circ \cdots f(x)$ converges to the sign function. This has the effect of reducing axis-aligned bias and encouraging exploration fo directions that would otherwise be suppressed. Also, muon can tolerate higher batch sizes.

## learning rates 

Learning rates have their own life cycle: they warmup (typically 1%-5% of training steps for short trainings, but large labs fix the warmup steps) from zero to avoid chaos, then anneal after settling into a good minimum. [Cosine annealing](https://arxiv.org/abs/1608.03983) was the go-to scheduler, but it's also inflexible due to the cosine period needing to match the total training duration. Alternatives include [warmup-stable-decay (WSD)](https://arxiv.org/abs/2404.06395) and [multi-step](https://arxiv.org/abs/2401.02954); in the last x% of tokens, the former linearly decays the learning rate whereas multi-step does discrete drops. [TODO: include image]. for WSD, typically 10-20% is allocated for the decay phase, matching cosine annealing; in multi-step, 80/10/10 also matches cosine annealing while 70/15/15 and 60/20/20 can outperform it. Deepseek-v3 used cosine annealing between the decay drops and added a constant phase before the final sharp step.

HF's ablations (on their 1B model) showed that WSD tended to underperform cosine annealing before WSD's decay began, but once it entered its decay phase, WSD showed nearly linear improvement in both loss and eval metrics, which allowed it to catch up to cosine annealing by the end. After running further ablations on the learning rate, the HF team settled on 2e-4; increasing led to potential increased risk of instability during long training runs.

WSD schedule especially helps with ablations sicne it does not research restarting the same run for different token counts, since we can retrain only the end portions (learning rate decay) while maintaining the front portion.

## batch size

There is a [critical batch size](https://arxiv.org/abs/1812.06162): too small and we may be underutilizing compute, but too large we the model needs more tokens to reach the same loss. Still, larger batch sizes given more efficient gradient estimations, and are preferred. 

A useful proxy is that for optimizers like AdamW or Muon, if the batch size squares up by $k$ then the learning rate should scale up by $\sqrt{k}$. This is because the covariance stricts by a factor of $k$, and based on the SGD parameter update $\Delta w = -\eta g_B$ , so $\text{Var}(\Delta w) \sim \eta^2 \frac{\Sigma}{B}$  where $B$ is the original batch size, so $\eta \sim \sqrt{k}$. 

As training progresses, the critical batch size grows. Initially, since the model is making large updates, $||g||^2$ is large so the model should have a small critical batch size. After the model stabilizes, larger batches become more effective. This motivates the idea of *batch size warmup*. 

## scaling laws

Scaling laws (e.g. [Chincilla scaling laws](https://arxiv.org/abs/2203.15556)) provide a useful proxy for determining how aggressively/conservatively to update hyperparameters as model size scales. 

First, $C \approx 6 \cdot N \cdot D$ where $C$ is the compute budget measured in FLOPs, N is the number of parameters, and $D$ is the number of training tokens. The 6 is dervied from empirical estimates for the number of FLOPs per parameter.

[TODO: add image of Deepseek]

Initially, [scaling laws](https://arxiv.org/abs/2001.08361) indicates that language model size was the main constraint, leading to a GPT-3 model with 175B parameters but only trained on 300B tokens. A [re-derivation](https://arxiv.org/abs/2203.15556) found that training duration could improve gains more than size; they found that compute-optimal training of GPT-3 should have consumed 3.7T tokens.

However, scaling laws are almost always never religiously followed. Recently, labs have been "overtraining" models beyond the training durations uggested by scaling laws (e.g. Qwen 3 being trained on 36T tokens).  Moreover, "compute-optimal" scaling laws don't account for larger models being more expensive after training due to inference. To that tend, HF decided to train of 11T tokens on a 3B model.

# data curation

Even with the perfect architecture, a model's performance is still heavily dependent on its training data; no amount of compute or optimization can compensate for training on the wrong content. To this end, it's about assembling the right **data mixture**, balancing training objectives and tuning data proportions. This is particularly difficult since across competiting objectives, for a fixed compute budget, increasing one proportion necessarily decreases another, hurting performance.

There already exist large corpa of pre-training datasets like [FineWeb2](https://arxiv.org/abs/2506.20920) and [The Pile](https://pile.eleuther.ai/). However, there are still a plethera of information gaps, so recent models additionally rely on specialized pre-trainind datasets for domains like math and coding. 

One consideration in **data quality**. Of course, training of the highest quality data possible is preferrable. But for a training budget of $X$ tokens, because high quality data is limited, only filtering for it would lead to repeated data, which [can be harmful](https://arxiv.org/abs/2305.16264). So, an ideal mixture includes both higher and lower quality data.

## multi-stage training

[Multi-stage training](https://arxiv.org/abs/2502.02737), the idea of evolving the data mixture as training progresses, can better maximize both high-quality and lower-quality data compared to a static mixture because a LM's final behavior is heavily dictated by the [data it sees at the end of training](https://arxiv.org/abs/2410.08527). So, this motivates the strategy of saving the higher quality data towards the end. This introduces another variable of when to begin changing mixtures, and a general principle to **performance-driven intervention**: if a benchmark begins to plateau, it's a signal to introduce high-quality data for that domain.

## ablation 

While architectural ablations are on with smaller models (e.g. on 1B models to train for 3B models), data mixtures is done at scale because Larger models have much larger capacities to understand a variety of domains. Moreover, **annealing ablations** are done on checkpoints of the main run (like 7T out of 11T tokens) to determine what datasets to introduce when. 

To determine optimal data proportions, recent models often use a validation loss or a holdout loss to minimize based on evaluation objectives and data domains. However, some of these methods tend to converge toward distributions that mirror the dataset size distribution, and they don't outperform careful manual ablations.

## data

### hugging face
HuggingFace's goal was to build a multi-lingual model that also excels on math and coding. In stage 1 of their multi-stage training, they use a 75/12/10/3 split among english web data, multilingual web data, code data, and math data.
    
- **English web data**: they ablate on a mixture of FineWeb-Edu (educational and STEm benchmarks) and DCLM (common sense reasoning), two strong open ENlgish web datasets at the time of training, finding that a 60/40 or a 50/50 split was best. Later, they add in other datasets including [Pes2o](https://huggingface.co/datasets/allenai/dolmino-mix-1124/tree/main/data/pes2o), [Wikipedia & Wikibooks](https://huggingface.co/datasets/allenai/dolmino-mix-1124/tree/main/data/wiki), and [StackExchange](https://huggingface.co/datasets/HuggingFaceTB/stackexchange_2025_md). 
- **Multilingual web data**: five European languages were chosen, with data from FineWeb2-HQ. Smaller portions of other languages, like Chinese or Arabic, were chosen to allow others to do continual pretraining of SmolLM3. Ultimately, they foudn that 12% multilingua lcontent in the web mix was best.
- **Code data**: primarily extracted from [The Stack v2 and StarCoder2](https://arxiv.org/abs/2402.19173), it includes 16 languages, Github PRs, Jupyter/Kaggle notebooks, Github issues, and StackExchange threads. Despite research showing that code improves LM performance beyond coding, they did not observe this effect (rather a degradation on English benchmarks) using the recommended code mixure. They delay adding their eductionally filtered subset, Stack-Edu, following the principle of delaying the best data until the end.
- **Math data**: using FineMath3+, InfiWebMath3+, [MegaMath](https://arxiv.org/abs/2504.02807), and instruction/reasoning datasets like [OpenMathInstruct](https://arxiv.org/abs/2402.10176) and [OpenMathReasoning](https://arxiv.org/abs/2504.16891).

For new stages (using a checkoint at around 7T out of the total 11T tokens), they use a 40/60 split between the baseline mixture and the new dataset. 

# the training marathon

Before the main training run starts, ensure the infrastructure is ready. This includes **Slurm reversations on clusters**, **stress-testing GPUs** ([GPU Fryer](https://github.com/huggingface/gpu-fryer) or [DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)), and **avoid storage bloat** by upiloading checkpoints to third parties and deleting local copies after saving the next. To this end, **checkpoint and auto-resume systems** are important.

**Evals** are also deceptively time-confusing (Allan Institute spent roughly 20% on compute on evals), so ensuring automation and logging (not just evaluation scores, but also throughput, loss, gradient nomr, and node health) is crucial. 

## resolving mysteries

### vanishing throughout 

HF observed a ~40% drop in throughout (14k to 8k tokens/sec/GPU) after a few hours of starting the main run. The issue came from data storage; their cluster uses a network-attached storage with a "keep-hot" caching model that stores frequently accessed files and evits "cold" files to third-party S3. With 24TB of training data, the storage was pushed to its limit, so it evicted dataset shards mid-training. This meant fetching them back and creating stalls that slowed throughout. 

The first fix can in the form of swapping the storage method by reserving a spare node with the dataset preloaded and copying using `fpsync` (`s5cmd` took double the time). This fixed the issue of a node dying and the replacement GPU having no data since by swapping it with the spare node, training could continue. So, the new spare, not to be wasted, could run evals or dev jobs.

Testing again, they found smaller by still prominent drops in throughput. After experimenting with individual nodes that yeilded the same result, they focused on the change in training steps and found that smaller step counts resulted in smaller throughput drops. The`nanotron` dataloader they were using was growing the lookup table making the training step to the next chunk of tokens to read instead of keeping it bounded or precomputed. Stored in global memory, the growing table causes allocation failures and page faults/worse cache locality. So, they switched to `Tokenizedbytes` dataloader, solving the throughout issue

### noisy loss

However, the loss curve looked more noisy. They found the issue with the dataloader because it reads sequences sequentially for each document. Without **shuffling of sequences**, batches are no longer representative of the overall data distribution, increasing gradient variance. Also, a long file (e.g. code) would supply many conseuctive sequences that would also spike loss. To fix, they reshuffled the tokenized sequences offline; an alternative was changing the dataloader to do random access, which has both higher memory usage and slower runtime.

### unsatisfactory performance

After two days and 1T tokens, evals showed that with a similar recipe, SmolLM2 (1.7B) was more performant at the same stage in training as SmolLM3 was. The team found the issue with **tensor parallelism**: the weights of SmolLM2 fit on a single GPU, whereas for SmolLM3, they had to be shared across 2 GPUs.

Further, the two TP ranks were initialised with the same random seed instead of different seeds, which causes similar activations/gradients, a loss of diversity of features, and lower convergence.

## staying on course

