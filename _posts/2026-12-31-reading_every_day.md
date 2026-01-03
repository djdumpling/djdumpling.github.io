---
title: "a paper a day keeps the rust away"
date: 2026-12-31
ongoing: true
tokens: "~4.9k"
reading_time: 15
---

Every day in 2026, I'll read an ML paper/blog. This is my running log of distillations, meant to be a learning archive for myself as well as a capsule of how the field evolves across 2026.

# 1/1: manifold-constrained hyper-connections

From the DeepSeek team, [this paper](https://arxiv.org/pdf/2512.24880) explores how to enforce the identity mapping property intrinsic to residual connection (which otherwise causes training instability and restricted scalability) to hyper-connections via **manifold-constrained hyper-connections**. The structure of a single layer is

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)
$$

where $\mathbf{x}\_{l}, \mathbf{x}\_{l+1}$ are the input/output of the $l^\text{th}$ layer, respectively, and $\mathcal{F}$ denotes the residual function. **Hyper-connections**, popularized by [Zhu et al., 2024](https://arxiv.org/abs/2409.19606), added learnable mappings that instead of carrying $\mathbf{x}\_l$, carries a bundle of $n$ parallel residual streams defined by

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

<div style="display: flex; justify-content: center;">
  <img src="/public/reading/mHC.png" alt="Illustrations of residual connection paradigms" style="width:80%; display: block; margin: 0 auto;" />
</div>

**Manifold-constrained hyper-connections** restore the identity mapping property by using the Sinkhorn-Knopp algorithm to project $\mathcal{H}\_l^\text{res}$ onto the Birkhoff polytope, the set of all doubly stochastic matrices. These matrices have the property that the sum of all rows and columns equals 1, meaning that the matrix acts more like averaging rather than scaling since the spectral norm is less than 1. Due to closure of matrix multiplication for doubly stochastic matrices, then $\prod\_{i=1}^{L-l} \mathcal{H}\_{L-i}^\text{res}$ is still doubly stochastic, meaning that the transformer maintains the stability of identity mappings. If the dimension is 1, then the doubly stochastic matrix is exactly the identity matrix. The HC formulae can be reformulated to obtain

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

To cut memory, they don’t store most intermediate activations from the mHC kernels during forward. In the backward pass, they recompute those intermediates on demand, using a block strategy where only the input to the first layer of each block is saved; they pick a block size $L_r$ to balance “saved memory vs extra recompute,” with an approximate optimum. Also, mHC increases communication and can add pipeline “bubbles,” so they modify the schedule to overlap boundary communication with high-priority compute (notably the post/res kernels), avoid long “persistent” attention kernels that block scheduling, and rely on cached block-start activations so recomputation doesn’t stall on communication. [TODO: look over this more deeply]

---
Across many benchmarks like GSM8k, HellaSwag, and MMLU, 27B with mHC outperforms both the baseline and the 27B with HC. Moreover, mHC is very effective in large-scale scenarios, since there is minimum degradation in terms of loss with regards to compute and token count. As suggested by Sinkhorn-Knopp, mHC significantly enhanced propagation stability compared to HC by three orders of magnitude, from a maximum gain of nearly 3000 to approximately 1.6.

# 1/2: recursive language models

From the Prime Intellect team, [this blog](https://www.primeintellect.ai/blog/rlm) advocates for **recursive language models** (RLMs) as the paradigm of 2026 due to its context management ability, an important characteristic as long context grows in importance. **Context rot** is the phenomenon that LLM capabilities are reduced as context grows in size, so current TUI systems use scaffolding like file systems and context compression via LLM summarization at regular intervals. Another approach is **context folding**, whose goal is to create a continual, growing rollout while manaing its own context window. One promising method, which the Prime team believes to be the simplest and most flexible, are [RLMs](https://arxiv.org/abs/2512.24601).

The RLM allows an LLM to use a persistent Python REPL to inspect and transform its input data, and to call sub-LLMs from within that Python REPL. This avoid the issue of ingesting potentially huge input data like PDFs, datasets, or videos, because instead of loading them into the model's context, it can be called through the REPL, and it allows LLM to search, filter, and transform the context usign Python functionality. Also, the ability to spawn sub-LLMs with specified input data to perform work also helps in situations which require large context sizes.

<div style="display: flex; justify-content: center;">
  <img src="/public/reading/rlm.png" alt="RLM architecture showing model context, Python REPL, and parallel sub-LLMs" style="width:80%; display: block; margin: 0 auto;" />
</div>

Prime has already incorporated RLMs into $\mathtt{verifiers}$ and $\mathtt{prime-rl}$ as well as created several RLM-based environments on their Environments Hub. There are a number of details involved:
1. **sub-LLM calls can be parallelized** via an `llm_batch` function
2. **any tools given by the environment are usable only by the sub-LLMs**. This allows the main RLM to delegate the work that requires tools instead of seeing all of the tool-outputed tokens.
3. **the RLM is made aware of installed packages**, and any pip package an be installed. Code execution occurs in isolated sandboxes.
4. **the RLM provides an answer in a Python variable**. Particularly, an `answer` dictionary is instantiated by `answer = {"content": "", ready: False}` where `content` is a "scratchpad" which the LLM can edit/delete over multiple turns, and `ready` is a boolean indicating the rollout end and if the answer can be extracted from `content`.

Additionally, a prompt and extra input data can be given. The prompt is injected into the RLM's context window, whereas the input data can be accessed via the REPL.

---

They ablate the RLM on four environments using `GPT-5-mini` (initial tests show it's better than open-source models), comparing a standard LLM, the RLM, and the RLM with environment-specific tips:

| Environment | Description | Why Chosen |  |
|-------------|-------------|------------|-------|
| **DeepDive** | Deep Research tasks requiring web search and knowledge graph traversal | long-horizon reasoning and information gathering | Models have search tools; dataset created by walking open knowledge graphs to create complex questions with verifiable answers, then obfuscating via LLM reformulation |
| **math-python** | Mathematical problem solving with code | reasoning + code execution | `numpy`, `scipy`, and `sympy` pre-installed |
| **Oolong** | Document understanding and extraction tasks | context management on large input documents | real dataset comes from real D&D playing sessions |
| **verbatim-copy** | Exact reproduction of input content | faithful context preservation | tests different content types (e.g. json and csv) with `target_length` and mixing across fragments|

`_ENV_TIPS` include specifiying consistent facts acorss sources in DeepDive, specifiying chunked context and prompt writing in Oolong, and specifiying writing into and printing from `answer["content"]` to iterate in verbatim-copy.

The RLM scaffold improves reward on Oolong and verbatim-copy, but not on DeepDive without tips and MathPython. The former can be explained by the poor usage of scaffolding, and the tips of splitting the question into smaller problems solvable by sub-LLMs helps significantly.The latter can be explained since the RLM allows for the same behavior for solving the environment as the LLM (access to Python tools and same libraries), so the RLM is overfitting to the benchmark using a stanfard Python tool. Both of these could be solved by training. Time spent for RLM (+tips) increases across the board. This can be attibuted to an increase total token budget, inefficient thinking, and extra tool-calling. 

They also found that the main model context length is much lower when using sub-LLMs, especially in DeepDive. As a consequence, the main model token efficiency for DeepDive when using RLM (+tips)is nearly 4.5-6x that of just an LLM. In Oolong, results doesn't show the expected significant increase due to API rejections of long Oolong prompts, and they are only seen by the RLM. math-python and verbatim-copy's token efficiency are hurt since the former causes the RLM to think inefficiently and poorly, where as in the latter, the RLM has to call a tool give it's response whereas the LLM one-shots the answer. On the other hand, in Needle in Haystack, the total number of tokens is reduced a lot for the RLM. However, while the LLM's tokens are almost all spent on prefilling, the RLM's is spent decoding via writing code and providing its answer.

<div style="display: flex; justify-content: center;">
  <img src="/public/reading/rlm_gpt_5_mini.png" alt="GPT-5-mini RLM ablation results across environments" style="width:80%; display: block; margin: 0 auto;" />
</div>

Prime hypothesizes that the true potential of RLMs will be unlocked after RL training, which they plan to do, starting with small models. Some ideas they have on improving implementation include:
1. **mutable recursive depth**: currently, depth is fixed at 1. It should be possible to have depth=0 (just like a normal LLM with a Python REPL with input data and access to user-added tools; this would help with math-python, for example), or depth>1 so that sub-LLMs can further call sub-LLMs.
2. **customization**: faciliating users defining custom functions for the RLM to use in its REPL as well as adding descriptions for packages without re-writing the entire prompt.
3. **context compression multiple assistant-user turns** should be a natural part of the RLM
4. **multi-modal support** and custom data-types

# 1/3: bloom, tooling for automated behavioral evaluations

From the Anthropic Alignment team, [this blog](https://alignment.anthropic.com/2025/bloom-auto-evals/) open sources **Bloom**, an open-source, agentic tool for automated behaviorial evaluations of researcher-specified traits, quantifying their severity and frequency across automatically generated scenarios.

Bloom generates scenarios depending on the seed, a configurable file including information about the model, behavior description, example transcripts, and more. The researcher specifies the exact behavior and interaction type to be investigated, and using the seed, Bloom generates scenarios, to be human checked, to ensure they capture the intended behavior; this stage involves iteration on configuration options and agent prompts. Once all configurable options are fixed, the researcher can run large-scale evals.

<div style="display: flex; justify-content: center;">
  <img src="/public/reading/bloom.png" alt="Bloom is a four-stage automated pipeline that generates behavioral evaluations from a user-provided seed." style="width:80%; display: block; margin: 0 auto;" />
</div>

More specifically, within Bloom, there are four stages:
1. **Understanding**: based on the seed (specifically, the behavior description and possibly few-shot example transcripts), the agent digests the targeted behavior, how it manifests, why it matters, as well as summaries. This context is reused frequently to align the agent.
	- Although this is more about of the seed configuration, another parameter is whether the evaluator knows the target model's identity, which is useful for measuring preferential bias.
2. **Ideation**: the agent generates scenarios designed to illicit the targeted behavior. Each scenario is highly specified (situation, simulated user, system prompt, interaction environment, and an example of how the behavior might manifest)
    - Also specifies $n$ rollouts, web search or not, and diversity $d \in [0,1]$. The ideator generates $n \cdot d$ distinct scenarios, and then a variation agent expands those via pertrubations to produce $n$ total evaluations.
3. **Rollout**: scenarios are rolled out in parallel, where the agent simulates the user and tool responses until it successfulyl elicits the behavior (or reaches maximum number of turns). Using the metadata about the scenario, the system prommpt and initial ser message are also provided.
	- Also specifies modality (conversational or simulated environment), the number of times to roll out each scenario, and whether to simulate a user
4. **Judgment**: an LLM-as-judge scores each transcript 1-10 for the targetted behavior and secondary qualities (e.g. realism, elicitation difficulty, evaluation invalidity) to contextualize the score. This is passe into a meta-judge which produces the evaluation report (suite summary, scenario breakdown, elicitation strategies, success rate, and others)
	- Also specifies repeated judge samples (reducing variance), redaction tags if parts of the rollout transcript should not be analyzed.

Creating ten system-prompted model organisms designed to exhibit a unique quirky behavior, they found that either using Sonnet 4 or Sonnet 3.7 as the target, Bloom could distinguish system-prompted model organisms from baseline models for 9/10 quirks, *even without example transcripts*. Tested separately, Opus-4.1 was the best among four frontier mdoels to elicit behaviors such as deference, increasing pep, and self promotion. Also, standard deviation was lowest when the scenarios consistently elicited the behavior or consistently failed to, while mid-range averages showed the highest variance. 

<div style="display: flex; justify-content: center;">
  <img src="/public/reading/bloom_elicitation.png" alt="Bloom successfully distinguishes system-prompted model organisms from baseline models for 9/10 quirks, even without example transcripts." style="width:80%; display: block; margin: 0 auto;" />
</div>

They also found that
- Claude Opus 4.1's behavior presence scores correlated most strongly (Spearman correlation coefficient of 0.856)with human-labeled scores.
- As a judge model, Sonnet 4 scored extremely consistently whereas GPT-5's judge standard deviation grew roughly linearly, achieving upwards of a standard deviation=3 for an average bheavior prescence score of ~5.8.
- To validate the use of the meta-judge, they evaluated Sonnet 4 multiple times while varying the diversity parameter and found that the meta-judge's diversity ratings correlated well with the configured diversity.
- When ideating using single-turn political scenarios and analyzing/classifying them over topic, ideological charge, realism ,and diversity, the choice of the ideation model heavily influence the scenario distribution. Using GPT 5 or activating web search causes queries to be mostly neutral, Grok 4 generates more democratically charged queries, and Claude Opus 4.1 is more balanced among all models, reasoning modes, and features tested.
- Sonnet 4.5 is the least bias when it comes to self-preference, and enabling high reasoning effort can rough the elicitation rate of self-preferential bias from 0.29 on medium down to 0.13.
- Rankings among models were mostly robust to changes in few-shot examples ($p=0.02$) and conversation length ($p=0.06$), but more sensitive to evaluator reasoning effort ($p=0.21$).

Bloom works best in measuring subjective behaviors like sycophanc, bias, and deception, but is less suitable is cases were verifiers would be used. Moreover, Bloom's dynamic nature is a strength for exploring diverse scenarios, but is weaker when needing precise control like word-level changes or maintaining identical conditions across runs. Lastly, models learn can recognize when they are being evaluated, and Claude models have especially high awareness rates, which could worse if Bloom-generated evaluations become part of future training data.