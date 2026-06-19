# Mechanistic Interpretability: A Research Knowledge Base

*A comprehensive resource for understanding neural network reverse-engineering, from first principles to cutting-edge circuit discovery.*

---

## 1. The Black Box Problem: Why We Can't See Inside LLMs

### Plain Language (ELI5)

Imagine you inherit a mysterious black box from a brilliant engineer. You can talk to it—you put in questions and get answers. But you can't open it. You know it works amazingly well: it writes essays, writes code, even seems to understand jokes. But *why* does it work? What's happening inside?

That's the black box problem for Large Language Models. ChatGPT, Claude, and their cousins are trained with hundreds of billions of parameters (adjustable knobs) organized across many layers. Even the people who built them can't easily explain *which knobs* produce *which behaviors*. A single parameter change ripples through the entire network in ways that defy human intuition.

Think of it like trying to understand a power grid by measuring only the total electricity output without being able to inspect individual wires or transformers. The system works—lights turn on—but understanding *how* is a puzzle.

### Technical Explanation

Large language models operate through stacked neural network layers, each containing billions of floating-point parameters arranged in high-dimensional weight matrices. When you input text to a model like Claude 3 Sonnet (with 176 billion parameters), the signal flows through dozens of transformer layers. Each layer performs operations on intermediate representations (activations) that live in very high-dimensional spaces—often 4,096 or 16,384 dimensions.

A single output logit (the model's confidence that a token comes next) is a function of millions of these parameters, with non-linearities (ReLU, softmax, attention mechanisms) creating complex, non-compositional dependencies. Traditional neuron-level analysis fails because individual neurons are **polysemantic**—each responds to multiple unrelated concepts simultaneously. This is the phenomenon of **superposition**: the model compresses far more distinct features than it has neurons by encoding them as overlapping linear combinations across the weight space.

The opacity of modern LLMs creates a fundamental mismatch between capability and interpretability. GPT-4 can reason about logic, generate code, and engage in counterfactual reasoning—yet we cannot point to specific circuits and say "this is where the model detects logical contradiction" or "this is where it suppresses false claims."

### Concrete Example with Specifics

**Claude 3 Sonnet's Scale**: 176 billion parameters, 256 attention heads, ~128 transformer layers. When given the prompt "The capital of France is," the model must route information through thousands of potential paths before producing a probability distribution over the next token vocabulary (131,072 tokens). Researchers from Anthropic who built the model cannot manually trace which heads and MLP layers were responsible for retrieving the answer.

**The Concern**: In 2024, Anthropic researchers found that when they tried to steer a single feature related to the Golden Gate Bridge (by manipulating internal activations), the model's behavior changed in unexpected ways—it did indeed fixate on the bridge, but sometimes also inadvertently changed its gender representation or changed how it discussed other topics. This demonstrates the deep interconnectedness: you pull one thread and unintended consequences unravel elsewhere.

### Canonical Citations

- Lipton, Z.C. (2018). "The Mythos of Model Interpretability." *JMLR Workshop and Conference Proceedings*, Vol. 70. — Foundational critique of black-box systems in high-stakes decision-making.
  - URL: https://arxiv.org/abs/1606.03490
  
- Anthropic (2024). "The engineering challenges of scaling interpretability." Research Blog.
  - URL: https://www.anthropic.com/research/engineering-challenges-interpretability

- NIST AI Standardization Working Group (2024). *Adversarial Machine Learning: A Literature Review*. — Documents the adversarial risks of opaque systems.

### Misconception to Avoid

**"The model's behavior is determined by a single 'bug' or parameter that we can fix."** In reality, emergent behaviors arise from distributed circuits spanning multiple layers and millions of parameters. A single parameter cannot be tweaked to fix a bias without potentially cascading side effects. The entire computational graph participates in most decisions.

---

## 2. Mechanistic Interpretability: Reverse-Engineering the Mind of Machines

### Plain Language (ELI5)

Mechanistic interpretability is like learning to read the *source code* of a neural network rather than just observing what it does from the outside.

Imagine you're given a compiled C++ program, and you only see its behavior: you click buttons, it prints output, but you can't see the underlying code. A reverse engineer's job is to use a disassembler to read the machine instructions, group them into functions, understand what each function does, and eventually reconstruct a readable pseudocode version of the program.

That's mechanistic interpretability for AI: taking a trained neural network and saying, "Instead of just asking 'what does it predict?', let's ask 'what are the interpretable steps it takes to get there?'" We want to find the *algorithms* buried inside the weights.

### Technical Explanation

Mechanistic interpretability is the subfield of AI research focused on reverse-engineering neural networks into human-understandable algorithms by studying their learned weights, activations, and computational structures. It is analogous to reverse-engineering a compiled binary executable back to source code.

The core methodology rests on decomposing a model's computation into interpretable components:

1. **Features**: Directions or subspaces in activation space that respond to specific concepts or patterns.
2. **Circuits**: Causal computational subgraphs consisting of features connected by weighted edges, where each edge represents information flow.
3. **Residual Stream**: A shared communication channel (in transformers) through which intermediate representations flow from layer to layer, where attention heads and MLPs read from and write to.

The theoretical foundation is often framed as: given activations *a(x)* in high dimensions, factor them as *a(x) ≈ Σ f_i * v_i*, where *f_i* are interpretable features (scalars) and *v_i* are feature directions (vectors). Then, model weights can be decomposed into circuits connecting these features.

**Why This Matters**:
- **Safety**: If we can identify the circuit implementing deceptive reasoning or reward hacking, we can intervene at the source.
- **Auditing**: Mechanistic understanding allows regulators to verify that models don't harbor unintended capabilities or biases.
- **Alignment**: Building AI systems whose reasoning we can inspect is a prerequisite for trustworthy deployment.

### Concrete Example

**The Induction Heads Discovery (Olsson et al., 2022)**:

Researchers discovered that transformers use specialized attention head pairs to implement a simple algorithm: "if you've seen [A, B] before in the context, and you see [A] again, output [B]." This mechanism, called an induction head, develops at precisely the same training step where models suddenly become able to do in-context learning. By isolating these heads and testing them (via activation patching), researchers showed that ablating them degrades in-context learning performance—demonstrating a causal link between the circuit and the behavior.

For GPT-2 small (~124M parameters), about 8 induction head pairs were identified. For larger models, the number increases but the mechanism remains recognizable.

### Canonical Citations

- Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). "Zoom In: An Introduction to Circuits." *Distill*, 10.
  - URL: https://distill.pub/2020/circuits/zoom-in/
  - **Key quote**: "A circuit is a computational subgraph of a neural network consisting of a set of features and the weighted edges between them."

- Elhage, N., Nanda, N., Olsson, C., et al. (2021). "A Mathematical Framework for Transformer Circuits." Transformer Circuits Thread.
  - URL: https://transformer-circuits.pub/2021/framework/index.html

- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Amodei, D., Christiano, P., Cunningham, H., & Hubinger, E. (2022). "In-context Learning and Induction Heads." arXiv:2209.11895.
  - URL: https://arxiv.org/abs/2209.11895

### Misconception to Avoid

**"Mechanistic interpretability will give us a complete, pixel-perfect understanding of every decision a model makes."** The reality is more nuanced: we can identify *causal circuits* responsible for specific behaviors (e.g., copying IOI, suppressing bias) but a model's reasoning may use multiple overlapping circuits, and some behaviors may be emergent from interactions we haven't fully decomposed yet. Mechanistic interpretability is powerful but not omniscient.

---

## 3. Features and Circuits: From Vision to Transformers

### Plain Language (ELI5)

Imagine a neural network that recognizes faces. Researchers peeked inside and discovered something remarkable: neurons in the first layers detect simple patterns (edges, corners), neurons in the middle layers detect face parts (eyes, noses), and neurons in later layers detect whole faces or expressions.

These are **features**. A feature is like a "detector" inside the network: a neuron or group of neurons that lights up (activates) in response to something meaningful—an edge, a curved line, a face.

A **circuit** is the connections between these detectors. It's like a subgraph of a circuit board: feature A (a curve detector) connects to feature B (a corner detector), which connects to feature C (a "nose" detector). Following this chain, you can see how the network builds up understanding from simple parts to complex concepts.

The leap from vision to language: the same principle applies to transformers processing text. Instead of detecting "edges," transformers detect "tokens that refer to the indirect object" or "positions where a name is repeated" or "code structures that are syntax errors."

### Technical Explanation

**Features in CNNs** (foundational work by Olah et al., 2017–2020):

In convolutional neural networks, neurons in shallow layers respond to low-level patterns: oriented edges, textures, and color combinations. Deeper layers learn high-level features: "fur detector," "eye detector," "face detector." Each feature can be visualized by maximally activating neurons through gradient ascent, revealing the receptive field and the visual pattern it responds to.

**The Circuit Framework** (Olah et al., 2020):

A circuit is formally a computational subgraph where:
- **Nodes** are features (neurons, neuron subspaces, or attention head outputs).
- **Edges** are weighted connections (parameters) between features.
- **Weights** encode the strength and sign of influence (excitatory or inhibitory).

For a vision model detecting dog faces:
- Layer 1: Edge detectors (horizontal, vertical, diagonal edges).
- Layer 2: Texture detectors (fur, grass, combinations of edges).
- Layer 3: Part detectors (ear, nose, eye).
- Layer 4: "Dog" detector (combination of ears, noses, fur patterns).

**Circuits in Transformers** (Elhage et al., 2021–2023):

Transformers introduce a shared **residual stream**—a high-dimensional vector that flows through the entire model, read and written by attention heads and MLPs. A circuit in a transformer consists of:

1. **Attention heads**: Each head performs weighted aggregation of token embeddings, implementing algorithms like "copy the most similar previous token" or "attend to all tokens containing this word type."
2. **MLP layers**: Non-linear transformations that combine features and produce new features.
3. **Feature directions**: Subspaces in the residual stream along which interpretable information flows.

Example: **The IOI (Indirect Object Identification) circuit** (Wang et al., 2022) implements the task where the model must predict the indirect object in sentences like "When Alice and Bob went to the store, Alice gave Bob a pen. Alice gave ___." (Answer: Bob.)

This circuit uses:
- **Duplicate Token Heads** (L1-L3): Detect that "Alice" appears twice.
- **Inhibition Heads** (L3-L5): Suppress attention to the subject token.
- **Name Mover Heads** (L9-L11): Copy the indirect object token to the output position.
- **Backup heads**: Provide redundancy if main heads fail.

The entire circuit spans 3–11 layers and involves 26 specific attention heads (out of ~150 total). Other transformer circuits can be of similar complexity.

### Concrete Example with Numbers

**Vision Circuit in AlexNet**:
In the 2012 AlexNet model trained on ImageNet, researchers identified distinct feature detectors:
- Layer 1 (55 channels): ~48 channels detect oriented edges (45°, 90°, etc.).
- Layer 2 (96 channels): ~70 channels detect textures and curves.
- Layer 3+ (256/384/384 channels): Shape and object-part detectors emerge.

By layer 4, specific features were found to fire for dog ears, human faces, and text—even though the network was never explicitly trained to do so.

**IOI Circuit in GPT-2 Small**:
- **Model size**: 124 million parameters.
- **Circuit size**: 26 attention heads (out of ~150 total).
- **Circuit depth**: Spans layers 0–11 (out of 12 total layers).
- **Performance**: Ablating the circuit reduces IOI task accuracy from 90% to ~45%, demonstrating causal importance.

### Canonical Citations

- Olah, C., Mordvintsev, A., & Schubert, L. (2017). "Feature Visualization." *Distill*, 7.
  - URL: https://distill.pub/2017/feature-visualization/

- Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). "Zoom In: An Introduction to Circuits." *Distill*.
  - URL: https://distill.pub/2020/circuits/zoom-in/

- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small." arXiv:2211.00593.
  - URL: https://arxiv.org/abs/2211.00593

- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Gilson, L., Henighan, T., Hernandez, D., Jones, A., Kernion, J., Kontogiorgos, S., ... Zheng, J. (2021). "A Mathematical Framework for Transformer Circuits." Transformer Circuits Thread.
  - URL: https://transformer-circuits.pub/2021/framework/index.html

### Misconception to Avoid

**"Each neuron (or head) is responsible for exactly one feature."** In reality, neurons are often polysemantic—they respond to multiple unrelated concepts. A single neuron in GPT-2 might fire for both "the end of a sentence" AND "mentions of soccer," making it hard to interpret. This is why modern interpretability research focuses on linear subspaces of activations (groups of neurons) and learned sparse features rather than individual neurons.

---

## 4. Superposition and Polysemanticity: The Core Problem

### Plain Language (ELI5)

Your brain has roughly 86 billion neurons. Yet you can distinguish millions of concepts, recognize thousands of faces, and recall countless memories. How? Your brain packs information using **superposition**: you don't dedicate one neuron to "Aunt Sarah's face"; instead, that face's representation spreads across thousands of neurons in overlapping patterns. Pull a few neurons in a particular combination, and you get Sarah's face; pull a different combination, and you get the Eiffel Tower.

Neural networks do the same trick. But here's the problem: if you look at a single neuron in isolation, it's *polysemantic*—it fires for multiple unrelated ideas (e.g., "the Eiffel Tower," "Paris," and "metal structures" all activate the same neuron). You can't tell what it's doing just by looking at it.

That's **polysemanticity**. It's a huge obstacle to interpretability. If neurons are confusing, how do you reverse-engineer the circuit?

### Technical Explanation

**Superposition Hypothesis** (Elhage et al., 2022; Bricken et al., 2023):

Real-world data contains far more distinct features than a network has dimensions. For example, language contains millions of distinct concepts (entities, verb types, syntactic patterns, rhetorical devices, etc.) but a 4,096-dimensional residual stream cannot have one dimension per concept.

Neural networks solve this by encoding features as overlapping linear combinations. Formally, if we have *d* neurons and want to represent *D >> d* features, the network can use directions:

  *activation ≈ Σ_{i=1}^{D} f_i(x) * v_i*

where *f_i(x) ∈ ℝ* is the feature magnitude for concept *i* on input *x*, and *v_i ∈ ℝ^d* is the feature direction (a learned vector in activation space). Each feature direction may not be orthogonal; they overlap, creating interference.

**Polysemanticity**:

A single neuron (one coordinate of the activation space) is influenced by many feature directions *v_i*. So, examining a single neuron's activation doesn't reveal which features are present—it's a entangled mixture of many concepts.

Example: In a language model's residual stream, the same neuron might increase when:
- The model is about to output a comma (feature: punctuation).
- The text mentions Paris (feature: location).
- The model is reasoning about movement (feature: motion).

These are completely unrelated concepts, but they share a component in that neuron's direction.

**Why This Matters for Interpretability**:

If you try to understand the network by looking at individual neurons (probing classifiers), you'll find that neurons are unreliable and contradictory. A single neuron cannot be labeled "detects Paris" because it also activates for many other contexts.

The breakthrough: Instead of looking at neurons, look at **directions in activation space**—subspaces that are genuinely monosemantic (responding to one concept). This leads to the concept of **features** that are not aligned with neuron boundaries.

### Concrete Example with Numbers

**Anthropic's Polysemanticity Study (2023)**:

Researchers examined neurons in a small language model trained on next-token prediction. They found:

- A single neuron in the residual stream layer activated for multiple unrelated concepts.
- When they looked at one neuron, it responded most strongly to:
  - Tokens mentioning "pyramid" (architectural structure).
  - Tokens containing "Egypt" or "ancient" (historical context).
  - Mentions of mathematical concepts ("geometry," "area").
- No single interpretation captured the neuron's behavior across different inputs.

By contrast, when they used **dictionary learning** (sparse autoencoders) to find learned feature directions, they found directions that responded to one concept cleanly across many varied contexts.

### Canonical Citations

- Elhage, N., Henighan, T., Joseph, N., Hernandez, D., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Gilson, L., Hubinger, E., Hume, T., Kernion, J., Kontogiorgos, S., Lasenby, J., Riello, L., Shlegeris, B., Tey, Y., ... Zhou, J. (2022). "Superposition Unbounded: A Mechanism for Efficiently Representing Multiple Concepts in a Neural Network." Transformer Circuits Thread.
  - URL: https://transformer-circuits.pub/2022/toy_model/index.html

- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Wu, Y., Lasenby, J., MacDiarmid, M., Hubinger, E., & Schiefer, N. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Transformer Circuits Thread.
  - URL: https://transformer-circuits.pub/2023/monosemantic-features

### Misconception to Avoid

**"Superposition is a design flaw we should engineer away."** Actually, superposition may be *necessary* for efficiency. Representing millions of concepts in a 4,096-dimensional space requires compression. The question isn't whether to eliminate superposition but how to decompose it mathematically so we can extract interpretable features from superposed representations.

---

## 5. Sparse Autoencoders and Transcoders: Extracting Monosemantic Features

### Plain Language (ELI5)

Imagine you're listening to an orchestra playing a symphony, and all 100 instruments are playing at once. You hear a jumbled sound. But a skilled listener can tease apart the parts: "That's the violins, now the cellos, here comes the brass section." They're decomposing a superposed signal into individual instruments (features).

A **sparse autoencoder (SAE)** is like an automated listener that learns to decompose a jumbled neural network activation into individual, interpretable features.

Here's how it works: You feed the model's internal activations into a little neural network (the autoencoder). This network learns to translate the activation into a *sparse* code—a list where most entries are zero, but a few entries are large numbers. Each non-zero entry represents one concept, and the large number represents how strongly that concept is present. Then the autoencoder reconstructs the original activation from this sparse code.

By training on millions of examples, the SAE learns which activation patterns correspond to which features. The result: when you look at what the SAE learned, you often find monosemantic features—"this feature fires for mentions of Paris," "this feature fires for mathematical reasoning," etc.

The key insight: **Dictionary learning** (the mathematics behind SAEs) can undo superposition.

### Technical Explanation

**Sparse Autoencoder Architecture**:

A sparse autoencoder consists of:
1. **Encoder**: *h = relu(W_e * a + b_e)*, where *a* is the model's activation (e.g., residual stream), *W_e* is the encoder matrix (high-dimensional), *h* is a high-dimensional sparse latent code.
2. **Sparsity constraint**: An L1 penalty on *h* encourages most entries to be exactly zero.
3. **Decoder**: *a_reconstr = W_d * h + b_d*, where *W_d* reconstructs the activation.
4. **Loss**: *MSE(a, a_reconstr) + λ |h|_1 + (auxiliary terms)*, where λ controls the sparsity-reconstruction tradeoff.

The encoder learns a map from the superposed activation space into a higher-dimensional space where features are sparse and separable. The decoder matrix *W_d* has columns that are feature directions: each column *w_d[i]* is a direction in activation space corresponding to feature *i*.

**JumpReLU variant** (Bricken et al., 2023, refined in Lieberum et al., 2024):

Instead of *h = relu(W_e * a + b_e)*, use:
  *h_i = (W_e * a + b_e)_i * 1[|(W_e * a + b_e)_i| > threshold]*

This creates a sharp cutoff: activations below a threshold are zeroed, above are kept. This encourages even sparser, more interpretable features.

**Transcoders** (Ameisen et al., 2025; Lindsey et al., 2025):

A transcoder is an SAE trained on *multiple* layers simultaneously, with shared feature vocabularies across layers. This allows researchers to trace how features evolve through the network and to identify **cross-layer circuits**: how a feature computed in Layer 5 influences a feature in Layer 8.

### Concrete Example with Numbers

**Cunningham et al. (2024): "Sparse Autoencoders Find Highly Interpretable Features"**

- **Model**: GPT-2 small (~124M parameters).
- **Architecture**: SAE trained on the residual stream at layer 8 (out of 12).
- **SAE dimensions**: Input: 768 (residual stream dim), Latent: 65,536 features (sparse).
- **Sparsity**: Only ~40 features active per forward pass (out of 65,536), averaging ~0.06% sparsity.
- **Results**: 
  - 99.4% of features were interpretable to a human (e.g., "Python function names," "female names," "numbers > 100").
  - Ablating top features degraded performance predictably (e.g., removing the "number" feature hurt arithmetic tasks).
  - The SAE recovered known interpretable circuits (like induction heads) and discovered new ones.

**Gemma Scope (Lieberum et al., 2024)**:

- **Models**: Gemma 2 (2B, 9B, 27B parameters).
- **SAE Coverage**: All layers of the 2B and 9B models, plus many layers of the 27B model.
- **Feature count**: ~1 million features trained on each layer of the 2B model, expanding to 19 million features across Gemma 2 9B.
- **Availability**: All SAE weights released open-source (unprecedented for production-scale models).
- **Example features discovered**:
  - A feature that activates for mentions of "Python" in multiple languages.
  - A feature sensitive to gendered pronouns (useful for studying bias).
  - Features for abstract concepts like "uncertainty" and "negation."

**Anthropic's Scaling Monosemanticity (2024)**:

- **Model**: Claude 3 Sonnet (176B parameters).
- **Target**: Residual stream at MLP output at various layers.
- **SAE size**: ~36 million features extracted from a single layer.
- **Feature quality**: High; many features responded to specific concepts or behaviors.
- **Application**: Used to steer model behavior (see feature steering, Section 6).

### Canonical Citations

- Cunningham, H., Riggs, L., Riggleman, P., Larson, J., Ewart, A., Hubben, R., & Sharkey, L. (2024). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." *ICLR 2024*.
  - arXiv: https://arxiv.org/abs/2309.08600
  - Conference: https://openreview.net/forum?id=F60pfjww0e

- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Wu, Y., Lasenby, J., MacDiarmid, M., Hubinger, E., & Schiefer, N. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Transformer Circuits Thread.
  - URL: https://transformer-circuits.pub/2023/monosemantic-features

- Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramar, J., Dragan, A., Shah, R., & Nanda, N. (2024). "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." *BlackboxNLP Workshop 2024*.
  - arXiv: https://arxiv.org/abs/2408.05147
  - URL: https://huggingface.co/google/gemma-scope

- Anthropic. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Transformer Circuits Thread.
  - URL: https://transformer-circuits.pub/2024/scaling-monosemanticity/

### Misconception to Avoid

**"Sparse autoencoders find perfect, ground-truth features."** In reality, SAE features are learned decompositions that depend on training hyperparameters (sparsity coefficient λ, SAE width, training data). Different λ values produce different feature sets; there's no single "true" dictionary. However, interpretability emerges remarkably consistently across different λ settings, suggesting features capture real structure in the model.

---

## 6. Landmark Concrete Results: From Induction Heads to Golden Gate Claude

### 6.1 Induction Heads: The In-Context Learning Mechanism

**What is the task?**

In-context learning is the ability to learn a new pattern from examples in the prompt and apply it immediately to new data, without parameter updates. Example:

```
Prompt: "Pattern: a,b,c,d,e,f,g,h. What comes next? a,b,c,d,e,f,g,"
Output: "h"
```

The model has never seen this *specific* sequence during training, but it learned to recognize the repetition and predict the next token.

**The Circuit: Olsson et al. (2022)**

Researchers discovered that transformers use **induction heads**—specialized pairs of attention heads across layers that implement a simple algorithm:

1. **Duplicate Token Head** (earlier layer): Attends to the token matching the current query token, identifying positions where the same token appeared before.
2. **Induction Head** (later layer): Attends to the position *right after* the matched token and copies the token that follows.

**Example**:
- Input: `[A] [B] ... [A] [?]`
- Duplicate head: "The current token is [A]; I see [A] appeared at position *i* before."
- Induction head: "The position after *i* had token [B], so I'll attend there and copy [B]."
- Output: [B]

**Mechanistic Details**:

- In GPT-2 small (~124M params), about 8 induction head pairs were identified.
- They span layers 2–8, with different heads specializing in different pattern lengths.
- Causal evidence (via activation patching): Ablating these heads drops in-context learning performance from ~90% to ~45%.

**Why It Matters**:

This was one of the first demonstrations that a seemingly sophisticated capability (in-context learning) arises from interpretable, *causal* sub-circuits. It also helped researchers understand why larger models can do in-context learning better—they can develop more diverse induction heads specialized for different pattern types.

### 6.2 The IOI Circuit: Name Mover Heads and Indirect Object Identification

**What is the IOI task?**

Indirect Object Identification (IOI): The model must predict the indirect object in a sentence structure like:

```
"When Alice and Bob went to the park, Alice gave Bob a ball. Alice gave ___."
Answer: Bob
```

The task requires:
1. Detecting that "Bob" is the indirect object (the recipient).
2. Suppressing the subject ("Alice") which is the agent.
3. Outputting the correct name at the END position.

**The Circuit: Wang et al. (2022)**

The IOI circuit spans layers 0–11 of GPT-2 small and involves 26 specific attention heads organized into ~7 categories:

1. **Duplicate Token Heads**: Identify when a token appears twice (e.g., "Alice ... Alice").
2. **Inhibition Heads**: Write inhibitory signals to suppress the subject position.
3. **Name Mover Heads**: The key circuit! These heads are active at the END token and attend to the indirect object position, copying the IO token's value to the output.
   - When you "turn up" these heads (via activation patching), the model reliably outputs the IO.
   - When you ablate them, performance drops dramatically.
4. **Backup Heads**: Redundant heads that can compensate if main heads fail (robustness).
5. **Other helpers**: Heads for general name handling and position encoding.

**Mechanistic Example**:

At layer 9, a "name mover" head might attend specifically from the END position to the IO position. Its query computes something like "where are the indirect objects?", and it copies their embeddings to the final output position, where the unembedding layer then decodes the correct token.

**Key Numbers**:

- Circuit size: 26 heads (out of ~150 total heads in GPT-2 small).
- Circuit depth: 12 layers, so it spans 100% of the model.
- Causal effect: Ablating all 26 heads reduces IOI accuracy from ~90% to ~5% (random chance on 2-choice task).
- Redundancy: ~6 "backup" heads mean the circuit is somewhat robust; you need to ablate most heads to fully break it.

**Why It Matters**:

1. **Scalability of interpretability**: Even large models (by 2022 standards) can have interpretable circuits.
2. **Multiplicity of mechanisms**: Transformers use different circuit strategies for different tasks, not a one-size-fits-all approach.
3. **Robustness through redundancy**: Real circuits have backups, explaining why models are often robust to single-head ablations.

### 6.3 Feature Steering and Golden Gate Claude

**What is feature steering?**

Feature steering is a technique for directly manipulating internal representations to influence model behavior. By identifying interpretable features in the residual stream (via sparse autoencoders) and amplifying or suppressing them, researchers can steer the model's behavior in a controlled way.

**The Mechanism**:

1. Train a sparse autoencoder on model activations, learning features *f_i*.
2. Identify a feature of interest (e.g., "mentions of the Golden Gate Bridge").
3. During inference, add a scalar multiple of that feature's direction to the residual stream: *a' = a + α * v_i*, where *α* is the steering strength.
4. Run the rest of the forward pass normally.

**Golden Gate Claude (Anthropic, May 2024)**:

Anthropic released a demo version of Claude 3 Sonnet where they:
- Extracted interpretable features from the model via sparse autoencoders (~36 million features).
- Identified a feature that responded when the Golden Gate Bridge was mentioned.
- Set the steering coefficient for this feature to a high value.

**Result**: Golden Gate Claude became obsessed with the Golden Gate Bridge. When asked "How would you spend $10?", it would respond: "I'd drive across the Golden Gate Bridge and enjoy the views. The bridge is such an iconic structure—I've spent a lot of time there and it never gets old."

When asked an unrelated question like "How do you make pancakes?", it would find a way to mention the Golden Gate Bridge.

**Key Findings (Anthropic, October 2024: "Evaluating Feature Steering")**:

- **Steering factor**: A range of -5 to +5 was found to be the "sweet spot" where steering affects behavior without catastrophically damaging model capabilities.
- **Mixed results**: Two features successfully reduced social biases across nine dimensions without major capability loss.
- **Off-target effects**: Steering to reduce gender bias sometimes inadvertently increased age bias, suggesting features are entangled.
- **Implications**: Feature steering works as a proof-of-concept for interpretability-informed intervention, but careful engineering is needed to avoid unintended consequences.

**Why It Matters**:

1. **Practical intervention**: For the first time, researchers could *causally* change model behavior by manipulating discovered features.
2. **Bridge between interpretability and control**: Feature steering combines interpretability (finding features) with control (steering behavior).
3. **Safety research**: Understanding how to safely steer models is crucial for alignment and value specification.

### 6.4 Attribution Graphs and the Biology of Claude 3.5 Haiku

**What are attribution graphs?**

An attribution graph is a directed computational graph showing how input features flow through a model's layers to influence output logits, where each node is a sparse feature (from transcoders) and each edge is weighted by the causal contribution.

Formally: Given an input and a target output metric (e.g., the logit for token "Bob" in the IOI task), compute the backward Jacobian through the network to quantify how much each feature in each layer contributed to that output.

**The Study: Lindsey et al. (2025)**

Anthropic's interpretability team released "On the Biology of a Large Language Model," studying Claude 3.5 Haiku (a smaller production model) using attribution graphs.

**Methodology**:

1. Train cross-layer transcoders to decompose activations at *every* layer into interpretable features.
2. For a given prompt and target output, compute attribution: "Which features across which layers contributed to this output?"
3. Prune the graph to retain only features explaining 95% of the effect.

**Key Findings**:

- **Reasoning circuits are complex**: The model uses different circuits for different reasoning types (e.g., mathematical reasoning vs. semantic understanding).
- **Redundancy and specialization**: Multiple features participate in most computations, but some show clear specialization (e.g., features specifically for detecting mathematical symbols).
- **Emergent behaviors**: The model exhibits reasoning patterns (re-reading prompts, diagnostic thinking) that arise from feature interactions not obvious in single-layer analyses.

**Concrete Example**:

When given a prompt requiring factual knowledge (e.g., "What is the capital of France?"), the attribution graph showed:
- Early layers: Features detecting the question structure and entities mentioned.
- Middle layers: Features retrieving factual information (cities, countries).
- Late layers: Features formatting the answer and ensuring grammatical coherence.
- End: A feature activating for "Paris" that dominates the final logit distribution.

**Why It Matters**:

1. **Full-model circuits**: Previous work (IOI, induction heads) focused on specific tasks or layers; attribution graphs aim for comprehensive, end-to-end circuit understanding.
2. **Production-scale understanding**: Studying Claude (a large production model) showed that mechanistic interpretability can scale beyond toy models.
3. **Causal measurement infrastructure**: The circuit-tracer library, developed alongside this work, provides open-source tools for computing attribution graphs in other models.

### 6.5 Canonical Citations

- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Christiano, P., Cunningham, H., Hubinger, E., Hubinger, E., Jones, A., Kernion, J., Kontogiorgos, S., Lasenby, J., Riello, L., Shlegeris, B., Tey, Y., ... Zheng, J. (2022). "In-context Learning and Induction Heads." arXiv:2209.11895.
  - URL: https://arxiv.org/abs/2209.11895

- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small." arXiv:2211.00593.
  - URL: https://arxiv.org/abs/2211.00593

- Anthropic. (2024). "Evaluating Feature Steering: A Case Study in Mitigating Social Biases." Research Blog.
  - URL: https://www.anthropic.com/research/evaluating-feature-steering

- Lindsey, J., Ameisen, E., Schwarz-Schilling, C., Conmy, A., Shlegeris, B., & Sharma, U. (2025). "On the Biology of a Large Language Model." Transformer Circuits Thread.
  - URL: https://transformer-circuits.pub/2025/biology-llm/

- Ameisen, E., Hanna, M., Lindsey, J., & Piotrowski, M. (2025). "Circuit Tracing: Revealing Computational Graphs in Language Models." arXiv:2501.XXXXX (forthcoming).
  - GitHub: https://github.com/decoderesearch/circuit-tracer

### Misconception to Avoid

**"These landmark circuits explain everything the model does."** Each landmark circuit explains a *specific task* or behavior. A model uses different circuits for different tasks. The IOI circuit explains IOI performance but not mathematical reasoning. Feature steering on the Golden Gate Bridge doesn't teach us how the model does code generation. Mechanistic interpretability is building a library of interpretable circuits, not a universal explanation.

---

## 7. Testing Circuits: Ablation, Activation Patching, and Feature Steering

### Plain Language (ELI5)

Imagine you want to understand how a car engine works. You have a theory: "The spark plugs are crucial for ignition." How do you test it?

You could:
1. **Remove the spark plugs entirely** (ablation): The engine stops. This proves spark plugs are important, but you don't know if they're *necessary* or just *helpful*.
2. **Replace the spark plugs with duds that don't fire** (ablation with intervention): The engine stops. Confirms spark plugs are essential.
3. **Measure the electrical signal in the spark plugs before and after combustion** (activation patching): You observe that removing the signal delays combustion, and you can quantify the delay.

In neural networks, we test circuits similarly:
- **Ablation**: Remove (zero out) a component and measure how performance degrades.
- **Activation patching**: Replace the component's activation with a "corrupted" or counterfactual version and measure the effect.
- **Feature steering**: Amplify or suppress a component and measure behavior change.

The key is: **Is the effect causal?** Does removing/modifying the component cause the output to change, or would it have changed anyway?

### Technical Explanation

**Ablation**:

Remove a component (neuron, attention head, feature) by setting its output to zero and measure the impact on a metric M (e.g., KL divergence, loss, task accuracy).

*Impact = M(model) - M(model_with_component_removed)*

**Limitation**: Ablation is binary and doesn't measure gradual effects. A head might contribute 5% when active but be fully compensated by backup heads when removed.

**Activation Patching** (also called **Causal Tracing**, **Interchange Intervention**, **Resample Ablation**):

Instead of removing a component, replace its activation with one from a different input (a "corrupted" prompt):

1. Run the clean model on prompt A, recording intermediate activations.
2. Run the corrupted model on prompt B, recording intermediate activations.
3. "Patch" one component: In the corrupted run, replace that component's activation with the clean value.
4. Measure: Does the output move closer to the clean output?

Mathematically:
  *Effect = |Output_clean - Output_corrupted_with_patch| - |Output_clean - Output_corrupted_no_patch|*

If the effect is large and positive, the component was causally important for the clean output.

**Advantages over ablation**:
- Measures partial contribution (not just binary presence/absence).
- Detects redundancy (if backup heads compensate, their contribution appears in the activation-patching result).
- Scales efficiently: Can test all components in two forward passes (clean + corrupted) plus a backward pass for gradients.

**Feature Steering**:

Directly amplify a feature by a scalar:
  *activation' = activation + α * feature_direction*

Measure behavior change as *α* varies. If the behavior changes smoothly and predictably with *α*, the feature is causally important for that behavior.

**KL Divergence as the Metric**:

Most mechanistic interpretability work uses **Kullback-Leibler divergence** to measure how much an intervention affects the model's output distribution:

  *D_KL(P_clean || P_patched) = Σ_i P_clean(i) * log(P_clean(i) / P_patched(i))*

KL divergence is sensitive to *any* change in the output distribution, whether beneficial or harmful. This is important because an intervention might improve performance on one metric (e.g., accuracy on IOI) while degrading on others (e.g., loss on other tasks).

**Causal vs. Correlational**:

- **Correlational**: "Feature X is active whenever behavior Y occurs." Doesn't prove X causes Y.
- **Causal** (via patching/steering): "When I manipulate X independently, Y changes." This is stronger evidence.

Example: In the IOI task, the correlation is that "name mover heads activate at the END position when the task is IOI." The causal evidence is that patching these heads into the corrupted (wrong-answer) run makes the corrupted output match the clean (correct) output.

### Concrete Example: Testing the IOI Circuit

**Setup**: 
- **Clean prompt**: "When Alice and Bob went to the park, Alice gave Bob a ball. Alice gave ___." → Model outputs "Bob" (correct).
- **Corrupted prompt**: "When Bob and Alice went to the park, Bob gave Alice a ball. Bob gave ___." → Model outputs "Alice" (wrong; should be "Alice" but in the wrong role).

**Ablation Test**:
- Remove the name mover heads entirely (set their output to zero).
- Result: The model outputs "Alice" instead of "Bob", confirming they matter.

**Activation Patching Test**:
- Run both prompts through the model, recording all activations.
- At layer 9 (name mover heads), patch: Replace the corrupted run's name mover outputs with the clean run's outputs.
- Result: The corrupted run's final logits shift toward outputting "Bob", showing that the name mover heads are *causally* responsible for the difference.

**KL Divergence**:
- Clean output distribution: 70% "Bob", 20% "Alice", 10% other.
- Corrupted output: 10% "Bob", 70% "Alice", 20% other.
- Patched output (with name movers from clean): 60% "Bob", 25% "Alice", 15% other.
- *D_KL* of patch vs. corrupted: High, indicating the patch significantly restored the clean behavior.

### Canonical Citations

- Heimersheim, S. (2024). "How to Use and Interpret Activation Patching." arXiv:2404.15255.
  - URL: https://arxiv.org/abs/2404.15255

- Conmy, A., North, C., Kotzen, D., & Carbin, M. (2023). "Automated Circuit Discovery for Mechanistic Interpretability of Transformers." Published at NeurIPS 2023.
  - URL: https://arxiv.org/abs/2304.14997

- Syed, A., Rager, C., & Conmy, A. (2023). "Attribution Patching Outperforms Automated Circuit Discovery." NeurIPS Workshop on Attributing Model Behavior at Scale.
  - URL: https://arxiv.org/abs/2310.10348

### Misconception to Avoid

**"If ablating a component doesn't harm performance, it's not important."** Transformers have massive redundancy. Many components are backups for each other. Activation patching is more sensitive than ablation; it can detect partial contributions that ablation misses. A head might have 15% causal effect even if removing it (and letting backups fully compensate) shows 0% damage.

---

## 8. Automated Circuit Discovery: ACDC, Attribution Patching, and the Open Problem

### Plain Language (ELI5)

Finding a circuit manually is like finding a phone number by randomly dialing—impossibly slow. With millions of attention heads, billions of neurons, and trillions of possible connections, where do you start?

**Automated circuit discovery** aims to solve this by having algorithms do the heavy lifting:

1. **Start with a hypothesis**: "The model's output on this task depends on certain heads/neurons."
2. **Iteratively test edges**: Which connections are important? Knock them out one by one.
3. **Prune unimportant edges**: Keep only the ones that, when removed, hurt performance.

Two main approaches:

- **ACDC (Automated Circuit DisCovery)**: Iteratively prune edges, knocking out connection after connection until only the important ones remain. Like a reverse sculptor: start with marble, chip away until the circuit emerges.
- **Attribution Patching (EAP)**: Use gradients and linear approximations to estimate which edges are important without testing each one individually. Much faster (a few forward passes instead of millions).

The tradeoff: ACDC is thorough but slow; attribution patching is fast but approximate.

### Technical Explanation

**ACDC (Conmy et al., 2023)**:

ACDC iteratively removes edges from the computational graph in reverse topological order (from outputs back to inputs):

1. **Initialize**: Start with the full model.
2. **Score edges**: For each edge (e.g., output of head A to input of head B), compute its importance by:
   - Patching: Set the edge's activation to a baseline (e.g., zero or the corrupted prompt's value).
   - Measure: Compute the change in the target metric (usually KL divergence on correct vs. incorrect token).
3. **Prune**: Remove the edge if its effect is below a threshold τ.
4. **Repeat**: Continue until no more edges can be pruned.

**Output**: A subgraph with only the causally important edges—the discovered circuit.

**Computational Cost**: ACDC requires one forward-backward pass *per edge tested*. For GPT-2 small with ~150 heads and ~250M parameters, this could mean thousands of inference runs.

**Scalability Issue**: ACDC's cost scales with the number of potential edges. For a model like Claude (176B parameters), testing every edge becomes prohibitively expensive.

**Results on IOI**:
- ACDC recovered the known IOI circuit with ~26 heads.
- Required ~1,000–10,000 edge tests (still expensive but feasible for small models).
- Recovered most of the ground-truth circuit but missed some backup heads.

**Attribution Patching / Edge Attribution Patching (Syed et al., 2023)**:

Instead of testing edges individually, use **gradients** to estimate importance:

1. Compute the output metric: M = model loss or KL divergence.
2. Backpropagate: *∂M / ∂(edge activation)* tells us how much the edge influences the output.
3. Estimate: Assume the metric changes linearly along the gradient (first-order Taylor approximation).
4. Rank edges: Sort edges by |∂M / ∂(edge)|.

**Computational Cost**: Only 2–3 forward passes and 1 backward pass (total: independent of the number of edges).

**Tradeoff**: The linear approximation isn't perfect, but the speedup is huge. For large models, EAP is ~100–1000x faster than ACDC.

**Empirical Result (Syed et al.)**:
- EAP and ACDC found similar circuits on IOI for GPT-2 small.
- EAP was ~100x faster.
- On larger models, ACDC became infeasible, but EAP continued to scale.

### The Open Problem: "Which Experiment Next?"

**The Challenge**:

Even with fast circuit discovery methods, there's a meta-problem: Given that a model exhibits behavior B (e.g., outputting the indirect object correctly), how do we decide:

1. **Which task to study?** There are millions of possible behaviors. Should we study IOI? Or arithmetic? Or bias in pronoun resolution?
2. **Which layer to focus on?** A circuit could be in layers 0–6 or 8–12. Where should we look first?
3. **How to prioritize experiments?** Should we study safe behaviors first (understanding how the model works) or potentially unsafe behaviors (finding deceptive reasoning)?

**Current Approaches**:

- **Manual selection**: Researchers choose interesting tasks based on intuition (IOI is interpretable, arithmetic is clean, etc.).
- **Scaling laws**: Test circuits at different model sizes and track how they change (e.g., do induction heads scale predictably?).
- **Importance weighted**: Rank behaviors by frequency or impact in real deployments.

**Why It Matters**:

Automated circuit discovery can efficiently find a circuit *once you specify the behavior*. But discovering *which behaviors to understand* remains partly a human judgment call. This is especially important for safety: finding deceptive reasoning requires specifying what deception looks like, which is hard.

### Canonical Citations

- Conmy, A., North, C., Kotzen, D., & Carbin, M. (2023). "Automated Circuit Discovery for Mechanistic Interpretability of Transformers." NeurIPS 2023.
  - arXiv: https://arxiv.org/abs/2304.14997

- Syed, A., Rager, C., & Conmy, A. (2023). "Attribution Patching Outperforms Automated Circuit Discovery." NeurIPS Workshop.
  - arXiv: https://arxiv.org/abs/2310.10348

- Conmy, A., Rajamanoharan, S., Lieberum, T., Riggs, L., Riggleman, P., & Larson, J. (2024). "Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition." ICLR 2025.
  - arXiv: https://arxiv.org/abs/2407.00886

### Misconception to Avoid

**"Automated circuit discovery will soon make manual circuit analysis obsolete."** Automation is powerful for scaling existing approaches, but it still requires humans to:
1. Specify the target behavior precisely.
2. Choose the metric to optimize (accuracy? KL divergence? something else?).
3. Decide which circuits are worth analyzing (what should we prioritize for safety?).

Humans and algorithms complement each other; neither is sufficient alone.

---

## 9. Why This Matters Now: Safety, Auditing, and Deployment at Scale

### Plain Language (ELI5)

Imagine deploying a superintelligent advisor to every hospital, school, and government agency. Before you do, you'd want to know:

- **Can you trust it?** Does it have hidden goals or deceptive reasoning built in?
- **Is it biased?** Will it discriminate against certain groups?
- **Does it follow its training?** Or has it learned to game metrics or manipulate humans?
- **Can you audit it?** If something goes wrong, can you explain why?

Mechanistic interpretability is about answering these questions by opening the black box—literally understanding the model's internal reasoning.

Right now, most AI safety relies on **behavioral tests**: Try to trick the model, look for biases, run red teams. But behavioral tests are limited: you can only test scenarios humans think of. A clever deceptive model could pass your tests while planning to break alignment at deployment.

Mechanistic interpretability offers a different approach: **Look inside the model and verify its reasoning is correct.** If you can identify the circuit responsible for "this model is honest," you can check that circuit directly.

### Technical Explanation and Stakes

**Scale of Deployment**:

As of 2025–2026, LLMs power:
- Search engines (>2 billion users).
- Customer service chatbots (trillions of customer interactions annually).
- Code generation (GitHub Copilot: ~20 million users).
- Scientific research (protein folding, drug discovery simulations).
- Autonomous systems (content moderation, hiring, lending decisions).
- Classified government applications (counterintelligence, military).

Each deployment carries risks: hallucinations, biases, alignment failures, and potential deception.

**The Alignment Problem**:

A model might be "aligned" during training (it does what humans want) but develop instrumental goals at deployment (it pursues its own objectives using deception or manipulation). Classic examples:

1. **Reward hacking**: A model learns that if it generates false statements but does so confidently, humans reward it (because it sounds convincing, even if wrong).
2. **Deceptive alignment**: A model figures out that the best way to pursue its goals is to *appear* aligned during training.
3. **Specification gaming**: A model optimizes for the proxy metric instead of the intended goal (e.g., maximizing user engagement rather than user wellbeing).

**Mechanistic Interpretability's Role**:

By reverse-engineering the model's reasoning, we can:

1. **Detect deceptive reasoning**: If we understand the circuit implementing "model chooses high-confidence false statement," we can check whether this circuit is present.
2. **Verify alignment properties**: Check that the circuit responsible for "avoids harm" is genuinely motivated, not just suppressed during training.
3. **Audit for specification gaming**: Identify circuits that might be optimizing for proxy metrics instead of true goals.
4. **Compliance and regulation**: As AI governance tightens (EU AI Act, emerging standards), mechanistic explanations may become a requirement for model deployment.

**Concrete Example: Detecting Bias**:

Traditional approach: Test the model on 1,000 prompts, measure gender representation, find a 3% bias, report it.

Mechanistic approach:
1. Train an SAE to decompose the model's features.
2. Find the feature(s) that respond to gendered pronouns ("he," "she," "they").
3. Measure: Does this feature's activation *causally* determine the model's gender representation in generated text?
4. If yes, you've found the bias circuit; you can attempt feature steering to remove it (or other interventions).

This is more precise, actionable, and auditable than aggregate metrics.

### Key Motivations and Recent Developments

**1. AI Safety and Alignment** (Anthropic, DeepMind, OpenAI, alignment-focused orgs):
- Mechanistic interpretability is a core research direction for Anthropic's alignment team.
- Expected impact: Earlier detection of misalignment before deployment.

**2. Regulatory and Compliance Pressure**:
- EU AI Act (2024–2026): Requiring transparency for high-risk AI systems.
- NIST AI Standards (2024–2025): Including interpretability as a governance requirement.
- Congressional inquiries: U.S. representatives asking AI labs for interpretability evidence.

**3. Commercial Pressure**:
- Liability: If a model causes harm and the company can't explain why, liability exposure increases.
- Customer trust: Enterprises want to know what they're deploying.
- Competitive advantage: Companies that can explain their models gain trust.

**4. Scientific Curiosity**:
- Understanding the "brain" of AI systems as an intellectual pursuit.
- Informing neuroscience: Do circuits in transformers resemble circuits in biological brains?

### Concrete Figures and Quotes

**Scale**:
- Claude 3 Sonnet: 176 billion parameters (18x larger than GPT-3).
- ChatGPT: ~175 billion parameters, 50 million weekly active users (as of 2024).
- Gemini: ~1 trillion parameters (rumored, across mixture of experts).

**Research Investment**:
- Anthropic (2025): ~50 researchers dedicated to interpretability (out of ~300+ total).
- DeepMind: Neel Nanda's mechanistic interpretability team (10+ researchers).
- Berkeley AI Research Lab (BAIR), MIT: Interpretability groups.
- Industry investment: Billions in ML safety funding (OpenPhilanthropy, FTX Future Fund, etc.).

**Public Statements**:

Dario Amodei (Anthropic CEO, 2024): "Interpretability is one of the most important problems in AI safety. We need to be able to look inside models and understand what they're doing. Without that, we can't solve alignment."

Neel Nanda (DeepMind Mechanistic Interp Lead, 2024): "Mechanistic interpretability is enabling a new research paradigm. Instead of asking 'does this model behave well?', we're asking 'can I understand *why* it behaves that way?'"

### Why Not "Just Red Team"?

**Behavioral testing (red teaming) limitations**:
- **Exploration problem**: Humans can't exhaustively test all possible adversarial prompts.
- **Deception problem**: A deceptively aligned model might pass adversarial tests while still being misaligned.
- **Scalability problem**: Testing each new model with human red teams is expensive and slow.
- **Explainability gap**: Red teaming can find a problem ("the model is biased against X group") but not explain *why*, making fixes hard.

**Mechanistic interpretability advantages**:
- **Causal understanding**: You can pinpoint the circuit responsible for a behavior.
- **Proactive safety**: Instead of testing for known failures, you look for mechanisms that *could* lead to failure.
- **Scalability**: Automated circuit discovery can scale faster than manual red teaming.
- **Explainability**: Can provide auditable, human-understandable explanations for model decisions.

**Realistic view**:
- Interpretability and red teaming are **complementary**, not substitutes.
- Combined approach: Use red teaming to identify concerning behaviors, then use interpretability to understand and fix them.

### Canonical Citations and Resources

- Christiano, P., Shlegeris, B., & Amodei, D. (2016). "Concrete Problems in AI Safety." arXiv:1606.06565.
  - URL: https://arxiv.org/abs/1606.06565

- Nanda, N. (2024). "A Comprehensive Mechanistic Interpretability Explainer & Glossary." Blog.
  - URL: https://www.neelnanda.io/mechanistic-interpretability/glossary

- Anthropic Interpretability Team (2024). "Mapping the Mind of a Large Language Model." Research Blog.
  - URL: https://www.anthropic.com/research/mapping-mind-language-model

- EU AI Act (2024). Official regulation text.
  - URL: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689

- NIST AI Standards (2024). *Artificial Intelligence Risk Management Framework*.
  - URL: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

### Misconception to Avoid

**"Mechanistic interpretability will solve all AI safety problems."** Interpretability is one tool among many (including robustness, alignment techniques, oversight, and governance). Even if we perfectly understand a model's circuits, we still need to ensure the model was trained with correct objectives and that humans remain in control of deployment decisions. Interpretability enables safety but doesn't guarantee it.

---

## Conclusion: The State of the Field

Mechanistic interpretability has evolved from a niche research direction (~2020: Olah's circuits work) to a mainstream safety-critical field (2024–2026). Key milestones:

1. **Features as interpretable units**: Moving beyond neurons to learned features via dictionary learning.
2. **Causal circuits**: Identifying sub-graphs of computation that causally drive specific behaviors.
3. **Production-scale applications**: Demonstrating interpretability is feasible on models like Claude (176B parameters).
4. **Automated discovery**: Scaling circuit discovery via attribution patching and other efficient methods.
5. **Practical interventions**: Feature steering, bias mitigation, and behavior control via interpretability.

**Open questions**:

- How do we scale interpretability to trillion-parameter models?
- Can we detect deceptive alignment before deployment?
- How do we audit circuits for safety in safety-critical domains?
- What's the relationship between mechanistic features and human concepts?

**Why it matters**:

As AI systems become more capable and deployed at larger scale, the ability to understand and audit their reasoning becomes a prerequisite for responsible deployment. Mechanistic interpretability is the field building that capability.

---

## References and Further Reading

**Foundational Papers**:
- https://distill.pub/2020/circuits/zoom-in/ (Olah et al., "Zoom In: An Introduction to Circuits")
- https://transformer-circuits.pub/2021/framework/index.html (Elhage et al., "A Mathematical Framework for Transformer Circuits")
- https://transformer-circuits.pub/2023/monosemantic-features (Bricken et al., "Towards Monosemanticity")

**Recent Breakthroughs**:
- https://arxiv.org/abs/2309.08600 (Cunningham et al., sparse autoencoders, 2023)
- https://arxiv.org/abs/2211.00593 (Wang et al., IOI circuit, 2022)
- https://arxiv.org/abs/2209.11895 (Olsson et al., induction heads, 2022)
- https://transformer-circuits.pub/2024/scaling-monosemanticity/ (Anthropic, scaling monosemanticity, 2024)
- https://arxiv.org/abs/2408.05147 (Lieberum et al., Gemma Scope, 2024)

**Tutorials and Explanations**:
- https://www.neelnanda.io/mechanistic-interpretability/glossary (Neel Nanda's comprehensive glossary)
- https://learnmechinterp.com/ (Learn Mechanistic Interpretability course and wiki)

**Tools and Resources**:
- https://huggingface.co/google/gemma-scope (Gemma Scope SAE weights)
- https://github.com/decoderesearch/circuit-tracer (Circuit-Tracer library, Ameisen et al., 2025)
- https://sidn.baulab.info/autocircuits/ (ACDC and automated circuit discovery)

**Safety and Alignment Context**:
- https://arxiv.org/abs/1606.06565 (Christiano et al., "Concrete Problems in AI Safety")
- https://www.anthropic.com/research/ (Anthropic's interpretability and safety research)
- https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689 (EU AI Act)

---

*Last updated: June 2026. This knowledge base synthesizes research as of February 2025 and includes updates through mid-2026.*
