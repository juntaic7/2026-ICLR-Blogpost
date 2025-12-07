---
layout: distill
title: How To Open the Black Box&#58 Modern Models for Mechanistic Interpretability
description: Understanding how transformers represent and transform internal features is a core challenge in mechanistic interpretability. Traditional tools like attention maps and probing reveal only partial structure, often blurred by polysemanticity and superposition. New model-based methods offer more principled insight&#58 Sparse Autoencoders extract sparse, interpretable features from dense activations; Semi-Nonnegative Matrix Factorization uncovers how neuron groups themselves encode concepts; Cross-Layer Transcoders track how these representations evolve across depth; and Weight-Sparse Transformers encourage inherently modular computation through architectural sparsity. Together, these approaches provide complementary pathways for opening the black box and understanding the circuits that underpin transformer behavior.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2026-04-27-interpret-model.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Traditional Interpretability Analysis
    subsections:
    - name: Attention Analysis
    - name: Probing
    - name: Why Traditional Methods Fall Short?
  - name: Sparse Autoencoder
    subsections:
    - name: Framework Overview
    - name: Layers & Activation
    - name: SAE Evaluation
    - name: Feature Evaluation
  - name: Semi-Nonnegative Matrix Factorization
    subsections:
    - name: Method
    - name: Discussions
  - name: Cross-Layer Transcoder
    subsections:
    - name: Architecture
    - name: Discussions
  - name: Weight-Sparse Transformer
    subsections:
    - name: Architecture
    - name: Discussions
  - name: Final Remarks

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction
As modern transformer models grow in scale and capability, understanding how they make decisions has become both more important and more difficult. We can observe the inputs we give them and the outputs they produce, but the computations inside each layer unfold within a dense, high-dimensional space where many interacting components influence one another. From the outside, these models appear to compute through a swirl of entangled activations, offering few clues about why a particular answer emerged. This opacity makes it hard to trust models, hard to diagnose their failures, and hard to improve them in principled ways.

Mechanistic interpretability (MI) aims to bridge this gap by treating neural networks not as black boxes but as algorithms whose internal structure we can study, decompose, and eventually understand. MI is guided by a few core questions <d-cite key='rai2025practicalreviewmechanisticinterpretability, sharkey2025openproblemsmechanisticinterpretability'></d-cite>:
1.	*What features do models represent internally?*
2.	*How are these features combined into computational pathways or "circuits"?*
3.	*And to what extent are these mechanisms universal across architectures and scales?*

The first of these problem, i.e. understanding what information the model represents, is the natural entry point. Before we can trace circuits or explain behaviors, we must identify the elementary pieces of computation: the internal features encoded in the residual stream. Early interpretability work approached this challenge through traditional tools such as attention visualizations, probing classifiers, and attribution methods. While these techniques provide useful glimpses into model behavior, they often expose only fragments of the underlying structure, obscured by polysemanticity and the superposition of many concepts within the same neurons.

This has led to a shift toward more model-based approaches that directly extract or induce interpretable structure from transformer activations. In this post, we explore four such methods: Sparse Autoencoders, which learn sparse latent features that disentangle the residual stream; Semi-Nonnegative Matrix Factorization, which decomposes MLP activations into neuron-grounded building blocks; Cross-Layer Transcoders, which connect these features across depth to reveal how computation unfolds layer by layer; and Weight-Sparse Transformers, which aim to build interpretability into the architecture itself by encouraging models to develop modular, monosemantic features during training. Together, these approaches form a growing toolkit for understanding the internal representations and circuits that govern transformer behavior.

## Traditional Interpretability Analysis
### Attention Analysis
Attention analysis examines the attention weights ($\alpha_{ij} = \text{softmax}_j(\frac{q_i\cdot k_j}{\sqrt{d_k}})$) inside a transformer to visualize which tokens the model attends to at each position. Different attention heads often specialize in distinct linguistic patterns, such as syntax, coreference, or semantic relations.
However, attention weights only show where information flows, not which features are being computed. They reveal connections between tokens, but they cannot tell us what internal concepts the model is using or how those concepts are represented.

<div style="display: flex; justify-content: center; margin: 20px 0;">
  <table id="tab:attention-example" style="border-collapse: collapse; width: auto;">
    <caption style="caption-side: top; padding: 8px; font-weight: bold; text-align: center;">Table 1. The Famous Attention Weights Example for Query Tokens</caption>
    <thead>
      <tr>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">Query</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">The</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">cat</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">sat</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">on</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">the</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">mat</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px;"><strong>"The"</strong></td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">1.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px;"><strong>"cat"</strong></td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.3</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.7</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px;"><strong>"sat"</strong></td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.1</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.7</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.2</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px;"><strong>"on"</strong></td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.05</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.1</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.6</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.25</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px;"><strong>"the"</strong></td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.1</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.1</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.2</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.3</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.3</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.0</td>
      </tr>
      <tr>
        <td style="padding: 6px 12px;"><strong>"mat"</strong></td>
        <td style="padding: 6px 12px; text-align: center;">0.0</td>
        <td style="padding: 6px 12px; text-align: center;">0.2</td>
        <td style="padding: 6px 12px; text-align: center;">0.1</td>
        <td style="padding: 6px 12px; text-align: center;">0.3</td>
        <td style="padding: 6px 12px; text-align: center;">0.2</td>
        <td style="padding: 6px 12px; text-align: center;">0.2</td>
      </tr>
    </tbody>
  </table>
</div>
The famous attention pattern example shown in [Table 1](#tab:attention-example) has a predominantly causal structure where each token attends primarily to itself and preceding tokens, with "sat" focusing heavily on "cat" (0.7) to capture the subject-verb relationship, and later tokens like "mat" distributing attention more broadly across the sequence.

However, attention analysis has significant limitations. High attention weights indicate correlation rather than causation, they show where the model looked but not whether that information actually influenced the output. Moreover, attention patterns become increasingly difficult to interpret in deeper layers where residual connections allow information to bypass attention mechanisms entirely.

### Probing
Probing evaluates what information is present in a model’s hidden states by freezing the model and training a simple classifier on top of its internal representations. If the probe succeeds, it indicates that the information is linearly accessible in that layer. Probing is therefore useful for discovering whether the model has learned to encode liguistic knowledge like part-of-speech tags, entities, sentiment, or syntactic structure.

However, a successful probe does not mean the model actually uses that information for its predictions, overly powerful probes may read out information that the model never makes use of, giving a misleading sense of interpretability.

### Why Traditional Methods Fall Short?
1. **Polysemanticity**
<br> Polysemanticity is like having a single light switch that controls three different rooms. It refers to the fact that a single neuron often responds to several unrelated concepts at once. Instead of representing one clean aspect, the neuron activates for a mixture of patterns. For example, the same neuron might correspond to a fruit ("apple"), a technology company ("Apple"), and even for unrelated acronyms like "doctor".<br>
  - **Attention analysis** only shows that the neuron or head activated, not *which meaning* triggered it.
  - **Probing** can confirm the presence of certain information, but not how multiple concepts are bundled together inside the same neuron.
2. **Superposition**
<br>Superposition means the representation is more like blended colors on a palette than neatly separated channels. It describes how transformers store more features than the neurons they have by overlapping multiple concepts in the same hidden space. Instead of assigning each feature its own neuron, the model encodes features as distinct patterns across many neurons. Different features reuse the same dimensions but with different activation signatures. <br>For example, [Table 2](#tab:superposition-example) shows an example where four neurons might simultaneously encode three different features, each represented by its own pattern across those same four coordinates.
   - **Attention analysis** reveals correlations but can't separate the mixed signals from overlapping features.
   - **Probing** detects that the model encodes a concept, but it also can't disentangle how many features are intertwined inside the same space.

<div style="display: flex; justify-content: center; margin: 20px 0;">
  <table id="tab:superposition-example" style="border-collapse: collapse; width: auto;">
    <caption style="caption-side: top; padding: 8px; font-weight: bold; text-align: center;">Table 2. Feature Superposition: A Toy Example</caption>
    <thead>
      <tr>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">Feature</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">N1</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">N2</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">N3</th>
        <th style="border-bottom: 1px solid #333; padding: 6px 12px;">N4</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px;"><strong>Feature A</strong></td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.1</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.2</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.3</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.4</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px;"><strong>Feature B</strong></td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.5</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.2</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.1</td>
        <td style="border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: center;">0.2</td>
      </tr>
      <tr>
        <td style="padding: 6px 12px;"><strong>Feature C</strong></td>
        <td style="padding: 6px 12px; text-align: center;">0.2</td>
        <td style="padding: 6px 12px; text-align: center;">0.2</td>
        <td style="padding: 6px 12px; text-align: center;">0.3</td>
        <td style="padding: 6px 12px; text-align: center;">0.3</td>
      </tr>
    </tbody>
  </table>
</div>

While attention patterns and probing results help us peek inside a model, they only scratch the surface. They show that information exists, but not how it is organized within the dense, superposed activation space. These limitations highlight the need for alternative interpretability methods for feature extraction!

## Sparse Autoencoder
Sparse Autoencoders (SAEs) are designed to solve one of the central challenges in mechanistic interpretability: the dense and superposed nature of transformer activations. Instead of working directly in the model’s tangled representation space, SAEs learn a new basis where features become sparse, separated, and often far more interpretable. This makes them a powerful tool for uncovering the building blocks of a model’s internal computation.

{% include figure.liquid 
   path="assets/img/2026-04-27-interpret-model/sae_framework.jpg" 
   class="img-fluid"
   id="fig:sae-framework"
   caption="Figure 1. The Framework of Sparse Autoencoder (SAE)<d-cite key='shu2025surveysparseautoencodersinterpreting'></d-cite>"
   zoomable=true 
%}


### Framework Overview
An overview of the SAE framwork is shown in [Figure 1](#Fig:sae-framework). We decompose the SAE framework into four main components: input representation, encoding, decoding, and training with the loss function.

#### Input
For a specific layer $l$ in the model we want to interpret (e.g. a Transformer), we denote the hidden representation of token $x_n$ as $z_n^{(l)}$. Each vector $z_n^{(l)}$ is treated as a single input for the SAE.

*Note:* The SAE takes one token’s representation at a time, not an entire sequence. However, this vector already encodes rich contextual information. By the time a token $x_n$ reaches layer $l$ in a Transformer, its representation already contains information about the all of the previous tokens, thanks to self-attention mechanism and positional encoding. In other words, the sequence context comes from the Transformer, while the SAE merely learns to decompose the resulting representation.

#### Encode
The input vector is transformed into a sparse activation $h(z)$ by:

$$h(z) = \sigma(W_{\text{enc}}z+b_{\text{enc}}),$$

where $\sigma$ is a sparsity-encouraging activation function (e.g., ReLU, Top-K, JumpReLU). The encoder maps the original $d$-dimensional activation into an overcomplete latent space of size $m$, where $m \gg d$. In practice, $m$ is often chosen to be $4\times$ to $8\times$ larger than the original dimension so that the model can represent many more features than the Transformer's native space allows.

#### Decode
Once the sparse activation $h(z)$ is computed, the SAE reconstructs the original representation through a linear decoding step:

$$\hat{z}= h(z)\cdot W_{\text{dec}}+b_{\text{dec}},$$

The decoder combines the active features in $h(z)$ to approximate the original input vector $z$. Each row of the decoder matrix $W_{\text{dec}}$ corresponds to a feature vector, namely a direction in activation space representing a distinct learned concept. The sparse activations in $h(z)$ forces to select a small subset of these feature vectors and combine them to approximate the original input vector $z$. Because only a few latent units are active for any given token, the reconstruction is built from a small, interpretable set of feature vectors, which is what gives SAEs their power for mechanistic analysis.

#### Loss Function
Training an SAE is all about balancing two goals:
1. *Reconstruct the original activation well.*
The SAE should be able to take a hidden representation from the transformer, break it into interpretable features, and then put it back together again. If the reconstruction is bad, the features aren’t capturing the right structure.
2. *Use as few features as possible.*
We want each input to activate only a small number of latent units so those activations are easy to interpret. If everything activates all the time, the resulting features won’t mean anything.

Hence, SAEs are trained with a loss that balances two objectives:
- **Reconsruction loss** to accurately reconstruct the original activation, and
- **Sparsity loss** to enforce sparsity in the latent activations.

The combined loss can be written as:

$$\mathcal{L}(z) \;=\;
\underbrace{\|z - \hat{z}\|_2^2}_{\text{Reconstruction loss}}
\;+\;
\underbrace{\lambda \|h(z)\|_1}_{\text{Sparsity loss}}.$$

Averaging over a dataset $\mathcal{X}$ gives:

$$\mathcal{L}(\mathcal{X})
= \frac{1}{|\mathcal{X}|}
\sum_{z \in \mathcal{X}}
\left(
\|z - \hat{z}\|_2^2
+
\lambda \|h(z)\|_1
\right).$$

The reconstruction loss is typically the mean squared error between the reconstructed vector $\hat{z}$ and the original input vector $z$, encouraging the decoder’s feature vectors to span the space of real activations.

The sparsity loss is usually an $L_1$ penalty on the latent activation $h(z)$, which encourages most latent units to stay inactive while allowing a few meaningful ones to activate.



*Note:*
The ideal sparsity measure is the $L_0$ norm that counts non-zero entries, but it is non-differentiable. The $L_1$ norm serves as its closest convex relaxation, making it practical for gradient-based optimization.

#### SAE vs. VAE
Since SAEs are a type of autoencoder, it’s natural to compare them to Variational Autoencoders (VAEs)—one of the most widely used autoencoder variants in deep learning. Although both architectures share the same high-level structure (an encoder, a latent space, and a decoder), they are designed for very different goals and impose different assumptions on what the latent space should look like. A detailed comparison is provided in [Table 3](tab:sae-vae-comparison).

<div class="table" id="tab:sae-vae-comparison">
  <table>
    <caption style="caption-side: top; padding: 8px; font-weight: bold; text-align: center;">Table 3. Comparison of Sparse Autoencoders and Variational Autoencoders</caption>
    <thead>
      <tr>
        <th>Model</th>
        <th>SAE (Sparse Autoencoder)</th>
        <th>VAE (Variational Autoencoder)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Goal</strong></td>
        <td>Learn sparse, interpretable features</td>
        <td>Learn continuous latent representations</td>
      </tr>
      <tr>
        <td><strong>Latent Space</strong></td>
        <td><strong>Overcomplete</strong> ($m \gg d$)</td>
        <td><strong>Compressed</strong> ($m \ll d$)</td>
      </tr>
      <tr>
        <td><strong>Constraint</strong></td>
        <td><strong>Sparsity</strong> (most activations = 0)</td>
        <td><strong>Probabilistic</strong> (latent distributions)</td>
      </tr>
      <tr>
        <td><strong>Architecture</strong></td>
        <td>Deterministic encoder/decoder</td>
        <td>Probabilistic encoder, deterministic decoder</td>
      </tr>
      <tr>
        <td><strong>Loss Function</strong></td>
        <td>Reconstruction + $L_1$ sparsity</td>
        <td>Reconstruction + KL divergence</td>
      </tr>
      <tr>
        <td><strong>Use Case</strong></td>
        <td><strong>Interpretability</strong> of neural networks</td>
        <td><strong>Generation</strong> and representation learning</td>
      </tr>
    </tbody>
  </table>
</div>

In summary, although SAEs and VAEs both belong to the autoencoder family, they operate in almost opposite regimes. VAEs **compress** information to learn smooth, generative latent spaces, while SAEs **expand** the latent space and enforce sparsity to uncover disentangled, human-readable features—making them especially well-suited for mechanistic interpretability.

### Layers & Activation
Sparse Autoencoders are intentionally designed to be shallow. In most interpretability work, an SAE consists of a single linear/ReLU encoder and a single linear decoder, sometimes with a small number of additional layers (but rarely more than 2–3). This stands in sharp contrast to the deep, nonlinear architecture of transformer models.

*Why SAE has so few layers?*
-	**Interpretability**: Every extra layer introduces more mixing and entanglement, making feature meanings harder to trace.
- **Simplicity**: Linear or single-nonlinearity encoders keep the learned features easy to inspect.
- **Sparsity**: Shallow networks respond more directly to $L_1$-induced sparsity and are less prone to hiding patterns behind multiple nonlinearities.

In short: *SAEs stay shallow so their learned features remain clean and interpretable.*

#### Activation
Transformers and other deep networks use smooth nonlinearities such as GeLU, SiLU, Tanh, or Sigmoid. These functions are excellent for training large models - yet terrible for learning sparse, interpretable features.

- **GeLU:** $\text{GeLU}(x)=x\cdot \Phi(x)$, where $\Phi(x)$ is the Gaussian CDF.
- **SiLU/Swish:** $\text{SiLU}(x) =  x\cdot \sigma(x)$, where $\sigma(x)$ is the logistic sigmoid.

These smooth activations rarely produce exact zeros; instead, they create soft, continuous outputs where nearly everything has a nonzero value - but SAEs need zeros, since sparsity is the entire idea!

**ReLU** ($\text{ReLU}(x) = \max(0, x)$) is the most common activation used in SAEs because it naturally encourages sparsity. Its hard cutoff at zero produces many exact zeros, making it much easier to determine when a particular feature is present or absent. This behavior aligns well with the interpretability goal: each active latent dimension can be treated as a clear, discrete signal.

That said, many modern SAE designs rely on even stronger sparsity mechanisms.

One example is the **Top-K** activation, which keeps only the K largest activations and sets all others to zero. This enforces a fixed level of sparsity, as exactly K "most relevant" features selected by the model are active per input. It avoids threshold tuning entirely, since K is a direct and intuitive hyperparameter. Top-K has become popular in interpretability work because it produces consistently clean, discrete feature usage across all samples.

Another sparsity-oriented activation used in SAEs is **JumpReLU**, which introduces a learnable activation threshold. Instead of activating as soon as the input becomes positive, JumpReLU only responds when the input exceeds a learned value $\theta$: 

$$\text{JumpReLU}(x) = \max(0,\, x - \theta).$$

This allows the model to determine how strong a signal must be before a feature should be "turned on" during training. The result follows a flexible but still highly sparse activation pattern. Many inputs fall below the learned threshold and produce exact zeros, while only sufficiently strong signals activate a feature. JumpReLU therefore becomes more adaptive than ReLU, but still far more interpretable than smooth activations like GeLU or SiLU.


### SAE Evaluation

[Table 4](tab:sae-eval) summarizes the key metrics used to evaluate SAEs across several models<d-cite key='shu2025surveysparseautoencodersinterpreting'></d-cite>. These metrics fall into two broad categories: structural metrics, which assess whether the SAE is a faithful surrogate for the original activations, and functional metrics, which assess whether the learned features behave as clean, interpretable units. Together, they provide a more complete picture of SAE quality than reconstruction loss alone.

#### Structural Metrics

Before assessing the interpretability of individual features, we must first determine whether an SAE is a reliable approximation of the original layer. Structural metrics answer this question by measuring both sparsity and reconstruction fidelity.

$L_0$ sparsity counts how many latents fire on a typical token. Lower values indicate a cleaner, more selective representation. However, overly aggressive sparsity can degrade reconstruction quality. Metrics such as MSE, cross-entropy loss, KL divergence, and explained variance quantify how well the SAE preserves the geometry and predictive behavior of the original activations. If a language model maintains similar next-token predictions when its hidden states are replaced by SAE reconstructions, the SAE sits on a good sparsity–fidelity frontier: sparse enough to be interpretable, but faithful enough not to distort the model's behavior.

#### Functional Metrics

Reconstruction quality alone does not guarantee interpretability. An SAE can produce low error while still learning "bad" features, e.g., latents that collapse multiple concepts or absorb unrelated signals. Functional metrics capture these failure modes.

**Absorption** measures how frequently the "correct" latent vector fails to activate and is replaced by an unrelated but correlated feature. Mean absorption tracks partial failures, while full absorption captures cases where none of the appropriate latents are activated. Low absorption indicates that concepts are represented consistently rather than being swallowed by a few dominant features.

**Spurious Correlation Removal** (SCR) tests whether the SAE isolates spurious features that contribute to shortcut behavior. By identifying and ablating latents most associated with a known spurious attribute, SCR quantifies how much debiasing occurs when removing the top 5, 50, or 500 such features. High SCR scores indicate that the SAE has cleanly separated true signal from superficial correlations.

Finally, **Sparse Probing** compares concept probes trained on SAE latents to probes trained on the model’s dense activations. When a probe using only a small number of SAE features matches or exceeds the dense baseline, it suggests that the SAE has discovered disentangled, concept-aligned representations. Conversely, poor sparse-probe performance shows that the SAE’s features, despite having good structural scores, are not semantically meaningful.

<div class="table" id="tab:sae-eval">
  <table>
    <caption style="caption-side: top; padding: 8px; font-weight: bold; text-align: center;">
      Table 4. Metrics for Evaluating SAEs
    </caption>
    <thead>
      <tr>
        <th>Category</th>
        <th>Metric</th>
        <th>What it Measures</th>
        <th>Interpretability Intuition</th>
      </tr>
    </thead>
    <tbody>
      <!-- Structural metrics -->
      <tr>
        <td rowspan="5"><strong>Structural</strong></td>
        <td><strong>L0 Sparsity</strong></td>
        <td>Average number of active latents per token.</td>
        <td>How sparse the SAE actually is; lower L0 means fewer features fire on each input.</td>
      </tr>
      <tr>
        <td><strong>MSE</strong></td>
        <td>Mean squared error between original activations and SAE reconstructions.</td>
        <td>Basic reconstruction fidelity: does the SAE preserve the underlying representation?</td>
      </tr>
      <tr>
        <td><strong>Cross-Entropy Loss</strong></td>
        <td>Next-token loss when the model is run on reconstructed activations instead of originals.</td>
        <td>Checks whether reconstruction is “good enough” for the language modeling task.</td>
      </tr>
      <tr>
        <td><strong>KL Divergence</strong></td>
        <td>KL between the original and SAE-reconstructed next-token distributions.</td>
        <td>Measures how much the SAE changes the model’s predictive distribution.</td>
      </tr>
      <tr>
        <td><strong>Explained Variance</strong></td>
        <td>Fraction of variance in activations captured by the SAE reconstruction.</td>
        <td>High variance explained means the SAE captures most of the geometry of the layer.</td>
      </tr>

      <!-- Functional: Absorption -->
      <tr>
        <td rowspan="2"><strong>Absorption</strong></td>
        <td><strong>Mean Absorption</strong></td>
        <td>Fraction of cases where the “correct” feature fails to activate and a similar latent fires instead.</td>
        <td>Lower is better; high values indicate that meaningful concepts are getting swallowed by unrelated latents.</td>
      </tr>
      <tr>
        <td><strong>Full Absorption</strong></td>
        <td>Stricter version where none of the correct latents activate and the concept is fully absorbed elsewhere.</td>
        <td>Detects severe failures where a concept is entirely misrepresented.</td>
      </tr>

      <!-- Functional: SCR -->
      <tr>
        <td rowspan="3"><strong>Spurious Correlation Removal (SCR)</strong></td>
        <td><strong>Top-5 SCR</strong></td>
        <td>Debiasing performance when ablating the 5 most spurious latents.</td>
        <td>Tests if a few well-chosen features can remove shortcut correlations.</td>
      </tr>
      <tr>
        <td><strong>Top-50 SCR</strong></td>
        <td>Same as above, but ablating the top 50 latents.</td>
        <td>Shows how much debiasing we gain with a modestly larger intervention.</td>
      </tr>
      <tr>
        <td><strong>Top-500 SCR</strong></td>
        <td>SCR when removing the top 500 spurious latents.</td>
        <td>Upper bound on how well the SAE separates true signal from spurious features.</td>
      </tr>

      <!-- Functional: Sparse probing -->
      <tr>
        <td rowspan="2"><strong>Sparse Probing</strong></td>
        <td><strong>LLM Probe</strong></td>
        <td>Probe accuracy using the original dense LLM activations.</td>
        <td>Baseline: how well concepts can be decoded from the unmodified model.</td>
      </tr>
      <tr>
        <td><strong>SAE Probe</strong></td>
        <td>Probe accuracy using a small number of SAE latents.</td>
        <td>Higher than or close to the LLM probe suggests clean, concept-aligned SAE features.</td>
      </tr>
    </tbody>
  </table>
</div>

Taken together, these structural and functional metrics tell us whether an SAE is a faithful and useful surrogate for the original model. However, they do not yet tell us what the individual features mean. Once we know that an SAE reconstructs well, maintains sparsity, and avoids major failure modes like absorption or spurious entanglement, we can shift our focus to the interpretability of the features themselves.

### Feature Evaluation
Natually, the next question is: **are the latent features actually correspond to meaningful concepts in the model?** Evaluation generally falls into two categories: input-based (what activates a feature) and output-based (what the feature does when changed).

#### Input-based Evaluation
Input-based analysis examines the inputs or hidden states that cause a feature to activate. Common methods include:
- Top activating examples: Inspect tokens or contexts where a feature is strongest. If they cluster around a clear linguistic pattern like plural nouns, numbers, or closing brackets, the feature is likely to be meaningful.
- Sparsity/selectivity measurements: Good features activate rarely and consistently for the same type of input.

These evaluations aims to answer the question: *"What concept is this feature detecting?"*

#### Output-based Evaluation
Output-based evaluation checks whether a feature plays a **causal** role in the model’s behavior:
- Activation patching: Replace or modify a feature’s activation during the forward pass (often using activations from a different run) to test whether that feature is necessary or sufficient for a behavior.
-	Feature-direction interventions: Add or subtract the feature’s decoder vector in Transformer's residual stream to examine whether that direction corresponds to a meaningful, causal concept.

These evaluations aim to address the question: *"Does this feature actually matter for the model’s computation?"*


## Semi-Nonnegative Matrix Factorization
SAEs have become the dominant tool for feature discovery in MI, largely because they provide a flexible, scalable way to learn disentangled directions in activation space. But SAEs also reveal an important limitation: they learn features from scratch, without reference to the model’s underlying mechanisms. In particular, SAEs trained on the residual stream often struggle to produce features that cleanly correspond to the computations inside the model’s MLP layers.
This motivates the next question:
*What if instead of learning new features, we directly decompose the model’s own MLP activations to reveal how neuron groups compose concepts?*

The recent Semi-Nonnegative Matrix Factorization (SNMF)<d-cite key='shafran2025decomposingmlpactivationsinterpretable'></d-cite> approach offers exactly this perspective. It bypasses the autoencoder architecture entirely and treats MLP activations themselves as the object to factorize, yielding features that are sparse combinations of real neurons, with coefficients that directly reveal which inputs activate which features.

### Method
The core idea behind SNMF is simple but powerful: instead of training a full encoder–decoder network like an SAE, directly factorize the MLP activation matrix into interpretable building blocks. This approach rests on the assumption that the MLP’s output to the residual stream can be expressed as a linear combination of underlying features. Because SNMF operates entirely on collected activations, it is a fully unsupervised, training-free method, since it does not modify the original model or require gradient-based optimization.

For a chosen MLP layer, SNMF gathers neuron activations across a sequence of n tokens, forming a matrix $A \in \mathbb{R}^{d_a \times n}$. The goal is to decompose this matrix as:

$$A \approx ZY,$$

where
- $$Z \in \mathbb{R}^{d_a \times k}$$ contains the MLP features, each column representing a sparse linear combination of neurons (a co-activation pattern), and
- $$Y \in \mathbb{R}^{k \times n}_{\ge 0}$$ is a nonnegative coefficient matrix indicating how strongly each feature contributes to each token’s activation vector.

The intuition is that neuron activations should combine additively and sparsely to produce higher-level concepts. Enforcing nonnegativity in $Y$ ensures these combinations remain parts-based: features cannot "subtract" from one another, making the representation more interpretable.

The factorization alternates between two update steps:
1.	Multiplicative updates for $Y$, which preserve nonnegativity:
$Y \leftarrow Y \odot \frac{Z^\top A}{Z^\top ZY + \epsilon},$
2.	Closed-form ridge-regression updates for $Z$:
$Z = A Y^\top (YY^\top + \lambda I)^{-1}.$

After each pair of updates, SNMF applies winner-take-all sparsification to the columns of $Z$, keeping only the largest-magnitude entries and setting the rest to zero. This encourages each feature to rely on a small, coherent subset of neurons.

Each SNMF feature can then be mapped back into the residual stream via the MLP's output projection matrix, making it directly comparable to SAE features and suitable for causal interventions such as steering or ablations. The result is a set of interpretable, neuron-grounded features that reveal how the MLP layer internally organizes semantic structure.

### Discussions
SNMF offers an appealing alternative to SAEs by grounding features directly in the model's own neurons rather than learning new latent directions. Because of the assumption that each feature is a sparse combination of real MLP neurons and each token’s activation is expressed as a nonnegative mixture of these features, the resulting representations are often more transparent. They reveal how groups of neurons cooperate to encode meaningful concepts, and the nonnegativity constraint in $Y$ provides a clean way to examine which tokens most strongly activate each feature. This gives SNMF a natural interpretability advantage: instead of discovering artificial directions in activation space, it exposes structure that already exists within the network.

However, despite its conceptual clarity, SNMF faces significant limitations when applied to modern language models. The method requires collecting large matrices of MLP activations and repeatedly factorizing them, which becomes computationally expensive as model width grows. Unlike SAEs, which can be trained incrementally using minibatches, SNMF depends on having access to large dense activation datasets at once, making memory and storage major bottlenecks. Its results also depend strongly on the dataset used to extract activations: if the chosen text distribution does not sufficiently cover the model's behaviors, important features may be omitted or fragmented. Furthermore, because SNMF restricts features to be linear combinations of neurons, it may fail to capture structure that emerges only at the level of directions or higher-dimensional subspaces; some concepts that SAEs can isolate simply cannot be expressed as sparse neuron combinations.

The optimization procedure itself can also be brittle. Multiplicative updates and sparsification thresholds may lead to unstable or inconsistent features, and small changes in hyperparameters can yield noticeably different decompositions. Finally, SNMF lacks the natural overcompleteness of SAEs: choosing too few features underfits the representation, while choosing too many risks redundancy or noise. In practice, these constraints make SNMF more suitable as a diagnostic tool for understanding local MLP structure rather than a scalable alternative to SAEs for whole-model feature extraction.

Even with these limitations, SNMF remains a valuable complement to SAE-based approaches. It provides insight into how the model's neurons themselves are organized and how co-activation patterns contribute to semantic representation. As a bridge between raw neuron-level analysis and learned feature extraction methods, SNMF helps illuminate the internal structure of MLP layers and sets the stage for deeper analyses of how features evolve across depth.


## Cross-Layer Transcoder
While SAEs and SNMF both uncover meaningful feature structure within a single layer, they offer only a static view of the model. Neither approach tells us how these features transform as they propagate forward through the network. In practice, transformer representations change dramatically from layer to layer—features can split into multiple subfeatures, merge into broader abstractions, fade out, or invert their meaning entirely. These dynamics are often invisible if we inspect each layer independently. To move from “feature discovery” to understanding computation as a multi-step process, we need a method that aligns feature spaces across depth and traces how information flows between layers. This is the motivation behind the Cross-Layer Transcoder(CLT)<d-cite key='dunefsky2024transcodersinterpretablellmfeature'></d-cite>: a small learned model that maps the activation features of one layer into the feature basis of another. Where SAEs focus on extracting sparse, interpretable features from a single layer, CLTs let us track how those features transform as the representation flows forward through the network.

### Architecture

As is shown in [Figure 2](fig:clt), CLT is designed to mirror how a transformer updates its residual stream across depth. It consists of a collection of "features" arranged into the same number of layers as the underlying transformer. Each layer of the CLT reads from the transformer’s residual stream at that depth and contributes to reconstructing the MLP outputs of that layer and all subsequent layers.

{% include figure.liquid 
   path="assets/img/2026-04-27-interpret-model/clt.jpg" 
   class="img-fluid"
   id="fig:clt"
   caption="Figure 2. Replacement Model Constructed by Cross-Layer Transcoder (CLT)<d-cite key='dunefsky2024transcodersinterpretablellmfeature'></d-cite>"
   zoomable=true 
%}

At layer $\ell$, the CLT first encodes the transformer’s residual activation $x_\ell$ using a learned linear map followed by a sparsifying nonlinearity (typically **JumpReLU**):

$$a_\ell = \text{JumpReLU}(W^{(\ell)}_{\text{enc}} x_\ell),$$

where $a_\ell$ is the vector of CLT feature activations for that layer. These features are "cross-layer" because they are not restricted to influencing only one layer. Instead, each feature at layer $\ell$ can help reconstruct the MLP outputs $y_{\ell',\, \ell' \ge \ell}$, via separate decoder weights for each downstream layer:

$$\hat{y}_{\ell} = \sum_{\ell' = 1}^{\ell} W^{(\ell')\to \ell}_{\text{dec}}\, a_{\ell'}.$$

In other words, the transcoder at layer $\ell$ has a shared encoder but multiple decoders, each responsible for predicting the MLP output of a different downstream layer. This design makes each feature a stable, reusable computational unit: the encoder determines what the feature detects at layer $\ell$, while the decoders determine where and how that feature influences the rest of the model.
Thus, the MLP output at a given layer is reconstructed jointly from all features in that layer and all earlier layers.

All cross-layer transcoders are trained jointly as a single end-to-end model. Although each layer $\ell$ has its own encoder and its own set of decoders targeting every downstream layer, the entire stack of encoders and decoders is optimized together. During training, the CLT reads the transformer’s residual stream at each layer to produce feature activations, and the combined contributions of all earlier features are used to reconstruct the MLP output at every layer. The loss sums **reconstruction** errors across all layers, together with a **sparsity penalty** that encourages selective activations and minimal decoder weights. Because gradients flow through all layers at once, features learned at early depths become reusable building blocks for predicting later MLP outputs, allowing the CLT to capture the model’s computation as a coherent, multi-step transformation rather than a collection of isolated layer-wise mappings.

In practice, CLTs can have as much as hundreds of thousands to tens of millions of features across layers, but still remain far smaller than the transformer models they interpret. They are trained on recorded activations from the base model, using standard optimization methods. Once trained, a CLT provides a structured, layer-aligned view of how representations evolve, merge, and transform as they propagate forward through the transformer.

### Discussions

Despite their conceptual appeal, CLTs are difficult to deploy at scale, as each layer must have its own encoder and multiple decoder matrices aimed at every subsequent layer. For modern LLMs with dozens of layers and thousands of channels, the total parameter count quickly explodes, even though the CLT is only modeling the MLP pathways. Training such a model requires storing the full residual stream and MLP outputs for vast numbers of tokens, making data collection and GPU memory usage significant bottlenecks.

Even after training, CLTs face interpretability challenges. Because reconstruction quality matters for every layer, early-layer features must support many downstream predictions, causing heavy coupling between layers. This often makes learned features diffuse rather than crisp, especially in deeper models where errors compound. Moreover, CLTs capture correlations in activations, not causal mechanisms; they approximate how the model tends to update its representations, but cannot guarantee that the learned mapping reflects the true internal computation. As depth increases, nonlinear or context-specific transformations are also harder to approximate with a simple transcoder.

Yet CLTs remain a valuable research tool when used appropriately. They provide a structured way to trace how representations flow across layers, offering a layer-aligned view of model computation that complements within-layer methods like SAEs. 

## Weight-Sparse Transformer
SAE, SNMF and CLT provide powerful post-hoc tools for disentangling features inside large language models, but they don’t change the model itself. A complementary line of work asks a more radical question: *What if we trained the model from scratch to be interpretable?*

This motivates Weight-Sparse Transformers (WSTs)  <d-cite key=gao2025weightsparsetransformersinterpretablecircuits></d-cite>, models in which most parameters are encouraged to become exactly zero during training. Instead of discovering meaningful features after the fact, WSTs reshape the model so that features are simpler, more local, and more monosemantic, thus can potentially make feature-extraction tools like SAEs and transcoders more faithful and informative.

### Architecture
WSTs preserve the layout of standard transformers, i.e. attention blocks, MLP blocks, and residual connections. However, they differ in one fundamental respect: every major weight matrix is kept sparse throughout training. Instead of learning dense projections, the model is forced to operate with only a small subset of nonzero weights, which in turn encourages each attention head and MLP neuron to interact with only a limited number of residual channels.

WSTs rely on a simple but powerful mechanism to enforce sparsity: deterministic magnitude pruning. After each optimizer update, the model applies an **AbsTopK** operation to *every* weight matrix, retaining only the weights with the largest absolute values and setting all others to zero. Because this pruning happens during every training step, the model adapts early on to the sparse connectivity pattern and learns to route computation through a stable, minimal set of connections. The global sparsity level is gradually increased over the course of training, guiding the transformer from a nearly dense regime to a highly sparse one. By the end of training, many rows and columns of the attention and MLP matrices consist entirely of zeros, and the remaining structure reveals a compact computational graph that the transformer actually uses.

The effects of this sparse architecture ripple through the entire model. In the attention layers, the Q, K, V, and output projection matrices contain many zero entries, so each head reads from and writes to only a few chosen channels. Attention patterns become localized, reducing entanglement and making the function of each head easier to understand. The same dynamic appears in the MLP blocks, where each neuron receives input from only a handful of residual coordinates and produces updates to only a few output dimensions. Rather than acting as diffuse nonlinear mixers, MLP neurons become simple feature detectors with clear roles.

Because both attention and MLP updates are sparse, each layer modifies only a small portion of the residual stream. Dense transformers rewrite nearly the entire hidden vector at every layer, mixing information in hard-to-decipher ways. In contrast, WSTs update only the coordinates that matter, producing cleaner, more modular computation. The resulting circuits are far smaller and easier to trace, enabling a level of mechanistic clarity that dense models rarely exhibit.

### Discussions

While WSTs offer clean, interpretable circuits, they come with notable trade-offs. Continuous magnitude pruning makes training less stable and less efficient, and the resulting models generally underperform dense transformers at the same scale. Because sparsity forces the network into a rigid connectivity pattern, WSTs struggle to match the flexibility and capacity of dense architectures.

These limitations make WSTs valuable primarily as scientific probes rather than practical LLM replacements. They are excellent for studying modular computation, but unlikely to scale to state-of-the-art performance. Future work may focus on hybrid models that blend sparse and dense components, or on training objectives that encourage structure without imposing hard constraints.

This perspective naturally bridges back to SAEs. SAEs provide a post-hoc interpretability layer for dense models, recovering disentangled features without altering the architecture, while WSTs build interpretability directly into the model by enforcing sparse computation. Together, they illustrate two complementary strategies: discover structure inside dense models and induce structure during training—a promising direction for future research.

## Final Remarks

Mechanistic interpretability is steadily moving beyond one-off visualizations and heuristic reasoning toward something more systematic—a science built on feature extraction, circuit reconstruction, and architecture-level constraints. Sparse Autoencoders have shown that dense layers contain rich, recoverable structure; Semi-Nonnegative Matrix Factorization reveals how those structures relate to the model’s own neurons; Cross-Layer Transcoders expose how representations transform as they propagate through depth; and Weight-Sparse Transformers explore how we might build models whose internal computations are disentangled from the start. Each technique brings its own strengths and limitations, but together they form an increasingly coherent toolkit for understanding the computations inside modern transformers.

Interpretability remains a profound challenge: models continue to grow, tasks become more complex, and no single method will decode every aspect of their internal reasoning. Yet the shift toward feature-level and circuit-level approaches marks real progress. By treating neural networks not as opaque function approximators but as systems composed of reusable computational primitives, we are beginning to see how high-level behaviors emerge from low-level structure. The work ahead will involve hybrid techniques, new representational abstractions, and architectures designed with interpretability in mind. But the direction is clear. Opening the black box is no longer merely a philosophical aspiration—it is becoming a practical engineering discipline, and its foundations are starting to solidify.