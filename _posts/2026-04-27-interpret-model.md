---
layout: distill
title: How To Open the Black Box&#58 Modern Models for Mechanistic Interpretability
description: Understanding how transformers represent and transform internal features is a core challenge in mechanistic interpretability. Traditional tools like attention maps and probing reveal only partial structure, often blurred by polysemanticity and superposition. More principled alternatives work by recovering interpretable structure directly from activations&#58 Sparse Autoencoders extract sparse, disentangled features from the residual stream; Semi-Nonnegative Matrix Factorization decomposes MLP activations into neuron-grounded building blocks; Cross-Layer Transcoders trace how these features propagate and transform across depth. Together, they form a coherent, feature-centric framework for understanding what transformers learn and how they compute.
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
  - name: Juntai Cao
    affiliations:
      name: University of British Columbia
  - name: Xiang Zhang
    affiliations:
      name: University of British Columbia
  - name: Raymond Li
    affiliations:
      name: University of British Columbia
  - name: Jiarui Ding
    affiliations:
      name: University of British Columbia
# must be the exact same name as your blogpost
bibliography: 2026-04-27-interpret-model.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Traditional Approaches
    subsections:
    - name: Attention Analysis
    - name: Linear Probing Classifier
    - name: Causal Perturbation
    - name: Why Traditional Methods Fall Short?
  - name: Sparse Autoencoder
    subsections:
    - name: Framework Overview
    - name: Layers & Activation
    - name: Implementation
    - name: SAE Evaluation
    - name: Feature Evaluation
  - name: Sparse Autoencoder Variants
    subsections:
    - name: Gated SAE
    - name: Matryoshka SAE
  - name: Semi-Nonnegative Matrix Factorization
    subsections:
    - name: Method
    - name: Discussion
  - name: Transcoder
    subsections:
    - name: Cross Layer Transcoder
    - name: Discussion
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
As transformer models continue to grow in scale and capability, understanding how they arrive at their decisions has become both more important and more challenging. While we can observe the inputs we provide and the outputs they produce, the computations in between unfold within dense, high-dimensional internal states where many interacting components jointly influence the final prediction. This limited visibility makes it difficult to build trust in model behavior, diagnose failures, or improve systems in a principled way.

Mechanistic interpretability (MI) aims to address this challenge by treating neural networks not as opaque black boxes but as computational systems whose internal algorithms can be analyzed and understood. Broadly, MI is guided by three central questions: <d-cite key='rai2025practicalreviewmechanisticinterpretability, sharkey2025openproblemsmechanisticinterpretability'></d-cite>:
1.	*What information do models represent internally?*
2.	*How is this information transformed into intermediate computations or circuits?*
3.	*How consistent are these mechanisms across models, scales, and training settings?*

A natural starting point is to understand what the model represents internally, i.e. **feature**-level interpretability. In this context, a feature is not synonymous with the hidden dimension $d$ in neural network representations.  
Rather, a *feature* refers to *properties of the input which a sufficiently large network will reliably dedicate a neuron to representing* <d-cite key='Engels2024NotAL,engels2025decomposing'></d-cite>, or, more succinctly, *properties of the input that activate particular mechanisms* <d-cite key='braun2025interpretabilityparameterspaceminimizing'></d-cite>. Unlike raw hidden-state activations, which are typically dense and heavily entangled, features are treated as approximately independent latent variables, each corresponding to a distinct computational factor. These factors may align with human-interpretable concepts, or may instead capture previously unknown but functionally meaningful patterns learned by the model.

Early interpretability work approached this problem using tools such as attention analysis, probing classifiers, and attribution methods. These techniques provide useful diagnostic signals indicating that particular information or behaviors are present and sometimes suggest where they may be expressed&mdash;for example, revealing token interaction patterns, decodable signals in representations, or sensitivities of outputs to inputs&mdash;though they do not by themselves establish the causal mechanisms responsible.
However, they primarily offer observational perspectives on representations rather than explicitly modeling the underlying computational variables, making their conclusions largely correlational. Because they do not identify the latent features that constitute the model’s internal basis, they provide limited insight into how representations are organized or combined, motivating the development of methods that directly extract interpretable features from activations.

These limitations have prompted a shift toward methods that directly extract interpretable structure from transformer activations. In this post, we explore three such approaches. Sparse Autoencoders learn sparse latent features that disentangle polysemantic representations in the residual stream. Semi-Nonnegative Matrix Factorization decomposes MLP activations into neuron-grounded building blocks, linking abstract features back to concrete model components. Cross-Layer Transcoders connect these features across depth, tracing how computation unfolds from layer to layer. Together, these approaches form a growing toolkit for understanding how transformers encode and transform information through learned features.


## Traditional Approaches
### Attention Analysis
Attention analysis <d-cite key="NIPS2017_3f5ee243, clark-etal-2019-bert"></d-cite> 
examines the attention weights 
inside a transformer to understand how the model routes information across token positions. 
Specifically, each weight $\alpha_{ij}=\text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)$ quantifies the degree to which position $i$ aggregates 
information from a previous position $j$ when constructing its updated representation, with the softmax  normalization ensuring these weights form a probability distribution over all key positions.  In this way, attention analysis reveals the token-to-token relevance structure the model  implicitly computes at each layer.

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

**Limitations.** While attention analysis offers an intuitive window into token-level interactions, its interpretive scope is constrained in several important ways:
1. Attention weights reflect where information flows, but not what features are being extracted or how they are transformed---two attention heads with identical weight patterns may compute entirely different value projections and thus contribute differently to downstream representations.
2. High attention weight between two tokens establishes a correlation in information routing, not a causal relationship: ablation studies have shown that many high-weight connections can be suppressed with minimal effect on model outputs, while some low-weight connections prove critical
3. Interpretability degrades in deeper layers, where residual connections allow representations to carry information forward without passing through attention at all, making the correspondence between attention patterns and model behavior increasingly indirect.


### Linear Probing Classifier

Linear probing classifiers <d-cite key="alain2017understanding,conneau-etal-2018-cram, tenney-etal-2019-bert"></d-cite> evaluate what information is encoded in a model's internal representations by training a lightweight classifier on top of frozen hidden states. Specifically, given a representation $ h_i^{(\ell)} \in \mathbb{R}^d $ from layer $ \ell $, a probe learns a mapping $ f_\phi : \mathbb{R}^d \rightarrow \mathcal{Y} $ that predicts a target property $ y \in \mathcal{Y} $, such as part-of-speech tags, syntactic depth, or named entity labels. 

During probing, the underlying model parameters remain fixed while only the probe parameters  are trained. The probe's accuracy on held-out data therefore provides an estimate of how easily the target property can be linearly decoded from the representation at layer $ \ell $. In other words, probing measures the *linear accessibility* of information contained in the representation rather than how the model actually computes or uses that information.

During probing, the underlying model parameters remain fixed while only the probe parameters $ \phi $ are optimized by minimizing a task-specific loss (typically cross-entropy for categorical properties over a labeled dataset)​. The probe's accuracy on held-out data therefore provides an estimate of how easily the target property can be linearly decoded from the representation at layer $\ell$. Critically, each probe tests for a single pre-specified property at a time; evaluating multiple properties requires training separate probes independently, each demanding its own labeled dataset, as illustrated in [Figure 1](#fig:linear-probe).

{% include figure.liquid 
   path="assets/img/2026-04-27-interpret-model/linear_probe.jpg" 
   class="img-fluid"
   id="fig:linear-probe"
   caption="Figure 1. Structure of Linear Probing Classifier"
   zoomable=true 
%}

**Limitations.** Despite their simplicity, linear probes provide only limited interpretive insight:

1. A successful probe demonstrates that a property is recoverable from a representation, but it does not imply that the model relies on that information during inference. The decoded feature may be an epiphenomenon of training rather than a causally active component of the model's computation.
2. Probing treats representations as static vectors and does not reveal how information is created, transformed, or routed through the network. As a result, probing offers a diagnostic view of representational content rather than an explanation of the model's internal computation.
3. The one-probe-one-feature design means that probing can only reveal the presence of features that researchers already hypothesize and operationalize as classification targets, leaving any unanticipated or unlabeled structure in the representation undetected.

### Causal Perturbation

Causal perturbation (or activation patching) <d-cite key="NEURIPS2020_92650b2e, 10.5555/3600270.3601532"></d-cite> methods  measure the causal contribution of internal model components by intervening on intermediate activations and observing the resulting change in model outputs. In transformer models, these interventions are typically applied to activations within specific layers and token positions.
Formally, let $h^{(\ell)}_i$ denote the activation vector at layer $\ell$ and token position $i$ produced during a clean run of the model on input $x$, as illustrated in [Figure 2](#fig:causal-intervention). Let $\tilde{h}^{(\ell)}_i$ denote the corresponding activation produced during a corrupted run, where the input has been modified to disrupt the behavior of interest. Then an intervened forward pass is constructed by which the clean activation is replaced with the corrupted activation at a chosen location $h^{(\ell)}_i \leftarrow \tilde{h}^{(\ell)}_i.$

Next, the effect of this intervention is measured by comparing the model’s output distribution before and after the patch:

$$\Delta p = p(y \mid \text{intervened run}) - p(y \mid \text{clean run}).$$

A large change in output probability $\Delta p$ indicates that the activation at position $i$ in layer $\ell$ plays a causal role in mediating the behavior under investigation.

{% include figure.liquid 
   path="assets/img/2026-04-27-interpret-model/causal-intervention.png" 
   class="img-fluid"
   id="fig:causal-intervention"
   caption="Figure 2. Structure of Causal Intervention"
   zoomable=true 
%}

**Limitations.** Unlike attention analysis and linear probing, which are purely observational, causal perturbation provides interventional evidence by modifying internal activations and measuring the resulting change in model outputs. These methods aim to identify components whose activations causally contribute to a given model behavior. Yet, causal perturbation has notable limitations as an interpretability method:
1. Causal perturbation identifies *where* causally relevant information is localized (i.e. a specific layer, position, or attention head,) but not *what* feature or concept is encoded there. The content of the patched activation remains opaque.
2. Much like linear probing, the method is also inherently hypothesis-driven: one must specify in advance both the behavior of interest and the corruption to apply, meaning unexpected or unlabeled computations go undetected.
3. Interpreting $\Delta p$ as a clean measure of causal importance is complicated by the fact that patching a single component may propagate effects through residual connections and subsequent layers, making it difficult to isolate the contribution of any single component from the broader computational graph.


### Why Traditional Methods Fall Short?
The three methods reviewed above — attention analysis, linear probing, and causal perturbation — each illuminate a different facet of model behavior. Yet all three share a common limitation: they operate on the surfaces of model representations rather than on the underlying structure of the feature space itself. This limitation becomes acute in the face of two well-documented phenomena in transformer representations: **polysemanticity** and **superposition**.

**Polysemanticity** refers to the tendency of individual neurons to respond to multiple semantically unrelated concepts. A single MLP neuron, for instance, may activate for the fruit "apple," the technology company "Apple," and the abbreviation "APL" — not because these concepts are related, but because the model finds it efficient to reuse representational capacity across rare features.

**Superposition** compounds this problem at the level of the entire representation space. Because transformers must compress a large number of real-world features into a relatively low-dimensional activation space, features are not allocated one neuron apiece — instead, they are encoded as distinct, near-orthogonal directions distributed across many neurons simultaneously (as illustrated in [Table 2](#tab:superposition-example)). As a result, no single neuron is a feature; features only exist as distributed patterns across the full activation vector.

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

All three methods fail for the same underlying reason: they treat the neuron as the fundamental unit of analysis, yet no single neuron cleanly encodes a single concept. Attention analysis fails at *detection*: it sees that a neuron activates but cannot determine which meaning triggered it. Linear probing fails at *decoding*: the direction it recovers is a blend of co-encoded concepts rather than a clean signal for any one of them. Causal perturbation fails at *attribution*: it can confirm that a neuron is causally relevant, but cannot identify which of its mixed representations is doing the causal work.

Taken together, these limitations point to a fundamental gap: the traditional toolkit can tell us *what* a model encodes and *where* it looks, but not how features are geometrically organized within the activation space. Addressing this gap requires methods capable of explicitly decomposing the dense, superposed activation space into interpretable, monosemantic components — motivating the sparse dictionary learning approaches discussed in the following sections.

## Sparse Autoencoder
Sparse Autoencoders (SAEs) are designed to solve one of the central challenges in mechanistic interpretability: the dense and superposed nature of transformer activations. Instead of working directly in the model’s tangled representation space, SAEs learn a new basis where features become sparse, separated, and often far more interpretable. This makes them a powerful tool for uncovering the building blocks of a model’s internal computation.

{% include figure.liquid 
   path="assets/img/2026-04-27-interpret-model/sae-diagram-light.png" 
   class="img-fluid"
   id="fig:sae-framework"
   caption="Figure 3. The Framework of Sparse Autoencoder (SAE)<d-cite key='shu2025surveysparseautoencodersinterpreting'></d-cite>"
   zoomable=true 
%}


### Framework Overview
An overview of the SAE framwork is shown in [Figure 1](#Fig:sae-framework). We decompose the SAE framework into four main components: input representation, encoding, decoding, and training with the loss function.

#### Input
For a specific layer $l$ in the model we want to interpret (e.g. a Transformer), we denote the hidden representation of token $x_n$ as $z_n^{(l)}$. Each vector $z_n^{(l)}$ is treated as a single input for the SAE.

*Note:* The SAE takes one token’s representation at a time, not an entire sequence. However, this vector already encodes rich contextual information. By the time a token $x_n$ reaches layer $l$ in a Transformer, its representation already contains information about all of the previous tokens, thanks to the self-attention mechanism and positional encodings. In other words, the sequence context comes from the Transformer, while the SAE merely learns to decompose the resulting representation.

#### Encode
The input vector is transformed into a sparse activation $h(z)$ by:

$$h(z) = \sigma(W_{\text{enc}}z+b_{\text{enc}}),$$

where $\sigma$ is a sparsity-encouraging activation function (e.g., ReLU, Top-K, JumpReLU). The encoder maps the original $d$-dimensional activation into an overcomplete latent space of size $m$, where the feature dictionary size $m \gg d$. In practice, $m$ is often chosen to be $4\times$ to $8\times$ larger than the original dimension so that the model can represent many more features than the Transformer's native space allows.

#### Decode
Once the sparse activation $h(z)$ is computed, the SAE reconstructs the original representation through a linear decoding step:

$$\hat{z}= h(z)\cdot W_{\text{dec}}+b_{\text{dec}},$$

The decoder combines the active features in $h(z)$ to approximate the original input vector $z$. Each row of the decoder matrix $W_{\text{dec}}$ corresponds to a feature vector, namely a direction in activation space representing a distinct learned concept. The sparse activations in $h(z)$ force the model to select a small subset of these feature vectors and combine them to approximate the original input vector $z$. Because only a few latent units are active for any given token, the reconstruction is built from a small, interpretable set of feature vectors, which is what gives SAEs their power for mechanistic analysis.

#### Loss Function
Training an SAE is all about balancing two goals:
1. *Reconstruct the original activation well.*
The SAE should be able to take a hidden representation from the transformer, break it into interpretable features, and then put it back together again. If the reconstruction is bad, the features aren’t capturing the right structure.
2. *Use as few features as possible.*
We want each input to activate only a small number of latent units so those activations are easy to interpret. If everything activates all the time, the resulting features won’t mean anything.

Hence, SAEs are trained with a loss that balances two objectives:
- **Reconstruction loss** to accurately reconstruct the original activation, and
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
The ideal sparsity measure is the $L_0$ norm that counts non-zero entries, but it is non-differentiable. The $L_1$ norm serves as its standard differentiable surrogate, making it practical for gradient-based optimization.

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

#### Why does SAEs have so few layers?
-	**Interpretability**: Every extra layer introduces more mixing and entanglement, making feature meanings harder to trace.
- **Simplicity**: Linear or single-nonlinearity encoders keep the learned features easy to inspect.
- **Sparsity**: Shallow networks respond more directly to $L_1$-induced sparsity and are less prone to hiding patterns behind multiple nonlinearities.

In short: *SAEs stay shallow so their learned features remain clean and interpretable.*

#### Activation
Transformers and other deep networks use smooth nonlinearities such as GeLU, SiLU, Tanh, or Sigmoid. These functions are excellent for training large models --- yet terrible for learning sparse, interpretable features.

- **GeLU:** $\text{GeLU}(x)=x\cdot \Phi(x)$, where $\Phi(x)$ is the Gaussian CDF.
- **SiLU/Swish:** $\text{SiLU}(x) =  x\cdot \sigma(x)$, where $\sigma(x)$ is the logistic sigmoid.

These smooth activations rarely produce exact zeros; instead, they create soft, continuous outputs where nearly everything has a nonzero value --- but SAEs need zeros, since sparsity is the entire idea!

**ReLU** ($\text{ReLU}(x) = \max(0, x)$) is the most common activation used in SAEs because it naturally encourages sparsity. Its hard cutoff at zero produces many exact zeros, making it much easier to determine when a particular feature is present or absent. This behavior aligns well with the interpretability goal: each active latent dimension can be treated as a clear, discrete signal.

That said, many modern SAE designs rely on even stronger sparsity mechanisms.

One example is the **Top-$K$** activation, which keeps only the $K$ largest activations and sets all others to zero. This enforces a fixed level of sparsity, as exactly $K$ features are active per input. It avoids threshold tuning entirely, since $K$ is a direct and intuitive hyperparameter. Top-$K$ has become popular in interpretability work because it produces consistently clean, discrete feature usage across all samples.

Another sparsity-oriented activation used in SAEs is the **JumpReLU** family, which introduces a learnable threshold $\theta$ per feature. Instead of activating as soon as the input becomes positive, these activations only respond when the input exceeds $\theta$. Two variants are commonly used:
 
$$\text{JumpReLU}_\theta(x) = \begin{cases} x & \text{if } x > \theta \\ 0 & \text{otherwise,} \end{cases}$$
 
$$\text{ShiftedReLU}_\theta(x) = \max(0,\, x - \theta).$$
 
Both variants allow the model to learn how strong a signal must be before a feature is activated. The key difference is in what they output when the gate is open: JumpReLU passes $x$ through unchanged, while ShiftedReLU subtracts the threshold, shifting the activation down by $\theta$. The result in either case is a flexible but still highly sparse activation pattern — many inputs fall below the learned threshold and produce exact zeros, while only sufficiently strong signals activate a feature. Both variants are therefore more adaptive than plain ReLU, yet far more interpretable than smooth activations like GeLU or SiLU.

### Implementation
 
A typical JumpReLU SAE implementation is shown below. Note that we use `nn.Parameter` directly rather than `nn.Linear`, since SAEs require custom constraints that `nn.Linear` doesn't support cleanly, as discussed after the code.
 
```python
class JumpReLUSAE(nn.Module):
    """
    Sparse Autoencoder (SAE) using the JumpReLU activation function.
 
    JumpReLU gates each pre-activation against a learned per-feature threshold,
    producing strictly sparser representations than standard ReLU by zeroing out
    weakly active features entirely rather than merely penalising them.
 
    Args:
        d_model:    Dimensionality of the input residual stream.
        d_features: Number of SAE dictionary features (typically d_features >> d_model).
    """
 
    def __init__(self, d_model: int, d_features: int):
        super().__init__()
        self.W_enc     = nn.Parameter(torch.rand(d_model,    d_features))  # (d_model,    d_features)
        self.W_dec     = nn.Parameter(torch.rand(d_features, d_model))     # (d_features, d_model)
        self.threshold = nn.Parameter(torch.rand(d_features))              # per-feature JumpReLU threshold
        self.b_enc     = nn.Parameter(torch.rand(d_features))              # encoder bias
        self.b_dec     = nn.Parameter(torch.rand(d_model))                 # decoder / re-centring bias
 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the input into the feature dictionary and apply JumpReLU.
 
        Args:
            x: Residual stream tensor of shape (..., d_model).
 
        Returns:
            Sparse feature activations of shape (..., d_features).
        """
        pre_activations = x @ self.W_enc + self.b_enc
        gate            = (pre_activations > self.threshold)  # hard binary mask per feature
        feature_acts    = gate * F.relu(pre_activations)      # zero out sub-threshold features
        return feature_acts
 
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the residual stream from sparse feature activations.
 
        Args:
            feature_acts: Sparse tensor of shape (..., d_features).
 
        Returns:
            Reconstructed activation tensor of shape (..., d_model).
        """
        return feature_acts @ self.W_dec + self.b_dec
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encode–decode pass.
 
        Args:
            x: Residual stream tensor of shape (..., d_model).
 
        Returns:
            Reconstructed tensor of shape (..., d_model).
        """
        return self.decode(self.encode(x))
```
 
Two constraints motivate the use of raw `nn.Parameter` over `nn.Linear`:
 
**1. Decoder column normalisation.** SAE training requires every decoder feature vector to have unit norm, i.e. $\lVert W_{\text{dec}}[j, :]\rVert_2 = 1$ for all $j$. This is enforced as a post-gradient projection after each optimiser step:
 
```python
with torch.no_grad():  # constraint projection, not a learned op — no gradient needed
    norms = model.W_dec.norm(dim=1, keepdim=True)   # (d_features, 1)
    model.W_dec.copy_(model.W_dec / norms)
```
 
Without this, the model can trivially reduce the sparsity penalty by shrinking feature activations and compensating by scaling up the corresponding decoder vectors — the norm constraint closes this loophole.
 
**2. Encoder–decoder weight tying.** Some SAE variants initialise or constrain $W_{\text{enc}} = W_{\text{dec}}^\top$, so the encoder and decoder share parameters up to a transpose. Expressing both as raw `nn.Parameter` makes this relationship explicit and easy to enforce, whereas accessing `.weight` and `.weight.T` across two separate `nn.Linear` modules is error-prone.
  
### SAE Evaluation

[Table 4](tab:sae-eval) summarizes the key metrics used to evaluate SAEs across several models <d-cite key='shu2025surveysparseautoencodersinterpreting'></d-cite>. These metrics fall into two broad categories: **structural metrics**, which assess whether the SAE is a faithful surrogate for the original activations, and **functional metrics**, which assess whether the learned features behave as clean, interpretable units. Together, they provide a more complete picture of SAE quality than reconstruction loss alone.

#### Structural Metrics

Before assessing the interpretability of individual features, we must first determine whether an SAE is a reliable approximation of the original layer. Structural metrics answer this question by measuring both sparsity and reconstruction fidelity.

$L_0$ sparsity counts how many latents fire on a typical token. Lower values indicate a cleaner, more selective representation. However, overly aggressive sparsity can degrade reconstruction quality. A separate set of fidelity metrics quantifies how well the SAE preserves the geometry and predictive behavior of the original activations:

- **MSE** measures the direct reconstruction error between original and reconstructed activations.
- **Cross-entropy loss** and **KL divergence** check whether the model's next-token predictions change when its hidden states are replaced by SAE reconstructions.
- **Explained variance** captures how much of the activation geometry the SAE retains.

If a language model maintains similar next-token predictions under SAE reconstruction, the SAE sits on a good sparsity–fidelity frontier: sparse enough to be interpretable, but faithful enough not to distort the model's behavior.

#### Functional Metrics

Reconstruction quality alone does not guarantee interpretability. An SAE can produce low error while still learning features that collapse multiple concepts (polysemanticity) or absorb unrelated signals. Functional metrics capture these failure modes.

**Absorption** measures how frequently the correct latent vector fails to activate and is replaced by an unrelated but correlated feature. Mean absorption tracks partial failures, while full absorption captures cases where none of the appropriate latents activate. Low absorption indicates that concepts are represented consistently rather than being swallowed by a few dominant features.

**Spurious Correlation Removal** (SCR) tests whether the SAE isolates spurious features that contribute to shortcut behavior. By identifying and ablating a varying number of latents most associated with a known spurious attribute, SCR quantifies how much debiasing occurs at different intervention scales. High SCR scores indicate that the SAE has cleanly separated true signal from superficial correlations.

Finally, **Sparse Probing** compares concept probes trained on SAE latents to probes trained on the model's dense activations. When a probe using only a small number of SAE features matches or exceeds the dense baseline, it suggests that the SAE has discovered disentangled, concept-aligned representations. Conversely, poor sparse-probe performance shows that the SAE's features, despite having good structural scores, are not semantically meaningful.

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
        <td>Checks whether reconstruction is "good enough" for the language modeling task.</td>
      </tr>
      <tr>
        <td><strong>KL Divergence</strong></td>
        <td>KL between the original and SAE-reconstructed next-token distributions.</td>
        <td>Measures how much the SAE changes the model's predictive distribution.</td>
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
        <td>Fraction of cases where the "correct" feature fails to activate and a similar latent fires instead.</td>
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

Taken together, these structural and functional metrics tell us whether an SAE is a faithful and useful decomposition of the original layer's representations. However, they do not yet tell us what the individual features mean. Once we know that an SAE reconstructs well, maintains sparsity, and avoids major failure modes like absorption or spurious entanglement, we can shift our focus to the interpretability of the features themselves.

### Feature Evaluation

Naturally, the next question is: **do the latent features actually correspond to meaningful concepts in the model?** Evaluation generally falls into two categories: input-based (what activates a feature) and output-based (what the feature does when changed).

#### Input-based Evaluation

Input-based analysis examines the inputs or hidden states that cause a feature to activate. Common methods include:

- **Top activating examples:** Inspect tokens or contexts where a feature is strongest. If they cluster around a clear linguistic pattern — such as plural nouns, numbers, or closing brackets — the feature is likely meaningful.
- **Sparsity/selectivity measurements:** Good features activate rarely and consistently for the same type of input.

These evaluations aim to answer the question: *"What concept is this feature detecting?"*

#### Output-based Evaluation

Output-based evaluation checks whether a feature plays a **causal** role in the model's behavior:

- **Activation patching:** Replace or modify a feature's activation during the forward pass to test whether that feature is necessary or sufficient for a specific behavior.
- **Feature-direction interventions:** Add or subtract the feature's decoder vector in the Transformer's residual stream to examine whether that direction corresponds to a meaningful, causal concept.

These evaluations aim to address the question: *"Does this feature actually matter for the model's computation?"*

## Sparse Autoencoder Variants

The standard SAE described above provides a strong baseline, but several architectural refinements have been proposed to address its limitations. Here we discuss two principled improvements: the **Gated SAE**, which targets the shrinkage problem inherent in $L_1$-penalised training, and the **Matryoshka SAE**, which addresses both the rigidity of a fixed dictionary size and the absorption problem identified in functional evaluation.

---
### Gated SAE

Gated SAEs <d-cite key='rajamanoharan2024improving'></d-cite> are motivated by a subtle but consequential failure mode of standard SAE training known as **shrinkage**.

#### The Shrinkage Problem

Recall that the standard SAE training loss applies an $L_1$ penalty directly to the feature activations $h(z)$:

$$\mathcal{L}(z) = \|z - \hat{z}\|_2^2 + \lambda \|h(z)\|_1.$$

The $L_1$ penalty serves double duty here in an undesirable way: it both gates inactive features toward zero *and* shrinks the magnitude of genuinely active ones. This means that *features which should fire strongly are systematically underestimated* — a phenomenon known as **shrinkage**. The decoder must then compensate by inflating its weight norms, distorting the geometry of the learned feature directions.

#### Architecture

Gated SAEs address shrinkage by decoupling the binary decision of *whether* a feature fires from the continuous decision of *how strongly* it fires, using two parallel projections from the same input.

Formally, given input $z$, the encoder computes two parallel projections:

$$\pi_{\text{gate}}(z) = W_{\text{gate}}\, z + b_{\text{gate}}, \qquad \pi_{\text{mag}}(z) = W_{\text{mag}}\, z + b_{\text{mag}},$$

where $\pi_{\text{gate}}$ determines which features are active and $\pi_{\text{mag}}$ determines their magnitudes. The sparse feature activations are then:

$$h(z) = \mathbf{1}[\pi_{\text{gate}}(z) > 0] \odot \text{ReLU}(\pi_{\text{mag}}(z)),$$

where $\mathbf{1}[\cdot]$ is the Heaviside step function producing a binary gate mask, and $\odot$ denotes elementwise multiplication. The decoder is unchanged from the standard SAE:

$$\hat{z} = h(z)\, W_{\text{dec}} + b_{\text{dec}}.$$

To keep the parameter count reasonable, Gated SAEs tie the two weight matrices via a learned per-feature log-scale $r \in \mathbb{R}^{d_{\text{features}}}$:

$$W_{\text{mag}} = \exp(r) \odot W_{\text{gate}}.$$

This means the gate and magnitude pathways share the same feature directions in activation space, differing only in per-feature scale.

#### Training Loss

The Heaviside step in the gate pathway is non-differentiable, so training uses an auxiliary reconstruction path as a straight-through estimator. The full loss is:

$$\mathcal{L}(z) = \underbrace{\|z - \hat{z}\|_2^2}_{\text{reconstruction}} + \underbrace{\lambda \|\pi_{\text{gate}}(z)\|_1}_{\text{sparsity}} + \underbrace{\|z - \hat{z}_{\text{aux}}\|_2^2}_{\text{auxiliary}},$$

where the auxiliary reconstruction

$$\hat{z}_{\text{aux}} = \text{ReLU}(\pi_{\text{gate}}(z))\, W_{\text{dec}} + b_{\text{dec}}$$

replaces the hard Heaviside gate $\mathbf{1}[\cdot]$ with a soft ReLU, keeping the rest of the forward pass (i.e., the decoder $W_{\text{dec}}$ and bias $b_{\text{dec}}$) identical to the main path. Because ReLU is differentiable everywhere except at zero, this gives the optimiser a valid gradient signal into $W_{\text{gate}}$ during backpropagation, which the non-differentiable Heaviside would otherwise block. The auxiliary reconstruction is used **only during training** to route gradients; at inference time only $\hat{z}$ is used.

Two design choices here are worth highlighting:

- **$L_1$ on the pre-gate signal.** The sparsity penalty is applied to $\pi_{\text{gate}}(z)$ rather than to $h(z)$ directly. This means active features are no longer penalised for being large — only for existing at all — which is precisely what eliminates shrinkage.
- **Weight tying.** Sharing directions between $W_{\text{gate}}$ and $W_{\text{mag}}$ keeps the encoder parameter count comparable to a standard SAE, avoiding the cost of a fully independent second projection.

#### Comparison to Standard SAE

| | Standard SAE | Gated SAE |
|---|---|---|
| **Sparsity mechanism** | $L_1$ on activations $h(z)$ | $L_1$ on gate pre-activations $\pi_{\text{gate}}$ |
| **Shrinkage** | Present — active features underestimated | Eliminated — magnitude pathway is penalty-free |
| **Parameter count** | $2 \times d_{\text{model}} \times d_{\text{features}}$ | $\approx 2 \times d_{\text{model}} \times d_{\text{features}}$ (via weight tying) |
| **Training** | Standard backprop | Requires auxiliary loss for straight-through gradient |

---

### Matryoshka SAE

A different limitation of the standard SAE is that its dictionary size $d_{\text{features}}$ is fixed at training time. If you need a coarser or finer decomposition — say, for a downstream task that requires fewer but more general features, or more but more specific ones — you must retrain from scratch. Beyond this inflexibility, recall from the functional evaluation section that **absorption** is a key failure mode of standard SAEs: a concept's correct latent fails to activate and is captured instead by an unrelated but correlated feature. Matryoshka SAEs <d-cite key='bussmann2025learning'></d-cite> address both problems by training a single SAE whose features are organised into a nested hierarchy of granularities, inspired by Matryoshka Representation Learning in the embedding literature. The nested training objective encourages coarse-level features to be maximally self-sufficient, which reduces the incentive for concepts to be absorbed across unrelated latents.

#### Architecture

A Matryoshka SAE partitions its $d_{\text{features}}$ dictionary dimensions into $L$ nested subsets of increasing size:

$$S_1 \subset S_2 \subset \cdots \subset S_L = \{1, \ldots, d_{\text{features}}\},$$

where $\lvert S_l \rvert = k_l$ and $k_1 < k_2 < \cdots < k_L = d_{\text{features}}$. The encoder and decoder are structurally identical to a standard SAE. The difference lies entirely in how the loss is computed: each nested subset $S_l$ is treated as an independent SAE, and the model is trained to reconstruct well at every level of the hierarchy simultaneously.

At inference time, using only the first $k_l$ features gives a valid, self-contained reconstruction at granularity $l$ — no retraining required. The smaller subsets capture broad, high-salience features, while the larger subsets add progressively finer-grained detail.

#### Training Loss

The Matryoshka loss is a weighted sum of reconstruction and sparsity losses across all $L$ nested levels:

$$\mathcal{L}(z) = \sum_{l=1}^{L} w_l \left( \|z - \hat{z}^{(l)}\|_2^2 + \lambda \|h^{(l)}(z)\|_1 \right),$$

where $\hat{z}^{(l)}$ and $h^{(l)}(z)$ denote the reconstruction and activations using only features in $S_l$, and $w_l > 0$ are level weights (typically decreasing with $l$ to prioritise coarse-level fidelity). The full-dictionary level $l = L$ recovers the standard SAE loss.

In practice the nested subsets are implemented simply by slicing the first $k_l$ columns of $W_{\text{dec}}$ and the corresponding rows of $W_{\text{enc}}$ at each level, so no additional parameters are introduced.

#### The Nested Feature Hierarchy

An important consequence of Matryoshka training is that features at smaller levels are incentivised to be maximally general — they must reconstruct well *on their own*, without relying on the additional features in larger levels. This induces a coarse-to-fine structure where:

- **Level $S_1$** (smallest): a compact set of high-salience, broad features that alone account for most of the reconstruction.
- **Level $S_L$** (largest): the full dictionary, adding fine-grained, context-specific features on top of the coarse ones.

Crucially, because each nested level must stand alone as a complete decomposition, a concept cannot be silently offloaded to a correlated latent at a different level — it must be explicitly represented at every level where it is relevant. This structural pressure directly counteracts absorption, encouraging concepts to be consistently and cleanly localised within the hierarchy.

#### Comparison to Standard SAE

| | Standard SAE | Matryoshka SAE |
|---|---|---|
| **Dictionary size** | Fixed at training time | Single model, multiple granularities at inference |
| **Retraining for new granularity** | Yes | No, only slice a nested subset |
| **Feature organisation** | Unordered | Coarse-to-fine hierarchy |
| **Absorption** | Prone: concepts absorbed by correlated latents | Reduced: nested levels enforce concept self-sufficiency |
| **Training cost** | Single loss | Summed loss over $L$ levels |
| **Parameter count** | $2 \times d_{\text{model}} \times d_{\text{features}}$ | $2 \times d_{\text{model}} \times d_{\text{features}}$ (shared weights) |

## Semi-Nonnegative Matrix Factorization

SAEs have become the dominant tool for feature discovery in MI, largely because they provide a flexible, scalable way to learn disentangled directions in activation space. But SAEs also reveal an important limitation: they learn features from scratch, without reference to the model's underlying mechanisms. In particular, SAEs trained on the residual stream often struggle to produce features that cleanly correspond to the computations inside the model's MLP layers.

This motivates the next question: *What if instead of learning new features, we directly decompose the model's own MLP activations to reveal how neuron groups compose concepts?*

The recent Semi-Nonnegative Matrix Factorization (SNMF) <d-cite key='shafran2025decomposingmlpactivationsinterpretable'></d-cite> approach offers exactly this perspective. It bypasses the autoencoder architecture entirely and treats MLP activations themselves as the object to factorize, yielding features that are sparse combinations of real neurons, with coefficients that directly reveal which inputs activate which features.

### Method

The core idea behind SNMF is simple but powerful: instead of training a full encoder–decoder network like an SAE, directly factorize the MLP activation matrix into interpretable building blocks. This approach rests on the assumption that the MLP's output to the residual stream can be expressed as a linear combination of underlying features. Because SNMF operates entirely on collected activations, it is a fully unsupervised, training-free method — it does not modify the original model or require gradient-based optimization.

For a chosen MLP layer, SNMF gathers neuron activations across a sequence of $n$ tokens, forming a matrix $A \in \mathbb{R}^{d_a \times n}$. The goal is to decompose this matrix as:

$$A \approx ZY,$$

where:
- $Z \in \mathbb{R}^{d_a \times k}$ contains the MLP features, each column representing a sparse linear combination of neurons (a co-activation pattern), and
- $Y \in \mathbb{R}^{k \times n}_{\geq 0}$ is a nonnegative coefficient matrix indicating how strongly each feature contributes to each token's activation vector.

The intuition is that neuron activations should combine additively and sparsely to produce higher-level concepts. Enforcing nonnegativity in $Y$ ensures these combinations remain parts-based: features cannot "subtract" from one another, making the representation more interpretable.

The factorization alternates between two update steps:

1. **Multiplicative updates for $Y$**, which preserve nonnegativity:
$$Y \leftarrow Y \odot \frac{Z^\top A}{Z^\top ZY + \epsilon},$$

2. **Closed-form ridge-regression updates for $Z$**:
$$Z = A Y^\top (YY^\top + \lambda I)^{-1}.$$

After each pair of updates, SNMF applies winner-take-all sparsification to the columns of $Z$, keeping only the largest-magnitude entries and setting the rest to zero. This encourages each feature to rely on a small, coherent subset of neurons.

Each SNMF feature can then be mapped back into the residual stream via the MLP's output projection matrix $W_V$:

$$f_i = W_V z_i = \sum_{j=1}^{d_a} z_{i,j} v_j,$$

making it directly comparable to SAE features and suitable for causal interventions such as steering or ablations. The result is a set of interpretable, neuron-grounded features that reveal how the MLP layer internally organizes semantic structure.

#### Connection to Dimensionality Reduction

At first glance, SNMF may appear similar to classical dimensionality reduction methods such as PCA and NMF, since they both factorize an activation matrix into a smaller set of components. However, the objectives and constraints differ in important ways.

PCA finds orthogonal directions of maximum variance, producing a compact basis for the data. While PCA components are globally optimal in a reconstruction sense, they are not constrained to be sparse or nonnegative, and they often mix positive and negative contributions from many neurons simultaneously. The resulting components tend to be dense, globally distributed, and difficult to map back to any interpretable neuron-level concept. Standard NMF enforces nonnegativity on *both* $Z$ and $Y$, which works well for data that is inherently nonnegative (such as image pixels), but MLP activations can be negative due to gating functions like SiLU or GeLU. Forcing $Z \geq 0$ in this setting would artificially constrain the feature directions.

SNMF occupies a middle ground: it enforces nonnegativity only in $Y$ (the coefficient matrix), leaving $Z$ (the feature directions) unconstrained. This means:
- The feature directions in $Z$ can point anywhere in the signed MLP activation space, faithfully capturing both positively and negatively activating neurons.
- The coefficient matrix $Y$ remains nonnegative, ensuring that token representations are built as additive, parts-based combinations of features — not cancellations.
- The winner-take-all sparsification on $Z$ further ensures each feature direction is a sparse combination of neurons rather than a dense global mixture.

This combination of unconstrained sparse feature directions with nonnegative and additive token coefficients is precisely what makes SNMF better suited than PCA or standard NMF for decomposing MLP activations into interpretable neuron groups.

### Discussion

SNMF offers an appealing alternative to SAEs by grounding features directly in the model's own neurons rather than learning new latent directions. Because each feature is a sparse combination of real MLP neurons and each token's activation is expressed as a nonnegative mixture of these features, the resulting representations are often more transparent. They reveal how groups of neurons cooperate to encode meaningful concepts, and the nonnegativity constraint in $Y$ provides a clean way to examine which tokens most strongly activate each feature. This gives SNMF a natural interpretability advantage: instead of discovering artificial directions in activation space, it exposes structure that already exists within the network.

**On polysemanticity and superposition.** Recall that polysemanticity refers to a single neuron encoding multiple unrelated concepts, and superposition refers to a model representing more features than it has dimensions by encoding them as near-orthogonal directions in a lower-dimensional space. SNMF partially addresses the former: the winner-take-all sparsification on $Z$ means each discovered feature activates only a small subset of neurons, so instead of one neuron carrying many meanings, each SNMF feature corresponds to a coherent co-activation pattern of a few neurons. The nonnegative constraint on $Y$ further ensures that token representations are built additively from these patterns, preventing the kind of interference between overlapping representations that superposition exploits. However, **SNMF does not eliminate superposition** — superposition is a property of how the model chooses to store information, and no post-hoc factorization can remove it. What SNMF offers instead is a more grounded view: by restricting features to sparse linear combinations of real neurons, it makes the superposition structure directly visible rather than hiding it behind learned encoder weights as SAEs do. Empirically, SNMF-derived features outperform SAEs on causal steering tasks across Llama 3.1, Gemma 2, and GPT-2 while performing comparably on concept detection, suggesting that neuron-grounded features better capture the causal mechanisms the model actually uses — even if the underlying superposition is not removed.

**Limitations.** Despite its conceptual clarity, SNMF faces significant practical constraints. The method requires collecting large matrices of MLP activations and repeatedly factorizing them, which becomes computationally expensive as model width grows. Unlike SAEs, which can be trained incrementally using minibatches, SNMF depends on access to large dense activation datasets at once, making memory and storage major bottlenecks. Its results also depend strongly on the dataset used to extract activations: if the chosen text distribution does not sufficiently cover the model's behaviors, important features may be omitted or fragmented. Furthermore, because SNMF restricts features to linear combinations of neurons, it may fail to capture structure that emerges only at the level of directions or higher-dimensional subspaces — some concepts that SAEs can isolate simply cannot be expressed as sparse neuron combinations. The optimization procedure itself can be brittle: multiplicative updates and sparsification thresholds may lead to unstable or inconsistent features, and small changes in hyperparameters can yield noticeably different decompositions. Finally, SNMF lacks the natural overcompleteness of SAEs: choosing too few features underfits the representation, while choosing too many risks redundancy or noise.

A summary comparison of the three approaches is provided below.

<div class="table" id="tab:snmf-comparison">
  <table>
    <caption style="caption-side: top; padding: 8px; font-weight: bold; text-align: center;">
      Table 6. Comparison of Feature Discovery Methods
    </caption>
    <thead>
      <tr>
        <th></th>
        <th>PCA / NMF</th>
        <th>SAE</th>
        <th>SNMF</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Feature basis</strong></td>
        <td>Statistical variance directions</td>
        <td>Learned encoder–decoder directions</td>
        <td>Sparse neuron co-activation patterns</td>
      </tr>
      <tr>
        <td><strong>Grounded in model neurons?</strong></td>
        <td>No</td>
        <td>No</td>
        <td>Yes</td>
      </tr>
      <tr>
        <td><strong>Sparsity</strong></td>
        <td>None (PCA); nonneg only (NMF)</td>
        <td>Enforced via $L_1$ / Top-$K$ / JumpReLU</td>
        <td>Enforced via winner-take-all on $Z$</td>
      </tr>
      <tr>
        <td><strong>Overcompleteness</strong></td>
        <td>No ($k \leq d$)</td>
        <td>Yes ($k \gg d$)</td>
        <td>Configurable via $k$</td>
      </tr>
      <tr>
        <td><strong>Training required?</strong></td>
        <td>No</td>
        <td>Yes</td>
        <td>No (matrix factorization only)</td>
      </tr>
      <tr>
        <td><strong>Polysemanticity addressed?</strong></td>
        <td>No</td>
        <td>Partially (via sparsity)</td>
        <td>Partially (via neuron-level sparsity)</td>
      </tr>
      <tr>
        <td><strong>Scalability</strong></td>
        <td>Moderate</td>
        <td>High (minibatch training)</td>
        <td>Low (requires dense activation matrix)</td>
      </tr>
    </tbody>
  </table>
</div>

In practice, these constraints make SNMF more suitable as a diagnostic tool for understanding local MLP structure rather than a scalable alternative to SAEs for whole-model feature extraction. Even with these limitations, SNMF remains a valuable complement to SAE-based approaches — it provides insight into how the model's neurons are organized and how co-activation patterns contribute to semantic representation, setting the stage for deeper analyses of how features evolve across depth.

## Transcoder
 
SAEs and SNMF both uncover meaningful feature structure within a single layer, but they offer only a static view of the model. Neither approach tells us how features transform as they propagate forward through the network. In practice, transformer representations change dramatically from layer to layer — features can split into multiple subfeatures, merge into broader abstractions, fade out, or invert their meaning entirely. These dynamics are invisible if we inspect each layer independently.
 
To move from feature discovery to understanding computation as a multi-step process, we need a method that can model how one layer's features *cause* the next layer's outputs. This is the motivation behind the **Transcoder** <d-cite key='dunefsky2024transcodersinterpretablellmfeature'></d-cite>: a small learned model that replaces an MLP sublayer with a sparse, interpretable approximation, mapping the residual stream input to the MLP output via an explicit feature basis.
 
Formally, a transcoder at layer $\ell$ approximates the MLP computation as:
 
$$\hat{y}_\ell = W_{\text{dec}} \text{JumpReLU}(W_{\text{enc}} x_\ell + b_{\text{enc}}) + b_{\text{dec}}$$

where $x_\ell$ is the residual stream at layer $\ell$ and $\hat{y}_\ell$ is the predicted MLP output. 
This is structurally similar to an SAE encoder&mdash;decoder, but with a crucial difference: an SAE reconstructs its own input ($\hat{x} \approx x$), whereas a transcoder predicts a *different* quantity, i.e. the MLP layer $\ell$.

The learned features therefore do not merely decompose what is present in the residual stream&mdash;they represent computations the MLP performs on it. Each active feature can be interpreted as a specific transformation the model applies at that layer, rather than a static direction in activation space.
 
A transcoder is trained with a combined reconstruction and sparsity loss:
 
$$\mathcal{L} = \|\hat{y}_\ell - y_\ell\|_2^2 + \lambda \|a_\ell\|_1,$$
 
where $y_\ell$ is the true MLP output recorded from a forward pass and $a_\ell = \text{JumpReLU}(W_{\text{enc}}\, x_\ell + b_{\text{enc}})$ are the sparse feature activations. Once trained, a transcoder can be dropped in as a replacement for the original MLP, enabling circuit-level analysis: which input features activate which transcoder features, and which transcoder features write which directions into the residual stream.
 
**Transcoders vs. SAEs.** The key distinction is the direction of approximation. SAEs are trained to reconstruct the *same* signal they receive — they are analysis tools. Transcoders are trained to predict a *downstream* signal — they are computation models. This makes transcoders better suited for tracing causal pathways through the model, but also harder to train: the target $y_\ell$ can be structurally very different from the input $x_\ell$, and the approximation error compounds as the transcoder is used in place of the original MLP during inference.
 
---
 
### Cross-Layer Transcoder
 
While single-layer transcoders model each MLP in isolation, they miss an important property of transformers: features do not act only at the layer where they are detected. A direction introduced at layer $\ell$ persists in the residual stream and may influence MLP outputs at layers $\ell + 1, \ell + 2, \ldots$ through the residual connections. Single-layer transcoders cannot represent this cross-layer influence, since their decoders only target one MLP output.
 
The **Cross-Layer Transcoder (CLT)** <d-cite key='dunefsky2024transcodersinterpretablellmfeature'></d-cite> generalises this design by allowing each feature detected at layer $\ell$ to contribute to MLP reconstructions at all subsequent layers $\ell' \geq \ell$. Where a standard transcoder is a per-layer model, a CLT is a single end-to-end model spanning the full depth of the transformer.
 
#### Architecture
 
As shown in [Figure 2](#fig:clt), the CLT mirrors how a transformer updates its residual stream across depth. It consists of a collection of features arranged into the same number of layers as the underlying transformer. Each layer of the CLT reads from the transformer's residual stream at that depth and contributes to reconstructing the MLP outputs of that layer and all subsequent layers.
 
{% include figure.liquid 
   path="assets/img/2026-04-27-interpret-model/clt.jpg" 
   class="img-fluid"
   id="fig:clt"
   caption="Figure 2. Replacement Model Constructed by Cross-Layer Transcoder (CLT)<d-cite key='dunefsky2024transcodersinterpretablellmfeature'></d-cite>"
   zoomable=true 
%}
 
At layer $\ell$, the CLT encodes the residual activation $x_\ell$ using a learned linear map followed by JumpReLU:
 
$$a_\ell = \text{JumpReLU}(W^{(\ell)}_{\text{enc}}\, x_\ell),$$
 
where $a_\ell$ is the vector of CLT feature activations for that layer. These features are cross-layer because each feature at layer $\ell$ can help reconstruct MLP outputs at all downstream layers $\ell' \geq \ell$, via separate decoder weights for each target layer:
 
$$\hat{y}_{\ell} = \sum_{\ell' = 1}^{\ell} W^{(\ell')\to \ell}_{\text{dec}}\, a_{\ell'}.$$
 
The MLP output at a given layer is thus reconstructed jointly from all features activated at that layer and all earlier layers. Each feature has a shared encoder that determines what it detects, and multiple decoders that determine where and how it influences the rest of the model — making each feature a stable, reusable computational unit across depth.
 
All CLT encoders and decoders are trained jointly as a single end-to-end model. The training loss sums reconstruction errors across all layers together with a sparsity penalty:
 
$$\mathcal{L} = \sum_{\ell=1}^{L} \left( \|\hat{y}_\ell - y_\ell\|_2^2 + \lambda \|a_\ell\|_1 \right) + \mu \sum_{\ell, \ell'} \|W^{(\ell) \to \ell'}_{\text{dec}}\|_F,$$
 
where the final term penalises decoder weight norms to discourage diffuse, weakly-influential cross-layer connections. Because gradients flow through all layers at once, features learned at early depths become reusable building blocks for predicting later MLP outputs, allowing the CLT to capture the model's computation as a coherent, multi-step transformation rather than a collection of isolated layer-wise mappings.
 
In practice, CLTs can have hundreds of thousands to tens of millions of features across layers, yet remain far smaller than the transformer models they interpret. Once trained, a CLT provides a structured, layer-aligned view of how representations evolve, merge, and transform as they propagate forward through the transformer.
 
### Discussion
 
Despite their conceptual appeal, CLTs are difficult to deploy at scale. Each layer must have its own encoder and multiple decoder matrices targeting every subsequent layer — for a transformer with $L$ layers and hidden dimension $d$, the number of decoder matrices grows as $O(L^2)$. For modern LLMs with dozens of layers and thousands of channels, the total parameter count quickly becomes substantial, even though the CLT only models the MLP pathways. Training requires storing the full residual stream and MLP outputs for vast numbers of tokens, making data collection and GPU memory usage significant bottlenecks.
 
Even after training, CLTs face interpretability challenges. Because reconstruction quality matters at every layer, early-layer features must support many downstream predictions, causing heavy coupling between layers. This often makes learned features diffuse rather than crisp, especially in deeper models where errors compound. Furthermore, CLTs capture correlations in activations, not causal mechanisms — they approximate how the model tends to update its representations, but cannot guarantee that the learned mapping reflects the true internal computation. As depth increases, nonlinear or context-specific transformations become harder to approximate with a linear transcoder design.
 
Yet CLTs remain a valuable research tool when used appropriately. They provide a structured way to trace how representations flow across layers, offering a layer-aligned view of model computation that complements within-layer methods like SAEs and SNMF. Together, the three methods form a complementary toolkit: SAEs expose what features are present at a given layer, SNMF reveals which neurons implement those features, and CLTs trace how those features cause downstream computations.

## Final Remarks

Mechanistic interpretability is steadily moving beyond one-off visualizations and heuristic reasoning toward something more systematic—a science built on feature extraction, circuit reconstruction, and architecture-level constraints. Sparse Autoencoders have shown that dense layers contain rich, recoverable structure; Semi-Nonnegative Matrix Factorization reveals how those structures relate to the model's own neurons; Cross-Layer Transcoders expose how representations transform as they propagate through depth. Each technique brings its own strengths and limitations, but together they form an increasingly coherent toolkit for understanding the computations inside modern transformers.

| | SAE | SNMF | CLT |
|---|---|---|---|
| **Target** | Residual stream activations | MLP neuron activations | MLP input → output mapping |
| **Scope** | Single layer | Single layer | All layers jointly |
| **Features grounded in neurons?** | No | Yes | No |
| **Captures cross-layer flow?** | No | No | Yes |
| **Training required?** | Yes | No | Yes |
| **Scalability** | High | Low | Moderate–Low |
| **Primary use** | Feature discovery | Neuron group analysis | Computation tracing |

Interpretability remains a profound challenge: models continue to grow, tasks become more complex, and no single method will illuminate every aspect of their internal computation. Yet the shift toward feature-level and circuit-level approaches marks real progress. By treating neural networks not as opaque function approximators but as systems composed of reusable computational primitives, we are beginning to see how high-level behaviors emerge from low-level structure. The work ahead will involve hybrid techniques, new representational abstractions, and architectures designed with interpretability in mind. But the direction is clear. Understanding how neural networks compute is no longer merely a philosophical aspiration—it is becoming a practical engineering discipline, and its foundations are starting to solidify.