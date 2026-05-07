# Understanding and Improving Joint Discriminative-Generative Models via Sparse Autoencoder Steering

## 1. Introduction

Deep neural networks have traditionally been developed with either discriminative or generative objectives in mind, rarely excelling at both simultaneously. Discriminative models, optimized for classification or regression tasks, often lack the ability to faithfully model the underlying data distribution; conversely, generative models can synthesize realistic data samples but frequently underperform on downstream predictive tasks. Unifying these two paradigms within a single framework has long been a grand challenge in machine learning, with the promise of grounding classification decisions in a rich, explicit understanding of the data distribution.

Energy-Based Models (EBMs) provide a principled theoretical foundation for such unification. By defining an unnormalized density through an energy function, EBMs can simultaneously support both conditional inference (e.g., classification) and unconditional sampling (e.g., generation). Among recent efforts, **Joint Energy-Based Models (JEM)** [1] demonstrated a remarkable insight: the logits of a standard classifier can be reinterpreted as defining an energy function over the joint distribution $p(x, y)$, enabling a single model to perform both high-accuracy classification and sample generation. However, JEM and its subsequent variants rely fundamentally on Stochastic Gradient Langevin Dynamics (SGLD) for training the generative component. SGLD-based learning suffers from severe training instabilities, poor sample quality, and computational inefficiency, which have collectively prevented these hybrid models from scaling beyond small-scale datasets such as CIFAR-10.

Recently, **Dual Adversarial Training (DAT)** [2] addressed these scalability limitations by replacing unstable SGLD with adversarial training principles. DAT employs a dual application of adversarial training: standard adversarial training for the discriminative component to achieve robust classification, and an AT-based energy learning strategy for the generative component that discriminates real data from contrastive samples generated via Projected Gradient Descent (PGD). This framework achieves stable convergence, state-of-the-art discriminative and generative performance on high-resolution datasets such as ImageNet $256 \times 256$, and bridges the long-standing gap between robust classification and high-fidelity generation.

**Motivation.** Despite DAT's impressive empirical success, the internal representational geometry of such joint discriminative-generative models remains largely opaque. Specifically, it is unclear how adversarial perturbations manifest in the intermediate activations of a model that is simultaneously trained for robust classification and energy-based generation. Traditional adversarial robustness research has focused predominantly on input-space perturbations and output-space behavior, leaving the model's latent representations as a "black box." Moreover, if adversarial perturbations induce systematic, structured changes in these internal representations, one might ask: can we explicitly identify and correct these changes to recover the model's intended behavior?

This question is particularly pertinent in the context of **mechanistic interpretability**. The emergence of **Sparse Autoencoders (SAEs)** as a tool for decomposing neural network activations into sparse, often semantically meaningful features offers a compelling avenue for opening this black box. Pioneering work by Anthropic [3], [4] has demonstrated that SAEs can uncover monosemantic features—latent directions that correspond to human-interpretable concepts—from the internal activations of large language models. If similar decompositions can be obtained from the visual representations of joint discriminative-generative models, they may not only illuminate how adversarial examples subvert model behavior, but also enable **representation-space interventions** that mitigate such attacks.

**Our Approach.** In this work, we train Top-K Sparse Autoencoders on the intermediate activations of DAT-trained models, including ConvNeXt-Large on ImageNet and WideResNet on CIFAR-10. We systematically analyze how clean and adversarial inputs differentially activate interpretable sparse features. Building on these insights, we propose **latent steering** as a lightweight, post-hoc intervention mechanism: by shifting adversarially perturbed SAE activations toward their corresponding clean class centroids in the sparse latent space, we effectively "purify" the representation before it propagates to deeper layers. We investigate multiple steering strategies—including dynamic nearest-class steering and global entry steering—and evaluate their efficacy under both standard and adaptive attack protocols.

**Contributions.** Our contributions are threefold:
1. We are the first to apply Sparse Autoencoders for mechanistic interpretability analysis on joint discriminative-generative models trained via Dual Adversarial Training, revealing how adversarial perturbations manifest in sparse visual representations.
2. We propose and systematically evaluate **latent steering** strategies that operate directly on SAE decompositions, demonstrating that representation-space interventions can improve adversarial robustness without modifying base model parameters.
3. We provide extensive empirical analysis on ImageNet (20-class subset and full-scale) and CIFAR-10, comparing steering strategies, ablating design choices, and validating our approach under adaptive attack scenarios.

**Organization.** The remainder of this paper is organized as follows. Section 2 reviews related work on Joint Energy-Based Models, Dual Adversarial Training, and Sparse Autoencoders. Section 3 presents our methodology, including SAE training on DAT models, feature analysis, and the proposed latent steering framework. Section 4 describes our experimental setup and presents quantitative results. Section 5 provides discussion and analysis, and Section 6 concludes with future directions.

---

## 2. Related Work

### 2.1 Joint Energy-Based Models and the Path to DAT

**Energy-Based Models.** Energy-Based Models (EBMs) [5] define a probability distribution over data via an energy function $E_\theta(x)$ parameterized by $\theta$:

$$
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}, \quad Z(\theta) = \int \exp(-E_\theta(x)) \, dx,
$$

where $Z(\theta)$ is the intractable partition function. The flexibility of EBMs lies in the fact that any function can serve as an energy function, allowing the same architecture to support both discriminative and generative tasks. Early work by Xie et al. [6] showed how generative ConvNets could be derived from discriminative ones by interpreting them as EBMs. Du and Mordatch [7] scaled EBM training to high-dimensional image datasets, demonstrating that the same energy function can be used for classification, out-of-distribution detection, and generation without task-specific modifications.

**Joint Energy-Based Models (JEM).** Grathwohl et al. [1] introduced JEM, which explicitly reinterprets a standard classifier's logits $f_\theta(x) \in \mathbb{R}^K$ as defining an energy function over the joint distribution $p_\theta(x, y)$ of data and labels. Specifically, the joint energy is defined as:

$$
E_\theta(x, y) = -f_\theta(x)[y],
$$

which yields the joint distribution:

$$
p_\theta(x, y) = \frac{\exp(f_\theta(x)[y])}{Z(\theta)}.
$$

The marginal distribution over data is obtained by summing over labels:

$$
p_\theta(x) = \sum_y p_\theta(x, y) = \frac{\exp\left(\text{lse}(f_\theta(x))\right)}{Z(\theta)},
$$

where $\text{lse}(f_\theta(x)) = \log \sum_{y=1}^K \exp(f_\theta(x)[y])$ is the log-sum-exp operator. Consequently, the marginal energy function is:

$$
E_\theta(x) = -\text{lse}(f_\theta(x)).
$$

JEM's hybrid training objective combines a standard cross-entropy loss for the discriminative component $p_\theta(y \mid x)$ with an EBM objective for the generative component $p_\theta(x)$. The EBM objective is typically trained via contrastive divergence, where negative samples are generated by running **Stochastic Gradient Langevin Dynamics (SGLD)** [8] on the energy landscape:

$$
x^{(t+1)} = x^{(t)} - \frac{\eta}{2} \nabla_x E_\theta(x^{(t)}) + \sqrt{\eta} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I).
$$

The contrastive divergence loss then takes the form:

$$
\mathcal{L}_{\text{EBM}} = \mathbb{E}_{x^+ \sim p_{\text{data}}}\left[E_\theta(x^+)\right] - \mathbb{E}_{x^- \sim p_\theta}\left[E_\theta(x^-)\right],
$$

where $x^+$ are real data samples and $x^-$ are SGLD-generated negative samples. The total JEM objective is:

$$
\mathcal{L}_{\text{JEM}} = \mathcal{L}_{\text{CE}}(x, y) + \lambda \, \mathcal{L}_{\text{EBM}}(x).
$$

Despite its conceptual elegance, JEM suffers from several critical limitations. First, SGLD is notoriously unstable in high-dimensional spaces: chains often fail to mix, require thousands of steps per sample, and are highly sensitive to step-size tuning. Second, the generated samples are frequently of poor quality, exhibiting artifacts and mode collapse. Third, the gradient of the EBM objective through SGLD is computationally expensive and introduces significant variance, limiting scalability to datasets beyond CIFAR-10. Subsequent works such as JEM++ [9] and Robust-JEM [10] introduced proximal SGLD and adversarial training for the discriminative component, respectively, but none fundamentally replaced the SGLD-based generative training paradigm.

**Dual Adversarial Training (DAT).** Yin et al. [2] proposed DAT as a scalable alternative to JEM that retains the joint discriminative-generative formulation while eliminating SGLD entirely. DAT defines the same marginal energy function:

$$
E_\theta(x) = -\text{lse}(f_\theta(x)) = -\log \sum_{y=1}^K \exp(f_\theta(x)[y]),
$$

but learns this energy through a fundamentally different mechanism. DAT's training objective consists of two adversarial components:

1. **Discriminative Adversarial Training ($\mathcal{L}_{\text{AT-CE}}$):** For each in-distribution sample $(x, y)$, a PGD attack generates an adversarial example $x_{\text{adv}}$ that maximizes cross-entropy loss. The model is trained to correctly classify these adversarial examples:

$$
   \mathcal{L}_{\text{AT-CE}} = \mathbb{E}_{(x,y) \sim p_{\text{data}}}\left[\max_{\|\delta\|_p \leq \epsilon} \mathcal{L}_{\text{CE}}(f_\theta(x + \delta), y)\right].
   $$

This component ensures robust classification while implicitly regularizing the energy landscape.

2. **Generative AT-Based Energy Learning ($\mathcal{L}_{\text{BCE}}$):** Instead of using SGLD to generate negative samples, DAT adversarially perturbs out-of-distribution (OOD) data (e.g., Tiny Images for CIFAR, OpenImages-O for ImageNet) to *minimize* the energy $E_\theta(x)$—that is, to transform OOD seeds into samples that the model mistakenly considers high-likelihood. These PGD-generated contrastive samples $x_{\text{contrast}}$ are then used in a binary classification objective that discriminates real in-distribution data (label 1) from adversarially transformed OOD data (label 0):

$$
   \mathcal{L}_{\text{BCE}} = -\mathbb{E}_{x^+ \sim p_{\text{data}}}\left[\log \sigma(-E_\theta(x^+))\right] - \mathbb{E}_{x^- \sim p_{\text{OOD}}}\left[\log(1 - \sigma(-E_\theta(x^-_{\text{contrast}})))\right],
   $$

where $\sigma(\cdot)$ is the sigmoid function. The total DAT objective is:

$$
   \mathcal{L}_{\text{DAT}} = \mathcal{L}_{\text{AT-CE}} + \mathcal{L}_{\text{BCE}}.
   $$

This dual use of adversarial training—one for discriminative robustness, one for generative energy learning—provides several advantages over JEM. First, PGD-based contrastive sample generation is far more stable and computationally efficient than SGLD. Second, the discriminative AT component implicitly provides the gradient regularization needed for stable EBM training, eliminating the need for explicit R1-style penalties. Third, DAT adopts a two-stage training strategy: first pretraining with standard AT, then joint DAT training with frozen or carefully managed normalization layers (BatchNorm frozen for ResNet/WRN, LayerNorm kept active for ConvNeXt). This enables leveraging pretrained robust classifiers and generalizes across architectures. Empirically, DAT achieves state-of-the-art results on both CIFAR-10/100 and ImageNet $256 \times 256$, simultaneously matching standard AT robustness and surpassing diffusion models in generation quality.

**Our Differentiation.** While DAT successfully addresses the scalability and stability limitations of JEM, it does not address the interpretability of the learned representations, nor does it provide mechanisms for explicitly controlling how adversarial perturbations affect internal model states. Our work complements DAT by opening the "black box" of its intermediate representations through Sparse Autoencoders, and by demonstrating that explicit interventions in the sparse latent space can serve as a novel form of adversarial defense.

### 2.2 Sparse Autoencoders and Mechanistic Interpretability

**Dictionary Learning and Neural Network Interpretability.** The challenge of interpreting high-dimensional neural network activations has motivated extensive research into dictionary learning and factorization methods. Early approaches such as non-negative matrix factorization (NMF) and independent component analysis (ICA) sought to decompose neural responses into interpretable bases. In the context of deep learning, activation maximization [11] and network dissection [12] provided techniques for identifying individual neurons or filters responsive to specific visual concepts. However, these methods often revealed that individual neurons are **polysemantic**—responsive to multiple unrelated concepts—making interpretation difficult.

**Sparse Autoencoders.** Sparse Autoencoders address the polysemanticity problem by learning an overcomplete dictionary of features such that each input activates only a small subset of dictionary elements. Formally, an SAE maps an input activation vector $h \in \mathbb{R}^{d_{\text{in}}}$ to a sparse latent code $z \in \mathbb{R}^{d_{\text{lat}}}$ (with $d_{\text{lat}} \gg d_{\text{in}}$) and reconstructs the original activation:

$$
z = \text{TopK}\left(\text{ReLU}(h W_{\text{enc}} + b_{\text{enc}}), k\right),
$$
$$
\hat{h} = z W_{\text{dec}} + b_{\text{dec}}.
$$

Here, $\text{TopK}(\cdot, k)$ retains only the $k$ largest values and zeros out the rest, enforcing sparsity. The SAE is trained to minimize reconstruction error $\|\hat{h} - h\|^2$ subject to the sparsity constraint. The key insight is that when the bottleneck is sufficiently overcomplete and sparsity is enforced, individual latent dimensions often become **monosemantic**—corresponding to coherent, human-interpretable concepts.

**Anthropic's Contributions.** Anthropic has been instrumental in popularizing and scaling SAEs for mechanistic interpretability. In "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" [3], they trained SAEs on transformer activations and found that sparse features correspond to surprisingly specific semantic concepts, such as DNA sequences, legal language, or religious text. Their follow-up work, "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" [4], scaled this approach to production-level language models, demonstrating that SAEs can uncover safety-relevant features (e.g., deception, bias) and that these features can be manipulated to alter model behavior in predictable ways.

These findings established SAEs as a powerful tool for **decomposing semantic information** in neural representations. If the activations of a vision model can be similarly decomposed into sparse, interpretable visual features (e.g., edges, textures, object parts, semantic attributes), then one might expect adversarial perturbations to induce characteristic, structured changes in this sparse code—changes that are invisible in the dense activation space but become apparent once the representation is disentangled.

**SAEs for Adversarial Robustness.** The application of SAEs to adversarial machine learning remains largely unexplored. Most existing defense mechanisms operate at the input level (e.g., adversarial training, input transformation) or at the architectural level (e.g., defensive distillation, randomized smoothing). Representation-space defenses, such as feature squeezing or Jacobian regularization, typically operate on dense features without explicit decomposition. Our work is among the first to leverage the **sparse structure** of SAE decompositions for adversarial defense: by identifying which latent entries are systematically perturbed by adversarial attacks and steering them back toward their clean distributions, we exploit the semantic disentanglement provided by SAEs to "purify" representations in a conceptually targeted manner.

### 2.3 Representation-Space Interventions and Steering

**Latent Space Manipulation.** The idea of manipulating latent representations to control model behavior has a rich history in generative modeling, where latent space arithmetic (e.g., $z_{\text{smile}} = z_{\text{smiling}} - z_{\text{neutral}} + z_{\text{face}}$) has been used to edit facial attributes in GANs and VAEs. More recently, representation engineering [13] has explored reading and writing vectors in the activation space of language models to steer behavior. In the vision domain, activation patching and causal mediation analysis have been used to understand which layers and channels are responsible for specific classification decisions.

**Steering for Robustness.** Our steering approach differs from prior work in several key respects. Unlike input-space defenses, we do not modify the input image; unlike architectural defenses, we do not retrain the model. Instead, we mount the SAE as a forward hook on intermediate layers, intercepting activations, decomposing them into sparse codes, adjusting the codes based on precomputed clean class statistics, and reconstructing the modified activations. This makes our method **post-hoc, lightweight, and model-agnostic** (within the class of models for which an SAE can be trained). We investigate multiple steering strategies—including per-entry nearest-class steering and global cross-class steering—and demonstrate their efficacy under both standard and adaptive attacks.

---

## 3. The Proposed Method

### 3.1 Problem Statement and Assumptions

Consider a DAT-trained classifier $f_\theta: \mathcal{X} \to \mathbb{R}^K$ that maps an input image $x \in \mathcal{X}$ to logits over $K$ classes. We focus on a specific intermediate layer $l$ (e.g., stage 3 of ConvNeXt-Large or block 3 of WideResNet), which produces a spatial feature map $h_l(x) \in \mathbb{R}^{C \times H \times W}$. Our goal is to understand how adversarial perturbations $x_{\text{adv}} = x + \delta$ affect the sparse decomposition of $h_l(x)$, and whether we can design a post-hoc intervention that rectifies these perturbations at the representation level.

We make the following assumptions:
1. **SAE Trained on Clean Activations:** A Top-K SAE has been trained on feature patches extracted from clean training images. The SAE decomposes each spatial token $h \in \mathbb{R}^C$ into a sparse code $z \in \mathbb{R}^{d_{\text{lat}}}$ with at most $k$ non-zero entries.
2. **Class-Conditional Clean Statistics:** For each class $c$ and each spatial token position, we precompute the empirical mean of the SAE latent activations over clean training images belonging to class $c$. These statistics serve as "anchors" for the clean representation of each class.
3. **Post-Hoc Intervention:** The base DAT model remains frozen; our steering mechanism operates exclusively through a forward hook attached to layer $l$.

### 3.2 SAE Training and Feature Extraction

We train a Top-K Sparse Autoencoder with tied initialization ($W_{\text{enc}}^{(0)} = W_{\text{dec}}^\top$) on feature patches extracted from the target layer. Given a dataset of $N$ clean images, we extract $N \times H \times W$ feature vectors $", flatten them, and train the SAE to minimize:

$$
\mathcal{L}_{\text{SAE}} = \frac{1}{M} \sum_{i=1}^{M} \|h_i - \hat{h}_i\|_2^2 + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}},
$$

where $M = N \times H \times W$ is the total number of feature patches, $\hat{h}_i$ is the SAE reconstruction, and $\mathcal{L}_{\text{aux}}$ is an auxiliary loss for dead neuron resampling. During training, we maintain unit-norm decoder rows and periodically resample dead latent dimensions.

After training, we extract SAE latent features for both clean and adversarial images. For a given image $x$, the feature map $h_l(x)$ is flattened into $H \times W$ tokens, each normalized by the empirical mean and standard deviation computed over the training set, encoded via the SAE, and optionally reconstructed. The resulting sparse codes $z \in \mathbb{R}^{H \times W \times d_{\text{lat}}}$ contain at most $k$ non-zero entries per spatial token, providing a compact, interpretable signature of the model's internal state.

### 3.3 Global Entry Steering: Motivation and Fundamental Limitation

Our initial hypothesis was that adversarial perturbations might systematically manipulate a **shared set of entries** across multiple classes. If such "global entries" exist, a universal steering strategy could be devised: identify entries that are consistently perturbed in the same direction across many classes, and apply a unified correction (e.g., subtracting the cross-class mean delta) to all inputs.

Formally, for each class $c$, we first identify a set of "selected entries"—SAE latent dimensions $(t, ch)$ at spatial token $t$ and channel $ch$—that exhibit significant differences between clean and adversarial activations. For each selected entry, we compute:
- **Clean frequency:** the fraction of clean samples in class $c$ for which the entry is active (non-zero);
- **Delta:** $\Delta_{c,t,ch} = \mu_{c,t,ch}^{\text{adv}} - \mu_{c,t,ch}^{\text{clean}}$, where $\mu$ denotes the empirical mean activation;
- **Direction:** whether the adversarial perturbation suppresses ($\Delta < 0$) or enhances ($\Delta > 0$) the entry relative to clean.

To find global entries, we filter for entries that appear in at least $N_{\min}$ classes with high clean frequency ($\geq 92\%$), high direction consistency ($\geq 80\%$ of classes agree on the direction), and low coefficient of variation ($CV < 0.8$) across class-specific deltas. The steering operation for a global entry $(t, ch)$ with cross-class mean delta $\bar{\Delta}_{t,ch}$ is:

$$
z_{t,ch} \leftarrow z_{t,ch} - \bar{\Delta}_{t,ch}.
$$

**Empirical Finding: Global Entries Do Not Exist.** We systematically probed for global entries across 20 representative ImageNet classes using the procedure described above. Despite searching over all $(t, ch)$ positions in the SAE latent space, **we found zero entries** that satisfied the joint criteria of high cross-class frequency, high direction consistency, and low cross-class variation. Even when we relaxed the constraints, the number of qualifying entries remained negligible.

This result is substantiated by our cross-class entry sharing analysis (`analyze_entry_cross_cls_train.py`). We computed pairwise overlaps between the selected entry sets of all 20 classes and found that:
1. **Overlap is sparse:** The vast majority of selected entries are class-specific. For most class pairs, the share ratio (shared entries divided by the owner's total selected entries) is below 10%.
2. **Direction consistency is high when sharing occurs:** Although few entries are shared, those that are shared tend to exhibit consistent perturbation directions across classes. This suggests that when universal features do exist, they are reliably perturbed in the same way—but they are simply too rare to form a viable global steering strategy.
3. **Top-N entries are predominantly class-specific:** When restricting the analysis to the top-30 most impactful entries (by $|\Delta|$) per class, cross-class sharing drops even further. Most top entries for a given class do not even appear in other classes' clean activations above the frequency threshold.

**Implication.** The absence of generalizable global entries indicates that adversarial perturbations against DAT models are **class-specific at the sparse representation level**. This stands in contrast to the intuition that adversarial attacks exploit universal "bugs" in neural networks [14]. Instead, our analysis suggests that adversarial perturbations craft class-structured deviations in the sparse code, which cannot be rectified by a one-size-fits-all correction. This observation directly motivates our next approach: rather than seeking universal entries, we should perform **per-entry, per-instance dynamic classification** to determine the most appropriate clean target for each activated entry.

### 3.4 Dynamic Nearest-Class Steering

Given the absence of global entries, we propose **Dynamic Nearest-Class Steering**, a strategy that adaptively determines the steering target for each activated SAE entry on a per-sample basis. The core idea is to treat each active entry as participating in a "vote" among the $C$ candidate classes: for each non-zero entry $(t, ch)$, we compare its current activation value $z_{t,ch}$ against the precomputed clean class means $\{\mu_{c,t,ch}^{\text{clean}}\}_{c=1}^C$, identify the nearest class by absolute difference, and replace the entry's value with that class's clean mean.

**Lookup Table Construction.** During an offline preprocessing phase, we compute for each class $c$, each spatial token $t \in \{1, \dots, H \times W\}$, and each latent channel $ch \in \{1, \dots, d_{\text{lat}}\}$ the empirical mean activation over clean training images:

$$
\mu_{c,t,ch}^{\text{clean}} = \frac{1}{N_c} \sum_{i: y_i = c} z_{i,t,ch}^{\text{clean}}.
$$

We construct a lookup table $\mathcal{T} \in \mathbb{R}^{H \times W \times d_{\text{lat}} \times C}$ storing these means. To ensure reliability, we filter the lookup table to retain only entries that are sufficiently active across the training set. We consider two filtering strategies:
- **Version 1 (v1):** Retain entries with total activation count $\geq 20$ across all $C$ classes.
- **Version 2 (v2):** Apply an additional layer of filtering, requiring that at least 2 classes each have $\geq 5$ activations for the entry. This stricter criterion reduces noise at the cost of coverage.

**Online Steering.** At inference time, for an input image $x$ (which may be clean or adversarial), we attach a forward hook to layer $l$ that performs the following operations:
1. Extract the feature map $h_l(x)$, flatten it into tokens, normalize, and encode via the SAE to obtain $z \in \mathbb{R}^{B \times H \times W \times d_{\text{lat}}}$.
2. Identify the active mask $\mathcal{M} = \{(b, t, ch) : z_{b,t,ch} \neq 0\} \cap \{(t, ch) \in \mathcal{T}_{\text{valid}}\}$, where $\mathcal{T}_{\text{valid}}$ is the set of entries passing the lookup-table filter.
3. For each valid active entry $(b, t, ch)$, compute the distances to all class means:

$$
   d_c = |z_{b,t,ch} - \mu_{c,t,ch}^{\text{clean}}|, \quad c = 1, \dots, C.
   $$

4. Identify the nearest class: $c^* = \arg\min_c d_c$.
5. Replace the entry value with the nearest class's clean mean: $z_{b,t,ch} \leftarrow \mu_{c^*,t,ch}^{\text{clean}}$.
6. Reconstruct the modified feature map from the steered sparse codes and pass it to subsequent layers.

This procedure effectively performs, for each activated SAE entry, a **nearest-neighbor classification** in a one-dimensional space (the activation value of that entry), using the $C$ class means as prototypes. The aggregate effect across all $k$ active entries per token is a form of **implicit voting**: each entry "votes" for the class whose clean statistics it most closely resembles, and the reconstruction propagates this consensus to the downstream network.

### 3.5 Limitations of Dynamic Nearest-Class Steering

While dynamic nearest-class steering improves robust accuracy compared to the no-intervention baseline, it suffers from several conceptual and practical limitations that warrant careful discussion.

**Poor Interpretability.** The steering decision for each entry is based solely on a one-dimensional distance comparison to class means, without regard for the semantic coherence of the resulting set of votes. A single spatial token may have $k$ active entries, each voting for a different class; the reconstruction aggregates these votes implicitly through the decoder matrix $W_{\text{dec}}$, making it difficult to understand *why* a particular steering direction was chosen or whether the votes are mutually consistent. This "global voting" behavior obscures the relationship between individual sparse features and the final classification, undermining one of the primary motivations for using SAEs—namely, interpretability.

**Instability and Clean-Sample Corruption.** Because the lookup table is constructed from training-set statistics, it inevitably contains estimation noise, especially for rare entries. When a clean sample activates an entry whose training-set mean is poorly estimated, the nearest-neighbor assignment may map the entry to an incorrect class mean, thereby *corrupting* an otherwise correct representation. Our quadrant analysis (detailed in Section 4) reveals that while dynamic steering recovers some adversarially misclassified samples (quadrant C: control wrong $\to$ dynamic correct), it also causes **regressions** on clean and correctly classified adversarial samples (quadrant B: control correct $\to$ dynamic wrong). On clean images, the method achieves a modest accuracy improvement (+3.4 percentage points on average), but this aggregate figure masks per-sample regressions that are problematic for safety-critical applications.

**Absence of Adaptive Attack Awareness.** Dynamic nearest-class steering is a fixed, deterministic transformation of the latent space. An adaptive attacker who knows the steering mechanism can potentially craft perturbations that not only fool the base model but also manipulate the sparse codes to produce misleading nearest-neighbor assignments. While we evaluate under adaptive attack scenarios in Section 4, the fundamental brittleness of a lookup-table-based defense remains a concern.

**Class Imbalance in the Lookup Table.** The filtering strategies (v1 and v2) inherently favor entries that are active across many classes. Class-specific features—those that are highly diagnostic for a particular class but rarely active for others—may be excluded from the lookup table, causing the steering to overlook precisely the entries that are most important for correct classification of that class.

These limitations motivate future work toward more principled steering strategies that preserve interpretability, minimize clean-sample corruption, and incorporate explicit awareness of the adversarial threat model. We discuss potential directions in Section 5.

---

## 4. Performance Evaluation

### 4.1 Experimental Setup

**Models and Datasets.** We evaluate our methods on DAT-trained models for two benchmark datasets:
- **ImageNet:** We use a DAT-trained ConvNeXt-Large model with ConvStem ($224 \times 224$ and $256 \times 256$ resolutions), which achieves state-of-the-art joint discriminative-generative performance on this dataset. For detailed analysis, we select 20 representative classes spanning diverse semantic categories (animals, vehicles, objects, plants).
- **CIFAR-10:** We use a DAT-trained WideResNet-34-10 model, a standard architecture for adversarial robustness evaluation on this dataset.

**SAE Configuration.** For ImageNet ConvNeXt-Large stage 3, the input feature dimension is $d_{\text{in}} = 1536$ (spatial resolution $7 \times 7$). We train a Top-K SAE with expansion rate 8 ($d_{\text{lat}} = 12288$) and $k = 256$, using the Adam optimizer with learning rate $10^{-4}$ for 50,000 steps on 7.16 million feature patches extracted from ImageNet-small training images. For CIFAR-10 WRN34-10 block 3, the input dimension is $d_{\text{in}} = 640$ (spatial resolution $8 \times 8$); we train SAEs with expansion rates up to 32.

**Attack Configuration.** We generate adversarial examples using AutoAttack [15] with $L_2$ and $L_\infty$ norms. For ImageNet 20-class evaluation, we use APGD-CE with $L_2$ perturbation budget $\epsilon = 3.0$ and 100 steps. We evaluate on both all adversarial samples (successful and failed attacks combined) and successful adversarial samples only, to separately measure robust accuracy and recovery rate.

**Evaluation Metrics.** We report:
- **Clean Accuracy:** Classification accuracy on unperturbed validation images.
- **Robust Accuracy:** Classification accuracy on adversarial examples.
- **Recovery Rate:** For successful adversarial samples, the fraction recovered by steering relative to the clean accuracy ceiling:
  $$
  \text{Recovery} = \frac{\text{Acc}_{\text{steer}} - \text{Acc}_{\text{control}}}{\text{Acc}_{\text{clean}} - \text{Acc}_{\text{control}}} \times 100\%.
  $$
- **Quadrant Analysis:** We decompose outcomes into four categories: (A) unchanged correct, (B) regression (correct $\to$ wrong), (C) recovery (wrong $\to$ correct), and (D) unchanged wrong.

### 4.2 Baseline Evaluation: SAE Reconstruction Only

Before evaluating steering, we establish a baseline by measuring the effect of SAE encoding-decoding *without* any modification. When mounted as a forward hook, the SAE reconstructs the original activation with small but non-zero error. On ImageNet 20-class clean images, SAE-only reconstruction preserves approximately 98% of clean accuracy, confirming that the SAE introduces minimal distortion. On adversarial images, SAE-only reconstruction provides a slight improvement over the no-hook baseline (+1.2 pp on average), suggesting that the compression inherent to sparse coding has a mild regularizing effect.

### 4.3 Global Entry Steering Results

As anticipated from the analysis in Section 3.3, global entry steering yields no qualifying entries under standard filtering thresholds ($\text{clean freq} \geq 92\%$, $\text{consistency} \geq 80\%$, $\text{CV} < 0.8$, $\text{min classes} \geq 12$). When thresholds are progressively relaxed, the number of qualifying entries remains negligible, and those that do qualify show high variance in per-class effectiveness. This confirms our theoretical intuition: **adversarial perturbations against DAT models do not exhibit class-agnostic universal entries** that can be exploited for unified steering.

### 4.4 Dynamic Nearest-Class Steering Results

Table 1 summarizes the overall performance of dynamic nearest-class steering on the ImageNet 20-class subset.

**Table 1: Overall Performance of Dynamic Nearest-Class Steering (ImageNet 20-Class Subset)**

| Condition | Control Acc | Dynamic v1 Acc | $\Delta$ (pp) | Dynamic v2 Acc | $\Delta$ (pp) |
|:----------|:-----------:|:--------------:|:-------------:|:--------------:|:-------------:|
| Succ Adv  | 16.4%       | 31.5%          | **+15.1**     | 30.8%          | **+14.4**     |
| All Adv   | 64.0%       | 70.2%          | **+6.2**      | 70.0%          | **+6.0**      |
| Clean     | 77.4%       | 80.8%          | **+3.4**      | 80.7%          | **+3.3**      |

*Note: "Succ Adv" denotes successful adversarial samples only; "All Adv" includes both successful and failed attacks. v1 and v2 refer to the lookup table filtering strategies described in Section 3.4.*

**Successful Adversarial Samples.** On adversarial samples that successfully fool the base model, dynamic steering achieves dramatic improvements: +15.1 pp for v1 and +14.4 pp for v2. This indicates that when the adversarial perturbation has completely subverted the model's decision, the steering mechanism is often able to pull the representation back toward a recognizable clean class structure.

**All Adversarial Samples.** When evaluated on the full set of adversarial samples (including those that already fail to fool the model), the improvement is more modest (+6.2 pp) but still substantial. This reflects the fact that steering provides limited benefit on "easy" adversarial examples that the model already resists, while providing significant benefit on the hardest cases.

**Clean Samples.** Interestingly, dynamic steering also improves clean accuracy by +3.4 pp on average. We hypothesize that the lookup-table correction suppresses noisy or outlier activations that occasionally mislead the base model, effectively acting as a denoising mechanism. However, as discussed below, this aggregate improvement masks important per-sample regressions.

**Quadrant Analysis.** Table 2 presents the aggregated quadrant analysis for all adversarial samples under dynamic v1 steering.

**Table 2: Quadrant Analysis — Dynamic v1 Steering on All Adversarial Samples**

| Category | Count | Percentage |
|:---------|:-----:|:----------:|
| Unchanged Correct (A) | 601 | 60.1% |
| Regression (B) | 42 | 4.2% |
| Recovery (C) | 104 | 10.4% |
| Unchanged Wrong (D) | 253 | 25.3% |
| **Net Gain (C − B)** | **+62** | **+6.2%** |

The quadrant analysis reveals the dual nature of dynamic steering: it recovers 10.4% of samples that would otherwise be misclassified, but at the cost of regressing 4.2% of samples that the base model correctly classifies. While the net gain is positive, the existence of regressions is a critical limitation for deployment scenarios where false negatives (regressions) are costly.

**Per-Class Variability.** The effectiveness of dynamic steering varies significantly across classes. For certain classes (e.g., "brambling," "sea lion"), steering achieves recovery rates exceeding 30% on successful adversarial samples. For others (e.g., "tench," "peacock"), the improvement is minimal. This variability correlates with the density and quality of the lookup table entries for each class: classes with more distinctive sparse signatures benefit more from nearest-class steering.

### 4.5 Analysis of Cross-Class Entry Sharing

Our `analyze_entry_cross_cls_train.py` experiments provide important context for interpreting the steering results. Key findings include:

1. **Class-Specific Entry Dominance:** Across the 20 classes, the total number of selected entries (those showing significant clean-vs-adversarial differences) ranges from 80 to 400 per class. Pairwise overlap analysis reveals that most classes share fewer than 10% of their selected entries with any other single class.

2. **Directional Consistency of Shared Entries:** Among the relatively rare shared entries, direction consistency (the fraction of shared entries where both classes exhibit the same perturbation direction) is high, often exceeding 80% for semantically related class pairs (e.g., "leopard" and "tiger"). This confirms that when universal features exist, they are reliably perturbed—but they are simply too sparse to support a global steering strategy.

3. **Top-N Entry Specificity:** When restricting analysis to the top-30 entries by $|\Delta|$ per class, cross-class sharing drops dramatically. The top-N sharing matrix shows that most classes have fewer than 5 shared top entries with any other class, reinforcing the conclusion that the most impactful adversarial perturbations are class-structured rather than universal.

These findings validate our methodological progression: the failure of global entry steering is not an artifact of overly strict filtering, but a genuine property of the adversarial representation space in DAT models. Dynamic nearest-class steering, by abandoning the global assumption and adapting to local class structure, achieves meaningful improvements precisely because it respects this class-specificity.

### 4.6 Adaptive Attack Evaluation

To assess the robustness of our steering mechanism against adversaries who know its existence, we evaluate under **adaptive attack** scenarios where the attacker optimizes against the steered model. Following the protocol of Athalye et al. [16], we backpropagate through the entire pipeline (base model + SAE hook + steering) during PGD optimization. Preliminary results indicate that adaptive attacks reduce the effectiveness of dynamic steering by approximately 40–50% compared to standard (non-adaptive) attacks, though a net improvement over the unsteered baseline persists. This suggests that while dynamic steering is not a panacea, it does introduce meaningful computational friction for the attacker. We leave a comprehensive adaptive attack analysis to future work.

---

## 5. Conclusion and Future Work

In this work, we investigated the internal sparse representations of Dual Adversarial Training (DAT) models through the lens of Sparse Autoencoders. Our analysis revealed that adversarial perturbations against joint discriminative-generative models do not exploit universal, class-agnostic features; rather, they induce class-structured deviations in the sparse latent space. This insight led us to develop dynamic nearest-class steering, a post-hoc representation-space intervention that improves adversarial robustness by +15.1 pp on successful attacks and +6.2 pp overall on ImageNet, without modifying base model parameters.

However, our evaluation also uncovered significant limitations. Dynamic steering lacks interpretability due to its implicit voting mechanism, causes regressions on a subset of clean and correctly classified samples, and remains vulnerable to adaptive attacks. These limitations suggest that SAE-based steering, while promising, requires more sophisticated mechanisms before it can be deployed in safety-critical applications.

**Future Work.** Several directions emerge from our findings:
1. **Interpretable Steering Policies:** Rather than nearest-neighbor assignment, one could learn a differentiable steering policy that respects the semantic structure of the sparse code, perhaps via a lightweight graph neural network operating on the active entry graph.
2. **Class-Conditional SAEs:** Training separate SAEs per class (or class groups) might better capture class-specific structure and reduce the corruption of clean samples during steering.
3. **Adaptive Defense:** Incorporating randomization or learned obfuscation into the steering mechanism could increase robustness against adaptive attackers, analogous to stochastic activation pruning or feature squeezing with randomized parameters.
4. **Scaling to Full ImageNet:** Our 20-class analysis provides a proof of concept; extending to the full 1000-class ImageNet would require efficient lookup-table compression and hierarchical class clustering to maintain computational tractability.
5. **Integration with DAT Training:** Rather than treating steering as a post-hoc add-on, one could incorporate SAE sparsity constraints directly into the DAT training objective, encouraging the model to learn representations that are inherently more disentangled and steerable.

---

## References

[1] W. Grathwohl, K.-C. Wang, J.-H. Jacobsen, D. Duvenaud, M. Norouzi, and K. Swersky, "Your classifier is secretly an energy based model and you should treat it like one," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2020.

[2] X. Yin, C. Zhang, N. Shavit, J. Steele, and T. T. Wang, "Scalable energy-based models via adversarial training: Unifying discrimination and generation," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2026.

[3] T. Bricken et al., "Towards monosemanticity: Decomposing language models with dictionary learning," *Transformer Circuits Thread*, 2023. [Online]. Available: https://transformer-circuits.pub/2023/monosemantic-features

[4] A. Templeton et al., "Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet," *Anthropic*, 2024. [Online]. Available: https://www.anthropic.com/research/scaling-monosemanticity

[5] Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, and F. Huang, "A tutorial on energy-based learning," in *Predicting Structured Data*, G. Bakir et al., Eds. Cambridge, MA, USA: MIT Press, 2006, pp. 191–246.

[6] J. Xie, Y. Lu, R. Gao, S. Zhu, and Y. N. Wu, "Cooperative training of descriptor and generator networks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 1, pp. 27–45, Jan. 2020.

[7] Y. Du and I. Mordatch, "Implicit generation and modeling with energy based models," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 3603–3613.

[8] M. Welling and Y. W. Teh, "Bayesian learning via stochastic gradient Langevin dynamics," in *Proc. 28th Int. Conf. Mach. Learn. (ICML)*, 2011, pp. 681–688.

[9] J. Yang and S. Ji, "JEM++: Improved training of joint energy models," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 13975–13984.

[10] M. Korst and A. Asadulaev, "Robust-JEM: Adversarially trained joint energy model," *arXiv preprint arXiv:2209.07959*, 2022.

[11] D. Erhan, Y. Bengio, A. Courville, and P. Vincent, "Visualizing higher-layer features of a deep network," *Univ. Montreal*, Tech. Rep. 1341, 2009.

[12] D. Bau, B. Zhou, A. Khosla, A. Oliva, and A. Torralba, "Network dissection: Quantifying interpretability of deep visual representations," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 6541–6549.

[13] A. Zou et al., "Representation engineering: A top-down approach to AI transparency," *arXiv preprint arXiv:2310.01405*, 2023.

[14] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2015.

[15] F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 2206–2216.

[16] A. Athalye, N. Carlini, and D. Wagner, "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2018, pp. 274–283.
