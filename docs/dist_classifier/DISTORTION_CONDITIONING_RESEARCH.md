# Distortion-conditioned IQA: literature review and experiment plan

**Date:** 2026-09-01

**Scope:** A KADID-10k proof of concept now, followed by transfer to the synthetic, authentic, and generative datasets in [datasets.md](datasets.md).

## Executive answer

The idea is scientifically sound, and it is worth running. It is also not new in its basic form: [MEON (Ma et al., 2018)](https://ece.uwaterloo.ca/~zduanmu/publications/tip2018biqa/paper/tip18MEON.pdf) already predicts a distortion probability vector and uses it to combine distortion-specific quality predictions. More recent work—especially [LIQE](https://arxiv.org/abs/2303.14968), [DB-CNN](https://arxiv.org/abs/1907.02665), and [CONTRIQUE](https://arxiv.org/abs/2110.13266)—supports the broader claim that distortion supervision can improve quality representations.

The current plan is a good **diagnostic proof of concept**, but a 25-way KADID classifier plus a quality model trained on ground-truth one-hot labels is not a solid transfer design. I recommend:

1. Keep the ResNet classifier experiment, but aggregate its 25 KADID probabilities into the repository's six KADID distortion groups plus a separate **no-distortion/pristine** state.
2. Train the quality model with the same kind of input it will receive at inference: calibrated, preferably out-of-fold, predicted probabilities. Use ground-truth one-hot groups only as an oracle upper bound.
3. Compare simple concatenation against a small **residual mixture of experts**:
   \[
   \hat q(x)=q_0(z)+\sum_g p_g(x)\,\Delta_g(z),
   \]
   where \(z\) is the image embedding, \(q_0\) is a shared scorer, and each \(\Delta_g\) is a small group-specific correction.
4. Treat classifier confidence as a possible severity leak, not automatically as clean distortion-type information.
5. For the eventual cross-dataset model, replace one flat softmax with a hierarchy: an image-source regime and multi-label impairment families, with an unknown route.

If oracle conditioning does not reliably beat the matched image-only baseline, stop pursuing the classifier branch. If oracle conditioning helps but predicted conditioning does not, the hypothesis is still viable but the classifier/taxonomy is the bottleneck.

## What the current repository is doing

The current distortion branch is clear and mostly well set up:

- [scripts/dist_classifier/train.sh](scripts/dist_classifier/train.sh) launches the configuration in [configs/dist_classifier.yaml](configs/dist_classifier.yaml).
- [models/dist_classifier/model.py](models/dist_classifier/model.py) fine-tunes an ImageNet-pretrained ResNet-18.
- [models/dist_classifier/dataset.py](models/dist_classifier/dataset.py) defines 26 classes: KADID's 25 exact distortion types and one pristine class.
- The train/validation split is by pristine reference, which is the right leakage boundary.
- Pristine references are added to the classifier set, and class-weighted cross entropy compensates for their lower count.
- The baseline in [train.py](train.py) uses a frozen CLIP/SigLIP vision embedding and a small MLP quality head.

There are four repository-specific issues to fix before drawing conclusions.

### 1. The classifier vocabulary and project vocabulary differ

The classifier predicts 25 KADID-specific types. The project explicitly defines a portable taxonomy in [prepare_data.py](prepare_data.py):

- compression
- generative
- blur
- noise
- color
- tone
- spatial
- authentic

KADID covers only compression, blur, noise, color, tone, and spatial. A KADID-only model cannot learn authentic or generative conditions. Worse, a 25-way classifier is forced to express an image from another dataset in KADID's vocabulary even when none of the classes is appropriate.

For the cheapest proof of concept, preserve the existing classifier and sum its probabilities according to \(p_g=\sum_{k\in g}p_k\). Also train a direct six-group-plus-pristine classifier as a comparison. Exact-type supervision may learn finer features; direct group supervision may be more transferable.

### 2. The quality loader does not currently return the group

[README.md](README.md) says every batch carries `group`, but [dataset.py](dataset.py) currently returns `distortion` and `level`, not `group`. The conditioned experiment needs a stable group-to-index mapping stored with the checkpoint.

### 3. Hard-label training followed by soft-label inference is a distribution mismatch

If the quality head sees perfect one-hot labels during training and noisy softmax vectors during inference, it has never learned how to respond to classifier ambiguity or errors. Feeding the classifier's in-sample predictions is better, but still optimistic because the classifier has already fitted those references.

Use reference-level cross-fitting:

1. Divide the quality-training references into folds.
2. Train the distortion classifier on all but one fold.
3. Produce probabilities for the omitted fold.
4. Repeat until every quality-training image has a prediction from a classifier that did not train on its reference.
5. Fit the final inference classifier on all quality-training references, then test on untouched references.

This makes the quality head's training condition resemble its inference condition. A jointly trained shared network, as in MEON, is the longer-term alternative.

### 4. Resolution is already affecting the classifier

The existing MLflow runs show:

- native KADID size: best validation accuracy **0.8051**;
- 224-pixel crop: best validation accuracy **0.6930**.

The native-resolution run finishes at 0.9250 training accuracy and 0.7693 validation accuracy, so its in-sample probabilities are substantially easier than its reference-disjoint predictions. This reinforces the need for out-of-fold probabilities.

The resolution result is also consistent with [MUSIQ](https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html), which argues that fixed resizing and cropping can alter the very quality an IQA model is meant to measure. Native KADID images happen to share a size, but cross-dataset batching will require aspect-preserving resize/padding or multi-crop/native-resolution aggregation.

## What the literature says

| Work | Main result relevant here | Implication |
| --- | --- | --- |
| [DIIVINE (Moorthy & Bovik, 2011)](https://pubmed.ncbi.nlm.nih.gov/21521667/) | An early general-purpose BIQA pipeline identifies the distortion before applying distortion-aware quality prediction. | Distortion routing has a long history; it is not merely a modern architectural trick. |
| [MEON (Ma et al., 2018)](https://ece.uwaterloo.ca/~zduanmu/publications/tip2018biqa/paper/tip18MEON.pdf) | Its distortion subnetwork outputs a probability vector. Its quality subnetwork predicts one score per distortion and returns their probability-weighted sum. The model is jointly optimized after distortion pretraining. | This is the closest precedent. Predicted probabilities should participate in quality training, not appear only at inference. |
| [DB-CNN (Zhang et al., 2020)](https://arxiv.org/abs/1907.02665) | Pretrains a synthetic-distortion branch to classify distortion type and level, combines it with a semantic branch, then fine-tunes the fused representation for IQA. | Distortion supervision is useful as representation learning, not only as a one-hot side channel. Type and severity are difficult to keep separate. |
| [MetaIQA (Zhu et al., 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_MetaIQA_Deep_Meta-Learning_for_No-Reference_Image_Quality_Assessment_CVPR_2020_paper.html) | Learns a quality prior across distortion-specific tasks and adapts it to unknown distortions, including authentic ones. | Closed-set KADID classes are not the only way to exploit distortion structure; learning what is shared across distortions is more transfer-oriented. |
| [HyperIQA (Su et al., 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.html) | Uses semantic content to generate adaptive quality-prediction parameters and explicitly models local distortions. | Quality is conditioned by content as well as distortion; a global class vector cannot replace local quality features. |
| [UNIQUE (Zhang et al., 2021)](https://arxiv.org/abs/2005.13983) | Identifies a strong synthetic/authentic distribution shift and trains a unified uncertainty-aware model from multiple databases using pairwise ranking. | Mixing datasets and modelling uncertainty are central to transfer; treating authentic as another synthetic class is too simple. |
| [MUSIQ (Ke et al., 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html) | Processes native-resolution images at multiple scales because resizing/cropping may change perceived quality. | The current square resize and pooled embedding are acceptable controls, but likely a larger bottleneck than conditioning later. |
| [CONTRIQUE (Madhusudana et al., 2022)](https://arxiv.org/abs/2110.13266) | Learns quality representations through distortion type/degree-aware contrastive training over synthetic and authentic images, then freezes the encoder for shallow regression. | A learned degradation representation may transfer better than a fixed closed-set posterior. Mixed synthetic/authentic training matters. |
| [LIQE (Zhang et al., 2023)](https://arxiv.org/abs/2303.14968) | Jointly learns quality, scene, and eleven broad distortion categories. Its ablation reports mean SRCC 0.915 for quality only, 0.920 for quality plus distortion, and 0.922 for all three tasks across six datasets. | Direct evidence supports the hypothesis, but the gain is modest. Broad categories, an “other” category, joint training, and multiple datasets are important design choices. |
| [QPT (Zhao et al., 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Quality-Aware_Pre-Trained_Models_for_Blind_Image_Quality_Assessment_CVPR_2023_paper.html) | Constructs roughly \(2\times10^7\) possible compositional degradations and learns with a quality-aware contrastive objective; the paper reports advantages over supervised classification pretexts. | For a stronger final system, quality-aware pretraining over composed degradations is more promising than scaling a ResNet type classifier alone. |
| [TOPIQ (Chen et al., 2024)](https://arxiv.org/abs/2308.03060) | Uses high-level semantics to guide attention toward local distortion evidence across feature scales. | Keep patch/local/multi-scale features on the roadmap; distortion labels do not solve spatial pooling. |
| [MDID (Sun et al., 2017)](https://doi.org/10.1016/j.patcog.2016.07.033) | Provides images containing one to four random distortions and shows multiply distorted IQA is challenging. | A single-label softmax is structurally wrong for many real images; the eventual impairment head should be multi-label or continuous. |

The overall picture is consistent: distortion supervision often helps, but the strongest transfer-oriented methods learn shared, local, multi-scale, uncertainty-aware representations rather than trusting a closed-set type label alone.

## Main scientific risks

### Taxonomy and open-set failure

A softmax must allocate all probability mass among known classes. An authentic low-light photograph, a restoration artifact, or a new codec can therefore receive a confident but meaningless KADID label. An “unknown/other” output helps, but it must be trained with outlier examples; temperature scaling alone does not solve open-set recognition.

For the final design, separate two concepts that the current eight groups mix:

- **source regime:** synthetic, authentic capture, generative/restoration, pristine, unknown;
- **impairment families:** blur, noise, compression, color, tone, spatial, other.

The source regime can be multiclass. The impairment families should eventually be sigmoid/multi-label because several may coexist.

### Probability calibration

Modern neural-network probabilities are often poorly calibrated; [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) found temperature scaling to be a strong simple baseline. Fit the temperature only on inner validation references, then report NLL, Brier score, and expected calibration error in addition to accuracy. Calibration on KADID still does not guarantee calibration after a dataset shift.

### Severity leakage

KADID's severe distortions are generally easier to classify than mild ones. Consequently, maximum probability and entropy can act as an implicit severity signal, and severity is strongly related to MOS. A gain from the soft posterior could therefore be “classifier confidence predicts severity,” not “knowing the type changes how quality should be judged.”

Required controls:

- ground-truth hard group versus predicted hard argmax;
- predicted hard argmax versus full soft posterior;
- posterior entropy/max-probability alone;
- the whole posterior without image features;
- correlation of entropy/max-probability with KADID level and MOS.

If the soft vector wins mainly because of confidence, decide whether this is acceptable. It may improve a metric, but it changes the scientific claim from type conditioning to learned degradation conditioning.

### Dataset and score shortcuts

In [datasets.md](datasets.md), several datasets have a constant group. “Authentic” or “generative” can therefore become a dataset identifier, and the quality head can learn dataset-specific score offsets. The group-only control is essential. SPAQ's constant condition is a particularly useful negative control, as the repository already notes.

For mixed-dataset training, form ranking pairs within a dataset or use explicit dataset-scale calibration. Do not interpret pooled correlation across heterogeneous MOS scales as quality generalization.

### Pristine is not authentic

The classifier's pristine examples are undistorted KADID references. The repository's authentic group means photographs not synthetically degraded, which may still contain sensor noise, blur, exposure errors, and processing artifacts. Do not map pristine directly to authentic.

KADID's pristine images also have no MOS rows in the quality-training CSV. A standalone pristine expert would therefore be unsupervised. The shared \(q_0\) term in the residual mixture handles this more safely: when the classifier predicts no distortion, group residuals can approach zero and the shared scorer remains active.

### Multiple and spatially local distortions

KADID has one global distortion per image. Authentic and restoration images often contain combinations and local artifacts. A whole-image multiclass label cannot say “blurred background, noisy shadows, blocking in one region.” This is another reason to keep patch tokens or multi-crop features and move to multi-label impairment probabilities later.

## Recommended KADID proof of concept

### Stage A — validate the condition itself

Use the exact same CLIP/SigLIP backbone, image split, loss, optimizer, and training budget for every quality-head comparison.

| ID | Image input | Condition | Fusion | Question |
| --- | --- | --- | --- | --- |
| B0 | yes | none | baseline MLP | Reference performance |
| B1 | no | predicted group posterior | small MLP | How much score prior is in the condition alone? |
| C1 | yes | oracle hard group | concatenate | Is useful condition information present at all? |
| C2 | yes | predicted hard group | concatenate | Does classifier top-1 preserve the oracle benefit? |
| C3 | yes | calibrated soft posterior | concatenate | Does uncertainty help? |
| C4 | yes | calibrated soft posterior | residual mixture of experts | Does explicit routing beat generic concatenation? |
| N1 | yes | shuffled within split | same as C4 | Is a correct condition necessary? |
| N2 | yes | uniform/zero condition | same as C4 | Is the model actually using the condition? |
| N3 | yes | entropy or max-probability only | concatenate | Is the gain mainly severity/confidence? |

Use a matched-capacity image-only baseline for C4. A mixture with seven full independent MLPs has far more parameters than the current head; prefer a shared head plus small, zero-initialized residual adapters.

### Stage B — train with realistic conditions

For C2–C4, use out-of-fold classifier predictions for the quality-training rows. At final evaluation:

- train the classifier only on quality-training references;
- calibrate it only on an inner split of those references;
- keep the final IQA validation/test references untouched;
- run classifier and quality inference exactly as deployment will.

Condition dropout is useful: with a small probability, replace (p) by uniform or zero during quality training. This discourages catastrophic dependence on a confidently wrong classifier. Mild probability smoothing can help, but it is not a substitute for out-of-fold predictions.

### Stage C — measure the right things

Classifier:

- exact-type top-1 accuracy and macro-F1;
- aggregated group accuracy and macro-F1;
- per-class recall and confusion matrices;
- NLL, Brier score, and ECE;
- accuracy/calibration by KADID severity;
- pristine recall separately.

Quality:

- SRCC and PLCC on KADID;
- macro SRCC/PLCC by broad group and by exact distortion;
- mean within-reference SRCC;
- within-distortion severity ranking;
- paired bootstrap confidence intervals over pristine references;
- at least five paired seeds or splits.

The current baseline already reports within-reference SRCC, which is valuable. Add the per-group, per-type, and confidence-interval views rather than relying on one global correlation.

### Decision rules

- **C1 does not consistently beat B0:** the condition is not useful under this backbone/head; stop or revisit the quality representation before improving the classifier.
- **C1 beats B0, but C2/C3 do not:** the classifier, taxonomy, or calibration is the bottleneck.
- **C3 beats C2, but N3 nearly matches C3:** most of the benefit is confidence/severity, not type.
- **B1 nearly matches C3/C4:** the result is largely a distortion-group score prior.
- **Shuffling/wrong groups do not hurt:** the quality model is ignoring the condition or exploiting a non-semantic shortcut.
- **KADID improves but the frozen design fails on TID2013/CSIQ:** the exact taxonomy or KADID appearance has been learned, not transferable distortion knowledge.

## Architecture recommendation

### First implementation: concatenation

Concatenate the calibrated group posterior to the frozen image embedding and use the current MLP. This is the smallest change and should remain in the ablation even if a stronger fusion is added.

Do not train this version with oracle labels and deploy it with classifier outputs. Use oracle labels only for C1.

### Preferred small model: residual mixture of experts

Let \(z=f(x)\) and let \(p_g(x)\) be the calibrated broad-group posterior. Use:

\[
\hat q(x)=q_0(z)+\sum_{g=1}^{G}p_g(x)\Delta_g(z).
\]

Advantages:

- \(q_0\) is always available when the classifier is uncertain or the distortion is unknown;
- group experts learn corrections rather than complete, data-hungry scoring functions;
- the contribution of each condition is inspectable;
- soft combinations naturally interpolate;
- experts can be zero-initialized so training begins at the image-only baseline.

A condition embedding with FiLM or a hypernetwork is a reasonable later alternative, inspired by content-adaptive methods such as HyperIQA, but it adds harder-to-interpret capacity before the basic hypothesis is established.

### Longer-term model: shared multitask backbone

After the controlled proof of concept, consider one backbone with:

- a quality head;
- a source-regime head;
- a multi-label impairment head;
- a differentiable posterior-to-quality path;
- a joint quality plus classification objective.

This follows the core lesson of MEON and LIQE, removes the two-backbone inference cost, and lets the quality loss shape the degradation representation. Pretrain the degradation task first, then jointly fine-tune with a smaller auxiliary-loss weight. Compare against a quality-aware pretrained encoder such as CONTRIQUE/QPT so that “classifier supervision” is not confused with “better low-level pretraining.”

## Transfer roadmap aligned with datasets.md

### Phase 1 — KADID only

Use six synthetic groups plus pristine/no-distortion. Establish the oracle gap, predicted-condition gap, calibration, severity confound, and negative controls. Do not claim cross-dataset transfer yet.

A useful internal generalization test is leave-one-exact-distortion-out within each broad group. For example, train a group classifier without one blur algorithm and test whether it still assigns the held-out algorithm to blur. This can be used during development without consulting the final held-out datasets.

### Phase 2 — mixed training datasets

Train on the datasets designated for training in [datasets.md](datasets.md), with sampling by dataset. Introduce the source-regime/impairment hierarchy.

Be cautious with labels that are constant for an entire dataset:

- SPAQ and GFIQA-20k provide authentic/source-regime evidence, but not detailed artifact labels.
- PIPAL and AIGCIQA2023 provide generative/source-regime evidence, but “generative” covers heterogeneous failures.
- Hold out an entire training dataset within a regime during model selection. Otherwise, classifier accuracy may measure dataset recognition.

For quality-score transfer, consider within-dataset pairwise/ranking loss in addition to the current regression loss, following the motivation of UNIQUE. Keep the loss unchanged in Phase 1 so the conditioning comparison remains isolated.

### Phase 3 — one final held-out evaluation

Freeze the taxonomy, classifier, calibration procedure, architecture, and hyperparameters before evaluating TID2013, CSIQ, CID2013, KonIQ-10k, CLIVE, AGIQA-3K, and UHD-IQA as specified in [datasets.md](datasets.md).

Report per-dataset results, not pooled correlations. The most diagnostic outcomes are:

- TID2013 and CSIQ: transfer of broad synthetic families;
- CID2013/KonIQ/CLIVE/UHD-IQA: whether authentic routing generalizes rather than identifying one training corpus;
- AGIQA-3K: transfer across generators;
- wrong, shuffled, zeroed, and uniform conditions on every dataset.

## What I would do next

In priority order:

1. Add `group` to [dataset.py](dataset.py) with a checkpointed stable mapping.
2. Add a utility that maps the existing 26-way logits to six group probabilities plus pristine.
3. Evaluate group-level confusion, calibration, and severity dependence for the existing checkpoints.
4. Implement B0, B1, C1, C2, C3, and the negative controls using concatenation.
5. Add the residual mixture only if C1 establishes a real oracle benefit.
6. Generate reference-level out-of-fold classifier predictions before treating C3/C4 as deployment evidence.
7. Only then expand the classifier to multi-dataset hierarchical/multi-label prediction.
8. After the conditioning question is answered, upgrade the image representation to patch-token, local, or multi-scale features. The current single pooled, square-resized CLIP embedding is likely to become the dominant ceiling.

## Bottom line

Proceed, but frame the first experiment correctly:

- **Research hypothesis:** a portable distortion-family condition improves a matched NR-IQA model.
- **Oracle test:** ground-truth broad group.
- **Deployment test:** calibrated out-of-fold predicted group distribution.
- **Preferred fusion:** shared scorer plus posterior-weighted residual experts.
- **Critical controls:** group-only, hard versus soft, entropy-only, shuffled/wrong/zero condition, and reference-level confidence intervals.
- **Transfer requirement:** hierarchical unknown-aware and eventually multi-label conditioning, not a flat KADID-specific softmax.

That design preserves the simplicity of your idea while making the result interpretable and much harder to explain away as label leakage, severity leakage, classifier overfitting, or dataset recognition.

## Sources

Sources were accessed on 2026-09-01. Primary papers, official proceedings/project pages, and the official KADID page were prioritized.

- A. K. Moorthy and A. C. Bovik, “Blind Image Quality Assessment: From Natural Scene Statistics to Perceptual Quality,” *IEEE Transactions on Image Processing*, 2011. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/21521667/)
- K. Ma et al., “End-to-End Blind Image Quality Assessment Using Deep Neural Networks,” *IEEE Transactions on Image Processing*, 2018. [Author-hosted paper](https://ece.uwaterloo.ca/~zduanmu/publications/tip2018biqa/paper/tip18MEON.pdf)
- W. Zhang et al., “Blind Image Quality Assessment Using a Deep Bilinear Convolutional Neural Network,” *IEEE Transactions on Circuits and Systems for Video Technology*, 2020. [Paper](https://arxiv.org/abs/1907.02665)
- H. Zhu et al., “MetaIQA: Deep Meta-Learning for No-Reference Image Quality Assessment,” *CVPR*, 2020. [CVF paper page](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_MetaIQA_Deep_Meta-Learning_for_No-Reference_Image_Quality_Assessment_CVPR_2020_paper.html)
- S. Su et al., “Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network,” *CVPR*, 2020. [CVF paper page](https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.html)
- W. Zhang et al., “Uncertainty-Aware Blind Image Quality Assessment in the Laboratory and Wild,” *IEEE Transactions on Image Processing*, 2021. [Paper](https://arxiv.org/abs/2005.13983)
- J. Ke et al., “MUSIQ: Multi-Scale Image Quality Transformer,” *ICCV*, 2021. [CVF paper page](https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html)
- P. C. Madhusudana et al., “Image Quality Assessment Using Contrastive Learning,” 2022. [Paper](https://arxiv.org/abs/2110.13266) and [official implementation](https://github.com/pavancm/CONTRIQUE)
- W. Zhang et al., “Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective,” *CVPR*, 2023. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Blind_Image_Quality_Assessment_via_Vision-Language_Correspondence_A_Multitask_Learning_CVPR_2023_paper.html)
- K. Zhao et al., “Quality-Aware Pre-Trained Models for Blind Image Quality Assessment,” *CVPR*, 2023. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Quality-Aware_Pre-Trained_Models_for_Blind_Image_Quality_Assessment_CVPR_2023_paper.html)
- C. Chen et al., “TOPIQ: A Top-Down Approach from Semantics to Distortions for Image Quality Assessment,” *IEEE Transactions on Image Processing*, 2024. [Paper](https://arxiv.org/abs/2308.03060)
- W. Sun, F. Zhou, and Q. Liao, “MDID: A Multiply Distorted Image Database for Image Quality Assessment,” *Pattern Recognition*, 2017. [Publisher page](https://doi.org/10.1016/j.patcog.2016.07.033)
- C. Guo et al., “On Calibration of Modern Neural Networks,” *ICML*, 2017. [PMLR paper page](https://proceedings.mlr.press/v70/guo17a.html)
- H. Lin, V. Hosu, and D. Saupe, “KADID-10k: A Large-Scale Artificially Distorted IQA Database,” *QoMEX*, 2019. [Official database page](https://database.mmsp-kn.de/kadid-10k-database.html)

## Research limits

This is a design review, not a new benchmark run. I inspected the existing code, checkpoints, and MLflow metrics but did not retrain models. The search focused on distortion-aware NR-IQA, auxiliary degradation learning, cross-dataset transfer, calibration, multiple distortions, and spatial/multi-scale representation. It was stopped after the central claims had direct primary support and additional papers were repeating the same design directions.
