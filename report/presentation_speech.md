# Presentation speech

Target duration: approximately 7 minutes 20 seconds at 120 words per minute. Slides 9–12 and 17–24 are intentionally omitted.

## Slide 1 — AI Summer Camp Challenge: Distortion Conditioned NR IQA

Hello everyone. We are Sergey Muravlev, Andrey Dongolenko, and Danila Evsiukov. Today we present our work on distortion-conditioned no-reference image quality assessment.

## Slide 2 — Task description

No-reference IQA predicts perceptual quality without a pristine reference. In many applications, however, we know the broad degradation type—such as blur, noise, compression, colour, spatial or generative artefacts. Acquisition and restoration pipelines often expose exactly this kind of metadata. Our question is: can this information make predictions agree better with human judgement?

## Slide 3 — Our baseline

Our baseline sends the image through a frozen CLIP or SigLIP encoder and trains only a small regression head to predict quality from zero to one. Conditioning enters this head, not the expensive backbone. Training therefore stays lightweight, while the visual representation remains identical between matched models. This isolates whether distortion information itself provides value.

## Slide 4 — Experimental setup

Our training pool covers synthetic distortions, authentic photographs, and generated or restoration artefacts. Final testing uses separate datasets with synthetic and real-world degradation. For reference-based data, all variants of one source image stay in the same split, preventing leakage. Multi-dataset results are macro-averaged so large datasets cannot dominate. We report both SRCC for ranking agreement and PLCC for linear agreement with human scores.

## Slide 5 — Training protocol

Matched comparisons use five epochs, batch size 32, AdamW at a learning rate of ten to the minus three, and Smooth L1 loss. The hidden width is 256, dropout is 0.1, and the seed is fixed. Only the prediction head is optimized.

## Slide 6 — Conditioning approaches

We study four conditioning levels: natural-language descriptions; oracle distortion labels; labels predicted automatically by ResNet-18; and richer intermediate features from a classifier or IQA model. Each trades convenience against information richness. This progression lets us compare how informative each condition is, whether external metadata is required, and what it costs.

## Slide 7 — Text conditioning

The frozen CLIP text tower encodes a distortion-specific instruction. We tested concatenation, residual correction, and interaction fusion. The strongest combines visual and text features with their elementwise interaction. On KADID-10k, SRCC rises from 0.7599 to 0.7847. Paraphrasing retains most performance, while generic, wrong, or shuffled prompts cause large drops. The model therefore uses prompt meaning, not merely extra parameters.

## Slide 8 — Datasets and multi-view experiments

For better generalization, we train on KADID-10k, SPAQ, and AIGCIQA2023 with equal sampling. We add one global crop and four local tiles because one crop may hide localized defects, then use monotonic MOS calibration to align score distributions. Across this development path, test SRCC rises from 0.4783 to 0.6612. This total change is not a conditioning-only effect, because the data and representation also change. Matched controls confirm separate gains from text and multi-view processing.

## Slide 13 — Label conditioning: description

Next, we ask whether coarse labels provide the same benefit more directly. Each of eight distortion groups receives a learned 32-dimensional embedding, supplied as ground truth at inference. This gives us an oracle upper bound when reliable metadata is available. We remove the label for ten percent of training examples so the model also learns an unconditional fallback.

## Slide 14 — Label conditioning: setup

The 32-dimensional bottleneck balances expressiveness with low parameter count. We evaluate both KADID-only training and the three-dataset mixture. Transfer is then measured on seven external datasets spanning synthetic, authentic, and generative distortions.

## Slide 15 — Label conditioning: single dataset

On KADID-10k, residual gating is the strongest CLIP-Base variant. It preserves an image-only prediction and adds a gated label-dependent correction. Validation SRCC improves from 0.7548 to 0.7658, and test SRCC from 0.4861 to 0.5230. This even exceeds the CLIP-Large baseline's test SRCC with far fewer parameters and roughly one tenth of its FLOPs.

## Slide 16 — Label conditioning: mixture of datasets

With three-dataset training, Input FiLM leads on validation, but the rank-four hypernetwork transfers best, reaching test SRCC 0.6380 and PLCC 0.6534. The residual gate is close at 0.6314 and runs faster. Wrong, zeroed, and shuffled-label interventions reduce performance, confirming that both models actually use group identity. So the best fusion depends on whether we prioritize validation accuracy, transfer, or throughput.

## Slide 25 — Classifier label conditioning

Oracle labels may be unavailable, so we train ResNet-18 to predict them. Hard labels, soft probabilities, and joint training all underperform the zero-condition control. Even the oracle label adds only 0.0044 test SRCC with this head. The categories encode distortion type but not its severity, and prediction errors erase the small available gain.

## Slide 26 — Feature conditioning

Intermediate classifier features preserve more information than the final class, including cues about strength and appearance. Layer-four features give the best KADID SRCC, 0.8906, while projected layer-three features give the best test SRCC, 0.5200. Gains vary sharply across datasets, so these features are valuable in-domain but do not provide uniformly robust transfer.

## Slide 27 — Full results

This table summarizes accuracy versus efficiency. It is not a strict leaderboard because test suites differ across experiment groups. Among the shown CLIP-Base systems, calibrated five-view text has the highest reported test SRCC, 0.6612. Pooled text retains nearly baseline throughput, while oracle residual gating is the fastest strong option. Those models require a known condition; classifier and ARNIQA features remove that requirement. ARNIQA helps, but greatly increases memory and latency.

## Slide 28 — Conclusion

In conclusion, distortion information can improve no-reference IQA, but its form matters. Oracle labels help cheaply; predicted labels are too coarse; and continuous features carry more information but add cost or transfer unevenly. Text offers the best overall balance: meaningful semantics, a frozen backbone, and strong performance when combined with multiple views. For deployment, the right choice depends on whether metadata, throughput, or maximum accuracy matters most. Thank you.
