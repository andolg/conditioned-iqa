# Datasets: what trains, what is held out, and why

The question is whether telling the metric what kind of distortion it is
looking at makes its scores agree better with human ones. The condition
names a **group**, not an individual type: no two releases share a type
vocabulary, so a per-type label teaches the corpus rather than the
distortion, and a fine taxonomy leaves most cells nearly empty.

## Training

| dataset | rows | group | why it is here |
| --- | --- | --- | --- |
| **KADID-10k** | 10,125 | blur · noise · compression · colour · tone · spatial | The only set where the same photograph appears clean and degraded — 81 references, 25 distortions, 5 severities. An effect is attributable to the condition rather than to the content, and this is the only source of the severity check. |
| **SPAQ** | 11,125 | `authentic` | Ordinary smartphone captures. A constant condition, which makes it the control: a model that improves when handed a constant has learned to recognise the dataset. |
| **GFIQA-20k** | 19,998 | `authentic` | Faces, where people notice softness they would forgive in a landscape. |
| **PIPAL** | 24,200 | `generative` (17,800 rows) | Artifacts that restoration algorithms produce — super-resolution, denoising, and the particular strangeness of GAN outputs. See the note below: its scores are Elo ratings and do not compare across references. |
| **AIGCIQA2023** | 2,400 | `generative` | Generated images, whose artifacts have no counterpart in anything a camera or codec does. Prompts are the split key. |

## Held out

Scored once, at the end, never consulted while anything is chosen.

| dataset | rows | group | what it tests |
| --- | --- | --- | --- |
| **TID2013** | 3,000 | six distortion groups | The taxonomy test. Twenty-four types in its own vocabulary folded into the same groups: a model that learned groups carries over, one that learned KADID's type table does not. |
| **CSIQ** | 866 | blur · noise · compression · tone | Whether that holds on another laboratory's stimuli. |
| **CID2013** | 474 | `authentic` | The pristine target — no modern IQA training corpus has touched it. |
| **KonIQ-10k** | 10,073 | `authentic` | Authentic photographs with no distortion labels: the near-domain check. |
| **CLIVE** | 1,162 | `authentic` | The same, from a different study. |
| **AGIQA-3K** | 2,982 | `generative` | Six different generators. |
| **UHD-IQA** | 6,073 | `authentic` | The high end of the scale, where models trained on visibly damaged pictures stop discriminating. |

## Reading the results

| | what it is | what makes it fail |
| --- | --- | --- |
| **Part 1** | the unconditioned scorer against its conditioned variants, trained separately at matched capacity; SRCC and PLCC per dataset | correlations pooled across datasets — the scales differ, and pooling measures the gap between them |
| **Part 2** | whether the condition did anything | a wrong group that does not hurt, a shuffle that does not erase the gain, a zeroed condition that changes nothing, permuted training that matches correct training |

A gain in Part 1 alone is also consistent with a lucky run, which is why
Part 2 exists.

## PIPAL: what its numbers mean

The release numbers its distortion classes and names them nowhere. Table 10
of the journal version ([arXiv:2011.15002](https://arxiv.org/abs/2011.15002))
lists seven sub-types, and each has a different number of parameter
variants — which is enough to recover the mapping, because those counts
match the number of variants per class in the data exactly:

| code | sub-type | variants |
| --- | --- | --- |
| `00` | traditional SR | 12 |
| `01` | PSNR-oriented SR | 16 |
| `02` | SR with kernel mismatch | 10 |
| `03` | GAN-based SR | 24 |
| `04` | denoising | 13 |
| `05` | SR and denoising jointly | 14 |
| `06` | traditional distortions, nine of them mixed | 27 |

They sum to the 116 distortion levels the paper reports. The filename is
`Aaaaa_bb_cc` — image, class, variant within the class — which the author
confirmed in issue 13 of the dataset repository. The first six classes are
restoration artifacts and carry the group `generative`. Class `06` mixes
blur, noise, two codecs, colour quantization and spatial warping; the order
of variants inside it follows Table 10 for most types but not all, so those
rows carry no group rather than a guessed one.

**The scores are Elo ratings, and they rank only within a reference.** Every
image starts at 1400 and moves through pairwise comparisons against other
restorations of the same picture, so 1400 here and 1400 there both mean
"typical for its own group" rather than equal quality. Nothing in PIPAL
compares images across references, so the data holds no information about
which picture is better in absolute terms — measured, the reference explains
0.1% of the variance, and the two hundred reference means span 22 points
against a 622-point spread within a single one.

What the ratings do share is a unit: 200 Elo points is 76% preference in any
group, by construction. So the scores are scaled globally like every other
dataset, not per reference — rescaling each reference separately would
destroy that unit. It was worth testing rather than assuming: trained both
ways at identical settings, the two targets reach the same within-reference
SRCC (0.183 against 0.182), and the gap that first appeared between them
turned out to be the loss seeing a target squeezed into half the range,
which an affine rescale fixes.

What does need care is the metric. `train.py` reports a within-reference
SRCC beside the global one, and on PIPAL that is the number to read: the
distortion class and its parameter variant together explain 65% of the
ratings, so a model that learns nothing but "ESRGAN usually scores 1550,
bicubic 1250" already correlates well globally. What the dataset was built
to measure — how a given algorithm did on a given picture — is the
remaining third, and it lives only inside a reference.

## Splits and scores

Splits go by reference. In KADID a hundred and twenty-five rows are one
photograph; splitting them across the boundary lets the model score by
recognising the picture — up to 0.44 SRCC of it on frozen features. For
photographs, splitting by image is fine.

Every dataset is min-maxed into [0, 1] with higher meaning better, and the
published number stays beside it in `original_subjective_score`. CSIQ's DMOS
runs backwards and is flipped during preparation.
