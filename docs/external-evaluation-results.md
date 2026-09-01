# Initial held-out dataset evaluation

These are zero-retraining tests of the seed-0 CLIP-Large checkpoints trained
on KADID-10k. The held-out datasets were not used for training, model choice,
or prompt selection. Higher SRCC is better.

| held-out dataset | image-only baseline | text interaction | interaction minus baseline |
| --- | ---: | ---: | ---: |
| TID2013 | 0.6458 | 0.6395 | -0.0062 |
| CSIQ | 0.8272 | 0.8333 | +0.0061 |
| CID2013 | 0.3739 | 0.2574 | -0.1166 |
| KonIQ-10k | 0.5454 | 0.5377 | -0.0077 |
| CLIVE | 0.5628 | 0.5200 | -0.0427 |
| UHD-IQA | 0.1597 | 0.1731 | +0.0134 |
| AGIQA-3K | 0.6775 | 0.7222 | +0.0447 |

The matched in-domain KADID validation scores are 0.8469 for the baseline and
0.8896 for text interaction. Thus, the KADID gain does **not** yet establish
general transfer: it is approximately neutral on the other synthetic sets,
positive on CSIQ and UHD-IQA, and harmful on several authentic-image sets.

The authentic datasets carry the same `authentic` condition for every image,
so they cannot test whether distortion-language semantics transfer. They do,
however, reveal that the KADID-trained conditional head does not automatically
generalize as an overall NR-IQA metric. In contrast, AGIQA-3K is a notable
positive transfer result: it is an unseen generated-image corpus and uses the
same broad `generative` condition. The next valid comparison is training on
the designated multi-dataset training suite and retaining all of these as
held-out final tests.

These result rows, including measured throughput and parameter size, are in
`runs/results.csv` and MLflow runs `external-large-baseline-s0` and
`external-large-interaction-s0`.
