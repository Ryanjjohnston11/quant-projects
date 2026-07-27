# DNA Promoter Classifier

A supervised machine learning pipeline that predicts whether a DNA
sequence contains a **promoter motif** — a short signal region that
marks where a gene begins — from the raw nucleotide sequence alone.

## The problem

Identifying functional regions in DNA is a core task in genomics and
genome engineering. This project builds a classifier that learns to
recognize promoter signals directly from sequence data, rather than
being told the pattern explicitly.

## Approach

- **Data:** A labeled dataset of DNA sequences, half containing an embedded promoter motif (the Pribnow box, `TATAAT`) and half random. Building the data with a known signal makes it possible to verify whether the model genuinely learns it.
- **Features:** Raw sequences are converted into numeric vectors using **k-mer frequency counts** (counts of every length-3 subsequence) — a standard technique for feeding sequence data to a model.
- **Model:** A logistic-regression classifier trained with a proper train/test split so performance is measured only on held-out data.
- **Evaluation:** Accuracy is reported against a majority-class baseline, with a confusion matrix for error analysis.

## Result

The model achieves **~79% held-out accuracy against a ~51% baseline** (+28% lift), confirming it learned a genuine biological signal rather than memorizing noise. The emphasis throughout is on honest evaluation: an accuracy number means nothing without a baseline to compare it to.

## Run it
