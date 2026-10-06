# Reference-conditioned Raman mixture identification

Code for reference-conditioned presence prediction (RGEP), auxiliary target-spectrum extraction, per-epoch synthetic mixture training, and classical Raman-mixture baselines.

This is a research extension of [CSAM-ResUNet](https://github.com/V1S10NAL/CSAM-ResUNet), not the original authors' official repository. RSSNet-inspired modules are an adapted implementation, not an official RSSNet reproduction.

## Scope

- Candidate-reference queries over seven material classes; the model predicts a presence score and an auxiliary extracted spectrum for each query.
- Pure-source duplicate-group isolation and fixed training/validation/test manifests.
- Fixed clean, fixed augmented, and per-epoch refreshed augmented training recipes.
- Validation-only checkpoint selection and threshold calibration.
- Matched-synthesis CSAM comparison, NNLS, background NNLS, NMF, and subset background NNLS with BIC-type selection.
- Binary-mixture evaluation and frozen-model single-/three-component checks.

Allowing threshold decisions to return a variable number of labels is distinct from proving unrestricted component-count generalization. Optical synthesis coefficients and extracted spectral intensities are not calibrated plastic mass fractions.

## Environment

The experiments used Python 3.8, TensorFlow/Keras 2.10, and the scientific packages listed in `requirements.txt`. CUDA/cuDNN and GPU architecture compatibility must be checked for the target machine; CPU diagnostics are supported by relevant entry points using `--cpu`.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data and saved experiment artifacts

Obtain the original data independently from [Science Data Bank](https://www.scidb.cn/s/NbEFze). Data, trained weights, saved predictions, logs, manuscript drafts, and local environments are intentionally excluded from this repository.

The included `dataset/splits/rssnet_pure_19580920.json` contains partition records and hashes, not spectral values. Training pure spectra and training references are drawn from the original study's reconstructed/normalized pure spectra. Raw measured mixtures are an out-of-processing-domain evaluation input.

Typical local layout:

```text
dataset/
  dataset_train_mix_33600_20251116.mat
  dataset_mixture_raw/
  dataset_mixture_denoised/
  splits/rssnet_pure_19580920.json
run/
  <evaluation anchor and locally generated experiment outputs>
```

Existing experiment drivers deliberately require saved validation anchors and source audits. These artifacts are not bundled here. Do not bypass a fingerprint mismatch or silently replace an anchor to claim reproduction of the reported experiment. Reproduction of the archived comparison requires its saved anchor configuration, training reference array, and audit records in addition to the dataset.

## Diagnostic run

After installing dependencies and obtaining the matching source data:

```bash
python -u train_reference_extractor.py \
  --source dataset/dataset_train_mix_33600_20251116.mat \
  --manifest dataset/splits/rssnet_pure_19580920.json \
  --training-data online_simulated --checkpoint-selection val_top2 \
  --version 2 --short-check --epochs 2 --batch-size 32 \
  --save-every-epoch --cpu --output-root run/diagnostic
```

This checks the workflow and does not produce a formal real-data benchmark. A full training run uses the same entry point without `--short-check` and supplies `--evaluation-anchor <ARCHIVED_RUN>`.

## Main entry points

| File | Purpose |
| --- | --- |
| `models/ReferenceGuidedExtractor.py` | Shared encoder, reference modulation, dual-path processing, extraction and presence heads |
| `reference_online_data.py` | Deterministic per-epoch synthetic mixture streams |
| `train_reference_extractor.py` | RGEP training and presence-only control |
| `run_reference_data_study.py` | Fixed-clean/fixed-augmented/refreshed-augmented recipe comparison |
| `run_reference_refresh_ablation.py` | Refresh-only control with archived anchor checks |
| `train_csam_rgep_aligned.py` | CSAM trained on the same synthetic mixtures and validation rule |
| `evaluate_reference_raw_nnls.py` | Raw or reconstructed measured inputs with reference-based baselines |
| `run_nmf_comparison.py` | NMF comparison |
| `run_subset_bic_comparison.py` | Subset NNLS with background and BIC-type count selection |
| `evaluate_cardinality_generalization.py` | Frozen-model single-/binary-/three-component evaluation |

CSAM and RGEP use different query structures. Matching synthetic samples and epochs does not automatically match optimizer updates or FLOPs; report actual budgets from each run.

## Tests

```bash
python -m unittest \
  tests.test_reference_online tests.test_reference_checkpoint \
  tests.test_reference_extractor tests.test_reference_refresh_ablation \
  tests.test_cardinality_generalization
```

TensorFlow-dependent tests require the model environment. Some experiment-level checks require separately obtained archived artifacts.

## Attribution and redistribution

See `NOTICE.md`. No new license is imposed on upstream-derived code. Check upstream authors' terms before redistributing or publicly licensing their material.

## References

- Huang et al., *Cascaded Improved Neural Network for the Reconstruction, Classification, and Unmixing of the Raman Spectra of Mixed Microplastics*, Analytical Chemistry (2026). DOI: https://doi.org/10.1021/acs.analchem.5c04049
- Long et al., *A Brain-Inspired Deep Separation Network for Single Channel Raman Spectra Unmixing*. https://arxiv.org/abs/2604.22324
- Ge et al., *SpEx+: A Complete Time Domain Speaker Extraction Network*. https://doi.org/10.21437/Interspeech.2020-1397

