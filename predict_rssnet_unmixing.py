"""Unmix NPZ x or evaluate a manifest-selected real plastic MATLAB condition.

No ground-truth class label is passed into the separator. Real SAM/pair scores
are reference-library PROXIES, not sample-aligned endpoint or mass-fraction truth.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from decoupled_data import DEFAULT_CATEGORIES, load_labeled_mixtures
from fixed_splits import file_sha256, manifest_indices, read_manifest
from models.RSSNetInspired import RSSNetInspired
from rssnet_data import normalize_input, write_json


def main(argv=None, model_cache=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True, help="New output directory (refuses overwrite)")
    p.add_argument("--input", help="NPZ with x; no labels needed; units restored in output")
    p.add_argument("--condition", type=int, choices=range(1, 10), default=8)
    p.add_argument("--partition", choices=("train", "validation", "test"), default="validation")
    p.add_argument("--split-manifest", default="dataset/splits/plastic_condition_holdout_1-7_8_9_v1.csv")
    p.add_argument("--raw-pattern", default="dataset/dataset_mixture_raw/dataset{index}_norm.mat")
    p.add_argument("--spectra-pattern", default="dataset/dataset_mixture_denoised/dataset{index}_denoised.mat")
    p.add_argument("--denoiser", help="Optional frozen multi-quality H5; overrides spectra-pattern, takes raw input")
    p.add_argument("--paper-model", help="Optional paper joint model: evaluate endpoints on IDENTICAL input rows")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="Debug only: first N selected rows; NOT a full benchmark")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.limit < 0:
        p.error("batch-size positive; limit nonnegative")
    if args.input and args.denoiser:
        p.error("NPZ input must already be preprocessed; --denoiser is for MATLAB conditions")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = json.loads((Path(args.run_dir) / "config.json").read_text(encoding="utf-8"))
    weights = Path(args.run_dir) / "checkpoint/best.weights.h5"
    weights_hash = file_sha256(weights)
    key = (str(Path(args.run_dir).resolve()), weights_hash, file_sha256(Path(args.run_dir) / "config.json"))
    if model_cache is None or not model_cache:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    if model_cache is not None and key in model_cache:
        model = model_cache[key]
    else:
        from matched_separators import load_run_model
        model = load_run_model(args.run_dir, config)
        model(tf.zeros((1, config["width"], 1)))
        model.load_weights(str(weights))
        if model_cache is not None:
            model_cache[key] = model
    provenance = dict(args=vars(args), model_sha256=weights_hash,
                      model_kind=config.get("model_kind", "rssnet"),
                      evaluation="debug subset" if args.limit else "full selected partition")
    pairs = None
    if args.input:
        with np.load(args.input, allow_pickle=False) as src:
            spectra = np.asarray(src["x"], dtype=np.float32)
        ids = np.arange(len(spectra))
        provenance["input_sha256"] = file_sha256(args.input)
    else:
        from train_recognition_guided_real_domain import load_condition, load_denoiser, denoiser_inputs
        _, meta = read_manifest(args.split_manifest)
        split_args = SimpleNamespace(raw_pattern=args.raw_pattern, num_features=config["width"],
            input_suffix="_norm", target_source="online", limit_per_condition=0,
            split_manifest=args.split_manifest, split_mode=meta.get("protocol", "within-condition"),
            train_conditions=meta.get("train_conditions", []),
            validation_conditions=meta.get("validation_conditions", []),
            test_conditions=meta.get("test_conditions", []))
        data = load_condition(args.condition, split_args)
        selected = manifest_indices(data, args.condition, split_args)[args.partition]
        if not len(selected):
            raise ValueError("Requested partition empty for this condition; e.g. Dataset 8=validation, 9=test")
        if args.limit:
            selected = selected[:args.limit]
        ids, pairs = data["sample_index"][selected], data["pairs"][selected]
        provenance["split_manifest_sha256"] = file_sha256(args.split_manifest)
        provenance["raw_sha256"] = file_sha256(args.raw_pattern.format(index=args.condition))
        if args.denoiser:
            denoiser = load_denoiser(args.denoiser)
            raw = data["raw"][selected]
            spectra = denoiser.predict(denoiser_inputs(denoiser, raw, data["condition"][selected]),
                                       batch_size=args.batch_size, verbose=0)
            # Prediction-only row normalization; no class/particle block information.
            spectra = np.maximum(spectra, 0)
            spectra /= np.maximum(np.max(spectra, axis=1, keepdims=True), 1e-8)
            provenance["denoiser_sha256"] = file_sha256(args.denoiser)
            provenance["upstream_warning"] = "Frozen does not prove no prior real-test exposure; audit upstream training separately"
            provenance["preprocessing"] = "online denoiser, clip nonnegative, per-row max; NOT paper block normalization"
        else:
            path = args.spectra_pattern.format(index=args.condition)
            all_x, labels, all_pairs = load_labeled_mixtures(path, num_features=config["width"])
            if not np.array_equal(labels, data["labels"]) or not np.array_equal(all_pairs, data["pairs"]) or all_x.shape != data["raw"].shape:
                raise ValueError("Processed/raw pair layout mismatch")
            spectra = all_x[selected]
            provenance["input_sha256"] = file_sha256(path)
            provenance["preprocessing"] = "supplied MAT; assumes within-pair row order preserved from raw"
    if args.input and args.limit:
        spectra, ids = spectra[:args.limit], ids[:args.limit]
    x, scale = normalize_input(spectra)
    if x.shape[1] != config["width"]:
        raise ValueError("Input spectral width differs from training")
    pred = model.predict(x, batch_size=args.batch_size, verbose=0) * scale[..., None]
    if not np.isfinite(pred).all():
        raise FloatingPointError("Non-finite separated spectra")
    with np.load(Path(args.run_dir) / "reference.npz", allow_pickle=False) as src:
        refs, names = src["reference"], src["names"]
    norms = np.linalg.norm(pred, axis=1)
    unit = pred / np.maximum(norms[:, None, :], 1e-8)
    similarity = np.einsum("nlc,kl->nck", unit, refs)
    nearest = np.argmax(similarity, axis=-1)
    # Explicitly flag undefined endpoints rather than pretending to identify them.
    valid = norms > 1e-8
    nearest = np.where(valid, nearest, -1)
    arrays = dict(input=np.asarray(spectra), components=pred, reconstruction=pred.sum(axis=-1),
                  sample_index=ids, endpoint_class_indices=nearest, names=names,
                  reference_cosine=similarity, valid_endpoints=valid)
    metrics = dict(samples=len(pred), undefined_components=int((~valid).sum()),
                   interpretation="weighted optical component spectra; NOT measured mass fractions",
                   input_reconstruction_rmse=float(np.sqrt(np.mean((pred.sum(axis=-1) - x[..., 0] * scale) ** 2))))
    if pairs is not None:
        from evaluate_unmixing_conditions import reference_proxy_metrics, build_true_indices
        truth = np.sort(build_true_indices(pairs, DEFAULT_CATEGORIES), axis=-1)
        correct = np.all(np.sort(nearest, axis=-1) == truth, axis=-1) & np.all(valid, axis=-1)
        metrics["independent_endpoint_pair_accuracy"] = float(correct.mean())
        proxy = reference_proxy_metrics(pred[..., 0], pred[..., 1], pairs, DEFAULT_CATEGORIES, refs)
        metrics["rssnet_reference_proxy"] = proxy
        metrics["separator_reference_proxy"] = proxy
        metrics["model_kind"] = config.get("model_kind", "rssnet")
        arrays.update(pairs=pairs, pair_correct=correct)
        if args.paper_model:
            from evaluate_denoiser_unmixer_factorial import load_paper_unmixer
            base = load_paper_unmixer(args.paper_model)
            paper_input = np.asarray(spectra)
            if paper_input.ndim == 2:
                paper_input = paper_input[..., None]
            a, b = base.predict(paper_input, batch_size=args.batch_size, verbose=0)
            metrics["paper_reference_proxy"] = reference_proxy_metrics(a, b, pairs, DEFAULT_CATEGORIES, refs)
            arrays.update(paper_a=a, paper_b=b)
            provenance["paper_model_sha256"] = file_sha256(args.paper_model)
        metrics["warning"] = "Reference centroid proxies, NOT sample-aligned endpoint truth. Same-input checkpoint comparison, NOT equal-training-data causal ablation."
    output.mkdir(parents=True)
    np.savez_compressed(output / "predictions.npz", **arrays)
    write_json(output / "metrics.json", metrics)
    write_json(output / "provenance.json", provenance)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
