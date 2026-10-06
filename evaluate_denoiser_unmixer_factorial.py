"""Evaluate a normalized 3x2 denoiser/unmixer factorial on matched samples.

The six cells are:

    A: paper denoised spectra -> paper joint model endpoint heads
    B: paper denoised spectra -> decoupled MSAG-CSAM unmixer
    C: online base + paper normalization -> paper endpoint heads
    D: online base + paper normalization -> decoupled unmixer
    E: adapted denoiser + paper normalization -> paper endpoint heads
    F: adapted denoiser + paper normalization -> decoupled unmixer

Measured plastic mixtures do not have sample-aligned endpoint or abundance
truth.  Consequently, the endpoint results reported here are class-centroid
proxy metrics and mixture-consistency metrics, not true endpoint errors.
"""

import argparse
import csv
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

import tools
from decoupled_data import DEFAULT_CATEGORIES, load_labeled_mixtures, save_json
from evaluate_unmixing_conditions import (
    load_reference_centroids,
    reference_proxy_metrics,
)
from train_recognition_guided_real_domain import (
    load_condition,
    load_denoiser,
    load_frozen_unmixer,
    split_condition_data,
    denoiser_inputs,
)
from evaluate_matched_denoising_sources import normalize_mixture_blocks


DEFAULT_PAPER_MODEL = (
    "./run/CSAM_ResUNet_classification_20251119_002532/checkpoint/"
    "CSAM_ResUNet_classification_model.h5"
)


def parse_conditions(value):
    values = tuple(int(item.strip()) for item in value.split(","))
    if not values or any(item < 1 or item > 9 for item in values):
        raise argparse.ArgumentTypeError("conditions must be in the range 1..9")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dual-run",
        default="./run/DualGuided_RealDomain_20260907_070438",
        help="Completed dual-guided run containing result/config.json",
    )
    parser.add_argument(
        "--base-denoiser",
        default=None,
        help="Defaults to the pretrained denoiser recorded by --dual-run",
    )
    parser.add_argument("--paper-model", default=DEFAULT_PAPER_MODEL)
    parser.add_argument(
        "--new-unmixing-weights",
        default=None,
        help="Defaults to the unmixing checkpoint recorded by --dual-run",
    )
    parser.add_argument(
        "--raw-pattern",
        default="./dataset/dataset_mixture_raw/dataset{index}_norm.mat",
    )
    parser.add_argument(
        "--paper-denoised-pattern",
        default=(
            "./dataset/dataset_mixture_denoised/"
            "dataset{index}_denoised.mat"
        ),
    )
    parser.add_argument(
        "--reference-path", default="./dataset/dataset_150mW_5000ms.mat"
    )
    parser.add_argument(
        "--conditions", type=parse_conditions, default=tuple(range(1, 10))
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def configure_runtime(seed, cpu=False):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_paper_unmixer(path):
    full_model = tf.keras.models.load_model(
        path,
        custom_objects=tools.load_custom_objects(
            "CSAM_ResUNet_classification"
        ),
        compile=False,
    )
    model = tf.keras.Model(
        full_model.input,
        [
            full_model.get_layer("spectrum_A").output,
            full_model.get_layer("spectrum_B").output,
        ],
        name="PaperEndpointHeads",
    )
    model.trainable = False
    return model


def endpoint_metrics(model, spectra, pair_names, centroids, batch_size):
    component_a, component_b = model.predict(
        spectra, batch_size=batch_size, verbose=0
    )
    proxy = reference_proxy_metrics(
        component_a,
        component_b,
        pair_names,
        DEFAULT_CATEGORIES,
        centroids,
    )
    residual = np.asarray(component_a) + np.asarray(component_b) - spectra
    mixture_rmse = float(np.sqrt(np.mean(np.square(residual))))
    mixture_mae = float(np.mean(np.abs(residual)))
    return {
        "reference_pair_accuracy": proxy["reference_pair_accuracy"],
        "reference_cosine_similarity": proxy[
            "reference_cosine_similarity"
        ],
        "reference_sam_degrees": proxy["reference_sam_degrees"],
        "reference_sam_median_degrees": proxy[
            "reference_sam_median_degrees"
        ],
        "collapsed_sample_fraction_at_0_05": proxy[
            "collapsed_sample_fraction_at_0_05"
        ],
        "smaller_component_fraction_mean": proxy[
            "smaller_component_fraction_mean"
        ],
        "mixture_rmse": mixture_rmse,
        "mixture_mae": mixture_mae,
    }


def mean_metrics(condition_results, cell):
    keys = tuple(condition_results[0][cell])
    return {
        key: float(np.mean([row[cell][key] for row in condition_results]))
        for key in keys
    }


def delta(left, right):
    """Return right-left; positive pair accuracy and negative SAM are better."""
    return {
        "reference_pair_accuracy_delta_pp": 100.0
        * (
            right["reference_pair_accuracy"]
            - left["reference_pair_accuracy"]
        ),
        "reference_sam_delta_degrees": (
            right["reference_sam_degrees"]
            - left["reference_sam_degrees"]
        ),
        "mixture_rmse_delta": right["mixture_rmse"] - left["mixture_rmse"],
    }


CELLS = ("A", "B", "C", "D", "E", "F")


def write_csv(path, rows):
    fields = ["condition", "samples"]
    for cell in CELLS:
        fields.extend(
            [
                f"{cell}_pair_accuracy",
                f"{cell}_sam_degrees",
                f"{cell}_mixture_rmse",
                f"{cell}_collapse_fraction",
            ]
        )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {"condition": row["condition"], "samples": row["samples"]}
            for cell in CELLS:
                values = row[cell]
                flat[f"{cell}_pair_accuracy"] = values[
                    "reference_pair_accuracy"
                ]
                flat[f"{cell}_sam_degrees"] = values[
                    "reference_sam_degrees"
                ]
                flat[f"{cell}_mixture_rmse"] = values["mixture_rmse"]
                flat[f"{cell}_collapse_fraction"] = values[
                    "collapsed_sample_fraction_at_0_05"
                ]
            writer.writerow(flat)


def main():
    args = parse_args()
    dual_run = Path(args.dual_run)
    config_path = dual_run / "result" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("split_mode") != "within-condition":
        raise ValueError(
            "The recorded dual run must use within-condition splitting"
        )
    seed = int(config["seed"])
    configure_runtime(seed, cpu=args.cpu)

    dual_denoiser_path = dual_run / "checkpoint" / "best_denoiser.h5"
    if not dual_denoiser_path.is_file():
        raise FileNotFoundError(dual_denoiser_path)
    new_weights = args.new_unmixing_weights or config.get(
        "unmixing_weights"
    )
    if not new_weights or not Path(new_weights).is_file():
        raise FileNotFoundError(new_weights)
    base_denoiser_path = args.base_denoiser or config.get(
        "pretrained_denoiser"
    )
    if not base_denoiser_path or not Path(base_denoiser_path).is_file():
        raise FileNotFoundError(base_denoiser_path)

    output_dir = Path(
        args.output_dir
        or dual_run / "result" / "unmixing_six_cell_normalized"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading paper endpoint model: {args.paper_model}", flush=True)
    paper_unmixer = load_paper_unmixer(args.paper_model)
    print(f"Loading new independent unmixer: {new_weights}", flush=True)
    new_unmixer, _ = load_frozen_unmixer(
        new_weights, args.batch_size, int(config["num_features"])
    )
    print(f"Loading dual-guided denoiser: {dual_denoiser_path}", flush=True)
    dual_denoiser = load_denoiser(str(dual_denoiser_path))
    dual_denoiser.trainable = False
    print(f"Loading online base denoiser: {base_denoiser_path}", flush=True)
    base_denoiser = load_denoiser(str(base_denoiser_path))
    base_denoiser.trainable = False
    centroids = load_reference_centroids(
        args.reference_path, DEFAULT_CATEGORIES, int(config["num_features"])
    )

    split_args = SimpleNamespace(
        split_manifest=config.get("split_manifest"),
        split_manifest_sha256=config.get("split_manifest_sha256"),
        raw_pattern=args.raw_pattern,
        target_pattern=args.paper_denoised_pattern,
        target_source="base-model",
        input_suffix=config.get("input_suffix", "_norm"),
        num_features=int(config["num_features"]),
        limit_per_condition=0,
        seed=seed,
        train_ratio=float(config["train_ratio"]),
        validation_ratio=float(config["validation_ratio"]),
        multi_quality_unpaired=bool(
            config.get("multi_quality_unpaired", False)
        ),
    )

    rows = []
    for condition in args.conditions:
        print(f"\nCondition {condition}: preparing independent test subset", flush=True)
        raw_all = load_condition(condition, split_args)
        test = split_condition_data(raw_all, condition, split_args)["test"]
        indices = test["sample_index"]
        paper_all, paper_labels, paper_pairs = load_labeled_mixtures(
            args.paper_denoised_pattern.format(index=condition),
            categories=DEFAULT_CATEGORIES,
            num_features=int(config["num_features"]),
            suffix=split_args.input_suffix,
        )
        if not (
            np.array_equal(raw_all["labels"], paper_labels)
            and np.array_equal(raw_all["pairs"], paper_pairs)
        ):
            raise ValueError(
                f"Condition {condition}: raw and paper-denoised rows differ"
            )
        paper_input = paper_all[indices]
        pair_names = test["pairs"]
        # Paper files normalize every complete 300-spectrum material-pair
        # block.  Predict the full condition first, apply that same transform,
        # and only then select the independent test rows.
        base_all = base_denoiser.predict(
            raw_all["raw"], batch_size=args.batch_size, verbose=0
        )
        dual_all = dual_denoiser.predict(
            denoiser_inputs(dual_denoiser, raw_all["raw"], raw_all["condition"]),
            batch_size=args.batch_size, verbose=0
        )
        base_normalized_all, base_scales = normalize_mixture_blocks(
            base_all, raw_all["pairs"]
        )
        dual_normalized_all, dual_scales = normalize_mixture_blocks(
            dual_all, raw_all["pairs"]
        )
        base_input = base_normalized_all[indices]
        dual_input = dual_normalized_all[indices]

        print("  A: paper denoising + paper endpoint model", flush=True)
        cell_a = endpoint_metrics(
            paper_unmixer, paper_input, pair_names, centroids, args.batch_size
        )
        print("  B: paper denoising + new independent unmixer", flush=True)
        cell_b = endpoint_metrics(
            new_unmixer, paper_input, pair_names, centroids, args.batch_size
        )
        print("  C: normalized online base + paper endpoint model", flush=True)
        cell_c = endpoint_metrics(
            paper_unmixer, base_input, pair_names, centroids, args.batch_size
        )
        print("  D: normalized online base + new independent unmixer", flush=True)
        cell_d = endpoint_metrics(
            new_unmixer, base_input, pair_names, centroids, args.batch_size
        )
        print("  E: normalized adapted denoising + paper endpoint model", flush=True)
        cell_e = endpoint_metrics(
            paper_unmixer, dual_input, pair_names, centroids, args.batch_size
        )
        print("  F: normalized adapted denoising + new independent unmixer", flush=True)
        cell_f = endpoint_metrics(
            new_unmixer, dual_input, pair_names, centroids, args.batch_size
        )
        row = {
            "condition": int(condition),
            "samples": int(len(indices)),
            "A": cell_a,
            "B": cell_b,
            "C": cell_c,
            "D": cell_d,
            "E": cell_e,
            "F": cell_f,
            "normalization_scales": {
                "online_base": base_scales,
                "adapted": dual_scales,
            },
        }
        rows.append(row)
        print(
            "  pair accuracy A/B/C/D/E/F: "
            f"{100*cell_a['reference_pair_accuracy']:.2f}% / "
            f"{100*cell_b['reference_pair_accuracy']:.2f}% / "
            f"{100*cell_c['reference_pair_accuracy']:.2f}% / "
            f"{100*cell_d['reference_pair_accuracy']:.2f}% / "
            f"{100*cell_e['reference_pair_accuracy']:.2f}% / "
            f"{100*cell_f['reference_pair_accuracy']:.2f}%",
            flush=True,
        )
        print(
            "  proxy SAM A/B/C/D/E/F: "
            f"{cell_a['reference_sam_degrees']:.3f} / "
            f"{cell_b['reference_sam_degrees']:.3f} / "
            f"{cell_c['reference_sam_degrees']:.3f} / "
            f"{cell_d['reference_sam_degrees']:.3f} / "
            f"{cell_e['reference_sam_degrees']:.3f} / "
            f"{cell_f['reference_sam_degrees']:.3f} deg",
            flush=True,
        )

    means = {cell: mean_metrics(rows, cell) for cell in CELLS}
    comparisons = {
        "new_vs_paper_unmixer_on_paper_denoising_B_minus_A": delta(
            means["A"], means["B"]
        ),
        "new_vs_paper_unmixer_on_online_base_D_minus_C": delta(
            means["C"], means["D"]
        ),
        "online_base_vs_paper_file_with_paper_unmixer_C_minus_A": delta(
            means["A"], means["C"]
        ),
        "online_base_vs_paper_file_with_new_unmixer_D_minus_B": delta(
            means["B"], means["D"]
        ),
        "adapted_vs_online_base_with_paper_unmixer_E_minus_C": delta(
            means["C"], means["E"]
        ),
        "adapted_vs_online_base_with_new_unmixer_F_minus_D": delta(
            means["D"], means["F"]
        ),
        "new_vs_paper_unmixer_on_adapted_F_minus_E": delta(
            means["E"], means["F"]
        ),
    }
    payload = {
        "cells": {
            "A": "paper denoised file -> paper joint-model endpoint heads",
            "B": "paper denoised file -> new independent unmixer",
            "C": "normalized online base -> paper joint-model endpoint heads",
            "D": "normalized online base -> new independent unmixer",
            "E": "normalized adapted denoiser -> paper joint-model endpoint heads",
            "F": "normalized adapted denoiser -> new independent unmixer",
        },
        "data": {
            "split_manifest": config.get("split_manifest"),
            "split_manifest_sha256": config.get("split_manifest_sha256"),
            "split": "pair-stratified independent test subset",
            "seed": seed,
            "train_ratio": split_args.train_ratio,
            "validation_ratio": split_args.validation_ratio,
            "conditions": list(args.conditions),
        },
        "models": {
            "paper_model": args.paper_model,
            "new_unmixing_weights": str(new_weights),
            "base_denoiser": str(base_denoiser_path),
            "dual_denoiser": str(dual_denoiser_path),
        },
        "warning": (
            "Measured mixtures have no sample-aligned endpoint or abundance "
            "truth. Endpoint centroid metrics are proxies, not true unmixing "
            "errors. All online denoiser outputs use the paper's complete "
            "condition/material-pair global-maximum normalization before "
            "test rows are selected. This reproduction transform relies on "
            "known pair groups and is not a blind-deployment protocol."
        ),
        "condition_mean": means,
        "comparisons": comparisons,
        "conditions": rows,
    }
    save_json(str(output_dir / "metrics.json"), payload)
    write_csv(output_dir / "condition_metrics.csv", rows)

    print("\nCondition-mean results", flush=True)
    for cell in CELLS:
        values = means[cell]
        print(
            f"{cell}: pair={100*values['reference_pair_accuracy']:.2f}%, "
            f"SAM={values['reference_sam_degrees']:.3f} deg, "
            f"mix_RMSE={values['mixture_rmse']:.6f}",
            flush=True,
        )
    print("\nIsolated effects", flush=True)
    for name, values in comparisons.items():
        print(
            f"{name}: pair {values['reference_pair_accuracy_delta_pp']:+.2f} pp, "
            f"SAM {values['reference_sam_delta_degrees']:+.3f} deg",
            flush=True,
        )
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
