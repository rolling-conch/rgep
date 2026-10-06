"""Evaluate raw and paper-normalized denoising sources on matched test rows.

The comparison is intentionally paired.  It reconstructs the split recorded by
a completed real-domain training run, then evaluates:

1. the paper-provided ``dataset*_denoised.mat`` spectra;
2. online predictions from the run's pretrained base denoiser;
3. those base predictions after the paper's mixture-block max normalization;
4. online predictions from the selected adapted denoiser; and
5. those adapted predictions after the same normalization.

Every source is passed through the same frozen classifier and converted to
exactly two labels with the same Top-2 rule.  Normalization is fitted on each
complete acquisition-condition/material-pair block before the independent test
indices are selected, matching how each ``PCPE_norm``-style MAT variable was
created from its corresponding reconstructed block.
"""

import argparse
import csv
import glob
import json
import os
from types import SimpleNamespace

import numpy as np

from decoupled_data import (
    DEFAULT_CATEGORIES,
    classification_metrics,
    load_labeled_mixtures,
    spectral_metrics,
    top_k_multihot,
)
from train_recognition_guided_real_domain import (
    load_condition,
    load_denoiser,
    load_frozen_teacher,
    probabilities,
    split_condition_data,
    denoiser_inputs,
)


SOURCE_NAMES = (
    "paper_file",
    "online_base",
    "online_base_block_norm",
    "adapted",
    "adapted_block_norm",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare paper-file, online-base, and adapted denoising before "
            "and after paper-style block normalization on the same samples"
        )
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Completed real-domain run; latest compatible run if omitted",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_denoiser.h5",
        choices=(
            "best_denoiser.h5",
            "best_candidate_denoiser.h5",
            "final_denoiser.h5",
            "initial_denoiser.h5",
        ),
    )
    parser.add_argument(
        "--paper-denoised-pattern",
        default=None,
        help=(
            "Paper-provided MAT pattern; defaults to target_pattern recorded "
            "in the run config"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def latest_compatible_run():
    patterns = (
        "./run/MultiQualityUnpaired_DualGuided_*",
        "./run/RecognitionGuided_RealDomain_*",
    )
    candidates = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isfile(os.path.join(path, "result", "config.json")):
                if os.path.isfile(
                    os.path.join(path, "checkpoint", "best_denoiser.h5")
                ):
                    candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No completed compatible run was found")
    return max(candidates, key=os.path.getmtime)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def config_namespace(config, batch_size):
    values = dict(config)
    for key in (
        "all_conditions",
        "train_conditions",
        "validation_conditions",
        "test_conditions",
        "report_conditions",
    ):
        if key in values:
            values[key] = tuple(values[key])
    if batch_size is not None:
        values["batch_size"] = int(batch_size)
    return SimpleNamespace(**values)


def label_name(multihot):
    return "+".join(
        name
        for name, selected in zip(DEFAULT_CATEGORIES, multihot)
        if selected
    )


def source_evaluation(teacher, spectra, labels, batch_size):
    source_probabilities = probabilities(teacher, spectra, batch_size)
    metrics, _, predictions = classification_metrics(
        labels,
        source_probabilities,
        DEFAULT_CATEGORIES,
        k=2,
    )
    correct = np.all(
        np.asarray(predictions, dtype=np.int32)
        == np.asarray(labels, dtype=np.int32),
        axis=1,
    )
    return metrics, source_probabilities, predictions, correct


def normalize_mixture_blocks(spectra, pair_names):
    """Divide each complete material-pair block by its global maximum.

    The paper MAT files store, for example, ``PCPE_norm = PCPE / max(PCPE)``.
    The maximum is taken over the full 300 x 1024 PCPE matrix, not separately
    for every spectrum.  ``pair_names`` therefore must describe the complete
    condition data, before train/validation/test row selection.
    """
    values = np.asarray(spectra, dtype=np.float32)
    pairs = np.asarray(pair_names)
    if len(values) != len(pairs):
        raise ValueError("spectra and pair_names must have the same length")
    normalized = np.array(values, copy=True)
    scales = {}
    for pair_name in np.unique(pairs):
        mask = pairs == pair_name
        scale = float(np.max(normalized[mask]))
        if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
            raise ValueError(
                f"Cannot normalize pair {pair_name!r}: maximum={scale}"
            )
        normalized[mask] /= scale
        scales[str(pair_name)] = scale
    return np.ascontiguousarray(normalized), scales


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, run_dir, conditions, means):
    lines = [
        "# Matched denoising evaluation with paper-style normalization",
        "",
        f"Run: `{run_dir}`",
        "",
        (
            "Every row compares the same independent test samples, frozen "
            "classifier, and exact Top-2 decision rule. Block normalization "
            "divides each complete condition/material-pair output matrix by "
            "its global maximum before test rows are selected."
        ),
        "",
        "| Dataset | N | Paper file | Online base | Base + block norm | "
        "Adapted | Adapted + block norm | Base norm - paper (pp) | "
        "Adapted norm - paper (pp) | Adapted norm - base norm (pp) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        lines.append(
            f"| {condition['dataset']} | {condition['samples']} | "
            f"{condition['paper_file']['subset_accuracy'] * 100:.2f}% | "
            f"{condition['online_base']['subset_accuracy'] * 100:.2f}% | "
            f"{condition['online_base_block_norm']['subset_accuracy'] * 100:.2f}% | "
            f"{condition['adapted']['subset_accuracy'] * 100:.2f}% | "
            f"{condition['adapted_block_norm']['subset_accuracy'] * 100:.2f}% | "
            f"{condition['online_base_block_norm_minus_paper_pp']:+.2f} | "
            f"{condition['adapted_block_norm_minus_paper_pp']:+.2f} | "
            f"{condition['adapted_block_norm_minus_online_base_block_norm_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Condition-balanced mean",
            "",
            f"- Paper file: {means['paper_file'] * 100:.3f}%",
            f"- Online base: {means['online_base'] * 100:.3f}%",
            (
                "- Online base + block normalization: "
                f"{means['online_base_block_norm'] * 100:.3f}%"
            ),
            f"- Adapted: {means['adapted'] * 100:.3f}%",
            (
                "- Adapted + block normalization: "
                f"{means['adapted_block_norm'] * 100:.3f}%"
            ),
            (
                "- Online base block norm - paper: "
                f"{means['online_base_block_norm_minus_paper_pp']:+.3f} pp"
            ),
            (
                "- Adapted block norm - paper: "
                f"{means['adapted_block_norm_minus_paper_pp']:+.3f} pp"
            ),
            (
                "- Adapted block norm - online base block norm: "
                f"{means['adapted_block_norm_minus_online_base_block_norm_pp']:+.3f} pp"
            ),
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    args = parse_args()
    run_dir = args.run_dir or latest_compatible_run()
    config_path = os.path.join(run_dir, "result", "config.json")
    config = load_json(config_path)
    config_args = config_namespace(config, args.batch_size)
    if config_args.split_mode != "within-condition":
        raise ValueError(
            "This evaluation requires a within-condition training run"
        )
    if config_args.limit_per_condition:
        raise ValueError(
            "Debug runs with limit_per_condition are not comparable to the "
            "paper-provided full condition files"
        )

    paper_pattern = args.paper_denoised_pattern or config_args.target_pattern
    adapted_path = os.path.join(
        run_dir, "checkpoint", args.checkpoint_name
    )
    for path in (
        config_args.pretrained_denoiser,
        config_args.teacher_model,
        adapted_path,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    output_dir = args.output_dir or os.path.join(
        run_dir, "result", "matched_normalized_evaluation"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Run: {run_dir}")
    print(f"Loading online base: {config_args.pretrained_denoiser}")
    base_denoiser = load_denoiser(config_args.pretrained_denoiser)
    print(f"Loading adapted denoiser: {adapted_path}")
    adapted_denoiser = load_denoiser(adapted_path)
    print(f"Loading frozen classifier: {config_args.teacher_model}")
    teacher = load_frozen_teacher(config_args.teacher_model)

    condition_results = []
    sample_rows = []
    for condition in config_args.report_conditions:
        print(f"\nDataset {condition}: reconstructing exact test split")
        complete = load_condition(condition, config_args)
        test = split_condition_data(
            complete, condition, config_args
        )["test"]
        selected = np.asarray(test["sample_index"], dtype=np.int64)

        paper_path = paper_pattern.format(index=condition)
        paper_full, paper_labels, paper_pairs = load_labeled_mixtures(
            paper_path,
            categories=DEFAULT_CATEGORIES,
            num_features=config_args.num_features,
            suffix=config_args.input_suffix,
        )
        paper_spectra = np.ascontiguousarray(paper_full[selected])
        if not (
            np.array_equal(test["labels"], paper_labels[selected])
            and np.array_equal(test["pairs"], paper_pairs[selected])
        ):
            raise ValueError(
                f"Dataset {condition}: paper and raw sample order differs"
            )

        # Predict the complete condition so that normalization uses the same
        # full 300-spectrum material-pair blocks as the paper MAT files.
        online_base_full = base_denoiser.predict(
            complete["raw"], batch_size=config_args.batch_size, verbose=1
        )
        adapted_full = adapted_denoiser.predict(
            denoiser_inputs(adapted_denoiser, complete["raw"], complete["condition"]),
            batch_size=config_args.batch_size, verbose=1
        )
        online_base_block_norm_full, base_scales = normalize_mixture_blocks(
            online_base_full, complete["pairs"]
        )
        adapted_block_norm_full, adapted_scales = normalize_mixture_blocks(
            adapted_full, complete["pairs"]
        )
        online_base = np.ascontiguousarray(online_base_full[selected])
        online_base_block_norm = np.ascontiguousarray(
            online_base_block_norm_full[selected]
        )
        adapted = np.ascontiguousarray(adapted_full[selected])
        adapted_block_norm = np.ascontiguousarray(
            adapted_block_norm_full[selected]
        )
        spectra_by_source = {
            "paper_file": paper_spectra,
            "online_base": online_base,
            "online_base_block_norm": online_base_block_norm,
            "adapted": adapted,
            "adapted_block_norm": adapted_block_norm,
        }
        evaluated = {}
        for source in SOURCE_NAMES:
            evaluated[source] = source_evaluation(
                teacher,
                spectra_by_source[source],
                test["labels"],
                config_args.batch_size,
            )

        condition_result = {
            "dataset": int(condition),
            "samples": int(len(selected)),
            "paper_path": paper_path,
            "sample_indices": selected.tolist(),
            "paper_file": evaluated["paper_file"][0],
            "online_base": evaluated["online_base"][0],
            "online_base_block_norm": evaluated[
                "online_base_block_norm"
            ][0],
            "adapted": evaluated["adapted"][0],
            "adapted_block_norm": evaluated["adapted_block_norm"][0],
            "normalization": {
                "method": "condition/material-pair global maximum",
                "online_base_pair_scales": base_scales,
                "adapted_pair_scales": adapted_scales,
            },
            "online_base_vs_paper_spectral": spectral_metrics(
                paper_spectra, online_base
            ),
            "online_base_block_norm_vs_paper_spectral": spectral_metrics(
                paper_spectra, online_base_block_norm
            ),
            "adapted_vs_paper_spectral": spectral_metrics(
                paper_spectra, adapted
            ),
            "adapted_block_norm_vs_paper_spectral": spectral_metrics(
                paper_spectra, adapted_block_norm
            ),
        }
        condition_result["online_base_block_norm_minus_paper_pp"] = 100.0 * (
            condition_result["online_base_block_norm"]["subset_accuracy"]
            - condition_result["paper_file"]["subset_accuracy"]
        )
        condition_result["adapted_block_norm_minus_paper_pp"] = 100.0 * (
            condition_result["adapted_block_norm"]["subset_accuracy"]
            - condition_result["paper_file"]["subset_accuracy"]
        )
        condition_result[
            "adapted_block_norm_minus_online_base_block_norm_pp"
        ] = 100.0 * (
            condition_result["adapted_block_norm"]["subset_accuracy"]
            - condition_result["online_base_block_norm"]["subset_accuracy"]
        )
        condition_results.append(condition_result)

        print(
            "Strict pair accuracy: "
            f"paper={condition_result['paper_file']['subset_accuracy'] * 100:.2f}%, "
            f"base={condition_result['online_base']['subset_accuracy'] * 100:.2f}%, "
            f"base+norm={condition_result['online_base_block_norm']['subset_accuracy'] * 100:.2f}%, "
            f"adapted={condition_result['adapted']['subset_accuracy'] * 100:.2f}%, "
            f"adapted+norm={condition_result['adapted_block_norm']['subset_accuracy'] * 100:.2f}%"
        )

        for row_index, sample_index in enumerate(selected):
            row = {
                "dataset": int(condition),
                "sample_index": int(sample_index),
                "particle_size_block": int(
                    test["particle_size_block"][row_index]
                ),
                "true_pair": str(test["pairs"][row_index]),
            }
            for source in SOURCE_NAMES:
                predictions = evaluated[source][2]
                correct = evaluated[source][3]
                row[f"{source}_prediction"] = label_name(
                    predictions[row_index]
                )
                row[f"{source}_correct"] = int(correct[row_index])
            sample_rows.append(row)

        del (
            paper_full,
            paper_spectra,
            online_base_full,
            online_base_block_norm_full,
            adapted_full,
            adapted_block_norm_full,
            online_base,
            online_base_block_norm,
            adapted,
            adapted_block_norm,
        )

    means = {
        source: float(
            np.mean(
                [item[source]["subset_accuracy"] for item in condition_results]
            )
        )
        for source in SOURCE_NAMES
    }
    means["online_base_block_norm_minus_paper_pp"] = 100.0 * (
        means["online_base_block_norm"] - means["paper_file"]
    )
    means["adapted_block_norm_minus_paper_pp"] = 100.0 * (
        means["adapted_block_norm"] - means["paper_file"]
    )
    means["adapted_block_norm_minus_online_base_block_norm_pp"] = 100.0 * (
        means["adapted_block_norm"] - means["online_base_block_norm"]
    )
    result = {
        "run_dir": run_dir,
        "checkpoint": adapted_path,
        "base_denoiser": config_args.pretrained_denoiser,
        "teacher_model": config_args.teacher_model,
        "paper_denoised_pattern": paper_pattern,
        "split": {
            "manifest": config.get("split_manifest"),
            "manifest_sha256": config.get("split_manifest_sha256"),
            "mode": config_args.split_mode,
            "train_ratio": config_args.train_ratio,
            "validation_ratio": config_args.validation_ratio,
            "seed": config_args.seed,
            "stratification": (
                "material_pair_and_particle_size_within_each_condition"
                if config_args.multi_quality_unpaired
                else "material_pair_within_each_condition"
            ),
        },
        "decision_rule": "exactly two labels selected by Top-2 probability",
        "normalization": (
            "divide every complete acquisition-condition/material-pair "
            "prediction matrix by its global maximum before selecting test rows"
        ),
        "condition_balanced_mean": means,
        "conditions": condition_results,
    }
    write_json(os.path.join(output_dir, "metrics.json"), result)
    write_rows(os.path.join(output_dir, "sample_predictions.csv"), sample_rows)
    write_report(
        os.path.join(output_dir, "report.md"),
        run_dir,
        condition_results,
        means,
    )
    print("\nCondition-balanced mean strict pair accuracy")
    print(
        f"paper={means['paper_file'] * 100:.3f}%, "
        f"base={means['online_base'] * 100:.3f}%, "
        f"base+norm={means['online_base_block_norm'] * 100:.3f}%, "
        f"adapted={means['adapted'] * 100:.3f}%, "
        f"adapted+norm={means['adapted_block_norm'] * 100:.3f}%"
    )
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
