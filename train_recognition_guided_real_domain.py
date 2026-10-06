"""Real-domain adaptation for recognition-guided Raman denoising.

The default path maps each measured raw spectrum to the base denoiser output
from that same spectrum.  ``--multi-quality-unpaired`` additionally uses
Dataset 1 as a group-level high-quality reference without assuming that rows
are paired across acquisition conditions.  By default, every condition and
material group is split into independent 70%/20%/10% subsets.
"""

import argparse
import gc
import glob
import json
import os
import shutil
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

import tools
from denoiser_ablation import apply_ablation_variant, active_weight_groups
from decoupled_data import (
    DEFAULT_CATEGORIES,
    classification_metrics,
    load_labeled_mixtures,
    save_history,
    spectral_metrics,
)
from models.RecognitionGuidedDenoiser import (
    DualGuidedRealDomainAdaptationModel,
    RealDomainAdaptationModel,
    _MultiQualityUnpairedMixin,
    build_recognition_guided_denoiser,
)
from evaluate_unmixing_conditions import (
    MODEL_NAME as UNMIXING_MODEL_NAME,
    load_reference_centroids,
    reference_proxy_metrics,
)


DEFAULT_TEACHER = (
    "./run/CSAM_ResUNet_classification_20251119_002532/checkpoint/"
    "CSAM_ResUNet_classification_model.h5"
)
DEFAULT_BASE_DENOISER = (
    "./run/CSAM_ResUNet_reconstruction_20250610_102330/checkpoint/"
    "CSAM_ResUNet_reconstruction_model.h5"
)
DEFAULT_UNMIXING_GLOB = (
    "./run/MSAG_CSAM_ResUNet_unmixing_only_*/checkpoint/best.weights.h5"
)
BASE_DENOISER_KINDS = (
    "CSAM_ResUNet_reconstruction",
    "MSAG_CSAM_ResUNet_reconstruction",
)
CONDITION_DETAILS = {
    1: (150, 5000, 750),
    2: (150, 3000, 450),
    3: (150, 1000, 150),
    4: (100, 5000, 500),
    5: (100, 3000, 300),
    6: (100, 1000, 100),
    7: (50, 5000, 250),
    8: (50, 3000, 150),
    9: (50, 1000, 50),
}


class MultiQualityUnpairedDualGuidedAdaptationModel(
    _MultiQualityUnpairedMixin,
    DualGuidedRealDomainAdaptationModel,
):
    """Dual-guided denoising with non-paired multi-quality supervision."""

    pass


def parse_conditions(value):
    try:
        conditions = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "conditions must be comma-separated integers"
        ) from error
    if not conditions or len(set(conditions)) != len(conditions):
        raise argparse.ArgumentTypeError(
            "conditions must be a non-empty list without duplicates"
        )
    invalid = [item for item in conditions if item not in CONDITION_DETAILS]
    if invalid:
        raise argparse.ArgumentTypeError(
            "condition indices must be between 1 and 9"
        )
    return conditions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adapt recognition-guided denoising to measured spectra"
    )
    parser.add_argument(
        "--raw-pattern",
        default="./dataset/dataset_mixture_raw/dataset{index}_norm.mat",
    )
    parser.add_argument(
        "--target-pattern",
        default=(
            "./dataset/dataset_mixture_denoised/"
            "dataset{index}_denoised.mat"
        ),
    )
    parser.add_argument(
        "--target-source",
        choices=("base-model", "file"),
        default="base-model",
        help=(
            "base-model generates denoising targets online with exactly the "
            "checkpoint used to initialize the residual model; file uses "
            "--target-pattern and must pass the baseline consistency check"
        ),
    )
    parser.add_argument(
        "--target-generation-batch-size", type=int, default=32
    )
    parser.add_argument("--target-alignment-samples", type=int, default=512)
    parser.add_argument("--target-mae-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--target-max-abs-tolerance", type=float, default=1e-5
    )
    parser.add_argument("--input-suffix", default="_norm")
    parser.add_argument("--split-manifest", default=None,
                        help="Immutable raw duplicate-group CSV shared by all models")
    parser.add_argument("--audit-split-only", action="store_true",
                        help="Verify full raw data against the fixed manifest, then exit before GPU inference")
    parser.add_argument(
        "--split-mode",
        choices=("within-condition", "condition-holdout"),
        default="within-condition",
        help=(
            "within-condition stratifies every acquisition condition into "
            "train/validation/test; condition-holdout retains whole-condition "
            "splits"
        ),
    )
    parser.add_argument(
        "--all-conditions",
        type=parse_conditions,
        default=tuple(range(1, 10)),
        help="conditions used by within-condition splitting",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument(
        "--train-conditions", type=parse_conditions, default=(1, 2, 3, 4, 5, 6, 7)
    )
    parser.add_argument(
        "--validation-conditions", type=parse_conditions, default=(8,)
    )
    parser.add_argument(
        "--test-conditions", type=parse_conditions, default=(9,)
    )
    parser.add_argument(
        "--report-conditions",
        type=parse_conditions,
        default=tuple(range(1, 10)),
    )
    parser.add_argument(
        "--pretrained-denoiser",
        default=DEFAULT_BASE_DENOISER,
        help="Frozen base denoiser; defaults to the original paper checkpoint",
    )
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument(
        "--unmixing-weights",
        default=None,
        help=(
            "Enable dual classification+unmixing guidance with this frozen "
            "decoupled-unmixer checkpoint. If omitted, legacy recognition-only "
            "adaptation is used."
        ),
    )
    parser.add_argument(
        "--reference-path", default="./dataset/dataset_150mW_5000ms.mat"
    )
    parser.add_argument("--output-root", default="./run")
    parser.add_argument("--num-features", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19580920)
    parser.add_argument(
        "--ablation-variant",
        choices=("C", "D", "E", "G"),
        default=None,
        help=(
            "Apply the fixed single-seed ablation contract. A is the frozen "
            "base output and is derived during matched evaluation."
        ),
    )
    parser.add_argument("--refinement-filters", type=int, default=32)
    parser.add_argument("--maximum-correction", type=float, default=0.01)
    parser.add_argument("--quality-conditioning", action="store_true",
                        help="Condition the residual gate on power, exposure and spectral noise")
    parser.add_argument(
        "--fine-tune-backbone",
        action="store_true",
        help="Also update the base denoiser; disabled by default for safety",
    )
    parser.add_argument(
        "--multi-quality-unpaired",
        action="store_true",
        help=(
            "Use the high-quality condition as a non-paired reference bank, "
            "plus condition-scaled synthetic degradation and group moments"
        ),
    )
    parser.add_argument("--high-quality-condition", type=int, default=1)
    parser.add_argument("--synthetic-pair-weight", type=float, default=0.02)
    parser.add_argument(
        "--synthetic-classification-weight", type=float, default=0.001
    )
    parser.add_argument(
        "--high-quality-identity-weight", type=float, default=1.0
    )
    parser.add_argument("--group-spectral-weight", type=float, default=0.05)
    parser.add_argument("--group-feature-weight", type=float, default=0.02)
    parser.add_argument("--synthetic-gaussian-min", type=float, default=0.001)
    parser.add_argument("--synthetic-gaussian-max", type=float, default=0.030)
    parser.add_argument("--synthetic-shot-noise-max", type=float, default=0.020)
    parser.add_argument("--synthetic-baseline-max", type=float, default=0.120)
    parser.add_argument(
        "--synthetic-spike-probability", type=float, default=0.0005
    )
    parser.add_argument("--synthetic-spike-amplitude", type=float, default=0.150)
    parser.add_argument("--synthetic-gain-variation", type=float, default=0.12)

    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=0.0005)
    parser.add_argument("--distillation-weight", type=float, default=0.002)
    parser.add_argument("--feature-weight", type=float, default=0.002)
    parser.add_argument("--sam-weight", type=float, default=0.05)
    parser.add_argument("--correction-weight", type=float, default=0.01)
    parser.add_argument("--safe-margin-weight", type=float, default=0.01)
    parser.add_argument("--confidence-margin", type=float, default=0.20)
    parser.add_argument("--hard-example-weight", type=float, default=3.0)
    parser.add_argument("--easy-identity-weight", type=float, default=4.0)
    parser.add_argument(
        "--unmixing-distillation-weight", type=float, default=0.002
    )
    parser.add_argument(
        "--endpoint-classification-weight", type=float, default=0.001
    )
    parser.add_argument(
        "--mixture-consistency-guidance-weight", type=float, default=0.05
    )
    parser.add_argument("--endpoint-temperature", type=float, default=0.10)
    parser.add_argument(
        "--no-pcgrad",
        action="store_false",
        dest="use_pcgrad",
        help="Disable gradient-conflict projection between the two guidance tasks",
    )
    parser.set_defaults(use_pcgrad=True)
    parser.add_argument(
        "--unmixing-accuracy-tolerance", type=float, default=1e-6
    )
    parser.add_argument(
        "--endpoint-sam-tolerance", type=float, default=1e-4
    )
    parser.add_argument("--peak-weight-scale", type=float, default=20.0)
    parser.add_argument("--sparse-loss-weight", type=float, default=0.002)
    parser.add_argument("--derivative-loss-weight", type=float, default=0.10)
    parser.add_argument(
        "--self-similarity-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of reference-defined peak structural self-similarity "
            "loss on measured training targets; zero preserves legacy runs"
        ),
    )
    parser.add_argument(
        "--synthetic-self-similarity-weight",
        type=float,
        default=0.0,
        help=(
            "Additional self-similarity weight for condition-scaled "
            "synthetic inputs against their high-quality references"
        ),
    )
    parser.add_argument("--self-similarity-patch-size", type=int, default=7)
    parser.add_argument("--self-similarity-search-radius", type=int, default=15)
    parser.add_argument("--self-similarity-search-step", type=int, default=2)
    parser.add_argument("--self-similarity-peak-fraction", type=float, default=0.25)
    parser.add_argument("--self-similarity-temperature", type=float, default=0.10)
    parser.add_argument("--self-similarity-l1-weight", type=float, default=0.50)
    parser.add_argument(
        "--limit-per-condition",
        type=int,
        default=0,
        help="Debug only: randomly retain at most this many samples per condition",
    )
    return parser.parse_args()


def validate_args(args):
    if args.high_quality_condition not in CONDITION_DETAILS:
        raise ValueError("high-quality-condition must be between 1 and 9")
    if args.multi_quality_unpaired and args.unmixing_weights is None:
        raise ValueError(
            "--multi-quality-unpaired currently requires --unmixing-weights "
            "because it extends the dual-guided training model"
        )
    active_training_conditions = (
        set(args.all_conditions)
        if args.split_mode == "within-condition"
        else set(args.train_conditions)
    )
    if (
        args.multi_quality_unpaired
        and args.high_quality_condition not in active_training_conditions
    ):
        raise ValueError(
            "the high-quality condition must be included in training"
        )
    if args.split_mode == "condition-holdout":
        condition_groups = (
            set(args.train_conditions),
            set(args.validation_conditions),
            set(args.test_conditions),
        )
        if condition_groups[0] & condition_groups[1]:
            raise ValueError("train and validation conditions must be disjoint")
        if condition_groups[0] & condition_groups[2]:
            raise ValueError("train and test conditions must be disjoint")
        if condition_groups[1] & condition_groups[2]:
            raise ValueError("validation and test conditions must be disjoint")
        if not set(args.report_conditions).issubset(condition_groups[2]):
            raise ValueError(
                "For condition-holdout evaluation, --report-conditions must "
                "contain only held-out test conditions"
            )
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train-ratio must be between 0 and 1")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")
    if args.train_ratio + args.validation_ratio >= 1.0:
        raise ValueError("train-ratio + validation-ratio must be less than 1")
    if (
        args.split_mode == "within-condition"
        and not set(args.report_conditions).issubset(args.all_conditions)
    ):
        raise ValueError(
            "report-conditions must be contained in all-conditions when "
            "using within-condition splitting"
        )
    if min(
        args.batch_size,
        args.epochs,
        args.num_features,
        args.early_stopping_patience,
        args.target_generation_batch_size,
        args.target_alignment_samples,
    ) <= 0:
        raise ValueError("batch size, epochs, features, and patience must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.refinement_filters < 6 or args.maximum_correction <= 0:
        raise ValueError("invalid refinement-head configuration")
    weights = (
        args.reconstruction_weight,
        args.classification_weight,
        args.distillation_weight,
        args.feature_weight,
        args.sam_weight,
        args.correction_weight,
        args.safe_margin_weight,
        args.hard_example_weight,
        args.easy_identity_weight,
        args.unmixing_distillation_weight,
        args.endpoint_classification_weight,
        args.mixture_consistency_guidance_weight,
        args.unmixing_accuracy_tolerance,
        args.endpoint_sam_tolerance,
        args.peak_weight_scale,
        args.sparse_loss_weight,
        args.derivative_loss_weight,
        args.self_similarity_weight,
        args.synthetic_self_similarity_weight,
        args.self_similarity_l1_weight,
        args.synthetic_pair_weight,
        args.synthetic_classification_weight,
        args.high_quality_identity_weight,
        args.group_spectral_weight,
        args.group_feature_weight,
        args.synthetic_gaussian_min,
        args.synthetic_gaussian_max,
        args.synthetic_shot_noise_max,
        args.synthetic_baseline_max,
        args.synthetic_spike_probability,
        args.synthetic_spike_amplitude,
        args.synthetic_gain_variation,
    )
    if min(weights) < 0:
        raise ValueError("loss weights must be nonnegative")
    if not -1.0 <= args.confidence_margin <= 1.0:
        raise ValueError("confidence-margin must be between -1 and 1")
    if args.endpoint_temperature <= 0:
        raise ValueError("endpoint-temperature must be positive")
    if args.limit_per_condition < 0:
        raise ValueError("limit-per-condition must be nonnegative")
    if min(args.target_mae_tolerance, args.target_max_abs_tolerance) < 0:
        raise ValueError("target consistency tolerances must be nonnegative")
    if args.synthetic_gaussian_min > args.synthetic_gaussian_max:
        raise ValueError("synthetic Gaussian-noise range is invalid")
    if (
        args.self_similarity_patch_size < 3
        or args.self_similarity_patch_size % 2 == 0
    ):
        raise ValueError("self-similarity-patch-size must be odd and at least 3")
    if min(
        args.self_similarity_search_radius,
        args.self_similarity_search_step,
    ) <= 0:
        raise ValueError("self-similarity search sizes must be positive")
    if not 0.0 < args.self_similarity_peak_fraction <= 1.0:
        raise ValueError("self-similarity-peak-fraction must be in (0, 1]")
    if args.self_similarity_temperature <= 0:
        raise ValueError("self-similarity-temperature must be positive")
    if (
        args.synthetic_self_similarity_weight > 0
        and not args.multi_quality_unpaired
    ):
        raise ValueError(
            "synthetic-self-similarity-weight requires --multi-quality-unpaired"
        )


def latest_pretrained_denoiser():
    pattern = os.path.join(
        ".",
        "run",
        "MSAG_CSAM_ResUNet_reconstruction_*",
        "checkpoint",
        "MSAG_CSAM_ResUNet_reconstruction_model.h5",
    )
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            "No MSAG reconstruction model found; pass --pretrained-denoiser"
        )
    return max(candidates, key=os.path.getmtime)


def reconstruction_custom_objects():
    """Merge the custom objects of every supported reconstruction backbone.

    The base denoiser may be either the original paper checkpoint
    (``CSAM_ResUNet_reconstruction``) or the newer multiscale variant
    (``MSAG_CSAM_ResUNet_reconstruction``).  Merging both dictionaries lets a
    single loader deserialize either one instead of guessing from the path.
    """
    merged = {}
    for kind in BASE_DENOISER_KINDS:
        merged.update(tools.load_custom_objects(kind) or {})
    return merged


def load_denoiser(path):
    return tf.keras.models.load_model(
        path, custom_objects=reconstruction_custom_objects(), compile=False
    )


def acquisition_quality(conditions):
    """Fixed physical-unit scaling, independent of labels and test statistics."""
    return np.asarray([
        [np.log1p(CONDITION_DETAILS[int(c)][0]) / np.log1p(150.0),
         np.log1p(CONDITION_DETAILS[int(c)][1]) / np.log1p(5000.0), 1.0]
        for c in conditions
    ], dtype=np.float32)


def denoiser_inputs(model, spectra, conditions):
    if len(model.inputs) == 2:
        return (spectra, acquisition_quality(conditions))
    return spectra


def load_frozen_teacher(path):
    custom_objects = tools.load_custom_objects(
        "CSAM_ResUNet_classification"
    )
    full_model = tf.keras.models.load_model(
        path, custom_objects=custom_objects, compile=False
    )
    classification_layer = full_model.get_layer("classification")
    teacher = tf.keras.Model(
        full_model.input,
        [classification_layer.input, classification_layer.output],
        name="FrozenRecognitionTeacher",
    )
    teacher.trainable = False
    return teacher


def latest_unmixing_weights():
    candidates = glob.glob(DEFAULT_UNMIXING_GLOB)
    if not candidates:
        raise FileNotFoundError(
            "No decoupled unmixer checkpoint found; pass --unmixing-weights"
        )
    return max(candidates, key=os.path.getmtime)


def load_unmixer_config(weights, batch_size):
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(weights)),
        "result",
        "training_config.json",
    )
    values = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            values = json.load(handle)
    defaults = {
        "num_features": 1024,
        "MP_type": list(DEFAULT_CATEGORIES),
        "base_filters": 64,
        "attention_ratio": 8,
        "peak_weight_scale": 10.0,
        "peak_weight_exponent": 2.0,
        "normalize_peak_weights": True,
        "sparse_loss_weight": 0.002,
        "derivative_loss_weight": 0.1,
        "sparse_threshold": 1e-4,
        "unmixing_loss_weight": 1.0,
        "mixture_consistency_weight": 0.05,
        "learning_rate": 1e-4,
    }
    defaults.update(values)
    defaults["batch_size"] = int(batch_size)
    defaults["num_substances"] = len(defaults["MP_type"])
    return SimpleNamespace(**defaults)


def load_frozen_unmixer(weights, batch_size, expected_features):
    config = load_unmixer_config(weights, batch_size)
    if list(config.MP_type) != list(DEFAULT_CATEGORIES):
        raise ValueError(
            "Frozen unmixer category order differs from classification labels"
        )
    if int(config.num_features) != int(expected_features):
        raise ValueError(
            "Frozen unmixer feature length differs from the denoiser input"
        )
    unmixer, _ = tools.built_model(UNMIXING_MODEL_NAME, config)
    unmixer.load_weights(weights)
    unmixer.trainable = False
    return unmixer, config


def load_condition(index, args):
    raw_path = args.raw_pattern.format(index=index)
    raw, labels, pair_names = load_labeled_mixtures(
        raw_path,
        categories=DEFAULT_CATEGORIES,
        num_features=args.num_features,
        suffix=args.input_suffix,
    )
    if args.target_source == "file":
        target_path = args.target_pattern.format(index=index)
        target, target_labels, target_pairs = load_labeled_mixtures(
            target_path,
            categories=DEFAULT_CATEGORIES,
            num_features=args.num_features,
            suffix=args.input_suffix,
        )
        if raw.shape != target.shape:
            raise ValueError(
                f"Condition {index}: raw {raw.shape} and target "
                f"{target.shape} differ"
            )
        if not (
            np.array_equal(labels, target_labels)
            and np.array_equal(pair_names, target_pairs)
        ):
            raise ValueError(f"Condition {index}: raw and target order differs")
    else:
        # A shape-compatible placeholder keeps splitting and indexing generic.
        # It is replaced by the frozen base model before any loss target or
        # downstream reference metric is computed.
        target = raw

    sample_index = np.arange(len(raw), dtype=np.int32)
    group_ids = np.empty(len(raw), dtype=np.int16)
    particle_size_blocks = np.empty(len(raw), dtype=np.int8)
    for pair_index, pair_name in enumerate(np.unique(pair_names)):
        pair_indices = np.flatnonzero(pair_names == pair_name)
        if len(pair_indices) % 3:
            raise ValueError(
                f"Condition {index}, pair {pair_name}: expected three "
                "equal particle-size blocks"
            )
        block_size = len(pair_indices) // 3
        local_blocks = np.arange(len(pair_indices)) // block_size
        particle_size_blocks[pair_indices] = local_blocks
        group_ids[pair_indices] = pair_index * 3 + local_blocks

    if args.limit_per_condition and len(raw) > args.limit_per_condition:
        rng = np.random.default_rng(args.seed + index)
        selected_parts = []
        strata = group_ids if args.multi_quality_unpaired else pair_names
        unique_pairs = np.unique(strata)
        base_quota, remainder = divmod(
            args.limit_per_condition, len(unique_pairs)
        )
        for pair_position, pair_name in enumerate(unique_pairs):
            quota = base_quota + int(pair_position < remainder)
            pair_indices = np.flatnonzero(strata == pair_name)
            if quota > len(pair_indices):
                raise ValueError(
                    f"Condition {index}, pair {pair_name}: debug quota "
                    f"{quota} exceeds {len(pair_indices)} samples"
                )
            if quota:
                selected_parts.append(
                    rng.choice(pair_indices, size=quota, replace=False)
                )
        selected = np.sort(np.concatenate(selected_parts))
        raw = raw[selected]
        target = target[selected]
        labels = labels[selected]
        pair_names = pair_names[selected]
        group_ids = group_ids[selected]
        particle_size_blocks = particle_size_blocks[selected]
        sample_index = sample_index[selected]
    return {
        "raw": raw,
        "target": target,
        "labels": labels,
        "pairs": pair_names,
        "condition": np.full(len(raw), index, dtype=np.int16),
        "group_id": group_ids,
        "particle_size_block": particle_size_blocks,
        # Preserve the row in the condition-level concatenated data.  This is
        # not consumed during training, but makes every independent-test
        # prediction traceable and allows exact paired significance tests.
        "sample_index": sample_index,
    }


def load_condition_group(indices, args):
    buckets = [load_condition(index, args) for index in indices]
    output = {
        key: np.concatenate([bucket[key] for bucket in buckets], axis=0)
        for key in buckets[0]
    }
    return output


def concatenate_data(buckets):
    if not buckets:
        raise ValueError("cannot concatenate an empty data group")
    return {
        key: np.concatenate([bucket[key] for bucket in buckets], axis=0)
        for key in buckets[0]
    }


def select_data(data, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return {key: value[indices] for key, value in data.items()}


def split_condition_data(data, index, args):
    """Stratify one condition without implying cross-condition row pairing."""
    if getattr(args, "split_manifest", None):
        from fixed_splits import manifest_indices
        indices = manifest_indices(data, index, args)
        return {name: select_data(data, rows) for name, rows in indices.items()}
    train_indices = []
    validation_indices = []
    test_indices = []
    strata = data["group_id"] if args.multi_quality_unpaired else data["pairs"]
    unique_pairs = np.unique(strata)
    for pair_position, pair_name in enumerate(unique_pairs):
        pair_indices = np.flatnonzero(strata == pair_name)
        sample_count = len(pair_indices)
        train_count = int(np.floor(sample_count * args.train_ratio))
        validation_count = int(
            np.floor(sample_count * args.validation_ratio)
        )
        test_count = sample_count - train_count - validation_count
        if min(train_count, validation_count, test_count) < 1:
            raise ValueError(
                f"Condition {index}, pair {pair_name}: {sample_count} samples "
                "are insufficient for the requested within-condition split"
            )
        rng = np.random.default_rng(
            args.seed + index * 1000 + pair_position
        )
        shuffled = rng.permutation(pair_indices)
        train_indices.extend(shuffled[:train_count])
        validation_indices.extend(
            shuffled[train_count : train_count + validation_count]
        )
        test_indices.extend(shuffled[train_count + validation_count :])

    train_indices = np.sort(np.asarray(train_indices, dtype=np.int64))
    validation_indices = np.sort(
        np.asarray(validation_indices, dtype=np.int64)
    )
    test_indices = np.sort(np.asarray(test_indices, dtype=np.int64))
    if (
        np.intersect1d(train_indices, validation_indices).size
        or np.intersect1d(train_indices, test_indices).size
        or np.intersect1d(validation_indices, test_indices).size
    ):
        raise RuntimeError(f"Condition {index}: split indices overlap")
    combined = np.concatenate(
        (train_indices, validation_indices, test_indices)
    )
    if not np.array_equal(
        np.sort(combined), np.arange(len(data["raw"]))
    ):
        raise RuntimeError(f"Condition {index}: split indices are incomplete")
    return {
        "train": select_data(data, train_indices),
        "validation": select_data(data, validation_indices),
        "test": select_data(data, test_indices),
    }


def load_within_condition_splits(args):
    train_buckets = []
    validation_buckets = []
    test_buckets = []
    per_condition_test = {}
    for index in args.all_conditions:
        print(f"Loading and stratifying condition {index}")
        condition_data = load_condition(index, args)
        condition_splits = split_condition_data(
            condition_data, index, args
        )
        train_buckets.append(condition_splits["train"])
        validation_buckets.append(condition_splits["validation"])
        test_buckets.append(condition_splits["test"])
        per_condition_test[index] = condition_splits["test"]
        print(
            f"  train={len(condition_splits['train']['raw'])}, "
            f"validation={len(condition_splits['validation']['raw'])}, "
            f"test={len(condition_splits['test']['raw'])}"
        )
    output = (
        concatenate_data(train_buckets),
        concatenate_data(validation_buckets),
        concatenate_data(test_buckets),
        per_condition_test,
    )
    return output


def load_condition_holdout_splits(args):
    """Load whole-condition roles, applying an optional frozen manifest."""
    role_conditions = {
        "train": args.train_conditions,
        "validation": args.validation_conditions,
        "test": args.test_conditions,
    }
    outputs = {}
    per_condition_test = {}
    for role, conditions in role_conditions.items():
        buckets = []
        for index in conditions:
            print(f"Loading condition {index} for {role}")
            data = load_condition(index, args)
            if args.split_manifest:
                from fixed_splits import manifest_indices

                indices = manifest_indices(data, index, args)
                selected = indices[role]
                excluded = len(indices["excluded"])
                print(
                    f"  retained={len(selected)}, excluded_cross_role_duplicates={excluded}"
                )
                data = select_data(data, selected)
            buckets.append(data)
            if role == "test":
                per_condition_test[index] = data
        outputs[role] = concatenate_data(buckets)
    return (
        outputs["train"],
        outputs["validation"],
        outputs["test"],
        per_condition_test,
    )


def teacher_targets(teacher, target, batch_size):
    features, probabilities = teacher.predict(
        target, batch_size=batch_size, verbose=1
    )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(probabilities, dtype=np.float32),
    )


def condition_severity_map(high_quality_condition):
    """Map acquisition energy to a 0..1 synthetic-noise severity."""
    energies = {
        index: float(details[2]) for index, details in CONDITION_DETAILS.items()
    }
    reference_energy = energies[high_quality_condition]
    difficulty = {
        index: np.sqrt(reference_energy / energy)
        for index, energy in energies.items()
    }
    reference = difficulty[high_quality_condition]
    maximum = max(difficulty.values())
    denominator = max(maximum - reference, np.finfo(np.float64).eps)
    return {
        index: float(np.clip((value - reference) / denominator, 0.0, 1.0))
        for index, value in difficulty.items()
    }


def attach_unpaired_reference_indices(data, bank_groups, severities, seed):
    """Assign an independent same-group high-quality reference to each row."""
    rng = np.random.default_rng(seed)
    reference_index = np.empty(len(data["raw"]), dtype=np.int32)
    for group_id in np.unique(data["group_id"]):
        destination = np.flatnonzero(data["group_id"] == group_id)
        candidates = np.flatnonzero(bank_groups == group_id)
        if not len(candidates):
            raise ValueError(
                f"high-quality reference bank has no samples for group {group_id}"
            )
        reference_index[destination] = rng.choice(
            candidates, size=len(destination), replace=True
        )
    severity = np.asarray(
        [severities[int(index)] for index in data["condition"]],
        dtype=np.float32,
    )
    unique_index = (
        data["condition"].astype(np.int64) * 100000
        + data["sample_index"].astype(np.int64)
    )
    data["reference_index"] = reference_index
    # Store severity as a fixed-point integer so both columns remain int64 and
    # can be used directly for stateless random seeds in TensorFlow.
    data["augmentation_index"] = np.stack(
        (unique_index, np.rint(severity * 1000000.0).astype(np.int64)),
        axis=1,
    )


def build_unpaired_resources(
    train_data,
    validation_data,
    test_data,
    train_features,
    train_probabilities,
    args,
):
    """Build leakage-safe group profiles from high-quality training rows."""
    high_mask = train_data["condition"] == args.high_quality_condition
    if not np.any(high_mask):
        raise ValueError("no high-quality training spectra were found")
    high_raw = np.ascontiguousarray(train_data["raw"][high_mask])
    high_target = np.ascontiguousarray(train_data["target"][high_mask])
    high_labels = np.ascontiguousarray(train_data["labels"][high_mask])
    high_probabilities = np.ascontiguousarray(
        train_probabilities[high_mask]
    )
    high_features = np.asarray(train_features[high_mask], dtype=np.float32)
    feature_norm = np.linalg.norm(high_features, axis=1, keepdims=True)
    high_features = high_features / np.maximum(
        feature_norm, np.finfo(np.float32).eps
    )
    bank_groups = np.asarray(train_data["group_id"][high_mask], dtype=np.int32)
    group_count = int(np.max(train_data["group_id"])) + 1
    spectral_shape = high_target.shape[1:]
    feature_shape = high_features.shape[1:]
    group_spectral_mean = np.empty(
        (group_count,) + spectral_shape, dtype=np.float32
    )
    group_spectral_std = np.empty_like(group_spectral_mean)
    group_feature_mean = np.empty(
        (group_count,) + feature_shape, dtype=np.float32
    )
    group_feature_std = np.empty_like(group_feature_mean)
    for group_id in range(group_count):
        mask = bank_groups == group_id
        if not np.any(mask):
            raise ValueError(
                f"high-quality training split is missing group {group_id}"
            )
        group_spectral_mean[group_id] = np.mean(
            high_target[mask], axis=0
        )
        group_spectral_std[group_id] = np.std(high_target[mask], axis=0)
        group_feature_mean[group_id] = np.mean(high_features[mask], axis=0)
        group_feature_std[group_id] = np.std(high_features[mask], axis=0)

    severities = condition_severity_map(args.high_quality_condition)
    for offset, data in enumerate(
        (train_data, validation_data, test_data), start=1
    ):
        attach_unpaired_reference_indices(
            data,
            bank_groups,
            severities,
            seed=args.seed + offset * 1000000,
        )
    print(
        "Condition-scaled synthetic-noise severity: "
        + ", ".join(
            f"D{index}={value:.3f}" for index, value in severities.items()
        )
    )
    print(
        f"High-quality reference bank: condition {args.high_quality_condition}, "
        f"samples={len(high_raw)}, groups={group_count}; no cross-condition "
        "row pairing is used"
    )
    return {
        "high_reference_raw": high_raw,
        "high_reference_target": high_target,
        "high_reference_labels": high_labels,
        "high_reference_probabilities": high_probabilities,
        "group_spectral_mean": group_spectral_mean,
        "group_spectral_std": group_spectral_std,
        "group_feature_mean": group_feature_mean,
        "group_feature_std": group_feature_std,
        "condition_severity": severities,
        "reference_bank_samples": int(len(high_raw)),
        "group_count": group_count,
    }


def unmixing_targets(unmixer, target, batch_size):
    component_a, component_b = unmixer.predict(
        target, batch_size=batch_size, verbose=1
    )
    return (
        np.asarray(component_a, dtype=np.float32),
        np.asarray(component_b, dtype=np.float32),
    )


def generate_base_model_targets(base_denoiser, data, batch_size, label):
    """Replace one split's targets with deterministic base-model inference."""
    print(
        f"Generating aligned denoising targets for {label} "
        f"({len(data['raw'])} spectra)"
    )
    target = base_denoiser.predict(
        data["raw"], batch_size=batch_size, verbose=1
    )
    if isinstance(target, (tuple, list)):
        raise ValueError("base denoiser must have exactly one spectral output")
    target = np.asarray(target, dtype=np.float32)
    if target.shape != data["raw"].shape:
        raise ValueError(
            f"Base-model target shape {target.shape} differs from raw shape "
            f"{data['raw'].shape} for {label}"
        )
    if not np.all(np.isfinite(target)):
        raise ValueError(f"Base-model targets contain NaN or Inf for {label}")
    data["target"] = np.ascontiguousarray(target)


def target_alignment_metrics(denoiser, data, batch_size, sample_count):
    """Compare initialized inference with the reconstruction loss targets."""
    count = min(int(sample_count), len(data["raw"]))
    indices = np.linspace(0, len(data["raw"]) - 1, count, dtype=np.int64)
    prediction = denoiser.predict(
        denoiser_inputs(denoiser, data["raw"][indices],
                        data["condition"][indices] if "condition" in data else None),
        batch_size=batch_size, verbose=0
    )
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(data["target"][indices], dtype=np.float32)
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    flat_prediction = prediction.reshape(count, -1).astype(np.float64)
    flat_target = target.reshape(count, -1).astype(np.float64)
    denominator = (
        np.linalg.norm(flat_prediction, axis=1)
        * np.linalg.norm(flat_target, axis=1)
    )
    cosine = np.sum(flat_prediction * flat_target, axis=1) / np.maximum(
        denominator, np.finfo(np.float64).eps
    )
    return {
        "samples": int(count),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_cosine_similarity": float(np.mean(cosine)),
    }


def enforce_target_alignment(denoiser, validation_data, args):
    metrics = target_alignment_metrics(
        denoiser,
        validation_data,
        args.target_generation_batch_size,
        args.target_alignment_samples,
    )
    print(
        "Initialized-model/target consistency: "
        f"MAE={metrics['mae']:.3e}, "
        f"RMSE={metrics['rmse']:.3e}, "
        f"max_abs={metrics['max_abs_error']:.3e}, "
        f"cosine={metrics['mean_cosine_similarity']:.9f}"
    )
    if (
        metrics["mae"] > args.target_mae_tolerance
        or metrics["max_abs_error"] > args.target_max_abs_tolerance
    ):
        raise ValueError(
            "Denoising targets do not match the initialized base model. "
            "Use --target-source base-model, or regenerate file targets with "
            "the same checkpoint and preprocessing before training."
        )
    return metrics


def make_dataset(
    data,
    features,
    probabilities,
    args,
    training,
    target_components=None,
):
    targets = {
        "target_spectrum": data["target"],
        "classification": data["labels"],
        "teacher_features": features,
        "teacher_probabilities": probabilities,
    }
    if target_components is not None:
        targets["target_component_a"] = target_components[0]
        targets["target_component_b"] = target_components[1]
    if args.multi_quality_unpaired:
        targets["group_id"] = data["group_id"]
        targets["reference_index"] = data["reference_index"]
        targets["augmentation_index"] = data["augmentation_index"]
    inputs = data["raw"]
    if getattr(args, "quality_conditioning", False):
        inputs = (inputs, acquisition_quality(data["condition"]))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if training:
        dataset = dataset.shuffle(
            min(len(data["raw"]), 8192),
            seed=args.seed,
            reshuffle_each_iteration=True,
        )
    if args.multi_quality_unpaired:
        dataset = dataset.apply(
            tf.data.experimental.group_by_window(
                key_func=lambda _, target: tf.cast(
                    target["group_id"], tf.int64
                ),
                reduce_func=lambda _, rows: rows.batch(args.batch_size),
                window_size=args.batch_size,
            )
        )
    else:
        dataset = dataset.batch(args.batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def probabilities(teacher, spectra, batch_size):
    _, output = teacher.predict(
        spectra, batch_size=batch_size, verbose=0
    )
    return output


def evaluate_condition(denoiser, teacher, index, args, data=None):
    if data is None:
        data = load_condition(index, args)
    guided = denoiser.predict(
        denoiser_inputs(denoiser, data["raw"], data["condition"]),
        batch_size=args.batch_size, verbose=0
    )
    result = {"samples": int(len(data["raw"]))}
    for name, spectra in (
        ("raw", data["raw"]),
        ("original_denoising", data["target"]),
        ("adapted_denoising", guided),
    ):
        metrics, _, _ = classification_metrics(
            data["labels"],
            probabilities(teacher, spectra, args.batch_size),
            DEFAULT_CATEGORIES,
            k=2,
        )
        result[name] = metrics
    result["adapted_spectral"] = spectral_metrics(data["target"], guided)
    result["adapted_minus_original_percentage_points"] = 100.0 * (
        result["adapted_denoising"]["subset_accuracy"]
        - result["original_denoising"]["subset_accuracy"]
    )
    del guided
    return result


def evaluate_unmixing_condition(
    denoiser, unmixer, centroids, index, args, data=None
):
    """Evaluate measured endpoints against explicitly labelled proxy metrics."""
    if data is None:
        data = load_condition(index, args)
    guided = denoiser.predict(
        denoiser_inputs(denoiser, data["raw"], data["condition"]),
        batch_size=args.batch_size, verbose=0
    )
    original_a, original_b = unmixer.predict(
        data["target"], batch_size=args.batch_size, verbose=0
    )
    guided_a, guided_b = unmixer.predict(
        guided, batch_size=args.batch_size, verbose=0
    )
    original_proxy = reference_proxy_metrics(
        original_a,
        original_b,
        data["pairs"],
        list(DEFAULT_CATEGORIES),
        centroids,
    )
    guided_proxy = reference_proxy_metrics(
        guided_a,
        guided_b,
        data["pairs"],
        list(DEFAULT_CATEGORIES),
        centroids,
    )
    result = {
        "ground_truth_status": (
            "pure-spectrum centroid proxy; no sample-aligned real endpoint truth"
        ),
        "original_denoising": original_proxy,
        "adapted_denoising": guided_proxy,
        "reference_pair_accuracy_delta_pp": 100.0
        * (
            guided_proxy["reference_pair_accuracy"]
            - original_proxy["reference_pair_accuracy"]
        ),
        "reference_sam_delta_degrees": (
            guided_proxy["reference_sam_degrees"]
            - original_proxy["reference_sam_degrees"]
        ),
    }
    del guided, original_a, original_b, guided_a, guided_b
    return result


class BestRealDomainCheckpoint(tf.keras.callbacks.Callback):
    """Keep a candidate and a baseline-constrained deployable checkpoint."""

    def __init__(
        self,
        path,
        candidate_path,
        initial_accuracy,
        initial_loss,
        reference_accuracy,
        accuracy_tolerance=1e-6,
    ):
        super().__init__()
        self.path = path
        self.candidate_path = candidate_path
        self.reference_accuracy = float(reference_accuracy)
        self.accuracy_tolerance = float(accuracy_tolerance)
        self.best_candidate_accuracy = float(initial_accuracy)
        self.best_candidate_loss = float(initial_loss)
        self.safe_constraint_met = (
            self.best_candidate_accuracy + self.accuracy_tolerance
            >= self.reference_accuracy
        )
        self.best_safe_accuracy = (
            self.best_candidate_accuracy
            if self.safe_constraint_met
            else -np.inf
        )
        self.best_safe_loss = (
            self.best_candidate_loss if self.safe_constraint_met else np.inf
        )

    @staticmethod
    def _is_better(accuracy, loss, best_accuracy, best_loss):
        return accuracy > best_accuracy + 1e-8 or (
            abs(accuracy - best_accuracy) <= 1e-8 and loss < best_loss
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get("val_guided_subset_accuracy")
        loss = logs.get("val_loss")
        if accuracy is None or loss is None:
            return
        accuracy = float(accuracy)
        loss = float(loss)
        if not np.isfinite(accuracy) or not np.isfinite(loss):
            return

        if self._is_better(
            accuracy,
            loss,
            self.best_candidate_accuracy,
            self.best_candidate_loss,
        ):
            self.best_candidate_accuracy = accuracy
            self.best_candidate_loss = loss
            self.model.denoiser.save(
                self.candidate_path, include_optimizer=False
            )
            print(
                f"Saved best unconstrained candidate: {self.candidate_path} "
                f"(val_accuracy={accuracy:.6f}, val_loss={loss:.6g})"
            )

        qualifies = (
            accuracy + self.accuracy_tolerance >= self.reference_accuracy
        )
        safe_improved = self._is_better(
            accuracy,
            loss,
            self.best_safe_accuracy,
            self.best_safe_loss,
        )
        if qualifies and safe_improved:
            self.safe_constraint_met = True
            self.best_safe_accuracy = accuracy
            self.best_safe_loss = loss
            self.model.denoiser.save(self.path, include_optimizer=False)
            print(
                f"Saved baseline-safe denoiser: {self.path} "
                f"(val_accuracy={accuracy:.6f} >= "
                f"reference={self.reference_accuracy:.6f})"
            )


class BestDualDomainCheckpoint(tf.keras.callbacks.Callback):
    """Save only epochs that do not degrade either downstream task."""

    def __init__(
        self,
        path,
        candidate_path,
        initial_metrics,
        accuracy_tolerance=1e-6,
        unmixing_accuracy_tolerance=1e-6,
        endpoint_sam_tolerance=1e-4,
    ):
        super().__init__()
        self.path = path
        self.candidate_path = candidate_path
        self.reference_accuracy = float(
            initial_metrics["target_subset_accuracy"]
        )
        self.reference_unmixing_accuracy = float(
            initial_metrics["target_unmix_pair_accuracy"]
        )
        self.reference_endpoint_sam = float(
            initial_metrics["target_endpoint_sam_degrees"]
        )
        self.accuracy_tolerance = float(accuracy_tolerance)
        self.unmixing_accuracy_tolerance = float(
            unmixing_accuracy_tolerance
        )
        self.endpoint_sam_tolerance = float(endpoint_sam_tolerance)
        initial_score = self._score(
            initial_metrics["guided_subset_accuracy"],
            initial_metrics["guided_unmix_pair_accuracy"],
            initial_metrics["guided_endpoint_sam_degrees"],
        )
        self.safe_constraint_met = self._qualifies(
            float(initial_metrics["guided_subset_accuracy"]),
            float(initial_metrics["guided_unmix_pair_accuracy"]),
            float(initial_metrics["guided_endpoint_sam_degrees"]),
        )
        self.best_safe_score = (
            initial_score if self.safe_constraint_met else -np.inf
        )
        self.best_safe_loss = (
            float(initial_metrics["loss"])
            if self.safe_constraint_met
            else np.inf
        )
        self.best_candidate_score = initial_score
        self.best_candidate_loss = float(initial_metrics["loss"])
        self.selected_metrics = (
            self._selected(initial_metrics)
            if self.safe_constraint_met
            else None
        )

    @staticmethod
    def _selected(values):
        keys = (
            "guided_subset_accuracy",
            "guided_unmix_pair_accuracy",
            "guided_endpoint_sam_degrees",
            "loss",
        )
        return {key: float(values[key]) for key in keys}

    def _score(self, classification_accuracy, unmixing_accuracy, endpoint_sam):
        # Accuracy terms have equal priority; SAM is a bounded tie breaker.
        return float(
            classification_accuracy
            + unmixing_accuracy
            - endpoint_sam / 180.0
        )

    @staticmethod
    def _is_better(score, loss, best_score, best_loss):
        return score > best_score + 1e-8 or (
            abs(score - best_score) <= 1e-8 and loss < best_loss
        )

    def _qualifies(self, accuracy, unmixing_accuracy, endpoint_sam):
        return (
            accuracy + self.accuracy_tolerance >= self.reference_accuracy
            and unmixing_accuracy + self.unmixing_accuracy_tolerance
            >= self.reference_unmixing_accuracy
            and endpoint_sam
            <= self.reference_endpoint_sam + self.endpoint_sam_tolerance
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        required = {
            "accuracy": "val_guided_subset_accuracy",
            "unmixing_accuracy": "val_guided_unmix_pair_accuracy",
            "endpoint_sam": "val_guided_endpoint_sam_degrees",
            "loss": "val_loss",
        }
        if any(logs.get(key) is None for key in required.values()):
            return
        values = {
            name: float(logs[key]) for name, key in required.items()
        }
        if not np.all(np.isfinite(list(values.values()))):
            return
        score = self._score(
            values["accuracy"],
            values["unmixing_accuracy"],
            values["endpoint_sam"],
        )
        if self._is_better(
            score,
            values["loss"],
            self.best_candidate_score,
            self.best_candidate_loss,
        ):
            self.best_candidate_score = score
            self.best_candidate_loss = values["loss"]
            self.model.denoiser.save(
                self.candidate_path, include_optimizer=False
            )
            print(
                "Saved best unconstrained dual-guided candidate: "
                f"{self.candidate_path} (score={score:.6f})"
            )

        if self._qualifies(
            values["accuracy"],
            values["unmixing_accuracy"],
            values["endpoint_sam"],
        ) and self._is_better(
            score, values["loss"], self.best_safe_score, self.best_safe_loss
        ):
            self.safe_constraint_met = True
            self.best_safe_score = score
            self.best_safe_loss = values["loss"]
            self.selected_metrics = {
                "guided_subset_accuracy": values["accuracy"],
                "guided_unmix_pair_accuracy": values["unmixing_accuracy"],
                "guided_endpoint_sam_degrees": values["endpoint_sam"],
                "loss": values["loss"],
            }
            self.model.denoiser.save(self.path, include_optimizer=False)
            print(
                f"Saved dual-safe denoiser: {self.path} "
                f"(classification={values['accuracy']:.6f}, "
                f"unmixing={values['unmixing_accuracy']:.6f}, "
                f"endpoint_SAM={values['endpoint_sam']:.4f} deg)"
            )


def json_safe_args(args):
    output = {}
    for key, value in vars(args).items():
        if isinstance(value, tuple):
            value = list(value)
        output[key] = value
    return output


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    apply_ablation_variant(args)
    if args.ablation_variant:
        args.ablation_active_weights = active_weight_groups(args)
        print(
            f"Ablation {args.ablation_variant}: "
            f"{args.ablation_components['description']}",
            flush=True,
        )
        print(
            json.dumps(args.ablation_active_weights, indent=2),
            flush=True,
        )
    validate_args(args)
    if args.split_manifest:
        from fixed_splits import read_manifest, manifest_indices
        if args.limit_per_condition:
            raise ValueError("Frozen manifest requires full, unsubsampled data")
        _, split_meta = read_manifest(args.split_manifest)
        requested_conditions = (
            set(args.all_conditions)
            if args.split_mode == "within-condition"
            else set(args.train_conditions)
            | set(args.validation_conditions)
            | set(args.test_conditions)
        )
        if set(map(int, split_meta["sources"])) != requested_conditions:
            raise ValueError("Manifest conditions and requested conditions must agree")
        manifest_protocol = split_meta.get("protocol", "within-condition")
        if manifest_protocol != args.split_mode:
            raise ValueError(
                f"Manifest protocol {manifest_protocol!r} does not match "
                f"--split-mode {args.split_mode!r}"
            )
        if args.split_mode == "condition-holdout":
            for name in ("train", "validation", "test"):
                requested = list(getattr(args, f"{name}_conditions"))
                recorded = list(map(int, split_meta[f"{name}_conditions"]))
                if requested != recorded:
                    raise ValueError(
                        f"Manifest {name} conditions and command-line conditions disagree"
                    )
        args.split_manifest_sha256 = split_meta["manifest_sha256"]
        args.split_training_seed = split_meta["seed"]
        if args.split_mode == "within-condition":
            if not np.isclose(args.train_ratio, split_meta["train_ratio"]) or not np.isclose(args.validation_ratio, split_meta["validation_ratio"]):
                raise ValueError("Requested ratios disagree with the immutable split; use its recorded ratios")
        # Fail before loading/predicting any neural model; preserve all groups.
        for condition in sorted(requested_conditions):
            manifest_indices(load_condition(condition, args), condition, args)
        print("Verified fixed duplicate-group split before denoising", flush=True)
        if args.audit_split_only:
            return
    else:
        if args.audit_split_only:
            raise ValueError("--audit-split-only requires --split-manifest")
        print("WARNING: legacy row split does not isolate duplicate spectra", flush=True)
    tools.set_seed(args.seed)

    pretrained_path = args.pretrained_denoiser or latest_pretrained_denoiser()
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(pretrained_path)
    if not os.path.isfile(args.teacher_model):
        raise FileNotFoundError(args.teacher_model)

    dual_guidance = args.unmixing_weights is not None
    if args.unmixing_weights == "latest":
        args.unmixing_weights = latest_unmixing_weights()
    if dual_guidance and not os.path.isfile(args.unmixing_weights):
        raise FileNotFoundError(args.unmixing_weights)
    if dual_guidance and not os.path.isfile(args.reference_path):
        raise FileNotFoundError(args.reference_path)

    if args.ablation_variant:
        prefix = f"Ablation_{args.ablation_variant}_"
        prefix += "MultiQuality_" if args.multi_quality_unpaired else "SingleQuality_"
        prefix += "DualGuided_"
    elif args.multi_quality_unpaired:
        prefix = "MultiQualityUnpaired_DualGuided_"
    elif dual_guidance:
        prefix = "DualGuided_RealDomain_"
    else:
        prefix = "RecognitionGuided_RealDomain_"
    if args.quality_conditioning:
        prefix += "QualityGate_"
    if (
        args.self_similarity_weight > 0
        or args.synthetic_self_similarity_weight > 0
    ):
        prefix += "PeakSelfSimilarity_"
    if args.split_manifest:
        prefix += "GroupSplit_"
    if args.split_mode == "condition-holdout":
        prefix += "ConditionHoldout_"
    run_name = prefix + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(args.output_root, run_name)
    checkpoint_path = os.path.join(run_path, "checkpoint")
    result_path = os.path.join(run_path, "result")
    os.makedirs(checkpoint_path, exist_ok=False)
    os.makedirs(result_path, exist_ok=False)
    if args.split_manifest:
        from pathlib import Path
        source_manifest = Path(args.split_manifest)
        pinned_manifest = Path(result_path) / "split_manifest.csv"
        shutil.copy2(source_manifest, pinned_manifest)
        shutil.copy2(source_manifest.with_suffix(".json"), pinned_manifest.with_suffix(".json"))
        args.split_manifest = str(pinned_manifest)
        args.independence_scope = "raw duplicate-group isolated; physical IDs and upstream pretraining exposure unknown"
    # Save configuration before expensive preprocessing/training for recovery.
    write_json(os.path.join(result_path, "config.json"), json_safe_args(args))

    print(f"Loading base denoiser: {pretrained_path}")
    base_denoiser = load_denoiser(pretrained_path)
    denoiser = build_recognition_guided_denoiser(
        base_denoiser,
        num_features=args.num_features,
        refinement_filters=args.refinement_filters,
        maximum_correction=args.maximum_correction,
        quality_conditioning=args.quality_conditioning,
    )
    base_denoiser.trainable = bool(args.fine_tune_backbone)
    print(
        "Base denoiser is "
        + ("trainable" if base_denoiser.trainable else "frozen")
    )
    print(f"Loading frozen recognition teacher: {args.teacher_model}")
    teacher = load_frozen_teacher(args.teacher_model)
    unmixer = None
    unmixer_config = None
    reference_centroids = None
    if dual_guidance:
        print(f"Loading frozen unmixing teacher: {args.unmixing_weights}")
        unmixer, unmixer_config = load_frozen_unmixer(
            args.unmixing_weights, args.batch_size, args.num_features
        )
        reference_centroids = load_reference_centroids(
            args.reference_path,
            list(DEFAULT_CATEGORIES),
            args.num_features,
        ).astype(np.float32)

    per_condition_test = {}
    if args.split_mode == "within-condition":
        print(
            "Using pair-stratified within-condition splitting: "
            f"train={args.train_ratio:.1%}, "
            f"validation={args.validation_ratio:.1%}, "
            f"test={1.0 - args.train_ratio - args.validation_ratio:.1%}"
        )
        (
            train_data,
            validation_data,
            test_data,
            per_condition_test,
        ) = load_within_condition_splits(args)
        split_description = {
            "mode": args.split_mode,
            "manifest": args.split_manifest,
            "manifest_sha256": getattr(args, "split_manifest_sha256", None),
            "independence_scope": getattr(args, "independence_scope", "legacy row split; duplicates may cross"),
            "conditions": list(args.all_conditions),
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": (
                1.0 - args.train_ratio - args.validation_ratio
            ),
            "stratification": (
                "material_pair_and_particle_size_within_each_condition"
                if args.multi_quality_unpaired
                else "material_pair_within_each_condition"
            ),
        }
    else:
        (
            train_data,
            validation_data,
            test_data,
            per_condition_test,
        ) = load_condition_holdout_splits(args)
        split_description = {
            "mode": args.split_mode,
            "train": list(args.train_conditions),
            "validation": list(args.validation_conditions),
            "test": list(args.test_conditions),
            "manifest": args.split_manifest,
            "manifest_sha256": getattr(args, "split_manifest_sha256", None),
            "independence_scope": getattr(
                args,
                "independence_scope",
                "whole-condition holdout without duplicate audit",
            ),
            "stratification": (
                "whole acquisition condition; pair and particle-size strata "
                "retained; global raw duplicate groups isolated"
            ),
        }
    print(
        f"Samples: train={len(train_data['raw'])}, "
        f"validation={len(validation_data['raw'])}, "
        f"test={len(test_data['raw'])}"
    )

    if args.target_source == "base-model":
        print(
            "Using the loaded base checkpoint as the single source of truth "
            "for denoising targets; no additional normalization is applied."
        )
        for label, data in (
            ("train", train_data),
            ("validation", validation_data),
            ("test", test_data),
        ):
            generate_base_model_targets(
                base_denoiser,
                data,
                args.target_generation_batch_size,
                label,
            )
        for index, condition_data in per_condition_test.items():
            condition_mask = test_data["condition"] == index
            aligned_raw = test_data["raw"][condition_mask]
            if np.array_equal(condition_data["raw"], aligned_raw):
                condition_data["target"] = np.ascontiguousarray(
                    test_data["target"][condition_mask]
                )
            else:
                generate_base_model_targets(
                    base_denoiser,
                    condition_data,
                    args.target_generation_batch_size,
                    f"condition-{index}-test",
                )
    else:
        print(
            "Using denoising targets from --target-pattern; training will "
            "stop if they do not match the initialized base checkpoint."
        )
    target_alignment = enforce_target_alignment(
        denoiser, validation_data, args
    )
    split_description["denoising_target_source"] = args.target_source
    split_description["target_checkpoint"] = pretrained_path
    split_description["target_preprocessing"] = (
        "MAT *_norm values -> float32 -> channel expansion; no additional "
        "normalization"
    )

    print("Precomputing frozen-teacher targets")
    train_features, train_probabilities = teacher_targets(
        teacher, train_data["target"], args.batch_size
    )
    validation_features, validation_probabilities = teacher_targets(
        teacher, validation_data["target"], args.batch_size
    )
    test_features, test_probabilities = teacher_targets(
        teacher, test_data["target"], args.batch_size
    )
    train_components = validation_components = test_components = None
    if dual_guidance:
        print("Precomputing frozen-unmixer endpoint targets")
        train_components = unmixing_targets(
            unmixer, train_data["target"], args.batch_size
        )
        validation_components = unmixing_targets(
            unmixer, validation_data["target"], args.batch_size
        )
        test_components = unmixing_targets(
            unmixer, test_data["target"], args.batch_size
        )
    unpaired_resources = None
    if args.multi_quality_unpaired:
        print(
            "Building non-paired multi-quality reference bank and group "
            "statistics from high-quality training rows"
        )
        unpaired_resources = build_unpaired_resources(
            train_data,
            validation_data,
            test_data,
            train_features,
            train_probabilities,
            args,
        )
    train_dataset = make_dataset(
        train_data,
        train_features,
        train_probabilities,
        args,
        training=True,
        target_components=train_components,
    )
    validation_dataset = make_dataset(
        validation_data,
        validation_features,
        validation_probabilities,
        args,
        training=False,
        target_components=validation_components,
    )
    test_dataset = make_dataset(
        test_data,
        test_features,
        test_probabilities,
        args,
        training=False,
        target_components=test_components,
    )

    if args.multi_quality_unpaired:
        model_class = MultiQualityUnpairedDualGuidedAdaptationModel
    elif dual_guidance:
        model_class = DualGuidedRealDomainAdaptationModel
    else:
        model_class = RealDomainAdaptationModel
    model_kwargs = {}
    if dual_guidance:
        model_kwargs.update(
            {
                "unmixer": unmixer,
                "reference_centroids": reference_centroids,
                "unmixing_distillation_weight": (
                    args.unmixing_distillation_weight
                ),
                "endpoint_classification_weight": (
                    args.endpoint_classification_weight
                ),
                "mixture_consistency_guidance_weight": (
                    args.mixture_consistency_guidance_weight
                ),
                "endpoint_temperature": args.endpoint_temperature,
                "use_pcgrad": args.use_pcgrad,
            }
        )
    if args.multi_quality_unpaired:
        model_kwargs.update(
            {
                key: unpaired_resources[key]
                for key in (
                    "high_reference_raw",
                    "high_reference_target",
                    "high_reference_labels",
                    "high_reference_probabilities",
                    "group_spectral_mean",
                    "group_spectral_std",
                    "group_feature_mean",
                    "group_feature_std",
                )
            }
        )
        model_kwargs.update(
            {
                "synthetic_pair_weight": args.synthetic_pair_weight,
                "high_reference_quality": acquisition_quality(
                    [args.high_quality_condition]
                )[0],
                "synthetic_classification_weight": (
                    args.synthetic_classification_weight
                ),
                "high_quality_identity_weight": (
                    args.high_quality_identity_weight
                ),
                "group_spectral_weight": args.group_spectral_weight,
                "group_feature_weight": args.group_feature_weight,
                "synthetic_self_similarity_weight": (
                    args.synthetic_self_similarity_weight
                ),
                "augmentation_seed": args.seed,
                "gaussian_std_min": args.synthetic_gaussian_min,
                "gaussian_std_max": args.synthetic_gaussian_max,
                "shot_noise_max": args.synthetic_shot_noise_max,
                "baseline_max": args.synthetic_baseline_max,
                "spike_probability": args.synthetic_spike_probability,
                "spike_amplitude": args.synthetic_spike_amplitude,
                "gain_variation": args.synthetic_gain_variation,
            }
        )
    training_model = model_class(
        denoiser=denoiser,
        teacher=teacher,
        reconstruction_loss_config={
            "intensity_weight_scale": args.peak_weight_scale,
            "intensity_weight_exponent": 2.0,
            "normalize_intensity": True,
            "sparse_strength": args.sparse_loss_weight,
            "derivative_strength": args.derivative_loss_weight,
        },
        reconstruction_weight=args.reconstruction_weight,
        classification_weight=args.classification_weight,
        distillation_weight=args.distillation_weight,
        feature_weight=args.feature_weight,
        sam_weight=args.sam_weight,
        correction_weight=args.correction_weight,
        safe_margin_weight=args.safe_margin_weight,
        self_similarity_weight=args.self_similarity_weight,
        self_similarity_loss_config={
            "patch_size": args.self_similarity_patch_size,
            "search_radius": args.self_similarity_search_radius,
            "search_step": args.self_similarity_search_step,
            "peak_fraction": args.self_similarity_peak_fraction,
            "temperature": args.self_similarity_temperature,
            "l1_weight": args.self_similarity_l1_weight,
        },
        confidence_margin=args.confidence_margin,
        hard_example_weight=args.hard_example_weight,
        easy_identity_weight=args.easy_identity_weight,
        name=(
            "MultiQualityUnpairedDualGuidedAdaptationModel"
            if args.multi_quality_unpaired
            else (
                "DualGuidedRealDomainAdaptationModel"
                if dual_guidance
                else "RealDomainAdaptationModel"
            )
        ),
        **model_kwargs,
    )
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate, clipnorm=1.0
        )
    )

    best_path = os.path.join(checkpoint_path, "best_denoiser.h5")
    candidate_path = os.path.join(
        checkpoint_path, "best_candidate_denoiser.h5"
    )
    final_path = os.path.join(checkpoint_path, "final_denoiser.h5")
    initial_path = os.path.join(checkpoint_path, "initial_denoiser.h5")
    denoiser.save(initial_path, include_optimizer=False)
    print("Evaluating initialized model on the validation split")
    initial_validation = training_model.evaluate(
        validation_dataset, return_dict=True, verbose=1
    )
    denoiser.save(best_path, include_optimizer=False)
    denoiser.save(candidate_path, include_optimizer=False)
    reference_accuracy = initial_validation["target_subset_accuracy"]
    print(
        "Validation reference accuracy from original denoised spectra: "
        f"{reference_accuracy:.6f}; zero-residual model: "
        f"{initial_validation['guided_subset_accuracy']:.6f}"
    )
    if dual_guidance:
        print(
            "Validation unmixing proxy reference from original denoised "
            f"spectra: pair_accuracy="
            f"{initial_validation['target_unmix_pair_accuracy']:.6f}, "
            f"endpoint_SAM="
            f"{initial_validation['target_endpoint_sam_degrees']:.4f} deg"
        )
        checkpoint = BestDualDomainCheckpoint(
            best_path,
            candidate_path=candidate_path,
            initial_metrics=initial_validation,
            accuracy_tolerance=1e-6,
            unmixing_accuracy_tolerance=args.unmixing_accuracy_tolerance,
            endpoint_sam_tolerance=args.endpoint_sam_tolerance,
        )
    else:
        checkpoint = BestRealDomainCheckpoint(
            best_path,
            candidate_path=candidate_path,
            initial_accuracy=initial_validation["guided_subset_accuracy"],
            initial_loss=initial_validation["loss"],
            reference_accuracy=reference_accuracy,
        )

    callbacks = [
        checkpoint,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.early_stopping_patience,
            restore_best_weights=False,
            verbose=1,
        ),
    ]
    history = training_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    denoiser.save(final_path, include_optimizer=False)
    save_history(os.path.join(result_path, "history.csv"), history.history)

    if checkpoint.safe_constraint_met:
        selection_reason = (
            "validation classification and unmixing proxy constraints met"
            if dual_guidance
            else (
                "validation accuracy met or exceeded the original-denoised "
                "reference"
            )
        )
    else:
        selection_reason = (
            "no trained epoch met all original-denoised validation "
            "constraints; retained the zero-residual initialization"
        )
        print("No trained epoch met the non-degradation constraint.")
        print("Using the zero-residual initialization as best_denoiser.h5.")

    best_denoiser = load_denoiser(best_path)
    best_model_kwargs = {}
    if dual_guidance:
        # Reuse the frozen teachers; only the selected denoiser is reloaded.
        best_model_kwargs.update(model_kwargs)
    best_training_model = model_class(
        denoiser=best_denoiser,
        teacher=teacher,
        reconstruction_loss_config={
            "intensity_weight_scale": args.peak_weight_scale,
            "intensity_weight_exponent": 2.0,
            "normalize_intensity": True,
            "sparse_strength": args.sparse_loss_weight,
            "derivative_strength": args.derivative_loss_weight,
        },
        reconstruction_weight=args.reconstruction_weight,
        classification_weight=args.classification_weight,
        distillation_weight=args.distillation_weight,
        feature_weight=args.feature_weight,
        sam_weight=args.sam_weight,
        correction_weight=args.correction_weight,
        safe_margin_weight=args.safe_margin_weight,
        confidence_margin=args.confidence_margin,
        hard_example_weight=args.hard_example_weight,
        easy_identity_weight=args.easy_identity_weight,
        **best_model_kwargs,
    )
    best_training_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate)
    )
    test_losses = best_training_model.evaluate(
        test_dataset, return_dict=True, verbose=1
    )

    condition_results = {}
    print("\nStrict pair accuracy by acquisition condition")
    for index in args.report_conditions:
        evaluation_data = per_condition_test.get(index)
        if evaluation_data is None:
            evaluation_data = load_condition(index, args)
            if args.target_source == "base-model":
                generate_base_model_targets(
                    base_denoiser,
                    evaluation_data,
                    args.target_generation_batch_size,
                    f"condition-{index}-evaluation",
                )
        metrics = evaluate_condition(
            best_denoiser,
            teacher,
            index,
            args,
            data=evaluation_data,
        )
        unmixing_metrics = None
        if dual_guidance:
            unmixing_metrics = evaluate_unmixing_condition(
                best_denoiser,
                unmixer,
                reference_centroids,
                index,
                args,
                data=evaluation_data,
            )
        if args.split_mode == "within-condition":
            data_role = "independent_test_subset"
        else:
            data_role = (
                "train"
                if index in args.train_conditions
                else "validation"
                if index in args.validation_conditions
                else "test"
                if index in args.test_conditions
                else "report_only"
            )
        condition_results[str(index)] = {
            "power_mw": CONDITION_DETAILS[index][0],
            "integration_ms": CONDITION_DETAILS[index][1],
            "energy_mj": CONDITION_DETAILS[index][2],
            "data_role": data_role,
            **metrics,
        }
        if unmixing_metrics is not None:
            condition_results[str(index)]["unmixing_proxy"] = (
                unmixing_metrics
            )
        print(
            f"Dataset {index} ({condition_results[str(index)]['data_role']}): "
            f"original={metrics['original_denoising']['subset_accuracy'] * 100:.2f}%, "
            f"adapted={metrics['adapted_denoising']['subset_accuracy'] * 100:.2f}%, "
            f"delta={metrics['adapted_minus_original_percentage_points']:+.2f} pp"
        )
        if unmixing_metrics is not None:
            print(
                "  unmixing proxy pair: "
                f"{unmixing_metrics['original_denoising']['reference_pair_accuracy'] * 100:.2f}% -> "
                f"{unmixing_metrics['adapted_denoising']['reference_pair_accuracy'] * 100:.2f}% "
                f"({unmixing_metrics['reference_pair_accuracy_delta_pp']:+.2f} pp); "
                "endpoint SAM delta="
                f"{unmixing_metrics['reference_sam_delta_degrees']:+.3f} deg"
            )

    mean_summary = {
        "original_subset_accuracy": float(
            np.mean(
                [
                    value["original_denoising"]["subset_accuracy"]
                    for value in condition_results.values()
                ]
            )
        ),
        "adapted_subset_accuracy": float(
            np.mean(
                [
                    value["adapted_denoising"]["subset_accuracy"]
                    for value in condition_results.values()
                ]
            )
        ),
        "original_macro_f1": float(
            np.mean(
                [
                    value["original_denoising"]["macro_f1"]
                    for value in condition_results.values()
                ]
            )
        ),
        "adapted_macro_f1": float(
            np.mean(
                [
                    value["adapted_denoising"]["macro_f1"]
                    for value in condition_results.values()
                ]
            )
        ),
    }
    mean_summary["adapted_minus_original_percentage_points"] = 100.0 * (
        mean_summary["adapted_subset_accuracy"]
        - mean_summary["original_subset_accuracy"]
    )
    if dual_guidance:
        proxy_values = [
            value["unmixing_proxy"] for value in condition_results.values()
        ]
        mean_summary.update(
            {
                "original_unmixing_proxy_pair_accuracy": float(
                    np.mean(
                        [
                            value["original_denoising"][
                                "reference_pair_accuracy"
                            ]
                            for value in proxy_values
                        ]
                    )
                ),
                "adapted_unmixing_proxy_pair_accuracy": float(
                    np.mean(
                        [
                            value["adapted_denoising"][
                                "reference_pair_accuracy"
                            ]
                            for value in proxy_values
                        ]
                    )
                ),
                "original_unmixing_proxy_sam_degrees": float(
                    np.mean(
                        [
                            value["original_denoising"][
                                "reference_sam_degrees"
                            ]
                            for value in proxy_values
                        ]
                    )
                ),
                "adapted_unmixing_proxy_sam_degrees": float(
                    np.mean(
                        [
                            value["adapted_denoising"][
                                "reference_sam_degrees"
                            ]
                            for value in proxy_values
                        ]
                    )
                ),
            }
        )
        mean_summary["unmixing_proxy_pair_accuracy_delta_pp"] = 100.0 * (
            mean_summary["adapted_unmixing_proxy_pair_accuracy"]
            - mean_summary["original_unmixing_proxy_pair_accuracy"]
        )
        mean_summary["unmixing_proxy_sam_delta_degrees"] = (
            mean_summary["adapted_unmixing_proxy_sam_degrees"]
            - mean_summary["original_unmixing_proxy_sam_degrees"]
        )
    print(
        "Condition-mean test accuracy: "
        f"original={mean_summary['original_subset_accuracy'] * 100:.2f}%, "
        f"adapted={mean_summary['adapted_subset_accuracy'] * 100:.2f}%, "
        f"delta={mean_summary['adapted_minus_original_percentage_points']:+.2f} pp"
    )
    if dual_guidance:
        print(
            "Condition-mean unmixing proxy pair accuracy: "
            f"{mean_summary['original_unmixing_proxy_pair_accuracy'] * 100:.2f}% -> "
            f"{mean_summary['adapted_unmixing_proxy_pair_accuracy'] * 100:.2f}% "
            f"({mean_summary['unmixing_proxy_pair_accuracy_delta_pp']:+.2f} pp)"
        )
        print(
            "Condition-mean endpoint proxy SAM: "
            f"{mean_summary['original_unmixing_proxy_sam_degrees']:.3f} -> "
            f"{mean_summary['adapted_unmixing_proxy_sam_degrees']:.3f} deg "
            f"({mean_summary['unmixing_proxy_sam_delta_degrees']:+.3f} deg)"
        )

    if dual_guidance:
        checkpoint_summary = {
            "policy": (
                "classification and proxy-unmixing non-degradation; "
                "proxy endpoint SAM non-increase"
            ),
            "reference_validation_accuracy": float(
                checkpoint.reference_accuracy
            ),
            "reference_validation_unmixing_proxy_accuracy": float(
                checkpoint.reference_unmixing_accuracy
            ),
            "reference_validation_endpoint_proxy_sam_degrees": float(
                checkpoint.reference_endpoint_sam
            ),
            "safe_constraint_met": bool(checkpoint.safe_constraint_met),
            "best_candidate_score": float(checkpoint.best_candidate_score),
            "selected_safe_score": (
                float(checkpoint.best_safe_score)
                if np.isfinite(checkpoint.best_safe_score)
                else None
            ),
            "selected_validation_metrics": checkpoint.selected_metrics,
            "selection_reason": selection_reason,
            "candidate_path": candidate_path,
            "selected_path": best_path,
        }
    else:
        checkpoint_summary = {
            "policy": "classification non-degradation",
            "reference_validation_accuracy": float(reference_accuracy),
            "safe_constraint_met": bool(checkpoint.safe_constraint_met),
            "best_candidate_validation_accuracy": float(
                checkpoint.best_candidate_accuracy
            ),
            "best_candidate_validation_loss": float(
                checkpoint.best_candidate_loss
            ),
            "selected_safe_validation_accuracy": (
                float(checkpoint.best_safe_accuracy)
                if np.isfinite(checkpoint.best_safe_accuracy)
                else None
            ),
            "selection_reason": selection_reason,
            "candidate_path": candidate_path,
            "selected_path": best_path,
        }

    write_json(
        os.path.join(result_path, "metrics.json"),
        {
            "model": (
                "MultiQualityUnpairedDualGuidedDenoiser"
                if args.multi_quality_unpaired
                else (
                    "DualGuidedRealDomainDenoiser"
                    if dual_guidance
                    else "RecognitionGuidedRealDomainDenoiser"
                )
            ),
            "pretrained_denoiser": pretrained_path,
            "teacher_model": args.teacher_model,
            "unmixing_weights": args.unmixing_weights,
            "unmixing_ground_truth_warning": (
                "Measured plastic mixtures have no sample-aligned endpoint or "
                "abundance truth. Unmixing selection uses pure-spectrum "
                "centroid proxy metrics and frozen-unmixer consistency only."
                if dual_guidance
                else None
            ),
            "denoising_target_alignment": {
                "source": args.target_source,
                "checkpoint": pretrained_path,
                "preprocessing": (
                    "MAT *_norm values -> float32 -> channel expansion; "
                    "no additional normalization"
                ),
                "validation_audit": target_alignment,
                "mae_tolerance": args.target_mae_tolerance,
                "max_abs_tolerance": args.target_max_abs_tolerance,
            },
            "multi_quality_supervision": (
                {
                    "pairing_status": (
                        "group-level only: material pair plus particle-size "
                        "block; no cross-condition row pairing"
                    ),
                    "high_quality_condition": args.high_quality_condition,
                    "reference_bank_source": "training split only",
                    "reference_bank_samples": unpaired_resources[
                        "reference_bank_samples"
                    ],
                    "group_count": unpaired_resources["group_count"],
                    "condition_severity": unpaired_resources[
                        "condition_severity"
                    ],
                    "paired_supervision": (
                        "online condition-scaled degradation of a "
                        "high-quality reference, supervised by that same "
                        "reference"
                    ),
                    "real_supervision": (
                        "same-spectrum base-denoiser fidelity plus "
                        "high-quality group mean/std alignment"
                    ),
                }
                if args.multi_quality_unpaired
                else None
            ),
            "spectral_self_similarity": {
                "enabled": bool(args.self_similarity_weight > 0),
                "measured_target_weight": args.self_similarity_weight,
                "synthetic_reference_weight": (
                    args.synthetic_self_similarity_weight
                ),
                "mask_source": (
                    "reference spectrum first-plus-second derivative; "
                    "prediction never selects supervised locations"
                ),
                "divergence": "symmetric Jensen-Shannon plus probability L1",
                "patch_size": args.self_similarity_patch_size,
                "search_radius": args.self_similarity_search_radius,
                "search_step": args.self_similarity_search_step,
                "peak_fraction": args.self_similarity_peak_fraction,
                "temperature": args.self_similarity_temperature,
                "l1_weight": args.self_similarity_l1_weight,
            },
            "condition_split": split_description,
            "initial_validation_losses": {
                key: float(value)
                for key, value in initial_validation.items()
            },
            "best_test_losses": {
                key: float(value) for key, value in test_losses.items()
            },
            "checkpoint_selection": checkpoint_summary,
            "condition_mean": mean_summary,
            "conditions": condition_results,
        },
    )
    write_json(
        os.path.join(result_path, "config.json"), json_safe_args(args)
    )
    print(f"\nBest inference model: {best_path}")
    print(f"Results: {result_path}")

    del train_data, validation_data, test_data
    gc.collect()


if __name__ == "__main__":
    main()
