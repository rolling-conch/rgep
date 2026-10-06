"""Data loading and evaluation helpers for decoupled training.

The training MATLAB file is expected to contain one mixture matrix per pair
(for example ``PCPE``) and its two endpoint matrices (``PCPE_PC`` and
``PCPE_PE``).  All returned spectra use ``float32`` and the Keras layout
``(samples, features, 1)`` to avoid the duplicate high-memory containers used
by the legacy training script.
"""

import csv
import json
from itertools import combinations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


DEFAULT_CATEGORIES = ["PC", "PE", "PP", "PS", "PV", "PM", "PT"]


def _load_mat(path, variable_names=None):
    return {
        key: value
        for key, value in loadmat(
            path, variable_names=variable_names
        ).items()
        if not key.startswith("__")
    }


def pair_definitions(categories):
    """Return ``(pair_key, first, second)`` for every material pair."""
    return [(a + b, a, b) for a, b in combinations(categories, 2)]


def _as_spectra(value, key, num_features):
    """Normalize a MATLAB matrix to ``(samples, num_features)`` float32."""
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(
            f"MAT variable {key!r} must be a 2-D spectral matrix; "
            f"got shape {array.shape}."
        )
    if array.shape[1] != num_features and array.shape[0] == num_features:
        array = array.T
    if array.shape[1] != num_features:
        raise ValueError(
            f"MAT variable {key!r} has {array.shape[1]} features; "
            f"expected {num_features}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"MAT variable {key!r} contains NaN or Inf values.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _require_key(mat_data, key, path):
    if key not in mat_data:
        available = ", ".join(sorted(mat_data)[:12])
        raise KeyError(
            f"Missing variable {key!r} in {path}. "
            f"First available variables: {available}"
        )
    return mat_data[key]


def load_training_splits(
    path,
    categories=None,
    train_ratio=0.7,
    seed=19580920,
    num_features=1024,
    include_components=True,
):
    """Load aligned mixtures/endmembers and split every pair independently."""
    categories = list(categories or DEFAULT_CATEGORIES)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("categories must contain at least two unique names")

    pairs = pair_definitions(categories)
    requested_keys = [pair_key for pair_key, _, _ in pairs]
    if include_components:
        for pair_key, first, second in pairs:
            requested_keys.extend(
                (f"{pair_key}_{first}", f"{pair_key}_{second}")
            )
    mat_data = _load_mat(path, variable_names=requested_keys)
    category_to_index = {
        category: index for index, category in enumerate(categories)
    }
    data_keys = ("x", "a", "b", "y") if include_components else ("x", "y")
    buckets = {
        split: {key: [] for key in data_keys}
        for split in ("train", "val", "test")
    }
    total_spectra = 0

    for pair_key, first, second in pairs:
        mixture = _as_spectra(
            _require_key(mat_data, pair_key, path),
            pair_key,
            num_features,
        )
        if include_components:
            component_a_key = f"{pair_key}_{first}"
            component_b_key = f"{pair_key}_{second}"
            component_a = _as_spectra(
                _require_key(mat_data, component_a_key, path),
                component_a_key,
                num_features,
            )
            component_b = _as_spectra(
                _require_key(mat_data, component_b_key, path),
                component_b_key,
                num_features,
            )
            if not (
                mixture.shape[0] == component_a.shape[0]
                == component_b.shape[0]
            ):
                raise ValueError(
                    f"Sample count mismatch for {pair_key}: mixture "
                    f"{mixture.shape[0]}, A {component_a.shape[0]}, "
                    f"B {component_b.shape[0]}."
                )
        if mixture.shape[0] < 4:
            raise ValueError(
                f"Pair {pair_key} needs at least four samples for the "
                "train/validation/test split."
            )

        label = np.zeros(len(categories), dtype=np.float32)
        label[category_to_index[first]] = 1.0
        label[category_to_index[second]] = 1.0
        labels = np.repeat(label[np.newaxis, :], mixture.shape[0], axis=0)
        total_spectra += mixture.shape[0]

        if include_components:
            source_arrays = (mixture, component_a, component_b, labels)
        else:
            source_arrays = (mixture, labels)
        train_arrays = train_test_split(
            *source_arrays,
            test_size=round(1.0 - train_ratio, 10),
            shuffle=True,
            random_state=seed,
        )
        train_parts = train_arrays[0::2]
        temporary_parts = train_arrays[1::2]
        validation_arrays = train_test_split(
            *temporary_parts,
            test_size=1.0 / 3.0,
            shuffle=True,
            random_state=seed,
        )
        validation_parts = validation_arrays[0::2]
        test_parts = validation_arrays[1::2]

        for split, arrays in (
            ("train", train_parts),
            ("val", validation_parts),
            ("test", test_parts),
        ):
            for key, array in zip(data_keys, arrays):
                buckets[split][key].append(array)

    del mat_data
    splits = {}
    for split, arrays in buckets.items():
        splits[split] = {}
        for key in data_keys:
            value = np.concatenate(arrays[key], axis=0)
            if key in ("x", "a", "b"):
                value = value[..., np.newaxis]
            splits[split][key] = value
    return splits, total_spectra


def load_labeled_mixtures(
    path,
    categories=None,
    num_features=1024,
    suffix="_norm",
):
    """Load real/simulated mixture keys and generate their two-hot labels."""
    categories = list(categories or DEFAULT_CATEGORIES)
    pairs = pair_definitions(categories)
    requested_keys = [pair_key + suffix for pair_key, _, _ in pairs]
    mat_data = _load_mat(path, variable_names=requested_keys)
    category_to_index = {
        category: index for index, category in enumerate(categories)
    }
    spectra = []
    labels = []
    pair_names = []
    for pair_key, first, second in pairs:
        data_key = pair_key + suffix
        values = _as_spectra(
            _require_key(mat_data, data_key, path), data_key, num_features
        )
        label = np.zeros(len(categories), dtype=np.float32)
        label[category_to_index[first]] = 1.0
        label[category_to_index[second]] = 1.0
        spectra.append(values)
        labels.append(np.repeat(label[np.newaxis, :], values.shape[0], axis=0))
        pair_names.extend([pair_key] * values.shape[0])
    del mat_data
    return (
        np.concatenate(spectra, axis=0)[..., np.newaxis],
        np.concatenate(labels, axis=0),
        np.asarray(pair_names),
    )


def top_k_multihot(probabilities, k=2):
    """Convert probabilities to a multi-hot array with exactly ``k`` labels."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (samples, classes)")
    if not 1 <= k <= probabilities.shape[1]:
        raise ValueError("k must be between 1 and the number of classes")
    top_indices = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    predictions = np.zeros_like(probabilities, dtype=np.int32)
    rows = np.arange(probabilities.shape[0])[:, np.newaxis]
    predictions[rows, top_indices] = 1
    return predictions


def classification_metrics(y_true, probabilities, categories, k=2):
    """Return JSON-safe multi-label metrics and a detailed text report."""
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = top_k_multihot(probabilities, k=k)
    metrics = {
        "subset_accuracy": float(accuracy_score(y_true, predictions)),
        "sample_jaccard_accuracy": float(
            jaccard_score(y_true, predictions, average="samples", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, predictions, average="micro", zero_division=0)
        ),
    }
    try:
        metrics["macro_auroc"] = float(
            roc_auc_score(y_true, probabilities, average="macro")
        )
        metrics["micro_auroc"] = float(
            roc_auc_score(y_true, probabilities, average="micro")
        )
        per_class_auc = roc_auc_score(
            y_true, probabilities, average=None
        )
        metrics["per_class_auroc"] = {
            category: float(value)
            for category, value in zip(categories, per_class_auc)
        }
    except ValueError:
        metrics["macro_auroc"] = None
        metrics["micro_auroc"] = None
        metrics["per_class_auroc"] = {}

    report = classification_report(
        y_true,
        predictions,
        target_names=list(categories),
        digits=6,
        zero_division=0,
    )
    return metrics, report, predictions


def spectral_metrics(y_true, y_pred):
    """Compute global regression, cosine-similarity, and SAM metrics."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)
    mse = mean_squared_error(y_true, y_pred)
    epsilon = np.finfo(np.float64).eps
    dot = np.sum(y_true * y_pred, axis=1)
    denominator = (
        np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    )
    cosine = np.clip(dot / np.maximum(denominator, epsilon), -1.0, 1.0)
    sam = np.arccos(cosine)
    return {
        "r2": float(r2_score(y_true.reshape(-1), y_pred.reshape(-1))),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "cosine_similarity": float(np.mean(cosine)),
        "sam_radians": float(np.mean(sam)),
        "sam_degrees": float(np.degrees(np.mean(sam))),
    }


def save_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def save_history(path, history):
    """Write a Keras History dictionary without requiring pandas."""
    columns = list(history)
    row_count = max((len(history[column]) for column in columns), default=0)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch"] + columns)
        for index in range(row_count):
            writer.writerow(
                [index + 1]
                + [
                    history[column][index]
                    if index < len(history[column]) else ""
                    for column in columns
                ]
            )
