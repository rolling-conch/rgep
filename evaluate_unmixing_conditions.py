"""Evaluate real-spectrum unmixing across all acquisition conditions.

Real mixtures do not contain sample-aligned endpoint ground truth.  Therefore
this script reports (1) mixture reconstruction consistency and (2) proxy
endpoint metrics against measured pure-spectrum class centroids.  The latter
are permutation-invariant and scale-invariant, but are not substitutes for
endpoint metrics computed against paired ground truth.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import argparse
import csv
import gc

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

import tools
from decoupled_data import (
    DEFAULT_CATEGORIES,
    load_labeled_mixtures,
    pair_definitions,
    save_json,
    spectral_metrics,
)


MODEL_NAME = "MSAG_CSAM_ResUNet_unmixing_only"
CONDITIONS = (
    (1, 150, 5000, 750),
    (2, 150, 3000, 450),
    (3, 150, 1000, 150),
    (4, 100, 5000, 500),
    (5, 100, 3000, 300),
    (6, 100, 1000, 100),
    (7, 50, 5000, 250),
    (8, 50, 3000, 150),
    (9, 50, 1000, 50),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the independent unmixer on data sets 1-9"
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--data_dir",
        default="./dataset/dataset_mixture_denoised",
    )
    parser.add_argument(
        "--filename_pattern",
        default="dataset{index}_denoised.mat",
    )
    parser.add_argument("--input_suffix", default="_norm")
    parser.add_argument(
        "--reference_path",
        default="./dataset/dataset_150mW_5000ms.mat",
    )
    parser.add_argument("--output_dir", default="./unmixing_condition_evaluation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_features", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=19580920)
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--attention_ratio", type=int, default=8)
    parser.add_argument("--peak_weight_scale", type=float, default=10.0)
    parser.add_argument("--peak_weight_exponent", type=float, default=2.0)
    parser.add_argument("--sparse_loss_weight", type=float, default=0.002)
    parser.add_argument("--derivative_loss_weight", type=float, default=0.1)
    parser.add_argument("--sparse_threshold", type=float, default=1e-4)
    parser.add_argument("--unmixing_loss_weight", type=float, default=1.0)
    parser.add_argument("--mixture_consistency_weight", type=float, default=0.05)
    parser.add_argument(
        "--MP_type", nargs="+", default=list(DEFAULT_CATEGORIES)
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    args.normalize_peak_weights = True
    args.train_data_ratio = 0.7
    args.num_spectra = 33600
    args.num_substances = len(args.MP_type)
    return args


def configure_runtime(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float64).eps)


def load_reference_centroids(path, categories, num_features):
    data = loadmat(path, variable_names=list(categories))
    centroids = []
    for category in categories:
        if category not in data:
            raise KeyError(f"Reference category {category!r} is missing in {path}")
        spectra = np.asarray(data[category], dtype=np.float64)
        spectra = np.squeeze(spectra)
        if spectra.ndim == 1:
            spectra = spectra[np.newaxis, :]
        if spectra.shape[1] != num_features and spectra.shape[0] == num_features:
            spectra = spectra.T
        if spectra.shape[1] != num_features:
            raise ValueError(
                f"Reference {category!r} has shape {spectra.shape}; "
                f"expected (*, {num_features})"
            )
        centroids.append(np.mean(spectra, axis=0))
    return normalize_rows(np.stack(centroids, axis=0))


def build_true_indices(pair_names, categories):
    category_to_index = {
        category: index for index, category in enumerate(categories)
    }
    lookup = {
        pair_key: (category_to_index[first], category_to_index[second])
        for pair_key, first, second in pair_definitions(categories)
    }
    try:
        return np.asarray([lookup[str(pair)] for pair in pair_names], dtype=np.int32)
    except KeyError as error:
        raise KeyError(f"Unknown mixture key: {error.args[0]}") from error


def reference_proxy_metrics(
    component_a,
    component_b,
    pair_names,
    categories,
    reference_centroids,
):
    normalized_a = normalize_rows(component_a)
    normalized_b = normalize_rows(component_b)
    cosine_a = np.clip(normalized_a @ reference_centroids.T, -1.0, 1.0)
    cosine_b = np.clip(normalized_b @ reference_centroids.T, -1.0, 1.0)
    sample_count, category_count = cosine_a.shape
    rows = np.arange(sample_count)
    true_indices = build_true_indices(pair_names, categories)
    true_first = true_indices[:, 0]
    true_second = true_indices[:, 1]

    direct_score = cosine_a[rows, true_first] + cosine_b[rows, true_second]
    swapped_score = cosine_a[rows, true_second] + cosine_b[rows, true_first]
    use_swapped = swapped_score > direct_score
    assigned_a = np.where(
        use_swapped,
        cosine_a[rows, true_second],
        cosine_a[rows, true_first],
    )
    assigned_b = np.where(
        use_swapped,
        cosine_b[rows, true_first],
        cosine_b[rows, true_second],
    )
    assigned_cosine = np.concatenate((assigned_a, assigned_b))
    assigned_sam = np.degrees(
        np.arccos(np.clip(assigned_cosine, -1.0, 1.0))
    )

    ordered_scores = cosine_a[:, :, np.newaxis] + cosine_b[:, np.newaxis, :]
    diagonal = np.arange(category_count)
    ordered_scores[:, diagonal, diagonal] = -np.inf
    best_ordered = np.argmax(
        ordered_scores.reshape(sample_count, -1), axis=1
    )
    predicted_first = best_ordered // category_count
    predicted_second = best_ordered % category_count
    predicted_pair = np.sort(
        np.stack((predicted_first, predicted_second), axis=1), axis=1
    )
    true_pair = np.sort(true_indices, axis=1)
    pair_correct = np.all(predicted_pair == true_pair, axis=1)

    nearest_a = np.argmax(cosine_a, axis=1)
    nearest_b = np.argmax(cosine_b, axis=1)
    independently_distinct = nearest_a != nearest_b

    norm_a = np.linalg.norm(
        np.asarray(component_a).reshape(sample_count, -1), axis=1
    )
    norm_b = np.linalg.norm(
        np.asarray(component_b).reshape(sample_count, -1), axis=1
    )
    smaller_fraction = np.minimum(norm_a, norm_b) / np.maximum(
        norm_a + norm_b, np.finfo(np.float64).eps
    )

    per_pair = {}
    for pair in np.unique(pair_names):
        selection = pair_names == pair
        pair_assigned_cosine = np.concatenate(
            (assigned_a[selection], assigned_b[selection])
        )
        per_pair[str(pair)] = {
            "reference_pair_accuracy": float(np.mean(pair_correct[selection])),
            "reference_cosine_similarity": float(
                np.mean(pair_assigned_cosine)
            ),
            "reference_sam_degrees": float(
                np.mean(
                    np.degrees(
                        np.arccos(
                            np.clip(pair_assigned_cosine, -1.0, 1.0)
                        )
                    )
                )
            ),
        }

    return {
        "reference_pair_accuracy": float(np.mean(pair_correct)),
        "reference_cosine_similarity": float(np.mean(assigned_cosine)),
        "reference_sam_degrees": float(np.mean(assigned_sam)),
        "reference_sam_median_degrees": float(np.median(assigned_sam)),
        "independent_endpoint_labels_distinct_fraction": float(
            np.mean(independently_distinct)
        ),
        "smaller_component_fraction_mean": float(np.mean(smaller_fraction)),
        "smaller_component_fraction_median": float(
            np.median(smaller_fraction)
        ),
        "collapsed_sample_fraction_at_0_05": float(
            np.mean(smaller_fraction < 0.05)
        ),
        "per_pair": per_pair,
    }


def write_summary_csv(path, rows):
    columns = (
        "dataset",
        "power_mw",
        "integration_ms",
        "energy_mj",
        "samples",
        "mixture_r2",
        "mixture_rmse",
        "mixture_mae",
        "mixture_cosine_similarity",
        "mixture_sam_degrees",
        "reference_pair_accuracy",
        "reference_cosine_similarity",
        "reference_sam_degrees",
        "smaller_component_fraction_median",
        "collapsed_sample_fraction_at_0_05",
    )
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def main():
    args = parse_args()
    configure_runtime(args.seed)
    categories = list(args.MP_type)
    reference_centroids = load_reference_centroids(
        args.reference_path, categories, args.num_features
    )

    model, _ = tools.built_model(MODEL_NAME, args)
    model.load_weights(args.weights)
    os.makedirs(args.output_dir, exist_ok=True)

    summary = []
    for index, power, integration, energy in CONDITIONS:
        input_path = os.path.join(
            args.data_dir,
            args.filename_pattern.format(index=index),
        )
        print(
            f"\nEvaluating data set {index}: {power} mW, "
            f"{integration} ms, {energy} mJ"
        )
        mixtures, _, pair_names = load_labeled_mixtures(
            input_path,
            categories=categories,
            num_features=args.num_features,
            suffix=args.input_suffix,
        )
        component_a, component_b = model.predict(
            mixtures, batch_size=args.batch_size, verbose=1
        )
        reconstructed = component_a + component_b
        mixture = spectral_metrics(mixtures, reconstructed)
        reference = reference_proxy_metrics(
            component_a,
            component_b,
            pair_names,
            categories,
            reference_centroids,
        )
        metrics = {
            "dataset": index,
            "power_mw": power,
            "integration_ms": integration,
            "energy_mj": energy,
            "samples": int(len(mixtures)),
            "input_path": input_path,
            "reference_path": args.reference_path,
            "ground_truth_note": (
                "No sample-aligned endpoint truth is available. Reference "
                "metrics use measured class-centroid spectra and are proxy "
                "metrics, not paired endpoint reconstruction metrics."
            ),
            "mixture_consistency": mixture,
            "reference_proxy": reference,
        }
        save_json(
            os.path.join(args.output_dir, f"dataset{index}_metrics.json"),
            metrics,
        )
        row = {
            "dataset": index,
            "power_mw": power,
            "integration_ms": integration,
            "energy_mj": energy,
            "samples": int(len(mixtures)),
            "mixture_r2": mixture["r2"],
            "mixture_rmse": mixture["rmse"],
            "mixture_mae": mixture["mae"],
            "mixture_cosine_similarity": mixture["cosine_similarity"],
            "mixture_sam_degrees": mixture["sam_degrees"],
            "reference_pair_accuracy": reference["reference_pair_accuracy"],
            "reference_cosine_similarity": reference[
                "reference_cosine_similarity"
            ],
            "reference_sam_degrees": reference["reference_sam_degrees"],
            "smaller_component_fraction_median": reference[
                "smaller_component_fraction_median"
            ],
            "collapsed_sample_fraction_at_0_05": reference[
                "collapsed_sample_fraction_at_0_05"
            ],
        }
        summary.append(row)
        print(
            f"data set {index}: mixture R2={mixture['r2']:.6f}, "
            f"mixture SAM={mixture['sam_degrees']:.3f} deg, "
            f"reference pair accuracy="
            f"{reference['reference_pair_accuracy']:.6f}, "
            f"reference SAM={reference['reference_sam_degrees']:.3f} deg"
        )
        del mixtures, pair_names, component_a, component_b, reconstructed
        gc.collect()

    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    write_summary_csv(os.path.join(args.output_dir, "summary.csv"), summary)

    print("\nSummary")
    print("set  condition       mix_R2   mix_SAM  ref_pair  ref_SAM  collapse")
    for row in summary:
        condition = (
            f"{row['power_mw']}mW/{row['integration_ms'] // 1000}s"
        )
        print(
            f"{row['dataset']:>3}  {condition:<14} "
            f"{row['mixture_r2']:>7.4f}  "
            f"{row['mixture_sam_degrees']:>7.3f}  "
            f"{row['reference_pair_accuracy'] * 100:>7.2f}%  "
            f"{row['reference_sam_degrees']:>7.3f}  "
            f"{row['collapsed_sample_fraction_at_0_05'] * 100:>7.2f}%"
        )
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
