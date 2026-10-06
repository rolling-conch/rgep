"""Frozen, same-input comparison: direct paper classifier vs RSSNet retrieval.

Reuses existing Dataset 8/9 endpoint predictions. No fitting or threshold tuning.
Pair confusion has 21 true rows and 22 prediction columns: the extra column
retains invalid/duplicate endpoint classifications rather than dropping errors.
"""
import argparse
import json
import os
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, jaccard_score
from decoupled_data import DEFAULT_CATEGORIES, load_labeled_mixtures, top_k_multihot
from fixed_splits import file_sha256, read_manifest


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def endpoint_labels(indices, classes):
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 2 or np.any(indices < -1) or np.any(indices >= classes):
        raise ValueError("Expected two valid class indices or -1 per sample")
    labels = np.zeros((len(indices), classes), dtype=np.int32)
    for i, row in enumerate(indices):
        labels[i, row[row >= 0]] = 1
    return labels


def score_labels(truth, prediction, names):
    truth, prediction = np.asarray(truth, dtype=int), np.asarray(prediction, dtype=int)
    if truth.shape != prediction.shape or not np.isin(prediction, [0, 1]).all() or not np.all(truth.sum(axis=1) == 2):
        raise ValueError("Expected binary labels and two true classes per spectrum")
    pairs = list(combinations(range(len(names)), 2))
    pair_names = [names[a] + names[b] for a, b in pairs]
    pair_lookup = {pair: i for i, pair in enumerate(pairs)}
    ti = np.asarray([pair_lookup[tuple(np.flatnonzero(row))] for row in truth])
    pi = np.asarray([pair_lookup.get(tuple(np.flatnonzero(row)), len(pairs)) for row in prediction])
    matrix = np.zeros((len(pairs), len(pairs) + 1), dtype=int)
    np.add.at(matrix, (ti, pi), 1)
    correct = np.all(truth == prediction, axis=1)
    metrics = dict(samples=len(truth), strict_pair_accuracy=float(correct.mean()),
        macro_f1_7labels=float(f1_score(truth, prediction, average="macro", zero_division=0)),
        micro_f1_7labels=float(f1_score(truth, prediction, average="micro", zero_division=0)),
        macro_precision_7labels=float(precision_score(truth, prediction, average="macro", zero_division=0)),
        macro_recall_7labels=float(recall_score(truth, prediction, average="macro", zero_division=0)),
        sample_jaccard=float(jaccard_score(truth, prediction, average="samples", zero_division=0)),
        macro_f1_21pairs=float(f1_score(ti, pi, labels=list(range(len(pairs))), average="macro", zero_division=0)),
        invalid_pair_count=int((pi == len(pairs)).sum()),
        per_class=classification_report(truth, prediction, target_names=names, output_dict=True, zero_division=0),
        per_pair_accuracy={name: float(correct[ti == i].mean()) if np.any(ti == i) else None
                           for i, name in enumerate(pair_names)})
    confusion = dict(true_labels=pair_names, predicted_labels=pair_names + ["INVALID"], counts=matrix.tolist(),
        note="INVALID includes duplicate endpoint classes or fewer than two valid endpoint predictions; all samples retained")
    return metrics, confusion, correct


def plot_confusion(confusion, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(confusion["counts"])
    fractions = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    im = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), confusion["predicted_labels"], rotation=90)
    ax.set_yticks(np.arange(matrix.shape[0]), confusion["true_labels"])
    ax.set_xlabel("Predicted combination (INVALID retained)")
    ax.set_ylabel("True combination")
    ax.set_title(title)
    for i, j in zip(*np.nonzero(matrix)):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                color="white" if fractions[i, j] > .5 else "black")
    fig.colorbar(im, ax=ax, label="Fraction within true combination", shrink=.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="run/RSSNetInspired_Plastic_20260920_075646_675976")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf
    import tools
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    run, output = Path(args.run_dir), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / "evaluation_source.py")
    directories = {8: "validation8_paper_denoised", 9: "test9_paper_denoised"}
    provenance = {c: json.loads((run / "result" / d / "provenance.json").read_text(encoding="utf-8"))
                  for c, d in directories.items()}
    for key in ("model_sha256", "paper_model_sha256", "split_manifest_sha256"):
        if provenance[8][key] != provenance[9][key]:
            raise ValueError("Different validation/test checkpoints or manifest")
    paper = provenance[8]["args"]["paper_model"]
    if file_sha256(paper) != provenance[8]["paper_model_sha256"]:
        raise ValueError("Paper classifier weights changed")
    print("Loading frozen paper classification head", flush=True)
    full = tf.keras.models.load_model(paper, custom_objects=tools.load_custom_objects("CSAM_ResUNet_classification"), compile=False)
    classifier = tf.keras.Model(full.input, full.get_layer("classification").output)
    classifier.trainable = False
    names = list(DEFAULT_CATEGORIES)
    with np.load(run / "reference.npz", allow_pickle=False) as f:
        references = f["reference"]
        if f["names"].tolist() != names:
            raise ValueError("Reference category order changed")
    summary = {}
    for condition, dirname in directories.items():
        src = run / "result" / dirname
        prov = provenance[condition]
        if prov["args"]["denoiser"] or prov["args"]["limit"]:
            raise ValueError("Expected full supplied-MAT evaluation, not a debug or new denoiser pipeline")
        manifest, _ = read_manifest(prov["args"]["split_manifest"], prov["split_manifest_sha256"])
        prediction_path = src / "predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as f:
            x, ids, pairs = f["input"], f["sample_index"], f["pairs"]
            indices, saved_correct = f["endpoint_class_indices"], f["pair_correct"]
            if f["names"].tolist() != names:
                raise ValueError("Prediction category order changed")
            # Check fixed retrieval rule against stored components and current reference.
            components = f["components"]
            norms = np.linalg.norm(components, axis=1)
            similarities = np.einsum("nlc,kl->nck", components / np.maximum(norms[:, None], 1e-8), references)
            recomputed = np.where(norms > 1e-8, np.argmax(similarities, axis=-1), -1)
            np.testing.assert_array_equal(indices, recomputed)
            del components, similarities
        expected_role = "validation" if condition == 8 else "test"
        expected_ids = {i for i, record in manifest[condition].items() if record["partition"] == expected_role}
        if len(np.unique(ids)) != len(ids) or set(map(int, ids)) != expected_ids:
            raise ValueError("Cached prediction row coverage differs from fixed split")
        mat_path = prov["args"]["spectra_pattern"].format(index=condition)
        if file_sha256(mat_path) != prov["input_sha256"]:
            raise ValueError("Processed input file changed")
        all_x, all_y, all_pairs = load_labeled_mixtures(mat_path)
        np.testing.assert_array_equal(x, all_x[ids])
        np.testing.assert_array_equal(pairs, all_pairs[ids])
        truth = all_y[ids].astype(int)
        del all_x, all_y, all_pairs
        rss_labels = endpoint_labels(indices, len(names))
        np.testing.assert_array_equal(saved_correct, np.all(rss_labels == truth, axis=1))
        print(f"Dataset {condition}: {len(x)} matched rows, direct classifier inference", flush=True)
        probabilities = classifier.predict(x, batch_size=args.batch_size, verbose=1)
        if probabilities.shape != truth.shape or not np.isfinite(probabilities).all():
            raise ValueError("Invalid classifier probabilities")
        direct_labels = top_k_multihot(probabilities, k=2)
        dest = output / f"dataset{condition}"
        dest.mkdir()
        results = {}
        corrects = {}
        for route, labels in (("paper_direct", direct_labels), ("rssnet_retrieval", rss_labels)):
            metric, confusion, correct = score_labels(truth, labels, names)
            results[route], corrects[route] = metric, correct
            save_json(dest / f"{route}_metrics.json", metric)
            save_json(dest / f"{route}_confusion.json", confusion)
            plot_confusion(confusion, f"Dataset {condition} | {route} | accuracy {100 * metric['strict_pair_accuracy']:.2f}%",
                           dest / f"{route}_confusion.png")
        a, b = corrects["paper_direct"], corrects["rssnet_retrieval"]
        results["paired_counts"] = dict(both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()),
            rssnet_only_correct=int((~a & b).sum()), paper_only_correct=int((a & ~b).sum()))
        np.savez_compressed(dest / "classification_predictions.npz", sample_index=ids, pairs=pairs,
            truth=truth, direct_probabilities=probabilities, direct_labels=direct_labels, rssnet_labels=rss_labels)
        save_json(dest / "comparison.json", results)
        summary[str(condition)] = {route: {key: results[route][key] for key in
            ("samples", "strict_pair_accuracy", "macro_f1_7labels", "micro_f1_7labels", "macro_f1_21pairs", "invalid_pair_count")}
            for route in ("paper_direct", "rssnet_retrieval")}
        summary[str(condition)]["paired_counts"] = results["paired_counts"]
        summary[str(condition)]["source_prediction_sha256"] = file_sha256(prediction_path)
        print(json.dumps(summary[str(condition)], indent=2), flush=True)
    summary["protocol"] = dict(no_fitting=True, direct_rule="paper classification head top-2",
        rssnet_rule="independent endpoint nearest-reference; duplicates/invalid endpoints are failures",
        reference_sha256=file_sha256(run / "reference.npz"), paper_model_sha256=file_sha256(paper),
        tensorflow_version=tf.__version__, cpu=args.cpu,
        warning="Dataset 9 already inspected in unmixing experiments; frozen supplementary comparison, not a new unseen test. Different training data and same-source RSSNet reference library preclude architecture-only causal conclusions.")
    save_json(output / "summary.json", summary)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
