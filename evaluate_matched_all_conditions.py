"""Frozen evaluation of the matched separators on all nine real conditions.

Keeps models in memory to avoid repeated CUDA startup. Reuses verified condition8
outputs. Never fits weights, thresholds, references, or chooses checkpoints.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime

if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from fixed_splits import file_sha256, read_manifest
from train_matched_separators import verify_matched
from evaluate_rssnet_classification import endpoint_labels, score_labels


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--output", required=True, help="New directory, never overwrite old results")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("Positive batch size required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "evaluation_source.py")
    shutil.copy2("predict_rssnet_unmixing.py", output / "prediction_source.py")
    save(output / "status.json", {"state": "initializing", "time": datetime.now().isoformat()})
    try:
        summary = read(Path(args.experiment) / "summary.json")
        paths = {k: Path(summary["runs"][k]) for k in ("paper", "rssnet")}
        audit = verify_matched(paths["paper"], paths["rssnet"])
        manifest_path = "dataset/splits/plastic_condition_holdout_1-7_8_9_v1.csv"
        manifest, _ = read_manifest(manifest_path)
        manifest_hash = file_sha256(manifest_path)
        frozen = {k: file_sha256(v / "checkpoint/best.weights.h5") for k, v in paths.items()}
        # Reference files must agree numerically, not merely have the same names.
        with np.load(paths["paper"] / "reference.npz", allow_pickle=False) as f:
            references, names = f["reference"].copy(), f["names"].tolist()
        with np.load(paths["rssnet"] / "reference.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(references, f["reference"])
            if names != f["names"].tolist():
                raise ValueError("Different reference category order")
        protocol = dict(args=vars(args), audit=audit, checkpoints=frozen,
            split_manifest_sha256=manifest_hash,
            note="Both separators synthetic-only trained. Historical manifest calls conditions1-7 train; here evaluation-only. All nine conditions previously explored; not new external blind tests.")
        save(output / "protocol.json", protocol)
        from predict_rssnet_unmixing import main as predict
        cache, rows, details, locations = {}, [], {}, {}
        for kind, run in paths.items():
            details[kind], locations[kind] = {}, {}
            for condition in range(1, 10):
                role = "train" if condition < 8 else "validation" if condition == 8 else "test"
                if file_sha256(run / "checkpoint/best.weights.h5") != frozen[kind]:
                    raise ValueError("Frozen checkpoint changed during evaluation")
                destination = output / kind / f"condition{condition}"
                cached = run / "result/condition8_frozen"
                reuse = condition == 8 and (cached / "predictions.npz").is_file()
                source = cached if reuse else destination
                save(output / "status.json", dict(state="running", model=kind, condition=condition,
                     reuse_existing=reuse, finished=len(rows), total=18))
                print(f"Evaluating {kind} Dataset {condition}; reuse={reuse}", flush=True)
                if not reuse:
                    cmd = ["--run-dir", str(run), "--output", str(destination), "--condition", str(condition),
                           "--partition", role, "--batch-size", str(args.batch_size)]
                    if args.cpu:
                        cmd.append("--cpu")
                    predict(cmd, model_cache=cache)
                prov = read(source / "provenance.json")
                if (prov["model_sha256"] != frozen[kind] or prov["split_manifest_sha256"] != manifest_hash or
                    prov["args"]["condition"] != condition or prov["args"]["partition"] != role or
                    prov["args"]["limit"] or prov["args"]["denoiser"]):
                    raise ValueError("Cached/evaluated protocol mismatch")
                for field, pattern in (("input_sha256", "dataset/dataset_mixture_denoised/dataset{}_denoised.mat"),
                                       ("raw_sha256", "dataset/dataset_mixture_raw/dataset{}_norm.mat")):
                    if prov[field] != file_sha256(pattern.format(condition)):
                        raise ValueError("Evaluation source changed")
                with np.load(source / "predictions.npz", allow_pickle=False) as f:
                    ids, correct = f["sample_index"], f["pair_correct"]
                    expected = {i for i, record in manifest[condition].items() if record["partition"] == role}
                    if len(np.unique(ids)) != len(ids) or set(map(int, ids)) != expected:
                        raise ValueError("Incomplete or duplicate condition row coverage")
                    from evaluate_unmixing_conditions import build_true_indices
                    true_indices = build_true_indices(f["pairs"], names)
                    truth = endpoint_labels(true_indices, len(names))
                    labels = endpoint_labels(f["endpoint_class_indices"], len(names))
                    metric, confusion, recomputed = score_labels(truth, labels, names)
                    np.testing.assert_array_equal(correct, recomputed)
                destination.mkdir(parents=True, exist_ok=True)
                save(destination / "classification_metrics.json", metric)
                save(destination / "pair_confusion.json", confusion)
                save(destination / "source.json", {"predictions": str(source / "predictions.npz"),
                    "predictions_sha256": file_sha256(source / "predictions.npz"), "reused": reuse})
                proxy = read(source / "metrics.json")["separator_reference_proxy"]
                row = dict(condition=condition, model=kind, samples=metric["samples"],
                    strict_pair_accuracy=metric["strict_pair_accuracy"], macro_f1=metric["macro_f1_7labels"],
                    micro_f1=metric["micro_f1_7labels"], forced_pair_accuracy=proxy["reference_pair_accuracy"],
                    reference_sam_degrees=proxy["reference_sam_degrees"])
                rows.append(row)
                details[kind][str(condition)] = row
                locations[kind][condition] = source
                save(output / "partial_summary.json", details)
                print(f"Finished {kind} Dataset{condition}: strict={row['strict_pair_accuracy']:.4%}", flush=True)
        for condition in range(1, 10):
            # Verify the actual paired input arrays and row order, not just paths.
            with np.load(locations["paper"][condition] / "predictions.npz") as a, np.load(locations["rssnet"][condition] / "predictions.npz") as b:
                for key in ("input", "sample_index", "pairs"):
                    np.testing.assert_array_equal(a[key], b[key])
        means = {k: {field: float(np.mean([v[field] for v in details[k].values()]))
                     for field in ("strict_pair_accuracy", "macro_f1", "micro_f1", "reference_sam_degrees")}
                 for k in paths}
        save(output / "summary.json", dict(protocol=protocol, per_condition=details, condition_means=means))
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save(output / "status.json", dict(state="completed", finished=18, summary=str(output / "summary.json")))
        print(f"Completed: {output}", flush=True)
    except Exception as exc:
        save(output / "status.json", dict(state="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
