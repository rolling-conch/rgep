"""Reuse the completed online arm; train only its fixed-augmentation control.

Default is a CPU-only protocol/data preflight. --train explicitly starts training.
Real inference is separate unless --evaluate-real is also supplied.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from reference_refresh_audit import (ONLINE_ANCHOR, array_digest, budget, load_anchor,
                                     read, validate_config, validate_epoch)
from reference_online_data import EpochMixtures
from rssnet_data import pure_partitions, simulate
from fixed_splits import file_sha256


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def preflight(anchor):
    cfg, rows = load_anchor(anchor)
    args = cfg["args"]
    pure, references, _ = pure_partitions(args["source"], args["manifest"], create=False)
    fixed = EpochMixtures(pure["train"], args["train_per_pair"], args["seed"], refresh=False)
    digests = [array_digest(fixed.generate(epoch)) for epoch in (0, 1)]
    if digests != [cfg["training_array_sha256"]] * 2:
        raise ValueError("Fixed epoch-zero data differs from online anchor; use the original runtime, never bypass this check")
    for role, offset in (("validation", 100), ("test", 200)):
        data = simulate(pure[role], 64, 19580920 + offset)
        if array_digest(data) != cfg[role + "_array_sha256"]:
            raise ValueError("Fixed evaluation arrays differ: " + role)
    with np.load(Path(anchor) / "reference.npz") as z:
        np.testing.assert_array_equal(references, z["reference"])
    for row in rows:
        query_order = np.random.default_rng(args["seed"] + row["epoch"]).permutation(
            cfg["training_recipe"]["mixtures_per_epoch"] * 7)
        if array_digest(dict(order=query_order)) != row["order_sha256"]:
            raise ValueError("Query order differs from historical run")
    return cfg, dict(passed=True, epochs_audited=len(rows), same_epoch_zero=True,
                    fixed_data_reused=True, references_equal=True,
                    validation_test_fingerprints_equal=True, query_orders_equal=True,
                    checkpoint_callback="execute exact historical callback saved in anchor")


def command_for(cfg, anchor, root):
    a = cfg["args"]
    cmd = [sys.executable, "-u", "train_reference_extractor.py",
           "--training-data", "fixed_simulated", "--evaluation-anchor", str(anchor),
           "--refresh-control-run", str(anchor), "--output-root", str(root / "training")]
    for name in ("source", "audit", "manifest", "epochs", "batch_size", "seed", "learning_rate",
                 "channels", "blocks", "version", "train_per_pair", "checkpoint_selection"):
        cmd += ["--" + name.replace("_", "-"), str(a[name])]
    if a["save_every_epoch"]:
        cmd.append("--save-every-epoch")
    if a["presence_only"]:
        cmd.append("--presence-only")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--online-run", default=str(ONLINE_ANCHOR))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate-real", action="store_true")
    args = parser.parse_args()
    if args.evaluate_real and not args.train:
        parser.error("--evaluate-real requires --train")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    try:
        cfg, audit = preflight(args.online_run)
        command = command_for(cfg, args.online_run, root)
        write(root / "protocol.json", dict(
            intervention="Only augmented-mixture refresh: repeat epoch zero vs regenerate each epoch",
            online_run=args.online_run, online_checkpoint_sha256=read(Path(args.online_run)/"status.json")["checkpoint_sha256"],
            preflight=audit, fixed_training_command=command, online_budget=budget(cfg, read(Path(args.online_run)/"status.json")["best_epoch"]),
            full_updates_equal=True, selected_updates="report separately; common selection does not force same epoch",
            primary="validation-locked exact-set accuracy, count accuracy and Macro-F1 on paper-denoised real inputs",
            secondary="Top-2, raw-input transfer, weak-coefficient synthetic strata",
            threshold="same synthetic validation only, independently locked per model before real inference",
            interpretation="Completed online arm reused; no completed fixed-arm accuracy yet; historical single paired seed",
            source_versions={str(p): file_sha256(p) for p in (Path(__file__), Path("reference_refresh_audit.py"), Path("train_reference_extractor.py"))}))
        if not args.train:
            write(root / "status.json", dict(state="prepared_not_training", preflight_passed=True, command=command))
            print("Preflight passed. No training started. Add --train with a NEW output directory.", flush=True)
            return
        write(root / "status.json", dict(state="training_fixed", command=command, log=str(root / "train.log")))
        with (root / "train.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        models = list((root / "training").glob("ReferenceExtractor_*"))
        if len(models) != 1:
            raise RuntimeError("Expected exactly one fixed-augmentation run")
        model = models[0]
        state, fixed_cfg = read(model / "status.json"), read(model / "config.json")
        if state["state"] != "completed":
            raise RuntimeError("Fixed training incomplete")
        validate_config(fixed_cfg, cfg)
        _, online_rows = load_anchor(args.online_run)
        fixed_rows = [json.loads(line) for line in (model/"result/query_audit.jsonl").read_text().splitlines() if line.strip()]
        if [r["epoch"] for r in fixed_rows] != list(range(cfg["args"]["epochs"])):
            raise ValueError("Fixed epoch audits incomplete")
        for row in fixed_rows:
            validate_epoch(row, online_rows)
        write(root / "paired_training_audit.json", dict(passed=True, fixed_run=str(model), online_run=args.online_run,
            fixed_budget=budget(fixed_cfg, state["best_epoch"]),
            online_budget=budget(cfg, read(Path(args.online_run)/"status.json")["best_epoch"]),
            full_budget_equal=True, references="same policy and fixed inference library; sampled references depend on targets"))
        if args.evaluate_real:
            for source in ("raw", "paper-denoised"):
                cmd = [sys.executable, "-u", "evaluate_reference_raw_nnls.py", "--run-dir", str(model),
                       "--output", str(root/source), "--input-source", source, "--batch-size", str(cfg["args"]["batch_size"])]
                if source == "paper-denoised":
                    cmd += ["--compare-evaluation", str(root/"raw")]
                write(root/"status.json", dict(state="evaluating_"+source, model=str(model), command=cmd))
                with (root/("evaluate_"+source+".log")).open("x", encoding="utf-8") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(root/"status.json", dict(state="completed", model=str(model),
            real_evaluation_completed=args.evaluate_real, paired_training_audit_passed=True))
        print("Completed fixed-augmentation control:", model, flush=True)
    except Exception as exc:
        write(root/"status.json", dict(state="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
