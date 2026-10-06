"""Sequential equal-data/equal-strategy paper-vs-RSSNet separator experiment.

No real mixture trains either model. Dataset 8 is a frozen post-training report;
Dataset 9 is not evaluated unless --evaluate-test is explicitly requested.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import shutil


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_matched(a, b):
    ca, cb = read(a / "config.json"), read(b / "config.json")
    for key in ("protocol", "pure_manifest_sha256", "validation_array_sha256", "test_array_sha256",
                "reference_array_sha256", "selection", "schedule", "initialization", "output_initialization"):
        if ca[key] != cb[key]:
            raise ValueError(f"Unmatched protocol: {key}")
    for key in ("epochs", "batch_size", "learning_rate", "seed", "split_seed", "train_per_pair", "eval_per_pair", "clean_input"):
        if ca["args"][key] != cb["args"][key]:
            raise ValueError(f"Unmatched training setting: {key}")
    audits = []
    for path in (a, b):
        rows = [json.loads(line) for line in (path / "result/data_audit.jsonl").read_text().splitlines()]
        if [r["epoch"] for r in rows] != list(range(ca["args"]["epochs"])):
            raise ValueError("Incomplete training data audit")
        complete = read(path / "result/completed.json")
        if complete["epochs"] != ca["args"]["epochs"]:
            raise ValueError("Incomplete epoch budget")
        audits.append(rows)
    if audits[0] != audits[1]:
        raise ValueError("Models did not see identical synthetic examples AND batch order")
    return {"passed": True, "epochs": len(audits[0]), "identical_batches": True,
            "limitations": "Same protocol, not equal parameter count/compute or identical initialization tensors; physical source overlap unverified"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=19580920)
    p.add_argument("--split-seed", type=int, default=19580920)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--train-per-pair", type=int, default=512)
    p.add_argument("--eval-per-pair", type=int, default=64)
    p.add_argument("--output-root", default="run")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="One epoch/tiny dataset; not formal evidence; no real evaluation")
    p.add_argument("--evaluate-test", action="store_true", help="Also report Dataset 9 after both checkpoints are frozen")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.train_per_pair, args.eval_per_pair) < 1 or args.learning_rate <= 0:
        p.error("Positive training settings required")
    if args.smoke:
        args.epochs, args.train_per_pair, args.eval_per_pair = 1, 1, 1
        args.batch_size = min(args.batch_size, 2)
    root = Path(args.output_root) / ("MatchedSeparators_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    shutil.copy2(__file__, root / "launcher_source.py")
    print(f"Experiment: {root}", flush=True)

    def execute(command, logfile):
        print("Starting: " + " ".join(command), flush=True)
        state = dict(state="running", command=command, logfile=str(logfile))
        (root / "status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        try:
            with logfile.open("x", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            state.update(state="failed", error=str(exc))
            (root / "status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            raise

    paths = {}
    for kind in ("paper", "rssnet"):
        out = root / kind
        cmd = [sys.executable, "-u", "train_rssnet_unmixing.py", "--matched-protocol", "--model", kind,
               "--seed", str(args.seed), "--split-seed", str(args.split_seed),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--learning-rate", str(args.learning_rate), "--train-per-pair", str(args.train_per_pair),
               "--eval-per-pair", str(args.eval_per_pair), "--output-root", str(out)]
        if args.cpu:
            cmd.append("--cpu")
        if args.smoke and kind == "rssnet":
            cmd.extend(["--channels", "8", "--blocks", "1", "--heads", "2", "--depth", "1"])
        execute(cmd, root / f"{kind}_train.log")
        found = list(out.glob("Matched_*"))
        if len(found) != 1:
            raise ValueError("Expected exactly one run per model")
        paths[kind] = found[0]
    audit = verify_matched(paths["paper"], paths["rssnet"])
    summary = {"audit": audit, "runs": {k: str(v) for k, v in paths.items()},
               "synthetic": {k: read(v / "result/synthetic_metrics.json") for k, v in paths.items()},
               "real": {}, "smoke_only": args.smoke}
    (root / "matched_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    # Selection was already finalized using synthetic validation only.
    for condition in ([] if args.smoke else ([8, 9] if args.evaluate_test else [8])):
        summary["real"][str(condition)] = {}
        for kind, path in paths.items():
            out = path / "result" / f"condition{condition}_frozen"
            cmd = [sys.executable, "-u", "predict_rssnet_unmixing.py", "--run-dir", str(path),
                   "--output", str(out), "--condition", str(condition),
                   "--partition", "validation" if condition == 8 else "test",
                   "--batch-size", str(args.batch_size)]
            if args.cpu:
                cmd.append("--cpu")
            execute(cmd, root / f"{kind}_condition{condition}.log")
            summary["real"][str(condition)][kind] = read(out / "metrics.json")
        pa, pb = [read(paths[k] / "result" / f"condition{condition}_frozen" / "provenance.json")
                  for k in ("paper", "rssnet")]
        for key in ("input_sha256", "raw_sha256", "split_manifest_sha256"):
            if pa[key] != pb[key]:
                raise ValueError("Real evaluation provenance mismatch: " + key)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (root / "status.json").write_text(json.dumps({"state": "completed", "summary": str(root / "summary.json")}), encoding="utf-8")
    print(f"Completed: {root}; matched audit passed", flush=True)


if __name__ == "__main__":
    main()
