"""CPU-safe controls for fixed-augmentation versus epoch-refresh ablations."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fixed_splits import file_sha256


ONLINE_ANCHOR = Path("run/ReferenceDataOnline_seed19580920/19580920/online_simulated/training/ReferenceExtractor_20260924_151025_266187")
CONTROL_SOURCES = ("models/ReferenceGuidedExtractor.py", "models/RSSNetInspired.py",
                   "reference_extraction_data.py",
                   "reference_online_data.py", "rssnet_data.py", "original_separator_data.py")
CONFIG_FIELDS = ("source", "source_manifest_sha256", "training_array_sha256",
                 "validation_array_sha256", "test_array_sha256", "architecture",
                 "parameters", "initial_weights_sha256", "selection", "selection_samples",
                 "losses", "reference_policy", "training_generator_sha256")
ARG_FIELDS = ("seed", "batch_size", "epochs", "learning_rate", "channels", "blocks",
              "version", "train_per_pair", "presence_only", "checkpoint_selection", "save_every_epoch")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_digest(arrays):
    # Identical byte protocol to matched_separators.array_digest, without TF import.
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_anchor(path):
    path = Path(path)
    cfg, state = read(path / "config.json"), read(path / "status.json")
    if state["state"] != "completed" or cfg["training_recipe"]["mode"] != "online_simulated":
        raise ValueError("Refresh control requires a completed online-simulated anchor")
    if cfg["args"].get("smoke") or cfg["args"].get("short_check"):
        raise ValueError("Cannot reuse a short or smoke training run")
    if state["epochs"] != cfg["args"]["epochs"]:
        raise ValueError("Anchor training budget incomplete")
    if file_sha256(path / "checkpoint/best.weights.h5") != state["checkpoint_sha256"]:
        raise ValueError("Anchor checkpoint fingerprint mismatch")
    for source in CONTROL_SOURCES:
        if file_sha256(source) != file_sha256(path / Path(source).name):
            raise ValueError("Control implementation differs from anchor: " + source)
    if not (path / "reference_checkpoint.py").is_file():
        raise ValueError("Anchor must retain its exact checkpoint callback implementation")
    if file_sha256(cfg["args"]["source"]) != cfg["source"]["source_sha256"]:
        raise ValueError("Pure-source file changed")
    if file_sha256(cfg["args"]["manifest"]) != cfg["source_manifest_sha256"]:
        raise ValueError("Pure-source partition manifest changed")
    rows = [json.loads(line) for line in
            (path / "result/query_audit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["epoch"] for row in rows] != list(range(cfg["args"]["epochs"])):
        raise ValueError("Missing or duplicate anchor epoch audits")
    if rows[0]["training_array_sha256"] != cfg["training_array_sha256"]:
        raise ValueError("Anchor epoch-zero fingerprint mismatch")
    if len({row["training_array_sha256"] for row in rows}) != len(rows):
        raise ValueError("Online anchor did not refresh every epoch")
    expected_n = cfg["training_recipe"]["mixtures_per_epoch"]
    if any(row["mixtures"] != expected_n for row in rows):
        raise ValueError("Anchor per-epoch sample budget differs")
    return cfg, rows


def validate_config(fixed, online):
    if fixed["training_recipe"]["mode"] != "fixed_simulated":
        raise ValueError("Expected fixed-simulated intervention")
    for key in CONFIG_FIELDS:
        if fixed.get(key) != online.get(key):
            raise ValueError("Refresh control mismatch: " + key)
    for key in ARG_FIELDS:
        if fixed["args"].get(key) != online["args"].get(key):
            raise ValueError("Refresh control argument mismatch: " + key)
    policy = lambda cfg: {k: v for k, v in cfg["training_recipe"].items()
                          if k not in ("mode", "refresh")}
    if policy(fixed) != policy(online):
        raise ValueError("Augmentation distribution differs")


def validate_epoch(row, anchor_rows):
    epoch = row["epoch"]
    anchor = anchor_rows[epoch]
    first = anchor_rows[0]
    if row["order_sha256"] != anchor["order_sha256"]:
        raise ValueError("Query ordering mismatch at epoch " + str(epoch))
    for key in ("training_array_sha256", "synthesis_seed", "mixtures", "alpha_min", "alpha_max"):
        if row[key] != first[key]:
            raise ValueError("Fixed augmentation changed at epoch %s: %s" % (epoch, key))


def budget(cfg, selected_epoch):
    mixtures = cfg["training_recipe"]["mixtures_per_epoch"]
    queries = mixtures * 7
    steps = (queries + cfg["args"]["batch_size"] - 1) // cfg["args"]["batch_size"]
    return dict(mixtures_per_epoch=mixtures, queries_per_epoch=queries, updates_per_epoch=steps,
                full_updates=steps * cfg["args"]["epochs"], selected_epoch=selected_epoch,
                selected_updates=steps * selected_epoch)
