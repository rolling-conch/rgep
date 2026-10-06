"""Frozen, content-verified partitions shared by training and evaluation."""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PARTITIONS = ("train", "validation", "test")
MANIFEST_PARTITIONS = PARTITIONS + ("excluded",)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def spectrum_hashes(spectrum):
    x = np.ascontiguousarray(spectrum, dtype=np.float32).reshape(-1).copy()
    if not np.isfinite(x).all() or np.any(x < -1e-6):
        raise ValueError("Raw spectrum must be finite and nonnegative")
    x = np.maximum(x, 0.)
    x[x == 0] = 0.  # canonicalize signed zero
    norm = np.linalg.norm(x)
    if norm <= 1e-8:
        raise ValueError("Zero input spectrum")
    return (hashlib.sha256(x.tobytes()).hexdigest(),
            hashlib.sha256((x/norm).tobytes()).hexdigest())


def grouped_partitions(records, seed, train_ratio=.7, validation_ratio=.2):
    """Union exact and unit-L2 duplicates globally, then balance row strata.

    Groups may cross acquisition conditions or particle-size blocks. They are
    indivisible; row counts only approximate requested ratios.
    """
    ratios = np.asarray([train_ratio, validation_ratio, 1-train_ratio-validation_ratio])
    if np.any(ratios <= 0):
        raise ValueError("All split ratios must be positive")
    parent = np.arange(len(records))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    seen = {}
    for i, row in enumerate(records):
        for field in ("raw_sha256", "unit_sha256"):
            key = (field, row[field])
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    strata = sorted({(r["condition"], r["pair"], r["particle_size_block"]) for r in records})
    lookup = {s: i for i, s in enumerate(strata)}
    row_strata = np.asarray([lookup[(r["condition"], r["pair"], r["particle_size_block"])] for r in records])
    totals = np.bincount(row_strata, minlength=len(strata))
    targets = totals[:, None]*ratios[None, :]
    counts = np.zeros_like(targets)
    rng = np.random.default_rng(seed)
    shuffled = list(groups.values())
    rng.shuffle(shuffled)
    # Large groups first, seed-random tie ordering; no spectral class-label fit.
    shuffled.sort(key=len, reverse=True)
    for members in shuffled:
        labels = {records[i]["pair"] for i in members}
        if len(labels) > 1:
            raise ValueError("Identical input spectra have conflicting material labels")
        ids, n = np.unique(row_strata[members], return_counts=True)
        before = counts[ids] - targets[ids]
        cost = np.sum((2*before*n[:, None] + n[:, None]**2) / np.maximum(targets[ids], 1.), axis=0)
        split = int(np.argmin(cost))
        counts[ids, split] += n
        group_key = min(records[i]["raw_sha256"] for i in members)
        for i in members:
            records[i]["duplicate_group"] = group_key
            records[i]["partition"] = PARTITIONS[split]
    if np.any(counts == 0):
        raise ValueError("At least one stratum lacks a partition; inspect group sizes before training")
    return records


def condition_holdout_partitions(
    records,
    train_conditions,
    validation_conditions,
    test_conditions,
):
    """Assign whole conditions while preventing cross-role duplicate leakage.

    Dataset membership defines the desired role.  If the same raw spectrum
    (or its unit-L2-normalized equivalent) occurs in more than one role, the
    row in the more protected role is retained and the lower-priority copy is
    marked ``excluded``.  Priority is test > validation > train.  This keeps
    the test set equal to the requested held-out conditions instead of moving
    training rows into the test set.
    """
    role_by_condition = {}
    for role, conditions in (
        ("train", train_conditions),
        ("validation", validation_conditions),
        ("test", test_conditions),
    ):
        for condition in conditions:
            condition = int(condition)
            if condition in role_by_condition:
                raise ValueError("Condition roles must be disjoint")
            role_by_condition[condition] = role
    present = {int(row["condition"]) for row in records}
    if present != set(role_by_condition):
        raise ValueError("Condition roles must cover every manifest condition")

    parent = np.arange(len(records))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    seen = {}
    for i, row in enumerate(records):
        for field in ("raw_sha256", "unit_sha256"):
            key = (field, row[field])
            if key in seen:
                left, right = root(i), root(seen[key])
                if left != right:
                    parent[left] = right
            else:
                seen[key] = i

    groups = {}
    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    priority = {"train": 0, "validation": 1, "test": 2}
    for members in groups.values():
        labels = {records[i]["pair"] for i in members}
        if len(labels) > 1:
            raise ValueError("Identical input spectra have conflicting material labels")
        protected_role = max(
            (role_by_condition[int(records[i]["condition"])] for i in members),
            key=priority.__getitem__,
        )
        group_key = min(records[i]["raw_sha256"] for i in members)
        for i in members:
            desired_role = role_by_condition[int(records[i]["condition"])]
            records[i]["duplicate_group"] = group_key
            records[i]["partition"] = (
                protected_role if desired_role == protected_role else "excluded"
            )
            records[i]["desired_partition"] = desired_role
    for role in PARTITIONS:
        if not any(row["partition"] == role for row in records):
            raise ValueError(f"Condition holdout produced an empty {role} partition")
    return records


@lru_cache(maxsize=8)
def _read_manifest(path, mtime, sidecar_mtime):
    manifest = Path(path)
    metadata = json.loads(manifest.with_suffix(".json").read_text(encoding="utf-8"))
    actual = file_sha256(manifest)
    if actual != metadata["manifest_sha256"]:
        raise ValueError("Split manifest checksum mismatch")
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_condition, hash_owner, group_owner, identities = {}, {}, {}, set()
    for r in rows:
        condition, index = int(r["condition"]), int(r["sample_index"])
        identity = (condition, index)
        if identity in identities or r["partition"] not in MANIFEST_PARTITIONS:
            raise ValueError("Duplicate row ID or invalid partition in manifest")
        identities.add(identity)
        for field in ("raw_sha256", "unit_sha256"):
            key = (field, r[field])
            owners = hash_owner.setdefault(key, set())
            if r["partition"] in PARTITIONS:
                owners.add(r["partition"])
            if len(owners) > 1:
                raise ValueError("Raw duplicate crosses active frozen partitions")
        g = r["duplicate_group"]
        owners = group_owner.setdefault(g, set())
        if r["partition"] in PARTITIONS:
            owners.add(r["partition"])
        if len(owners) > 1:
            raise ValueError("Duplicate group crosses active frozen partitions")
        by_condition.setdefault(condition, {})[index] = r
    if len(rows) != metadata["rows"]:
        raise ValueError("Manifest row count mismatch")
    return by_condition, metadata


def read_manifest(path, expected_sha256=None):
    p = Path(path).resolve()
    rows, meta = _read_manifest(str(p), p.stat().st_mtime_ns, p.with_suffix(".json").stat().st_mtime_ns)
    if expected_sha256 and meta["manifest_sha256"] != expected_sha256:
        raise ValueError("Run's pinned split checksum does not match manifest")
    return rows, meta


def manifest_indices(data, condition, args):
    rows, meta = read_manifest(args.split_manifest, getattr(args, "split_manifest_sha256", None))
    if getattr(args, "limit_per_condition", 0):
        raise ValueError("Debug subsampling is incompatible with a frozen full-data manifest")
    expected = rows.get(int(condition), {})
    actual_ids = set(map(int, data["sample_index"]))
    if actual_ids != set(expected):
        raise ValueError(f"Condition {condition}: manifest/data row coverage differs")
    result = {s: [] for s in MANIFEST_PARTITIONS}
    for local, sample_id in enumerate(data["sample_index"]):
        r = expected[int(sample_id)]
        h, hn = spectrum_hashes(data["raw"][local])
        if h != r["raw_sha256"] or hn != r["unit_sha256"] or str(data["pairs"][local]) != r["pair"]:
            raise ValueError(f"Condition {condition}, row {sample_id}: raw data changed since split creation")
        if int(data["particle_size_block"][local]) != int(r["particle_size_block"]):
            raise ValueError("Particle-size block changed since split creation")
        result[r["partition"]].append(local)
    split_mode = getattr(args, "split_mode", "within-condition")
    if split_mode == "within-condition":
        if any(not result[s] for s in PARTITIONS):
            raise ValueError("Empty manifest partition")
        if result["excluded"]:
            raise ValueError("Within-condition manifest unexpectedly excludes rows")
    else:
        roles = {
            "train": set(map(int, args.train_conditions)),
            "validation": set(map(int, args.validation_conditions)),
            "test": set(map(int, args.test_conditions)),
        }
        expected_role = next(
            (role for role, conditions in roles.items() if int(condition) in conditions),
            None,
        )
        if expected_role is None or not result[expected_role]:
            raise ValueError(f"Condition {condition}: missing active holdout partition")
        unexpected = [
            role for role in PARTITIONS
            if role != expected_role and result[role]
        ]
        if unexpected:
            raise ValueError(
                f"Condition {condition}: rows assigned to unexpected roles {unexpected}"
            )
    return {
        s: np.asarray(v, dtype=np.int64)
        for s, v in result.items()
    }
