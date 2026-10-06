"""Input preparation and leakage-aware splitting for the physical AE."""
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def spectra_matrix(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("Expected finite (samples, bands) spectra")
    if np.any(value < -1e-6):
        raise ValueError("Negative spectra: supply consistently baseline-corrected, nonnegative input")
    return np.maximum(value, 0)


def grouped_split(x, groups, seed):
    """Group physical IDs AND identical spectra together (transitively)."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(x):
        raise ValueError("One group ID is required per spectrum")
    parent = np.arange(len(x))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, (row, group) in enumerate(zip(x, groups)):
        for key in (("group", str(group)), ("spectrum", hashlib.sha256(row.tobytes()).hexdigest())):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    roots = np.asarray([root(i) for i in range(len(x))])
    unique = np.unique(roots)
    if len(unique) < 5:
        raise ValueError("At least five independent groups required")
    np.random.default_rng(seed).shuffle(unique)
    n = max(1, int(round(len(unique) * 0.15)))
    return {name: np.flatnonzero(np.isin(roots, ids)) for name, ids in (
        ("validation", unique[:n]), ("test", unique[n:2*n]), ("train", unique[2*n:]))}


def load_data(args):
    if args.dataset == "sugars":
        from benchmark_raman_sugars import (load_condition, polynomial_baseline_correct,
                                          stratified_well_split, reference_endmembers, COMPONENTS)
        stem = "Sugar_Concentration_Test" + ("_Fast" if args.condition == "low" else "")
        data = load_condition(Path(args.input), stem, (400, 1800))
        data["spectra"] = polynomial_baseline_correct(data["spectra"])
        split = stratified_well_split(data, args.seed)
        refs, _ = reference_endmembers(data, split["reference"])
        return dict(x=data["spectra"].astype(np.float32), groups=data["well"],
                    ids=data["filename"], names=list(COMPONENTS), reference=refs,
                    abundance_truth=data["abundances"],
                    splits={key: np.flatnonzero(np.isin(data["well"], split[key]))
                            for key in ("train", "validation", "test")},
                    preprocessing="400-1800 cm-1, degree-3 polynomial baseline, percentile offset; NOT paper ASPLS",
                    wave=data["wave"], group_note="well/formulation-disjoint; pure wells excluded from all splits")
    if args.dataset == "plastic":
        from decoupled_data import load_labeled_mixtures, DEFAULT_CATEGORIES, _as_spectra
        x, labels, pairs = load_labeled_mixtures(args.input, num_features=args.num_features,
                                                suffix=args.mat_suffix)
        x = spectra_matrix(x)
        # Group IDs for repeated acquisitions can be supplied by the user.
        groups = np.load(args.group_ids, allow_pickle=False).astype(str) if args.group_ids else np.arange(len(x)).astype(str)
        reference = None
        if args.reference_mat:
            raw = loadmat(args.reference_mat, variable_names=DEFAULT_CATEGORIES)
            reference = np.stack([np.median(_as_spectra(raw[n], n, args.num_features), axis=0)
                                  for n in DEFAULT_CATEGORIES])
            reference = spectra_matrix(reference)
        return dict(x=x, groups=groups, ids=np.asarray([f"{pair}:{i}" for i, pair in enumerate(pairs)]),
                    names=list(DEFAULT_CATEGORIES), reference=reference, presence_truth=labels,
                    splits=grouped_split(x, groups, args.seed),
                    wave=np.linspace(132.57, 4051.96, x.shape[1]),
                    preprocessing="already preprocessed MATLAB spectra, no per-row normalization",
                    group_note="physical group IDs + exact duplicates" if args.group_ids else
                    "exact-duplicate grouping only; physical replicate IDs unavailable; independence not established")
    with np.load(args.input, allow_pickle=False) as source:
        x = spectra_matrix(source["x"])
        groups = source["groups"].astype(str)
        names = source["names"].astype(str).tolist()
        data = dict(x=x, groups=groups, ids=np.arange(len(x)).astype(str), names=names,
                    reference=spectra_matrix(source["reference"]) if "reference" in source else None,
                    wave=source["wave"] if "wave" in source else np.arange(x.shape[1]),
                    preprocessing="preprocessed NPZ spectra", group_note="provided groups + exact duplicates")
        for key in ("abundance_truth", "presence_truth"):
            if key in source:
                value = source[key].copy()
                if value.shape != (len(x), len(names)) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{key} must be finite (samples, components)")
                if key == "abundance_truth" and (np.any(value < 0) or
                        not np.allclose(value.sum(axis=1), 1, atol=1e-5)):
                    raise ValueError("abundance_truth must be nonnegative fractions summing to one")
                if key == "presence_truth" and not np.isin(value, [0, 1]).all():
                    raise ValueError("presence_truth must contain only 0/1 labels")
                data[key] = value
        data["splits"] = grouped_split(x, groups, args.seed)
        return data


def validate_data(data):
    x = spectra_matrix(data["x"])
    if np.asarray(data["names"]).ndim != 1 or len(data["names"]) < 2:
        raise ValueError("At least two component names are required")
    if np.asarray(data["wave"]).shape != (x.shape[1],):
        raise ValueError("wave must contain one coordinate per spectral band")
    if len(data["names"]) != len(set(data["names"])):
        raise ValueError("Component names must be unique")
    if data["reference"] is not None and data["reference"].shape != (len(data["names"]), x.shape[1]):
        raise ValueError("Reference shape must be (components, bands)")
    seen_groups, seen_rows = set(), set()
    for indices in data["splits"].values():
        groups = set(data["groups"][indices])
        rows = set(hashlib.sha256(row.tobytes()).hexdigest() for row in x[indices])
        if seen_groups & groups or seen_rows & rows:
            raise ValueError("A physical group or exact duplicate crosses partitions")
        seen_groups.update(groups)
        seen_rows.update(rows)
    data["x"] = x
    return data
