"""Pure-source-disjoint synthetic two-component data. No mass-fraction claim."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from decoupled_data import DEFAULT_CATEGORIES, _as_spectra
from fixed_splits import file_sha256, spectrum_hashes
from physics_ae_data import grouped_split


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def normalize_input(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite (N,L) or (N,L,1) spectra")
    scale = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(scale < 1e-8):
        raise ValueError("Cannot unmix a zero spectrum")
    return (x / scale)[..., None], scale


def pure_partitions(path, manifest_path, seed=19580920, width=1024,
                    struct="Data_pure_norm_Struct", create=False):
    names = list(DEFAULT_CATEGORIES)
    source = loadmat(path, simplify_cells=True, variable_names=[struct] if struct else names)
    source = source[struct] if struct else source
    spectra, records = [], []
    global_owner = {}
    for category in names:
        rows = _as_spectra(source[category], category, width)
        if np.any(rows < 0):
            raise ValueError("Pure spectra must be nonnegative and consistently preprocessed")
        for index, row in enumerate(rows):
            raw_hash, unit_hash = spectrum_hashes(row)
            if unit_hash in global_owner and global_owner[unit_hash] != category:
                raise ValueError("Identical pure spectra carry conflicting labels")
            global_owner[unit_hash] = category
            records.append(dict(category=category, row=index, raw_sha256=raw_hash, unit_sha256=unit_hash))
        spectra.append(normalize_input(rows)[0][..., 0])
    metadata = dict(version=1, source_sha256=file_sha256(path), seed=seed, width=width,
                    struct=struct, categories=names)
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["metadata"] != metadata or [{k: r[k] for k in records[0]} for r in saved["rows"]] != records:
            raise ValueError("Pure source/config changed relative to fixed manifest")
        records = saved["rows"]
    else:
        if not create:
            raise FileNotFoundError(manifest_path)
        offset = 0
        for i, rows in enumerate(spectra):
            splits = grouped_split(rows, np.arange(len(rows)).astype(str), seed + i)
            for partition, indices in splits.items():
                for index in indices:
                    records[offset + index]["partition"] = partition
            offset += len(rows)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create: never silently overwrite a shared partition.
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(dict(metadata=metadata, rows=records), stream, indent=2)
    owners = {}
    for record in records:
        role = record["partition"]
        if role not in ("train", "validation", "test"):
            raise ValueError("Invalid pure partition")
        for key in ("raw_sha256", "unit_sha256"):
            h = record[key]
            if h in owners and owners[h] != role:
                raise ValueError("Pure duplicate crosses partitions")
            owners[h] = role
    result = {s: [] for s in ("train", "validation", "test")}
    for category, rows in zip(names, spectra):
        for role in result:
            ids = [r["row"] for r in records if r["category"] == category and r["partition"] == role]
            if not ids:
                raise ValueError("Empty pure partition")
            result[role].append(rows[ids])
    references = np.stack([np.median(rows, axis=0) for rows in result["train"]])
    references = normalize_input(references)[0][..., 0]
    return result, references, metadata


def simulate(pure, per_pair, seed, noise=True, fixed_alpha=None,
             disable_noise=False, disable_background=False):
    """Random optical mixing weights; targets preserve their shared input scale."""
    if per_pair < 1:
        raise ValueError("per_pair must be positive")
    rng = np.random.default_rng(seed)
    x_all, y_all, pair_all, alpha_all = [], [], [], []
    for a, b in combinations(range(len(pure)), 2):
        sa = pure[a][rng.integers(len(pure[a]), size=per_pair)]
        sb = pure[b][rng.integers(len(pure[b]), size=per_pair)]
        alpha = rng.uniform(0.1, 0.9, (per_pair, 1)).astype("float32")
        # Always consume the same random draws, even for ablated factors. Thus
        # pure-source indices and remaining augmentations match the legacy F run.
        if fixed_alpha is not None:
            if not 0 < fixed_alpha < 1:
                raise ValueError("fixed_alpha must lie strictly between 0 and 1")
            alpha = np.full_like(alpha, fixed_alpha)
        targets = np.stack((sa * alpha, sb * (1 - alpha)), axis=-1)
        mixture = targets.sum(axis=-1)
        if noise:
            white = rng.normal(size=mixture.shape)
            brown = np.cumsum(rng.normal(size=mixture.shape), axis=1)
            brown -= brown.mean(axis=1, keepdims=True)
            brown /= np.maximum(brown.std(axis=1, keepdims=True), 1e-8)
            n = white + brown
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
            snr = rng.uniform(10, 30, (per_pair, 1))
            noisy = mixture + n * np.linalg.norm(mixture, axis=1, keepdims=True) * 10 ** (-snr / 20)
            # Mild positive smooth fluorescence-like nuisance (not a third target).
            u = np.linspace(0, 1, mixture.shape[1])[None, :]
            background = rng.uniform(0, .08, (per_pair, 1)) * noisy.max(axis=1, keepdims=True) * (u - .5) ** 2
            mixture = mixture if disable_noise else noisy
            if not disable_background:
                mixture += background
            mixture = np.maximum(mixture, 0)
        x, scale = normalize_input(mixture)
        x_all.append(x)
        y_all.append(targets / scale[..., None])
        pair_all.extend([(a, b)] * per_pair)
        alpha_all.append(alpha[:, 0])
    return dict(x=np.concatenate(x_all).astype("float32"),
                y=np.concatenate(y_all).astype("float32"),
                pair=np.asarray(pair_all), alpha=np.concatenate(alpha_all))


def spectral_metrics(truth, prediction):
    """One fixed MSE-minimizing source permutation, shared by all metrics."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[-1] != 2:
        raise ValueError("Expected aligned (N,L,2) arrays")
    direct = np.sum((truth - prediction) ** 2, axis=(1, 2))
    swap = np.sum((truth - prediction[..., ::-1]) ** 2, axis=(1, 2)) < direct
    aligned = np.where(swap[:, None, None], prediction[..., ::-1], prediction)
    error = aligned - truth
    norm_product = np.linalg.norm(truth, axis=1) * np.linalg.norm(aligned, axis=1)
    valid = norm_product > 1e-12
    cos = np.clip(np.sum(truth * aligned, axis=1) / np.maximum(norm_product, 1e-12), -1, 1)
    t = truth - truth.mean(axis=1, keepdims=True)
    p = aligned - aligned.mean(axis=1, keepdims=True)
    projection = t * (np.sum(t * p, axis=1, keepdims=True) / (np.sum(t * t, axis=1, keepdims=True) + 1e-12))
    si = 10 * np.log10((np.sum(projection ** 2, axis=1) + 1e-12) /
                      (np.sum((p - projection) ** 2, axis=1) + 1e-12))
    denominator = np.sum((truth - truth.mean()) ** 2)
    return dict(r2_flattened=float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else None,
                rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))),
                cosine_similarity=float(cos[valid].mean()) if valid.any() else None,
                sam_degrees=float(np.degrees(np.arccos(cos[valid])).mean()) if valid.any() else None,
                undefined_sam_components=int((~valid).sum()), si_snr_db=float(si.mean()),
                assignment="minimum joint endpoint MSE over two permutations",
                samples=len(truth))
