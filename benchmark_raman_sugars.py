"""Backtest quantitative Raman unmixing on the public Raman Sugars dataset.

The split unit is a physical well/formulation, never an individual repeated
spectrum.  This prevents replicate leakage between train/validation/test.

Two deliberately different baselines are evaluated:
  * FCLS: library-based, fully constrained least-squares unmixing using the
    five separately measured pure wells as reference endmembers.
  * PLS: supervised chemometric abundance regression.  Both a high-SNR-only
    training regime and a mixed-SNR training regime are reported.

The script does not load the plastic classifier/unmixer: its seven plastic
classes and two-endmember output have no valid semantic mapping to five sugar
solution components.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score, r2_score


COMPONENTS = ("Sucrose", "Fructose", "Maltose", "Glucose", "Water")
CONCENTRATION_COLUMNS = tuple(f"{name} [ul]" for name in COMPONENTS)
SUGAR_COUNT = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="dataset/external/Raman_Sugars",
        help="Directory containing the compact spectra and metadata CSV files.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=19580920)
    parser.add_argument("--min-wavenumber", type=float, default=400.0)
    parser.add_argument("--max-wavenumber", type=float, default=1800.0)
    parser.add_argument("--baseline-degree", type=int, default=3)
    parser.add_argument("--pls-components", type=int, default=12)
    parser.add_argument("--fcls-iterations", type=int, default=500)
    return parser.parse_args()


def well_ids(metadata):
    return (
        metadata["row"].astype(str)
        + metadata["col"].astype(str)
        + "_"
        + metadata["plate"].astype(str)
    ).to_numpy()


def load_condition(data_dir: Path, stem: str, wavenumber_range):
    metadata_path = data_dir / f"{stem}_ALL_metadata.csv"
    spectra_path = data_dir / f"{stem}_ALL_spectra.csv"
    metadata = pd.read_csv(metadata_path)

    # The compact files are bands x observations; reorder the observations to
    # exactly match metadata instead of relying on repository column order.
    spectra_frame = pd.read_csv(spectra_path, dtype=np.float32)
    wave = spectra_frame.iloc[:, 0].to_numpy(dtype=np.float64)
    keep = (wave >= wavenumber_range[0]) & (wave <= wavenumber_range[1])
    requested = metadata["filename"].astype(str).tolist()
    missing = sorted(set(requested) - set(spectra_frame.columns[1:]))
    if missing:
        raise ValueError(f"{len(missing)} spectra named by metadata are missing")
    spectra = spectra_frame.loc[keep, requested].to_numpy(dtype=np.float32).T
    abundances = metadata.loc[:, CONCENTRATION_COLUMNS].to_numpy(dtype=np.float64)
    totals = metadata["Total Volume [ul]"].to_numpy(dtype=np.float64)
    abundances = abundances / totals[:, None]
    return {
        "wave": wave[keep],
        "spectra": spectra,
        "abundances": abundances,
        "well": well_ids(metadata),
        "filename": np.asarray(requested),
    }


def polynomial_baseline_correct(spectra, degree=3):
    """Fast deterministic baseline estimate used for this repository backtest."""
    spectra = np.asarray(spectra, dtype=np.float64)
    axis = np.linspace(-1.0, 1.0, spectra.shape[1])
    design = np.polynomial.legendre.legvander(axis, degree)
    coefficients = spectra @ np.linalg.pinv(design).T
    corrected = spectra - coefficients @ design.T
    # Offset by a low percentile, then enforce physically non-negative signal.
    corrected -= np.percentile(corrected, 1.0, axis=1, keepdims=True)
    return np.maximum(corrected, 0.0)


def formulation_table(condition):
    frame = pd.DataFrame(condition["abundances"], columns=COMPONENTS)
    frame.insert(0, "well", condition["well"])
    grouped = frame.groupby("well", sort=True).first()
    spread = frame.groupby("well", sort=True)[list(COMPONENTS)].std().fillna(0.0)
    if spread.to_numpy().max() > 1e-12:
        raise ValueError("Concentration labels disagree within a well")
    return grouped


def stratified_well_split(condition, seed):
    """70/15/15-ish split within each sugar-presence combination."""
    formulations = formulation_table(condition)
    values = formulations.loc[:, COMPONENTS].to_numpy()
    pure = np.count_nonzero(values > 0, axis=1) == 1
    reference = formulations.index[pure].to_numpy()
    candidates = formulations.loc[~pure]
    presence = candidates.loc[:, COMPONENTS[:SUGAR_COUNT]].to_numpy() > 0
    masks = presence @ (1 << np.arange(SUGAR_COUNT))
    rng = np.random.default_rng(seed)
    split = {"train": [], "validation": [], "test": [], "reference": list(reference)}
    for label in np.unique(masks):
        group = candidates.index[masks == label].to_numpy().copy()
        rng.shuffle(group)
        if len(group) < 3:
            raise ValueError(f"Presence class {label} has fewer than three formulations")
        n_validation = max(1, int(round(0.15 * len(group))))
        n_test = max(1, int(round(0.15 * len(group))))
        split["validation"].extend(group[:n_validation])
        split["test"].extend(group[n_validation:n_validation + n_test])
        split["train"].extend(group[n_validation + n_test:])
    return {name: np.asarray(sorted(wells)) for name, wells in split.items()}


def select(condition, wells):
    keep = np.isin(condition["well"], wells)
    return condition["spectra"][keep], condition["abundances"][keep], keep


def balanced_select(condition, wells, repeats_per_well, seed):
    """Select the same number of repeats from every well deterministically."""
    rng = np.random.default_rng(seed)
    selected = []
    for well in sorted(wells):
        indices = np.flatnonzero(condition["well"] == well)
        if len(indices) < repeats_per_well:
            raise ValueError(f"Well {well} has only {len(indices)} repeats")
        selected.extend(rng.choice(indices, repeats_per_well, replace=False))
    selected = np.asarray(sorted(selected))
    return condition["spectra"][selected], condition["abundances"][selected]


def project_simplex(values):
    """Euclidean projection of every row onto {x >= 0, sum(x) = 1}."""
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1)
    valid = ordered - cumulative / indices > 0
    rho = valid.sum(axis=1) - 1
    theta = cumulative[np.arange(len(values)), rho] / (rho + 1)
    return np.maximum(values - theta[:, None], 0.0)


def fcls(spectra, endmembers, iterations=500):
    """Batched projected-gradient fully constrained least squares."""
    gram = endmembers @ endmembers.T
    cross = spectra @ endmembers.T
    step = 1.0 / max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
    abundance = np.full((len(spectra), len(endmembers)), 1.0 / len(endmembers))
    for _ in range(iterations):
        abundance = project_simplex(abundance + step * (cross - abundance @ gram))
    return abundance


def reference_endmembers(condition, reference_wells):
    references = []
    labels = []
    for index in range(len(COMPONENTS)):
        wanted = np.zeros(len(COMPONENTS))
        wanted[index] = 1.0
        is_reference = np.isin(condition["well"], reference_wells) & np.all(
            np.isclose(condition["abundances"], wanted), axis=1
        )
        if not np.any(is_reference):
            raise ValueError(f"Missing pure reference spectra for {COMPONENTS[index]}")
        references.append(np.median(condition["spectra"][is_reference], axis=0))
        labels.append(COMPONENTS[index])
    return np.asarray(references), labels


def sam_degrees(first, second):
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def choose_presence_threshold(truth, prediction):
    truth = truth[:, :SUGAR_COUNT] > 0
    best = None
    for threshold in np.linspace(0.005, 0.12, 116):
        predicted = prediction[:, :SUGAR_COUNT] >= threshold
        strict = np.mean(np.all(predicted == truth, axis=1))
        macro = f1_score(truth, predicted, average="macro", zero_division=0)
        candidate = (strict, macro, -threshold)
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold))
    return best[1]


def metrics(truth, prediction, spectra, endmembers, threshold):
    prediction = project_simplex(prediction)
    truth_presence = truth[:, :SUGAR_COUNT] > 0
    pred_presence = prediction[:, :SUGAR_COUNT] >= threshold
    reconstructed = prediction @ endmembers
    squared = (truth - prediction) ** 2
    per_component = {}
    for index, name in enumerate(COMPONENTS):
        per_component[name] = {
            "rmse": float(np.sqrt(np.mean(squared[:, index]))),
            "mae": float(np.mean(np.abs(truth[:, index] - prediction[:, index]))),
            "r2": float(r2_score(truth[:, index], prediction[:, index])),
        }
    return {
        "samples": int(len(truth)),
        "abundance_rmse": float(np.sqrt(np.mean(squared))),
        "abundance_mae": float(np.mean(np.abs(truth - prediction))),
        "abundance_r2": float(r2_score(truth, prediction, multioutput="variance_weighted")),
        "strict_sugar_presence_accuracy": float(np.mean(np.all(pred_presence == truth_presence, axis=1))),
        "macro_sugar_presence_f1": float(
            f1_score(truth_presence, pred_presence, average="macro", zero_division=0)
        ),
        "presence_threshold": float(threshold),
        "reconstruction_sam_degrees": float(np.mean(sam_degrees(spectra, reconstructed))),
        "reconstruction_cosine_similarity": float(
            np.mean(np.cos(np.radians(sam_degrees(spectra, reconstructed))))
        ),
        "per_component": per_component,
    }


def fit_pls(x, y, n_components):
    count = min(n_components, x.shape[1], x.shape[0] - 1)
    model = PLSRegression(n_components=count, scale=True, max_iter=1000, tol=1e-7)
    model.fit(x, y)
    return model


def write_predictions(path, condition_name, model_name, condition, keep, prediction):
    truth = condition["abundances"][keep]
    output = pd.DataFrame({
        "condition": condition_name,
        "model": model_name,
        "well": condition["well"][keep],
        "filename": condition["filename"][keep],
    })
    for index, name in enumerate(COMPONENTS):
        output[f"true_{name}"] = truth[:, index]
        output[f"pred_{name}"] = prediction[:, index]
    output.to_csv(path, mode="a", index=False, header=not path.exists())


def plot_predictions(path, records):
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(17, 3.5), sharex=True, sharey=True)
    colors = {"high_snr": "#1665D8", "low_snr": "#E05A33"}
    for axis, component in zip(axes, COMPONENTS):
        index = COMPONENTS.index(component)
        for record in records:
            axis.scatter(
                record["truth"][:, index], record["prediction"][:, index],
                s=8, alpha=0.35, color=colors[record["condition"]],
                label=record["condition"],
            )
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(component)
        axis.set_xlabel("True fraction")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted fraction")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=2, frameon=False,
    )
    fig.suptitle(
        "Raman Sugars held-out-formulation backtest: condition-specific PLS",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.91))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output = Path(args.output_dir) if args.output_dir else Path("run") / (
        "RamanSugars_backtest_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    ) / "result"
    output.mkdir(parents=True, exist_ok=True)

    stems = {
        "high_snr": "Sugar_Concentration_Test",
        "low_snr": "Sugar_Concentration_Test_Fast",
    }
    conditions = {}
    for name, stem in stems.items():
        print(f"Loading {name} ...", flush=True)
        condition = load_condition(
            data_dir, stem, (args.min_wavenumber, args.max_wavenumber)
        )
        condition["spectra"] = polynomial_baseline_correct(
            condition["spectra"], args.baseline_degree
        )
        conditions[name] = condition

    if not np.array_equal(conditions["high_snr"]["wave"], conditions["low_snr"]["wave"]):
        raise ValueError("High- and low-SNR wavenumber axes differ")
    split = stratified_well_split(conditions["high_snr"], args.seed)
    all_high_wells = set(conditions["high_snr"]["well"])
    all_low_wells = set(conditions["low_snr"]["well"])
    if all_high_wells != all_low_wells:
        raise ValueError("High- and low-SNR conditions do not contain identical wells")

    manifest = []
    for partition, wells in split.items():
        manifest.extend({"well": well, "partition": partition} for well in wells)
    pd.DataFrame(manifest).sort_values(["partition", "well"]).to_csv(
        output / "split_manifest.csv", index=False
    )

    endmembers = {}
    for name, condition in conditions.items():
        endmembers[name], _ = reference_endmembers(condition, split["reference"])

    # Prepare well-disjoint train/validation arrays.  Pure references are also
    # included in PLS training as calibration endpoints, never in evaluation.
    train_wells = np.concatenate((split["train"], split["reference"]))
    train_high_x, train_high_y, _ = select(conditions["high_snr"], train_wells)
    train_low_x, train_low_y, _ = select(conditions["low_snr"], train_wells)
    balanced_low_x, balanced_low_y = balanced_select(
        conditions["low_snr"], train_wells, repeats_per_well=8, seed=args.seed
    )
    mixed_x = np.concatenate((train_high_x, balanced_low_x))
    mixed_y = np.concatenate((train_high_y, balanced_low_y))
    print(f"Fitting PLS on {len(train_high_x)} high-SNR spectra ...", flush=True)
    high_pls = fit_pls(train_high_x, train_high_y, args.pls_components)
    print(f"Fitting PLS on {len(train_low_x)} low-SNR spectra ...", flush=True)
    low_pls = fit_pls(train_low_x, train_low_y, args.pls_components)
    print(f"Fitting PLS on {len(mixed_x)} balanced mixed-SNR spectra ...", flush=True)
    mixed_pls = fit_pls(mixed_x, mixed_y, args.pls_components)

    predictions_path = output / "test_predictions.csv"
    results = {
        "dataset": str(data_dir),
        "split_unit": "physical well / unique formulation",
        "seed": args.seed,
        "components": list(COMPONENTS),
        "wavenumber_range_cm-1": [args.min_wavenumber, args.max_wavenumber],
        "bands": int(len(conditions["high_snr"]["wave"])),
        "well_counts": {key: int(len(value)) for key, value in split.items()},
        "models": {},
    }
    plot_records = []
    for condition_name, condition in conditions.items():
        val_x, val_y, _ = select(condition, split["validation"])
        test_x, test_y, test_keep = select(condition, split["test"])
        predictions = {
            "fcls_pure_reference": (
                fcls(val_x, endmembers[condition_name], args.fcls_iterations),
                fcls(test_x, endmembers[condition_name], args.fcls_iterations),
            ),
            "pls_high_snr_train": (
                project_simplex(high_pls.predict(val_x)),
                project_simplex(high_pls.predict(test_x)),
            ),
            "pls_condition_specific": (
                project_simplex(
                    (high_pls if condition_name == "high_snr" else low_pls).predict(val_x)
                ),
                project_simplex(
                    (high_pls if condition_name == "high_snr" else low_pls).predict(test_x)
                ),
            ),
            "pls_balanced_mixed_snr_train": (
                project_simplex(mixed_pls.predict(val_x)),
                project_simplex(mixed_pls.predict(test_x)),
            ),
        }
        for model_name, (val_prediction, test_prediction) in predictions.items():
            threshold = choose_presence_threshold(val_y, val_prediction)
            result = metrics(
                test_y, test_prediction, test_x, endmembers[condition_name], threshold
            )
            results["models"].setdefault(model_name, {})[condition_name] = result
            write_predictions(
                predictions_path, condition_name, model_name,
                condition, test_keep, test_prediction,
            )
            print(
                f"{model_name:24s} {condition_name:8s}: "
                f"RMSE={result['abundance_rmse']:.4f}, "
                f"MAE={result['abundance_mae']:.4f}, "
                f"R2={result['abundance_r2']:.4f}, "
                f"strict={100*result['strict_sugar_presence_accuracy']:.2f}%, "
                f"F1={result['macro_sugar_presence_f1']:.4f}, "
                f"SAM={result['reconstruction_sam_degrees']:.3f} deg",
                flush=True,
            )
            if model_name == "pls_condition_specific":
                plot_records.append({
                    "condition": condition_name,
                    "truth": test_y,
                    "prediction": test_prediction,
                })

    summary_rows = []
    for model_name, by_condition in results["models"].items():
        for condition_name, value in by_condition.items():
            summary_rows.append({
                "model": model_name,
                "condition": condition_name,
                **{key: val for key, val in value.items() if key != "per_component"},
            })
    pd.DataFrame(summary_rows).to_csv(output / "metrics_summary.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_predictions(output / "abundance_predictions.png", plot_records)
    print(f"Saved backtest to: {output}")


if __name__ == "__main__":
    main()
