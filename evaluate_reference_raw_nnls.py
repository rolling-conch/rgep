"""Frozen RGEP versus plain/background NNLS on unreconstructed published spectra.

Model, background degree and thresholds are fixed using synthetic validation only.
No baseline removal or denoising is applied to the real mixtures; only shared L2.
"""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from scipy.io import loadmat
from decoupled_data import DEFAULT_CATEGORIES, pair_definitions, _as_spectra
from rssnet_data import normalize_input, write_json
from fixed_splits import file_sha256
from reference_background_nnls import (BackgroundNNLS, labels_top2, evaluate_labels,
                                       validation_threshold, select_background)


def load_raw(path, width=1024, suffix=""):
    # Exact requested variables only; never silently substitute an input source.
    definitions = pair_definitions(DEFAULT_CATEGORIES)
    keys = [p+suffix for p, _, _ in definitions]
    content = loadmat(path, variable_names=keys)
    spectra, truth, pairs = [], [], []
    for key, first, second in definitions:
        variable = key+suffix
        if variable not in content:
            raise KeyError(f"Missing requested variable {variable} in {path}")
        x = _as_spectra(content[variable], variable, width)
        if len(x) != 300:
            raise ValueError(f"Expected all 300 rows for {key}, got {len(x)}")
        y = np.zeros((len(x), 7), np.int32)
        y[:, [DEFAULT_CATEGORIES.index(first), DEFAULT_CATEGORIES.index(second)]] = 1
        spectra.append(x); truth.append(y); pairs.extend([key]*len(x))
    x, _ = normalize_input(np.concatenate(spectra))
    return x, np.concatenate(truth), np.asarray(pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--raw-root", default="dataset/dataset_mixture_raw")
    p.add_argument("--input-source", choices=("raw", "paper-denoised"), default="raw")
    p.add_argument("--denoised-root", default="dataset/dataset_mixture_denoised")
    p.add_argument("--compare-evaluation", help="Completed raw evaluation to lock choices and check row identities")
    p.add_argument("--conditions", nargs="+", type=int, choices=range(1,10), default=list(range(1,10)))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--validate-only", action="store_true", help="Check protocol on validation; never load real/test spectra")
    a = p.parse_args()
    if a.batch_size < 1 or len(set(a.conditions)) != len(a.conditions):
        p.error("Positive batch-size and unique conditions required")
    run, out = Path(a.run_dir), Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    cfg, state = read(run/"config.json"), read(run/"status.json")
    if state["state"] not in ("completed","short_check_completed"):
        raise ValueError("Training must finish before evaluation")
    if cfg["args"].get("checkpoint_selection") != "val_top2":
        raise ValueError("This protocol requires full-validation classification checkpoint selection")
    if not a.validate_only and (cfg["args"].get("smoke") or cfg["args"].get("short_check") or state["state"] != "completed"):
        raise ValueError("Smoke/short weights cannot be used for formal raw evaluation")
    weights = run/"checkpoint/best.weights.h5"
    if file_sha256(weights) != state["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    # Importing training utilities only after CLI CPU selection preserves GPU policy.
    from matched_separators import array_digest
    with np.load(run/"result/validation_data.npz") as f:
        validation = {k:f[k] for k in f.files}
    if array_digest(validation) != cfg["validation_array_sha256"]:
        raise ValueError("Validation arrays changed")
    truth = np.zeros((len(validation["x"]),7),np.int32)
    np.put_along_axis(truth,validation["pair"],1,axis=1)
    with np.load(run/"result/selected_validation_predictions.npz") as f:
        np.testing.assert_array_equal(truth,f["truth"])
        presence = f["probabilities"].astype(np.float64)
        np.testing.assert_array_equal(labels_top2(presence),f["labels"])
    selection_record = read(run/"result/checkpoint_selection.json")
    if selection_record["epoch"] != state["best_epoch"] or int(
            np.all(labels_top2(presence) == truth,axis=1).sum()) != selection_record["correct_samples"]:
        raise ValueError("Selected validation predictions do not match the checkpoint record")
    with np.load(run/"reference.npz") as f:
        refs, names = f["reference"], f["names"]
    if names.tolist() != DEFAULT_CATEGORIES:
        raise ValueError("Class order mismatch")
    degree, candidates, val_scores = select_background(refs, validation["x"], truth)
    selections = {k:validation_threshold(truth, scores) for k,scores in dict(
        RGEP=presence, NNLS=val_scores[-1], NNLS_background=val_scores[degree]).items()}
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(run=str(run), checkpoint_epoch=state["best_epoch"],
        model_sha256=file_sha256(weights), reference_sha256=file_sha256(run/"reference.npz"),
        validation_array_sha256=cfg["validation_array_sha256"],
        validation_prediction_sha256=file_sha256(run/"result/selected_validation_predictions.npz"),
        source_manifest_sha256=cfg["source_manifest_sha256"],
        background_degree=degree, background_candidates=candidates,
        degree_selection="maximum synthetic validation Top-2 correct count, tie lower degree; -1=no background",
        background="unconstrained smooth polynomial in spectral index; material coefficients nonnegative, not mass fractions",
        threshold_selection="maximum validation exact-set; tie higher threshold; locked before real input loading",
        selections=selections, conditions=a.conditions, validate_only=a.validate_only,
        input_policy="un-suffixed raw pair variables; common L2 only; no denoising/baseline subtraction",
        reference_policy="same training-only reconstructed pure median library for both methods",
        scope="raw means before reconstruction, not verified untouched detector counts; known library, binary truths",
        script_sha256=file_sha256(__file__), solver_sha256=file_sha256("reference_background_nnls.py"))
    protocol["input_source"] = a.input_source
    if a.input_source == "paper-denoised":
        protocol["input_policy"] = "paper-provided network-denoised *_norm variables; common L2; no additional denoising"
        protocol["scope"] = "published cached network-denoised mixtures, not newly generated local denoiser predictions; known library, binary truths"
    if a.compare_evaluation:
        previous = Path(a.compare_evaluation)
        if read(previous/"status.json")["state"] != "completed":
            raise ValueError("Comparison evaluation must be completed")
        prior = read(previous/"protocol.json")
        for key in ("model_sha256", "reference_sha256", "validation_array_sha256", "validation_prediction_sha256",
                    "source_manifest_sha256", "background_degree", "conditions", "selections", "solver_sha256"):
            if protocol[key] != prior[key]:
                raise ValueError(f"Paired comparison changed locked field: {key}")
        protocol["compare_evaluation"] = str(previous)
        protocol["compare_protocol_sha256"] = file_sha256(previous/"protocol.json")
    write_json(out/"protocol.json", protocol)
    np.savez_compressed(out/"validation_scores.npz", truth=truth, RGEP=presence,
                        NNLS=val_scores[-1], NNLS_background=val_scores[degree])
    if a.validate_only:
        write_json(out/"status.json", dict(state="validation_check_completed"))
        print("Validation only:", selections, "background degree:", degree, flush=True)
        return
    write_json(out/"status.json", dict(state="initializing", checkpoint_epoch=state["best_epoch"],
                                      background_degree=degree, thresholds_locked=True))
    print("Validation choices locked. Initializing frozen model:", run, flush=True)
    import tensorflow as tf
    from models.ReferenceGuidedExtractor import ReferenceGuidedExtractor
    from reference_extraction_data import query_all
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = ReferenceGuidedExtractor(**cfg["architecture"])
    model(dict(mixture=validation["x"][:1],reference=refs[:1,:,None]))
    model.load_weights(str(weights))
    inference = tf.function(model, reduce_retracing=True)
    solvers = dict(NNLS=BackgroundNNLS(refs), NNLS_background=BackgroundNNLS(refs,degree))
    results, all_truth, all_labels = [], [], {}
    try:
        for condition in a.conditions:
            write_json(out/"status.json",dict(state="evaluating",condition=condition))
            if a.input_source == "paper-denoised":
                path = Path(a.denoised_root)/f"dataset{condition}_denoised.mat"
                x, y, pairs = load_raw(path, suffix="_norm")
            else:
                path = Path(a.raw_root)/f"dataset{condition}_norm.mat"
                x, y, pairs = load_raw(path)
            if a.compare_evaluation:
                with np.load(Path(a.compare_evaluation)/f"condition{condition}_predictions.npz") as f:
                    np.testing.assert_array_equal(y, f["truth"])
                    np.testing.assert_array_equal(pairs, f["pairs"])
                    np.testing.assert_array_equal(np.arange(len(y)), f["sample_index"])
            spectra, scores_rg = query_all(inference,x,refs,a.batch_size)
            del spectra
            scores = dict(RGEP=scores_rg.astype(np.float64))
            for key, solver in solvers.items():
                scores[key], _ = solver.predict(x)
            labels, measured = {}, {}
            for key, value in scores.items():
                labels[key+"_top2"] = labels_top2(value)
                labels[key+"_threshold"] = value >= selections[key]["threshold"]
            for key, value in labels.items():
                measured[key] = evaluate_labels(y,value)
                all_labels.setdefault(key,[]).append(value)
            all_truth.append(y)
            results.append(dict(condition=condition, source=str(path),
                input_sha256=file_sha256(path), normalized_input_sha256=array_digest(dict(x=x)),
                metrics=measured))
            np.savez_compressed(out/f"condition{condition}_predictions.npz",
                truth=y,pairs=pairs,sample_index=np.arange(len(y)),**scores,**labels)
            write_json(out/"partial_results.json",results)
            print(a.input_source,"condition",condition,measured,flush=True)
        y = np.concatenate(all_truth)
        pooled = {key:evaluate_labels(y,np.concatenate(values)) for key,values in all_labels.items()}
        write_json(out/"results.json",dict(protocol=protocol,conditions=results,pooled=pooled))
        write_json(out/"status.json",dict(state="completed",samples=len(y)))
        print("Saved:",out,flush=True)
    except Exception as exc:
        write_json(out/"status.json",dict(state="failed",error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
