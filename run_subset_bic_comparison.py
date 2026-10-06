"""CPU validation-locked subset NNLS+BIC, without a two-component constraint."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from subset_background_bic import SubsetBackgroundBIC
from reference_background_nnls import evaluate_labels
from evaluate_reference_raw_nnls import load_raw
from fixed_splits import file_sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def array_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load_arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def truth_of(data):
    y = np.zeros((len(data["x"]), 7), dtype=bool)
    np.put_along_axis(y, data["pair"], True, axis=1)
    return y


def paired(rgep, baseline, groups):
    _, inv = np.unique(groups, return_inverse=True)
    delta = rgep.astype(float) - baseline.astype(float)
    sums, sizes = np.bincount(inv, weights=delta), np.bincount(inv)
    draws = np.random.default_rng(19580920).integers(len(sums), size=(10000, len(sums)))
    difference = 100*sums[draws].sum(1)/sizes[draws].sum(1)
    return dict(delta_pp=float(delta.mean()*100), ci95_pp=np.percentile(difference, [2.5, 97.5]).tolist(),
                groups=len(sums), rgep_only_correct=int(np.sum(rgep & ~baseline)),
                baseline_only_correct=int(np.sum(baseline & ~rgep)))


def checked_penalties(values):
    """Keep the original grid by default; validate explicit sensitivity grids."""
    penalties = [float(v) for v in values]
    if (not penalties or not np.isfinite(penalties).all()
            or any(v <= 0 for v in penalties)
            or len(set(penalties)) != len(penalties) or 1. not in penalties):
        raise ValueError("Penalty grid must contain unique finite positive values and lambda=1")
    return sorted(penalties)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--penalties", type=float, nargs="+",
                        default=[.25, .5, 1., 2., 4., 8., 16., 32.],
                        help="Explicit lambda grid; default preserves the original protocol")
    args = parser.parse_args()
    try:
        penalties = checked_penalties(args.penalties)
    except ValueError as exc:
        parser.error(str(exc))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def log(message):
        print(message, flush=True)
        with (out/"progress.log").open("a", encoding="utf-8") as f:
            f.write(message+"\n")
    try:
        write(out/"status.json", dict(state="validating"))
        roots = {"raw": Path("run/ReferenceRawNNLS_online_seed19580920_20260925"),
                 "paper-denoised": Path("run/ReferenceDenoisedNNLS_online_seed19580920_20260925")}
        protocol = read(roots["raw"]/"protocol.json")
        anchor = Path(protocol["run"]); cfg = read(anchor/"config.json")
        assert file_sha256(anchor/"reference.npz") == protocol["reference_sha256"]
        refs = load_arrays(anchor/"reference.npz")["reference"]
        validation = load_arrays(anchor/"result/validation_data.npz")
        assert array_digest(validation) == protocol["validation_array_sha256"]
        truth = truth_of(validation)
        np.testing.assert_array_equal(truth, load_arrays(roots["raw"]/"validation_scores.npz", ["truth"])["truth"])
        degrees = [-1, 0, 1, 2, 3]
        audit = dict(anchor=str(anchor), source_manifest_sha256=protocol["source_manifest_sha256"],
                     reference_sha256=protocol["reference_sha256"],
                     validation_array_sha256=protocol["validation_array_sha256"],
                     criterion="n*log(RSS/n) + lambda*(active_material_count + background_rank)*log(n)",
                     n=1024, supports="all 128 subsets of seven classes, including empty and singleton",
                     degrees=degrees, penalties=penalties,
                     selection="validation exact-set; tie macro-F1, lambda closest to 1 in log scale, lower degree",
                     rules={"standard_BIC": "lambda=1; validation-selected degree",
                            "calibrated_BIC": "validation-selected lambda and degree; not textbook BIC"},
                     numerical_tolerance="active c >1e-10*input L2; relative RSS floor 1e-14",
                     limitations="Correlated Raman points, approximate library and nonnegative boundaries violate simple BIC assumptions. Binary-only validation can implicitly favor two outputs. Not mass fractions.",
                     hardware="CPU, BLAS 4 threads", script_sha256=file_sha256(__file__),
                     solver_sha256=file_sha256("subset_background_bic.py"))
        write(out/"protocol.json", audit)
        for name in (__file__, "subset_background_bic.py", "reference_background_nnls.py"):
            shutil.copy2(name, out/Path(name).name)
        candidates = []; solutions = {}
        def key(row):
            m = row["validation"]
            return (m["exact_set_accuracy"], m["macro_f1"], -abs(np.log(row["penalty"])), -row["degree"])
        for degree in degrees:
            solver = SubsetBackgroundBIC(refs, degree)
            rss, norm2 = solver.residuals(validation["x"])
            for penalty in penalties:
                result = solver.select(rss, norm2, penalty)
                row = dict(degree=degree, penalty=penalty, validation=evaluate_labels(truth, result["labels"]))
                candidates.append(row); solutions[(degree, penalty)] = result
            log(f"Validation degree {degree}: best exact-set {max(r['validation']['exact_set_accuracy'] for r in candidates if r['degree']==degree):.6f}")
        selected = {"standard_BIC": max((r for r in candidates if r["penalty"] == 1), key=key),
                    "calibrated_BIC": max(candidates, key=key)}
        write(out/"validation_candidates.json", candidates); write(out/"selection.json", selected)
        np.savez_compressed(out/"validation_predictions.npz", truth=truth,
                            **{k: solutions[(v["degree"],v["penalty"])]["labels"] for k,v in selected.items()})
        log("LOCKED before synthetic-test/real loading: "+json.dumps(selected))
        write(out/"status.json", dict(state="selection_locked"))
        if args.validate_only:
            write(out/"status.json", dict(state="validation_completed")); return
        solvers = {k: SubsetBackgroundBIC(refs,v["degree"]) for k,v in selected.items()}
        test = load_arrays(anchor/"result/synthetic_predictions.npz", ["x", "y", "pair", "alpha"])
        assert array_digest(test) == cfg["test_array_sha256"]
        ty = truth_of(test); test_metrics = {}; test_predictions = {}
        for method, solver in solvers.items():
            labels = solver.predict(test["x"], selected[method]["penalty"])["labels"]
            test_metrics[method] = evaluate_labels(ty, labels); test_predictions[method] = labels
        write(out/"synthetic_metrics.json", test_metrics)
        np.savez_compressed(out/"synthetic_predictions.npz", truth=ty, alpha=test["alpha"], **test_predictions)
        all_results = {}
        for source, previous in roots.items():
            folder = out/source; folder.mkdir()
            prior = read(previous/"results.json")
            assert prior["protocol"]["reference_sha256"] == audit["reference_sha256"]
            rows = []; collected = {k: [] for k in selected}; truths = []; groups = []; rg_ok = []
            for condition in range(1, 10):
                write(out/"status.json", dict(state="evaluating", source=source, condition=condition))
                path = Path("dataset/dataset_mixture_raw")/f"dataset{condition}_norm.mat" if source == "raw" else \
                       Path("dataset/dataset_mixture_denoised")/f"dataset{condition}_denoised.mat"
                x,y,pairs = load_raw(path, suffix="" if source=="raw" else "_norm")
                old = load_arrays(previous/f"condition{condition}_predictions.npz")
                record = next(r for r in prior["conditions"] if r["condition"]==condition)
                assert file_sha256(path) == record["input_sha256"]
                assert array_digest(dict(x=x)) == record["normalized_input_sha256"]
                for a,b in ((y,old["truth"]), (pairs,old["pairs"]), (np.arange(len(y)),old["sample_index"])):
                    np.testing.assert_array_equal(a,b)
                group = np.empty(len(y), dtype="U32")
                for pair in np.unique(pairs):
                    ids = np.flatnonzero(pairs==pair); assert len(ids)==300
                    group[ids] = [f"{pair}:{i//100}" for i in range(300)]
                metrics = {}; saved = {}
                for method, solver in solvers.items():
                    result = solver.predict(x, selected[method]["penalty"])
                    metrics[method] = evaluate_labels(y,result["labels"])
                    collected[method].append(result["labels"])
                    saved.update({method+"_"+k:v for k,v in result.items()})
                np.savez_compressed(folder/f"condition{condition}_predictions.npz",
                                    truth=y, pairs=pairs, sample_index=np.arange(len(y)), **saved)
                rows.append(dict(condition=condition, normalized_input_sha256=record["normalized_input_sha256"], metrics=metrics))
                truths.append(y); groups.append(group); rg_ok.append(np.all(old["RGEP_threshold"]==y,1))
                write(folder/"partial_results.json", rows)
                log(source+" condition "+str(condition)+" "+json.dumps(metrics))
            y = np.concatenate(truths); group = np.concatenate(groups); rg = np.concatenate(rg_ok)
            pooled = {k:evaluate_labels(y,np.concatenate(v)) for k,v in collected.items()}
            comparisons = {k:paired(rg,np.all(np.concatenate(v)==y,1),group) for k,v in collected.items()}
            all_results[source] = dict(conditions=rows, pooled=pooled,
                rgep_locked_threshold=prior["pooled"]["RGEP_threshold"],
                previous_coefficient_threshold={k:prior["pooled"][k+"_threshold"] for k in ("NNLS","NNLS_background")},
                paired_rgep_minus_bic_proxy63=comparisons)
            write(folder/"results.json",all_results[source])
        lines=["# 子集背景NNLS：组分数量自由选择", "",
               "允许0–7类，无Top-2及材料系数阈值；所有配置仅由合成验证集确定。", "",
               "| 方法 | 验证集合/% | 合成测试集合/% | 重构前集合/% | 去噪集合/% | 重构前数量/% | 去噪数量/% |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for method in selected:
            a,b=all_results["raw"]["pooled"][method],all_results["paper-denoised"]["pooled"][method]
            values=[selected[method]["validation"]["exact_set_accuracy"],test_metrics[method]["exact_set_accuracy"],
                    a["exact_set_accuracy"],b["exact_set_accuracy"],a["count_accuracy"],b["count_accuracy"]]
            lines.append("| "+method+" | "+" | ".join(f"{v*100:.2f}" for v in values)+" |")
        lines += ["", "## 锁定配置", "", json.dumps(selected,ensure_ascii=False,indent=2),
                  "", "## RGEP锁定阈值减BIC：63代理组配对区间", ""]
        for source,data in all_results.items():
            for method,m in data["paired_rgep_minus_bic_proxy63"].items():
                lines.append(f"- {source} / {method}: {m['delta_pp']:+.2f} pp, [{m['ci95_pp'][0]:+.2f}, {m['ci95_pp'][1]:+.2f}]。")
        lines += ["", "验证及实测真值均为二元混合，数量自由不等于验证了任意真实组分数。",
                  "相关光谱点、参考误差和非负边界使BIC渐近假设未保证；校准BIC不是标准BIC。",
                  "区间基于行块代理，非已确认独立物理样本，也不覆盖训练随机性。"]
        (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        write(out/"status.json",dict(state="completed",seconds=time.perf_counter()-start,samples_per_source=56700))
        log("COMPLETED "+str(out))
    except Exception as exc:
        write(out/"status.json",dict(state="failed",error=repr(exc))); raise


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
