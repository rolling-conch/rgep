"""Matched online presence-only arm, followed by locked real evaluations.

No real labels are used for training, checkpoint, or threshold choice.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ANCHOR = Path("run/ReferenceDataOnline_seed19580920/19580920/online_simulated/training/ReferenceExtractor_20260924_151025_266187")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def summarize(root, model):
    import numpy as np
    from reference_background_nnls import evaluate_labels
    joint_roots = {"raw":Path("run/ReferenceRawNNLS_online_seed19580920_20260925"),
                   "paper-denoised":Path("run/ReferenceDenoisedNNLS_online_seed19580920_20260925")}
    cfg, state = read(model/"config.json"), read(model/"status.json")
    if not cfg["args"]["presence_only"] or not cfg["comparison"]["audit_passed"]:
        raise ValueError("Expected audited presence-only model")
    acfg = read(ANCHOR/"config.json")
    def audit(path):
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if audit(model/"result/query_audit.jsonl") != audit(ANCHOR/"result/query_audit.jsonl"):
        raise ValueError("Completed epoch audit mismatch")
    with np.load(model/"result/selected_validation_predictions.npz") as z:
        val_y, val_p = z["truth"].astype(bool), z["probabilities"]
    low = float(np.where(~val_y, val_p, -np.inf).max())
    high = float(np.where(val_y, val_p, np.inf).min())
    report = dict(anchor=str(ANCHOR), presence_only_model=str(model), paired_audit_passed=True,
                  joint_selected_epoch=read(ANCHOR/"status.json")["best_epoch"],
                  presence_selected_epoch=state["best_epoch"],
                  total_updates_each=117600,
                  presence_selected_updates=2352*state["best_epoch"],
                  joint_selected_updates=2352*read(ANCHOR/"status.json")["best_epoch"],
                  full_budget_equal_selected_updates_may_differ=True,
                  spectral_supervision=False, real={})
    lines=["# 在线配方：联合模型与仅存在性模型", "",
           "同初始化、50轮逐轮合成谱、查询顺序及选模；光谱损失权重1→0。所选轮次可不同。",
           "仅存在性光谱头未受监督，其输出不作为提取质量证据。", "",
           "| 输入 | 模型 | Top-2/% | 固定0.5集合/% | 验证阈值集合/% | 数量/% | Macro-F1 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for source,joint_root in joint_roots.items():
        presence_root=root/source
        p_result=read(presence_root/"results.json")
        joint_result=read(joint_root/"results.json")
        scores={"joint":[], "presence":[]}; truths=[]; groups=[]
        for condition in range(1,10):
            with np.load(joint_root/f"condition{condition}_predictions.npz") as j, \
                 np.load(presence_root/f"condition{condition}_predictions.npz") as p:
                for field in ("truth","pairs","sample_index"):
                    np.testing.assert_array_equal(j[field],p[field])
                truths.append(p["truth"]); scores["joint"].append(j["RGEP"]); scores["presence"].append(p["RGEP"])
                g=np.empty(len(p["truth"]),dtype="U32")
                for pair in np.unique(p["pairs"]):
                    ix=np.flatnonzero(p["pairs"]==pair); assert len(ix)==300
                    g[ix]=[f"{pair}:{i//100}" for i in range(300)]
                groups.append(g)
        y=np.concatenate(truths).astype(bool); group=np.concatenate(groups)
        pooled={}; correct={}
        for arm,result in (("joint",joint_result),("presence",p_result)):
            sc=np.concatenate(scores[arm])
            threshold=result["protocol"]["selections"]["RGEP"]["threshold"]
            labels=sc>=threshold; correct[arm]=np.all(labels==y,1)
            fixed=evaluate_labels(y,sc>=.5)
            pooled[arm]=dict(top2=result["pooled"]["RGEP_top2"], threshold=result["pooled"]["RGEP_threshold"],
                             fixed05=fixed, locked_threshold=threshold)
            np.testing.assert_allclose(correct[arm].mean(), pooled[arm]["threshold"]["exact_set_accuracy"])
            values=[pooled[arm]["top2"]["exact_set_accuracy"],fixed["exact_set_accuracy"],
                    pooled[arm]["threshold"]["exact_set_accuracy"],pooled[arm]["threshold"]["count_accuracy"]]
            lines.append("| "+source+" | "+arm+" | "+" | ".join(f"{v*100:.2f}" for v in values)+
                         f" | {pooled[arm]['threshold']['macro_f1']:.4f} |")
        from run_subset_bic_comparison import paired
        interval=paired(correct["joint"],correct["presence"],group)
        # Fixed grid selected before this run. Descriptive only; never choose a new threshold.
        grid=np.unique(np.r_[np.round(np.arange(.1,.901,.05),2),pooled["presence"]["locked_threshold"]])
        scan=[dict(threshold=float(t),validation=evaluate_labels(val_y,val_p>=t)["exact_set_accuracy"],
                   real=evaluate_labels(y,np.concatenate(scores["presence"])>=t)["exact_set_accuracy"])
              for t in grid]
        plateau=None
        if low<high:
            s=np.concatenate(scores["presence"])
            lower=np.where(~y,s,-np.inf).max(1);upper=np.where(y,s,np.inf).min(1)
            events=np.unique(np.r_[low,high,lower[(lower>low)&(lower<high)],upper[(upper>low)&(upper<high)]])
            ts=np.unique(np.r_[events[1:],np.nextafter(events[:-1],np.inf)])
            valid=lower<upper
            accuracy=(np.searchsorted(np.sort(lower[valid]),ts,side="left")-
                      np.searchsorted(np.sort(upper[valid]),ts,side="left"))/len(y)
            plateau=dict(validation_perfect_interval_lower_exclusive=low,upper_inclusive=high,
                         real_accuracy_range=[float(accuracy.min()),float(accuracy.max())])
        report["real"][source]=dict(pooled=pooled,paired_joint_minus_presence_proxy63=interval,
                                    presence_threshold_grid=scan,presence_perfect_plateau=plateau)
    report["limitations"]="Single paired seed; proxy groups not verified independent specimens; non-significance is not equivalence; no automatic causal claim."
    lines+=["", "所选轮次：joint="+str(report["joint_selected_epoch"])+", presence="+str(state["best_epoch"]),
            "", report["limitations"]]
    for source, item in report["real"].items():
        v=item["paired_joint_minus_presence_proxy63"]
        lines.append(f"- {source}：联合减仅存在性阈值差 {v['delta_pp']:+.2f} pp，95%代理组区间 {v['ci95_pp']}。")
        if item["presence_perfect_plateau"] is not None:
            lines.append("- 仅存在性验证100%阈值区间扫描："+json.dumps(item["presence_perfect_plateau"]))
    write(root/"comparison.json",report)
    (root/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    start=time.time()
    def execute(command,logname,stage):
        write(root/"status.json",dict(state=stage,command=command,log=str(root/logname)))
        with (root/logname).open("x",encoding="utf-8") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        from fixed_splits import file_sha256
        for source in ("models/ReferenceGuidedExtractor.py","reference_extraction_data.py","reference_online_data.py","rssnet_data.py"):
            if file_sha256(source)!=file_sha256(ANCHOR/Path(source).name):
                raise ValueError("Implementation differs from joint anchor: "+source)
        write(root/"protocol.json",dict(anchor=str(ANCHOR),seed=19580920,epochs=50,batch_queries=32,
             intervention="Only spectral loss weight 1 -> 0; full architecture retained",
             controls="Same init fingerprint, data/query fingerprints every epoch, references, LR, optimizer, validation/test",
             checkpoint="full synthetic validation Top-2; tie balanced BCE; then earliest",
             real="raw and paper-denoised; validation-only locked thresholds per arm; no joint threshold reuse",
             interpretation="Do not assume extraction helps. If no robust benefit, describe reference-conditioned presence identification.",
             estimated_training_hours=5.3,
             note="Model initialization and final real evaluation additional; prediction branch is not supervised extraction."))
        command=[sys.executable,"-u","train_reference_extractor.py","--presence-only",
                 "--training-data","online_simulated","--checkpoint-selection","val_top2","--save-every-epoch",
                 "--version","2","--epochs","50","--batch-size","32","--seed","19580920",
                 "--learning-rate","0.0003","--train-per-pair","512",
                 "--evaluation-anchor",str(ANCHOR),"--compare-run",str(ANCHOR),
                 "--output-root",str(root/"training")]
        execute(command,"train.log","training")
        runs=list((root/"training").glob("ReferenceExtractor_*"))
        if len(runs)!=1 or read(runs[0]/"status.json")["state"]!="completed":
            raise RuntimeError("Training incomplete")
        model=runs[0]
        for source in ("raw","paper-denoised"):
            cmd=[sys.executable,"-u","evaluate_reference_raw_nnls.py","--run-dir",str(model),
                 "--output",str(root/source),"--input-source",source,"--batch-size","32"]
            if source=="paper-denoised":
                cmd+=["--compare-evaluation",str(root/"raw")]
            execute(cmd,"evaluate_"+source+".log","evaluating_"+source)
        write(root/"status.json",dict(state="summarizing",model=str(model)))
        summarize(root,model)
        write(root/"status.json",dict(state="completed",model=str(model),seconds=time.time()-start,
                                      report=str(root/"report.md")))
    except Exception as exc:
        write(root/"status.json",dict(state="failed",error=repr(exc)));raise


if __name__=="__main__":
    main()
