"""Sequential, auditable data-recipe experiment. No automatic real test inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ANCHOR = "run/ReferenceCheckpointAlignment_20260924_seed19580920/joint/training/ReferenceExtractor_20260923_170259_526394"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--recipes",nargs="+",choices=["original_p","fixed_simulated","online_simulated"],
                   default=["original_p","fixed_simulated","online_simulated"])
    p.add_argument("--seeds",nargs="+",type=int,default=[19580920])
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--evaluation-anchor",default=ANCHOR)
    p.add_argument("--presence-only",action="store_true")
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or min(a.seeds)<0 or len(set(a.seeds))!=len(a.seeds) or len(set(a.recipes))!=len(a.recipes):
        p.error("Unique recipes/seeds and positive budgets required")
    root=Path(a.output)
    root.mkdir(parents=True,exist_ok=False)
    write(root/"protocol.json",dict(args=vars(a),
        comparison="original fixed P vs fixed augmented synthesis vs per-epoch fresh augmented synthesis",
        budget="10752 mixtures, 75264 queries per epoch; identical architecture and schedule",
        selection="full synthetic validation Top-2, tie balanced presence BCE; secondary best joint loss also saved",
        real_evaluation="separate explicit command after completion; never used to pick recipe/threshold",
        limitations="binary optical synthesis from reconstructed pure spectra; raw mixtures are domain-shift test, not guaranteed gain"))
    runs={}
    try:
        for seed in a.seeds:
            matched=None
            for recipe in a.recipes:
                key=f"{seed}/{recipe}"
                folder=root/str(seed)/recipe
                folder.mkdir(parents=True)
                command=[sys.executable,"-u","train_reference_extractor.py",
                    "--training-data",recipe,"--checkpoint-selection","val_top2","--save-every-epoch",
                    "--version","2","--epochs",str(a.epochs),"--batch-size",str(a.batch_size),
                    "--seed",str(seed),"--train-per-pair","512",
                    "--evaluation-anchor",a.evaluation_anchor,"--output-root",str(folder/"training")]
                if a.presence_only:command.append("--presence-only")
                if a.cpu:command.append("--cpu")
                write(root/"status.json",dict(state="training",arm=key,command=command,runs=runs))
                print("Starting",key,"log:",folder/"train.log",flush=True)
                with (folder/"train.log").open("x",encoding="utf-8") as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                found=list((folder/"training").glob("ReferenceExtractor_*"))
                if len(found)!=1 or read(found[0]/"status.json")["state"]!="completed":
                    raise RuntimeError("Training output incomplete")
                cfg=read(found[0]/"config.json")
                comparison={k:cfg[k] for k in ["source_manifest_sha256","validation_array_sha256",
                    "test_array_sha256","architecture","parameters","initial_weights_sha256","selection"]}
                if matched is not None and comparison!=matched:
                    raise RuntimeError("Cross-recipe configuration/initialization mismatch")
                matched=comparison
                runs[key]=str(found[0])
                print("Completed",key,flush=True)
        write(root/"status.json",dict(state="completed",runs=runs,real_evaluation_started=False))
    except Exception as exc:
        write(root/"status.json",dict(state="failed",error=repr(exc),runs=runs))
        raise


if __name__=="__main__":
    main()
