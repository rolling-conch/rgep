"""Matched CSAM direct-classification control; no real-mixture training."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from fixed_splits import file_sha256
from rssnet_data import pure_partitions, simulate, write_json, normalize_input
from matched_separators import array_digest, matched_learning_rate
from evaluate_rssnet_classification import endpoint_labels, score_labels
from decoupled_data import DEFAULT_CATEGORIES, top_k_multihot, load_labeled_mixtures


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fresh_classifier(template):
    import tools
    full = tf.keras.models.load_model(template,
        custom_objects=tools.load_custom_objects("CSAM_ResUNet_classification"), compile=False)
    branch = tf.keras.Model(full.input, full.get_layer("classification").output)
    fresh = tf.keras.models.clone_model(branch)
    fresh.trainable = True
    return fresh


def joint_model_from_classifier(template, classifier):
    """Same shared initial tensors as F; additional decoder randomly initialized."""
    import tools
    full = tf.keras.models.load_model(template,
        custom_objects=tools.load_custom_objects("CSAM_ResUNet_classification"), compile=False)
    def clone_layer(layer):
        config = layer.get_config()
        if layer.name in ("spectrum_A", "spectrum_B"):
            config["kernel_initializer"] = tf.keras.initializers.serialize(tf.keras.initializers.RandomNormal(stddev=1e-3))
            config["bias_initializer"] = tf.keras.initializers.serialize(tf.keras.initializers.Constant(.005))
        return layer.__class__.from_config(config)
    joint = tf.keras.models.clone_model(full, clone_function=clone_layer)
    for layer in classifier.layers:
        if layer.weights:
            joint.get_layer(layer.name).set_weights(layer.get_weights())
    return joint


class MatchedMixtures(tf.keras.utils.Sequence):
    def __init__(self, pure, settings, group="F", original_data=None):
        self.pure, self.settings, self.epoch, self.group = pure, settings, -1, group
        self.original_data = original_data
        self.on_epoch_end()

    def on_epoch_end(self):
        self.epoch += 1
        s = self.settings
        data_seed = s["seed"] + (0 if self.group in ("A", "AN", "P") else self.epoch)
        self.data = self.original_data if self.group == "P" else simulate(self.pure, s["train_per_pair"], data_seed, noise=not s["clean_input"],
                             fixed_alpha=.5 if self.group == "B" else None,
                             disable_noise=self.group in ("C", "AN"), disable_background=self.group == "D")
        self.order = np.random.default_rng(s["seed"] + self.epoch).permutation(len(self.data["x"]))
        self.labels = endpoint_labels(self.data["pair"], len(DEFAULT_CATEGORIES)).astype("float32")

    def __len__(self):
        return (len(self.order) + self.settings["batch_size"] - 1) // self.settings["batch_size"]

    def __getitem__(self, index):
        batch = self.settings["batch_size"]
        ids = self.order[index * batch:(index + 1) * batch]
        if self.group == "E":
            return self.data["x"][ids], dict(spectrum_A=self.data["y"][ids, :, :1],
                spectrum_B=self.data["y"][ids, :, 1:], classification=self.labels[ids])
        return self.data["x"][ids], self.labels[ids]


def evaluate(model, run, baseline, real_root, batch):
    # All model selection is finished before any real evaluation starts.
    rows = []
    for condition in range(1, 10):
        write_json(run / "status.json", dict(state="evaluating", condition=condition))
        source = read(real_root / "paper" / f"condition{condition}" / "source.json")
        path = Path(source["predictions"])
        if file_sha256(path) != source["predictions_sha256"]:
            raise ValueError("Baseline predictions changed")
        prov = read(path.parent / "provenance.json")
        if prov["model_sha256"] != read(baseline / "result/completed.json")["checkpoint_sha256"]:
            raise ValueError("Wrong separation baseline")
        mat = f"dataset/dataset_mixture_denoised/dataset{condition}_denoised.mat"
        if file_sha256(mat) != prov["input_sha256"]:
            raise ValueError("Real input file changed")
        all_x, all_y, all_pairs = load_labeled_mixtures(mat)
        with np.load(path, allow_pickle=False) as f:
            ids, inputs = f["sample_index"], f["input"]
            np.testing.assert_array_equal(inputs, all_x[ids])
            np.testing.assert_array_equal(f["pairs"], all_pairs[ids])
            sep_labels = endpoint_labels(f["endpoint_class_indices"], len(DEFAULT_CATEGORIES))
        truth = all_y[ids].astype(int)
        del all_x, all_y, all_pairs
        probabilities = model.predict(normalize_input(inputs)[0], batch_size=batch, verbose=0)
        if probabilities.shape != truth.shape or not np.isfinite(probabilities).all():
            raise ValueError("Invalid classifier probabilities")
        labels = top_k_multihot(probabilities, k=2)
        direct, confusion, correct = score_labels(truth, labels, DEFAULT_CATEGORIES)
        fixed05, _, _ = score_labels(truth, (probabilities > .5).astype(int), DEFAULT_CATEGORIES)
        separation, _, sep_correct = score_labels(truth, sep_labels, DEFAULT_CATEGORIES)
        dest = run / "result" / f"condition{condition}"
        dest.mkdir()
        comparison = dict(direct=direct, direct_fixed05=fixed05, separation=separation,
            delta_direct_minus_separation_pp=100 * (direct["strict_pair_accuracy"] - separation["strict_pair_accuracy"]),
            paired_counts=dict(direct_only=int((correct & ~sep_correct).sum()), separation_only=int((sep_correct & ~correct).sum()),
                               both_correct=int((correct & sep_correct).sum()), both_wrong=int((~correct & ~sep_correct).sum())))
        write_json(dest / "comparison.json", comparison)
        write_json(dest / "direct_confusion.json", confusion)
        write_json(dest / "provenance.json", dict(baseline_source=source, input_sha256=prov["input_sha256"],
            split_manifest_sha256=prov["split_manifest_sha256"], classifier_sha256=file_sha256(run / "checkpoint/best.weights.h5")))
        np.savez_compressed(dest / "predictions.npz", sample_index=ids, truth=truth, probabilities=probabilities,
                            direct_labels=labels, separation_labels=sep_labels)
        rows.append(dict(condition=condition, samples=len(truth), direct_accuracy=direct["strict_pair_accuracy"],
                         separation_accuracy=separation["strict_pair_accuracy"], direct_macro_f1=direct["macro_f1_7labels"],
                         separation_macro_f1=separation["macro_f1_7labels"]))
        write_json(run / "result/partial_real.json", rows)
        print(f"Dataset{condition}: direct={direct['strict_pair_accuracy']:.4%}; separation={separation['strict_pair_accuracy']:.4%}", flush=True)
    write_json(run / "result/real_summary.json", dict(per_condition=rows,
        condition_means={k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("condition", "samples")},
        warning="Same inputs/batch order/encoder graph family; different supervision and output heads. Repeatedly explored real benchmark, not new blind external test."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="run/matched_seed19580920/MatchedSeparators_20260920_100128_343744/paper/Matched_paper_20260920_100132_682707")
    p.add_argument("--real-results", default="run/matched_seed19580920/MatchedSeparators_20260920_100128_343744/all_conditions_20260920")
    p.add_argument("--output", required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Two epochs, 21 examples; no real evaluation; not formal comparison")
    p.add_argument("--group", choices=list("FABCDE")+["AN", "P"], default="F")
    p.add_argument("--source-audit", default="run/Original_Source_Audit_20260921")
    p.add_argument("--a-run", default="run/CSAM_Recipe_Ablation_20260921_seed19580920/A")
    p.add_argument("--f-run", default="run/CSAM_Direct_Matched_seed19580920")
    args = p.parse_args()
    run, baseline = Path(args.output), Path(args.baseline)
    run.mkdir(parents=True, exist_ok=False)
    (run / "checkpoint").mkdir()
    (run / "result").mkdir()
    write_json(run / "status.json", dict(state="initializing", time=datetime.now().isoformat()))
    try:
        config = read(baseline / "config.json")
        s = dict(config["args"])
        if args.group != "F" and not args.smoke:
            f = Path(args.f_run)
            if read(f / "status.json")["state"] != "completed" or read(f / "config.json")["settings"] != s:
                raise ValueError("F run is incomplete or has different settings")
            if file_sha256(f / "checkpoint/best.weights.h5") != read(f / "result/training_completed.json")["checkpoint_sha256"]:
                raise ValueError("F checkpoint changed")
        if config.get("protocol") != "matched-separation-v1" or config["model_kind"] != "paper":
            raise ValueError("Expected matched paper separator")
        if file_sha256(s["paper_template"]) != config["paper_template_sha256"]:
            raise ValueError("Paper architecture template changed")
        if args.smoke:
            s.update(epochs=2, train_per_pair=1, eval_per_pair=1, batch_size=2)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.utils.set_random_seed(s["seed"])
        manifest = baseline / "pure_manifest.json"
        if file_sha256(manifest) != config["pure_manifest_sha256"]:
            raise ValueError("Pure split changed")
        pure, _, _ = pure_partitions(s["source"], manifest, s["split_seed"], struct=s["pure_struct"])
        original_data = None
        if args.group == "P":
            audit_path = Path(args.source_audit)
            audit = read(audit_path / "report.json")
            if (not audit['ready_for_P'] or audit['source_sha256'] != config['pure_source']['source_sha256'] or
                audit['manifest_sha256'] != config['pure_manifest_sha256'] or
                file_sha256(audit_path/'P_train.npz') != audit['P_train_sha256']):
                raise ValueError('Original data mapping/provenance mismatch')
            with np.load(audit_path/'P_train.npz', allow_pickle=False) as data:
                original_data = {key: data[key].copy() for key in ('x','pair')}
            if len(original_data['x']) != 21 * config['args']['train_per_pair']:
                raise ValueError('Original-data budget differs from A')
            if args.smoke:
                ids = [int(np.flatnonzero(np.all(original_data['pair']==pair,axis=1))[0])
                       for pair in np.unique(original_data['pair'],axis=0)]
                original_data = {k:v[ids] for k,v in original_data.items()}
            shutil.copy2(audit_path/'report.json', run/'original_source_audit.json')
            shutil.copy2(audit_path/'P_selected_rows.json', run/'original_selected_rows.json')
        validation = simulate(pure["validation"], s["eval_per_pair"], s["split_seed"] + 100, noise=not s["clean_input"])
        test = simulate(pure["test"], s["eval_per_pair"], s["split_seed"] + 200, noise=not s["clean_input"])
        for key, data in (("validation", validation), ("test", test)):
            if not args.smoke and array_digest(data) != config[f"{key}_array_sha256"]:
                raise ValueError("Different held-out synthetic data")
        expected_audit = [json.loads(line) for line in (baseline / "result/data_audit.jsonl").read_text().splitlines()]
        if not args.smoke and len(expected_audit) != s["epochs"]:
            raise ValueError("Baseline audit incomplete")
        for name in (__file__, "rssnet_data.py", "matched_separators.py", "evaluate_rssnet_classification.py"):
            shutil.copy2(name, run / Path(name).name)
        shutil.copy2(manifest, run / "pure_manifest.json")
        model = fresh_classifier(s["paper_template"])
        initial_hash = array_digest({str(i): w for i, w in enumerate(model.get_weights())})
        if args.group in ('P','AN') and not args.smoke:
            if initial_hash != read(Path(args.a_run)/'config.json')['classifier_initial_sha256']:
                raise ValueError('Initial weights differ from A')
        if args.group == "E":
            model = joint_model_from_classifier(s["paper_template"], model)
        (run / "architecture.json").write_text(model.to_json(), encoding="utf-8")
        write_json(run / "config.json", dict(settings=s, baseline=str(baseline), smoke=args.smoke, group=args.group,
            original_source_audit=args.source_audit if args.group=='P' else None,
            classifier_initial_sha256=initial_hash,
            ablation="A fixed epoch0 data; AN fixed epoch0 without noise; P mapped original matched-size data; B alpha=.5; C no noise; D no background; E joint; F full",
            joint_loss="E: endpoint MSE + .005 sparse + .005 smooth each; weights A=1 B=1 classification=.5; no ramp; output init std=.001 bias=.005" if args.group == "E" else None,
            parameters=model.count_params(), selection="minimum synthetic validation BCE; no real selection",
            initialization="random weights, not pretrained; same encoder graph as paper separator, classifier branch retained",
            classification_rule="top2 sigmoid probabilities; two known distinct classes",
            baseline_config_sha256=file_sha256(baseline / "config.json"),
            architecture_sha256=file_sha256(run / "architecture.json")))
        val_targets = endpoint_labels(validation["pair"], 7)
        monitor = "val_loss"
        if args.group == "E":
            from models.CSAM_ResUNet import DynamicHybridPhysicsLoss
            spectral_loss = DynamicHybridPhysicsLoss(mse_weight=1., sparse_strength=.005, smooth_strength=.005)
            model.compile(optimizer=tf.keras.optimizers.Adam(s["learning_rate"], clipnorm=5.),
                loss=dict(spectrum_A=spectral_loss, spectrum_B=spectral_loss, classification="binary_crossentropy"),
                loss_weights=dict(spectrum_A=1., spectrum_B=1., classification=.5))
            val_targets = dict(spectrum_A=validation["y"][:, :, :1], spectrum_B=validation["y"][:, :, 1:], classification=val_targets)
            monitor = "val_classification_loss"
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(s["learning_rate"], clipnorm=5.), loss="binary_crossentropy")
        mixtures = MatchedMixtures(pure["train"], s, args.group, original_data)
        class Audit(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                if mixtures.epoch != epoch:
                    raise ValueError("Sequence epoch mismatch")
                data_seed = s["seed"] + (0 if args.group in ("A", "AN", "P") else epoch)
                record = dict(epoch=epoch, data_seed=data_seed, sha256=array_digest(dict(mixtures.data, order=mixtures.order)))
                if not args.smoke and args.group in ("F", "E") and record != expected_audit[epoch]:
                    raise ValueError("Training examples/order differ from baseline")
                with (run / "result/data_audit.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                write_json(run / "status.json", dict(state="training", epoch=epoch+1, total_epochs=s["epochs"]))
        best = run / "checkpoint/best.weights.h5"
        print(f"Run: {run}; parameters={model.count_params()}; epochs={s['epochs']}", flush=True)
        history = model.fit(mixtures, validation_data=(validation["x"], val_targets),
            epochs=s["epochs"], shuffle=False, workers=1, max_queue_size=2, verbose=2,
            callbacks=[Audit(), tf.keras.callbacks.CSVLogger(str(run / "result/history.csv")),
                tf.keras.callbacks.LearningRateScheduler(lambda e: matched_learning_rate(e, s["learning_rate"])),
                tf.keras.callbacks.ModelCheckpoint(str(best), monitor=monitor, save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.TerminateOnNaN()])
        if len(history.history["loss"]) != s["epochs"] or not np.isfinite(history.history["loss"] + history.history["val_loss"] + history.history[monitor]).all():
            raise ValueError("Incomplete/nonfinite training")
        model.save_weights(str(run / "checkpoint/final.weights.h5"))
        model.load_weights(str(best))
        if args.group == "E":
            model = tf.keras.Model(model.input, model.get_layer("classification").output)
        probabilities = model.predict(test["x"], batch_size=s["batch_size"], verbose=0)
        metric, _, _ = score_labels(endpoint_labels(test["pair"], 7), top_k_multihot(probabilities, k=2), DEFAULT_CATEGORIES)
        write_json(run / "result/synthetic_metrics.json", metric)
        write_json(run / "result/training_completed.json", dict(epochs=s["epochs"],
            best_epoch=int(np.argmin(history.history[monitor]))+1, matched_data_audit=not args.smoke and args.group in ("F", "E"),
            checkpoint_sha256=file_sha256(best)))
        if not args.smoke:
            evaluate(model, run, baseline, Path(args.real_results), s["batch_size"])
        write_json(run / "status.json", dict(state="completed", smoke=args.smoke))
        print(f"Completed: {run}", flush=True)
    except Exception as exc:
        write_json(run / "status.json", dict(state="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
