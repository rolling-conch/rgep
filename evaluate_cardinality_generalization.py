"""Frozen binary-trained models on held-out singleton/binary/ternary spectra.

Inference only: never fit a model, pick a checkpoint, or calibrate a threshold.
New noise samples share the legacy augmentation law. Binary arrays are reused.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
import numpy as np
from cardinality_data import simulate_cardinality
from fixed_splits import file_sha256
from rssnet_data import pure_partitions, write_json
from reference_background_nnls import BackgroundNNLS, evaluate_labels
from subset_background_bic import SubsetBackgroundBIC

ANCHOR = Path('run/ReferenceDataOnline_seed19580920/19580920/online_simulated/training/ReferenceExtractor_20260924_151025_266187')
PRESENCE_ROOT = Path('run/ReferenceOnlinePresenceOnly_seed19580920_20260925')
CSAM = Path('run/CSAM_RGEP_OnlineAligned_seed19580920_20260925')
CLASSIC = Path('run/ReferenceDenoisedNNLS_online_seed19580920_20260925')
NMF = Path('run/NMF_OnlineComparison_seed19580920_20260925')
BIC = Path('run/SubsetBackgroundBIC_grid128_20261005')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def digest(values):
    import hashlib
    h = hashlib.sha256()
    for name in sorted(values):
        a = np.ascontiguousarray(values[name])
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def statistics(truth, labels):
    m = evaluate_labels(truth, labels)
    m['exact_set_correct'] = int(np.all(truth == labels, axis=1).sum())
    m['count_correct'] = int((truth.sum(1) == labels.sum(1)).sum())
    m['predicted_count_histogram_0_to_7'] = np.bincount(labels.sum(1).astype(int), minlength=8).tolist()
    return m


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', default='run/Cardinality_Frozen_20261006')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Use a new output directory; frozen results are not overwritten')
    out.mkdir(parents=True)
    started = time.monotonic()
    frozen = {}

    def lock(path):
        frozen[str(path)] = file_sha256(path)
        return read(path)

    cfg = lock(ANCHOR/'config.json')
    protocol = lock(CLASSIC/'protocol.json')
    presence_comparison = lock(PRESENCE_ROOT/'comparison.json')
    presence = Path(presence_comparison['presence_only_model'])
    lock(presence/'config.json')
    csam_cfg = lock(CSAM/'config.json')
    thresholds = {k: v['threshold'] for k, v in protocol['selections'].items()}
    thresholds['presence_only'] = presence_comparison['real']['paper-denoised']['pooled']['presence']['locked_threshold']
    thresholds['CSAM'] = lock(CSAM/'result/threshold.json')['threshold']
    nmf_selections = lock(NMF/'selection.json')
    for key in ('NMF_batch', 'NMF_online'):
        thresholds[key] = nmf_selections[key]['threshold']['threshold']
    bic_selection = lock(BIC/'selection.json')
    for model in (ANCHOR, presence, CSAM):
        state = lock(model/'status.json')
        weights = model/'checkpoint/best.weights.h5'
        frozen[str(weights)] = file_sha256(weights)
        assert state['state'] == 'completed' and frozen[str(weights)] == state['checkpoint_sha256']
    assert frozen[str(ANCHOR/'checkpoint/best.weights.h5')] == protocol['model_sha256']
    manifest = Path(cfg['args']['manifest'])
    frozen[str(manifest)] = file_sha256(manifest)
    assert frozen[str(manifest)] == cfg['source_manifest_sha256'] == protocol['source_manifest_sha256']
    frozen[str(ANCHOR/'reference.npz')] = file_sha256(ANCHOR/'reference.npz')
    assert frozen[str(ANCHOR/'reference.npz')] == protocol['reference_sha256']
    libraries = {}
    for key in ('NMF_batch', 'NMF_online'):
        path = NMF/f'{key}_frozen.npz'
        frozen[str(path)] = file_sha256(path)
        libraries[key] = arrays(path)['library']
    # All inference rules and checkpoint identities are locked before test loading.
    write_json(out/'protocol.json', dict(thresholds=thresholds, bic_selection=bic_selection,
        frozen_files=frozen, train_reference_policy='training-partition medians only',
        seed=20261006, samples_per_synthetic_combination=64,
        ternary_weights='uniform simplex with every weight >=.1: .1+.7*Dirichlet(1,1,1)',
        noise='legacy white+cumulative noise, SNR U(10,30) dB, U-shape background a U(0,.08)',
        measured_singleton='107 held-out reconstructed pure records, no added perturbation',
        scope='binary-trained checkpoints and binary-validation-locked rules; no retraining or calibration',
        grouping='synthetic rows reuse held-out source records; not independent physical acquisitions'))
    pure, refs, metadata = pure_partitions(cfg['args']['source'], manifest, create=False)
    assert metadata == cfg['source']
    np.testing.assert_array_equal(refs, arrays(ANCHOR/'reference.npz')['reference'])
    single = np.concatenate(pure['test'])[..., None]
    y_single = np.eye(7, dtype=bool)[np.concatenate([np.full(len(p), i, int) for i, p in enumerate(pure['test'])])]
    assert len(single) == 107
    binary_saved = arrays(ANCHOR/'result/synthetic_predictions.npz')
    binary = {k: binary_saved[k] for k in ('x', 'y', 'pair', 'alpha')}
    assert digest(binary) == cfg['test_array_sha256']
    y_binary = np.zeros((len(binary['x']), 7), bool)
    np.put_along_axis(y_binary, binary['pair'], True, axis=1)
    datasets = {'measured_1': dict(x=single, truth=y_single),
        'synthetic_1': simulate_cardinality(pure['test'], 64, np.random.SeedSequence([20261006, 0x43415244, 1]), 1),
        'synthetic_2': dict(x=binary['x'], truth=y_binary, classes=binary['pair'], alpha=binary['alpha']),
        'synthetic_3': simulate_cardinality(pure['test'], 64, np.random.SeedSequence([20261006, 0x43415244, 3]), 3)}
    predictions = {name: {} for name in datasets}
    results = {}
    for name, data in datasets.items():
        np.savez_compressed(out/f'{name}_inputs.npz', **data)
        write_json(out/'status.json', dict(state='evaluating_classical', dataset=name))
        print('CLASSICAL', name, len(data['x']), flush=True)
        for method, solver in [('NNLS', BackgroundNNLS(refs)),
                               ('NNLS_background', BackgroundNNLS(refs, protocol['background_degree']))] + [
                               (k, BackgroundNNLS(lib)) for k, lib in libraries.items()]:
            scores, _ = solver.predict(data['x'])
            predictions[name][method] = scores >= thresholds[method]
            np.savez_compressed(out/f'{name}_{method}.npz', scores=scores, labels=predictions[name][method])
        for method in ('standard_BIC', 'calibrated_BIC'):
            selection = bic_selection[method]
            labels = SubsetBackgroundBIC(refs, selection['degree']).predict(data['x'], selection['penalty'])['labels']
            predictions[name][method] = labels

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    from models.ReferenceGuidedExtractor import ReferenceGuidedExtractor
    from reference_extraction_data import query_all
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    for method, model_run in [('RGEP', ANCHOR), ('presence_only', presence)]:
        mc = read(model_run/'config.json')
        model = ReferenceGuidedExtractor(**mc['architecture'])
        model(dict(mixture=np.zeros((1, 1024, 1), np.float32), reference=refs[:1, :, None]))
        model.load_weights(str(model_run/'checkpoint/best.weights.h5'))
        infer = tf.function(model, reduce_retracing=True)
        for name, data in datasets.items():
            write_json(out/'status.json', dict(state='evaluating_network', method=method, dataset=name))
            print('NETWORK', method, name, flush=True)
            # Reuse the already-frozen binary test predictions, not a new forward pass.
            if name == 'synthetic_2':
                path = model_run/'result/synthetic_predictions.npz'
                scores = arrays(path)['probabilities']
            else:
                _, scores = query_all(infer, data['x'], refs, 32)
            if scores.shape != data['truth'].shape or not np.isfinite(scores).all():
                raise ValueError('Invalid scores')
            predictions[name][method] = scores >= thresholds[method]
            np.savez_compressed(out/f'{name}_{method}.npz', scores=scores, labels=predictions[name][method])
        del model, infer
        tf.keras.backend.clear_session()
    from train_matched_csam_classifier import fresh_classifier
    model = fresh_classifier(csam_cfg['args']['template'])
    model.load_weights(str(CSAM/'checkpoint/best.weights.h5'))
    infer = tf.function(model, reduce_retracing=True)
    for name, data in datasets.items():
        write_json(out/'status.json', dict(state='evaluating_network', method='CSAM', dataset=name))
        print('NETWORK CSAM', name, flush=True)
        scores = (arrays(CSAM/'result/synthetic_predictions.npz')['probabilities'] if name == 'synthetic_2'
                  else np.concatenate([np.asarray(infer(data['x'][i:i+32], training=False))
                                       for i in range(0, len(data['x']), 32)]))
        predictions[name]['CSAM'] = scores >= thresholds['CSAM']
        np.savez_compressed(out/f'{name}_CSAM.npz', scores=scores, labels=predictions[name]['CSAM'])

    # Reuse all 56,700 measured binary predictions from the published main table.
    pooled_truth, pooled = [], {key: [] for key in predictions['measured_1']}
    for condition in range(1, 10):
        filename = f'condition{condition}_predictions.npz'
        base = arrays(CLASSIC/filename)
        truth = base['truth'].astype(bool)
        pooled_truth.append(truth)
        other = {**{k: base[k+'_threshold'] for k in ('RGEP', 'NNLS', 'NNLS_background')},
            'presence_only': arrays(PRESENCE_ROOT/'paper-denoised'/filename)['RGEP_threshold'],
            'CSAM': arrays(CSAM/'result/paper-denoised'/filename)['probabilities'] >= thresholds['CSAM']}
        for root in (PRESENCE_ROOT/'paper-denoised', CSAM/'result/paper-denoised', NMF/'paper-denoised', BIC/'paper-denoised'):
            np.testing.assert_array_equal(arrays(root/filename)['truth'].astype(bool), truth)
        nm = arrays(NMF/'paper-denoised'/filename)
        for key in libraries:
            other[key] = nm[key] >= thresholds[key]
        bi = arrays(BIC/'paper-denoised'/filename)
        for key in ('standard_BIC', 'calibrated_BIC'):
            other[key] = bi[key+'_labels']
        for key in pooled:
            pooled[key].append(other[key])
    predictions['measured_2'] = {k: np.concatenate(v) for k, v in pooled.items()}
    datasets['measured_2'] = dict(truth=np.concatenate(pooled_truth))
    for name, data in datasets.items():
        truth = data['truth']
        results[name] = dict(samples=len(truth), true_count=int(truth.sum(1)[0]),
            methods={k: statistics(truth, v) for k, v in predictions[name].items()})
        np.savez_compressed(out/f'{name}_labels.npz', truth=truth, **predictions[name])
    for path, expected in frozen.items():
        assert file_sha256(path) == expected, 'Frozen input changed during evaluation: '+path
    write_json(out/'results.json', dict(protocol=read(out/'protocol.json'), datasets=results,
        dataset_array_sha256={k: digest(v) for k, v in datasets.items()}, frozen_files_unchanged=True,
        elapsed_seconds=time.monotonic()-started, tensorflow_version=tf.__version__,
        inference_device='CPU' if args.cpu else 'TensorFlow default device'))
    lines = ['# Frozen cardinality generalization', '',
             'All rules locked on the original binary validation set; no Top-2, oracle count, retraining or reselection.', '',
             '| Input | True count | N | Method | Count accuracy (%) | Exact-set accuracy (%) | Macro-F1 |',
             '|---|---:|---:|---|---:|---:|---:|']
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['input', 'true_count', 'N', 'method', 'count_accuracy', 'exact_set_accuracy', 'macro_f1', 'undercount', 'overcount'])
        for name, data in results.items():
            for key, m in data['methods'].items():
                writer.writerow([name, data['true_count'], data['samples'], key, m['count_accuracy'], m['exact_set_accuracy'], m['macro_f1'], m['undercount'], m['overcount']])
                lines.append(f"| {name} | {data['true_count']} | {data['samples']} | {key} | {100*m['count_accuracy']:.2f} | {100*m['exact_set_accuracy']:.2f} | {m['macro_f1']:.4f} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(out/'status.json', dict(state='completed', elapsed_seconds=time.monotonic()-started))
    print('COMPLETED', out, flush=True)


if __name__ == '__main__':
    main()
