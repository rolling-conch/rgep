"""All-class extraction on the exact cached real inputs used by R0."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from models.ReferenceGuidedExtractor import ReferenceGuidedExtractor
from reference_extraction_data import query_all, score_presence
from rssnet_data import normalize_input, write_json
from fixed_splits import file_sha256
from decoupled_data import DEFAULT_CATEGORIES

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--conditions', nargs='+', type=int, choices=range(1, 10), required=True)
    p.add_argument('--baseline', default='run/RSSNet_OriginalP_20260921_seed19580920/Matched_rssnet_20260921_065731_269111')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    if a.batch_size < 1 or len(set(a.conditions)) != len(a.conditions):
        p.error('Positive batch size and unique conditions required')
    run, baseline, output = Path(a.run_dir), Path(a.baseline), Path(a.output)
    cfg = json.loads((run / 'config.json').read_text())
    state = json.loads((run / 'status.json').read_text())
    weights = run / 'checkpoint/best.weights.h5'
    if state['state'] != 'completed' or state['checkpoint_sha256'] != file_sha256(weights):
        raise ValueError('Run not completed or checkpoint changed')
    if cfg['args']['smoke'] or cfg['args'].get('short_check'):
        raise ValueError('Refusing formal real evaluation of smoke-trained weights')
    with np.load(run / 'reference.npz') as f:
        refs, names = f['reference'], f['names']
    assert names.tolist() == DEFAULT_CATEGORIES
    with np.load(baseline / 'reference.npz') as f:
        np.testing.assert_array_equal(refs, f['reference'])
        np.testing.assert_array_equal(names, f['names'])
    bc = json.loads((baseline / 'config.json').read_text())
    assert cfg['original_selected_rows_sha256'] == bc['original_selected_rows_sha256']
    assert cfg['source_manifest_sha256'] == bc['pure_manifest_sha256']
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = ReferenceGuidedExtractor(**cfg['architecture'])
    model(dict(mixture=np.zeros((1,1024,1),dtype='float32'), reference=refs[:1,:,None]))
    model.load_weights(str(weights))
    inference = tf.function(model, reduce_retracing=True)
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for condition in a.conditions:
        src = baseline / f'result/condition{condition}'
        prov = json.loads((src / 'provenance.json').read_text())
        with np.load(src / 'predictions.npz', allow_pickle=False) as f:
            inputs, ids, pairs = f['input'], f['sample_index'], f['pairs']
            baseline_correct = f['pair_correct']
        x, scale = normalize_input(inputs)
        print(f'Dataset {condition}: {len(x)} spectra, querying all seven references', flush=True)
        write_json(output / 'status.json', dict(state='evaluating', condition=condition))
        spectra, probs = query_all(inference, x, refs, a.batch_size)
        # Labels first used here, never used to choose queries or model inputs.
        truth = np.zeros((len(x), 7), dtype='int32')
        for i, pair in enumerate(pairs):
            truth[i, names.tolist().index(pair[:2])] = 1
            truth[i, names.tolist().index(pair[2:])] = 1
        metrics, labels = score_presence(truth, probs)
        correct = np.all(labels == truth, axis=1)
        metrics.update(R0_independent_pair_accuracy=float(baseline_correct.mean()),
            delta_vs_R0_pp=float(100*(correct.mean()-baseline_correct.mean())),
            warning='Reference-conditioned presence Top2 vs blind separation+matching; NOT identical decision rule or compute. Previously inspected benchmark. No real endpoint truth.')
        if cfg['args'].get('presence_only'):
            metrics['warning'] += ' Presence-only ablation: saved spectra are NOT trained extraction outputs.'
        dest = output / f'condition{condition}'
        dest.mkdir()
        write_json(dest / 'metrics.json', metrics)
        write_json(dest / 'provenance.json', dict(args=vars(a), model_sha256=file_sha256(weights),
            reference_sha256=file_sha256(run / 'reference.npz'), source_predictions_sha256=file_sha256(src / 'predictions.npz'),
            input_sha256=prov['input_sha256'], split_manifest_sha256=prov['split_manifest_sha256']))
        np.savez_compressed(dest / 'predictions.npz', sample_index=ids, pairs=pairs, truth=truth,
            probabilities=probs, labels=labels, all_query_spectra=spectra*scale[...,None], names=names)
        summary[str(condition)] = metrics
        write_json(output / 'partial_summary.json', summary)
        print(condition, metrics, flush=True)
    write_json(output / 'summary.json', summary)
    n = sum(m['samples'] for m in summary.values())
    write_json(output / 'aggregate.json', dict(samples=n,
        top2_pair_accuracy=sum(m['samples']*m['top2_pair_accuracy'] for m in summary.values())/n,
        fixed05_subset_accuracy=sum(m['samples']*m['fixed05_subset_accuracy'] for m in summary.values())/n,
        R0_independent_pair_accuracy=sum(m['samples']*m['R0_independent_pair_accuracy'] for m in summary.values())/n))
    write_json(output / 'status.json', dict(state='completed', conditions=a.conditions, samples=n))

if __name__ == '__main__':
    main()
