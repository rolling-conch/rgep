"""CSAM direct classification on the exact audited online RGEP mixture stream.

Data/epoch and checkpoint criterion matched; optimizer-update budgets NOT equal.
Never uses real-mixture labels for training, selection, or threshold calibration.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
if '--cpu' in sys.argv or '--check-only' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from fixed_splits import file_sha256
from matched_separators import array_digest, matched_learning_rate
from rssnet_data import pure_partitions, write_json
from reference_online_data import EpochMixtures, synthesis_seed
from reference_checkpoint import ClassificationCheckpoint, SELECTION_RULE
from reference_background_nnls import labels_top2, evaluate_labels, validation_threshold
from train_matched_csam_classifier import fresh_classifier


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def truth_labels(data):
    y = np.zeros((len(data['x']), 7), dtype=np.float32)
    np.put_along_axis(y, data['pair'], 1, axis=1)
    if not np.all(y.sum(1) == 2):
        raise ValueError('Expected two different components')
    return y


def assert_epoch(data, epoch, audit, seed):
    row = audit[epoch]
    if (row['epoch'] != epoch or row['synthesis_seed'] != synthesis_seed(seed, epoch)
            or row['mixtures'] != len(data['x'])
            or row['training_array_sha256'] != array_digest(data)):
        raise ValueError(f'RGEP training stream mismatch at epoch {epoch+1}')


class AlignedMixtures(tf.keras.utils.Sequence):
    def __init__(self, generator, audit, batch, epochs, smoke=False):
        self.generator, self.audit, self.batch = generator, audit, batch
        self.epochs, self.smoke, self.epoch = epochs, smoke, -1
        self.on_epoch_end()

    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch >= self.epochs:
            return
        self.data = self.generator.generate(self.epoch)
        assert_epoch(self.data, self.epoch, self.audit, self.generator.seed)
        self.labels = truth_labels(self.data)
        # Each mixture once. RGEP shuffles 7 query records per mixture instead.
        self.order = np.random.default_rng(self.generator.seed+self.epoch).permutation(len(self.labels))

    def __len__(self):
        full = (len(self.order)+self.batch-1)//self.batch
        return min(2, full) if self.smoke else full

    def __getitem__(self, index):
        ids = self.order[index*self.batch:(index+1)*self.batch]
        return self.data['x'][ids], self.labels[ids]


def direct_predict(model, x, batch):
    return np.concatenate([np.asarray(model(x[i:i+batch], training=False))
                           for i in range(0, len(x), batch)])


def decisions(y, p, threshold):
    return {rule: evaluate_labels(y, labels) for rule, labels in {
        'top2': labels_top2(p), 'fixed05': p >= .5, 'threshold': p >= threshold}.items()}


def prepare(anchor):
    cfg, state = read(anchor/'config.json'), read(anchor/'status.json')
    a = cfg['args']
    if (state['state'] != 'completed' or a.get('training_data') != 'online_simulated'
            or a.get('smoke') or a.get('short_check') or a['checkpoint_selection'] != 'val_top2'):
        raise ValueError('Expected completed formal online RGEP anchor')
    if file_sha256(anchor/'checkpoint/best.weights.h5') != state['checkpoint_sha256']:
        raise ValueError('Anchor checkpoint changed')
    if file_sha256(a['manifest']) != cfg['source_manifest_sha256']:
        raise ValueError('Pure manifest changed')
    if file_sha256('rssnet_data.py') != cfg['training_generator_sha256']:
        raise ValueError('Generator implementation changed')
    pure, refs, meta = pure_partitions(a['source'], a['manifest'], seed=19580920, create=False)
    if meta != cfg['source']:
        raise ValueError('Pure source changed')
    with np.load(anchor/'reference.npz') as f:
        np.testing.assert_array_equal(refs, f['reference'])
    data = {}
    for kind, filename in [('validation', 'validation_data.npz'), ('test', 'synthetic_predictions.npz')]:
        with np.load(anchor/'result'/filename) as f:
            data[kind] = {k:f[k] for k in ('x', 'y', 'pair', 'alpha')}
        if array_digest(data[kind]) != cfg[kind+'_array_sha256']:
            raise ValueError('Held-out arrays changed: '+kind)
    audit = [json.loads(line) for line in (anchor/'result/query_audit.jsonl').read_text().splitlines()]
    if len(audit) != a['epochs'] or [v['epoch'] for v in audit] != list(range(a['epochs'])):
        raise ValueError('Incomplete epoch audit')
    generator = EpochMixtures(pure['train'], a['train_per_pair'], a['seed'])
    return cfg, state, generator, refs, data, audit


def real_evaluation(model, run, threshold, batch, anchors, anchor_sha):
    from evaluate_reference_raw_nnls import load_raw
    report = {}
    for source, previous in anchors.items():
        prior = read(previous/'results.json')
        if (read(previous/'status.json')['state'] != 'completed'
                or prior['protocol']['model_sha256'] != anchor_sha
                or prior['protocol'].get('input_source', 'raw') != source
                or prior['protocol']['conditions'] != list(range(1, 10))):
            raise ValueError('Wrong real comparison anchor')
        dest = run/'result'/source
        dest.mkdir()
        ys, ps, rows = [], [], []
        for i in range(1, 10):
            write_json(run/'status.json', dict(state='evaluating_real', source=source, condition=i))
            path = (Path('dataset/dataset_mixture_raw')/f'dataset{i}_norm.mat' if source == 'raw'
                    else Path('dataset/dataset_mixture_denoised')/f'dataset{i}_denoised.mat')
            x, y, pairs = load_raw(path, suffix='' if source == 'raw' else '_norm')
            provenance = prior['conditions'][i-1]
            if (file_sha256(path) != provenance['input_sha256']
                    or array_digest(dict(x=x)) != provenance['normalized_input_sha256']):
                raise ValueError('Real input differs from RGEP evaluation')
            with np.load(previous/f'condition{i}_predictions.npz') as f:
                np.testing.assert_array_equal(y, f['truth'])
                np.testing.assert_array_equal(pairs, f['pairs'])
                np.testing.assert_array_equal(np.arange(len(y)), f['sample_index'])
            p = direct_predict(model, x, batch)
            if p.shape != y.shape or not np.isfinite(p).all():
                raise ValueError('Invalid real predictions')
            m = decisions(y, p, threshold)
            np.savez_compressed(dest/f'condition{i}_predictions.npz', truth=y, probabilities=p,
                                pairs=pairs, sample_index=np.arange(len(y)))
            rows.append(dict(condition=i, metrics=m, input_sha256=file_sha256(path)))
            ys.append(y); ps.append(p)
            write_json(dest/'partial_results.json', rows)
            print('REAL', source, i, m, flush=True)
        report[source] = dict(conditions=rows, pooled=decisions(np.concatenate(ys),np.concatenate(ps),threshold),
                              RGEP=prior['pooled'], comparison_source=str(previous))
        write_json(dest/'results.json', report[source])
    write_json(run/'result/real_comparison.json', report)
    lines = ['# CSAM与在线RGEP：同一逐轮合成谱与分类选模', '',
             '单种子；数据与选模对齐，更新次数和计算量不相等。实测阈值在合成验证集上锁定。', '',
             '| 输入 | CSAM Top-2/% | RGEP Top-2/% | CSAM阈值/% | RGEP阈值/% |',
             '|---|---:|---:|---:|---:|']
    for source, r in report.items():
        lines.append(f"| {source} | {100*r['pooled']['top2']['exact_set_accuracy']:.2f} | "
                     f"{100*r['RGEP']['RGEP_top2']['exact_set_accuracy']:.2f} | "
                     f"{100*r['pooled']['threshold']['exact_set_accuracy']:.2f} | "
                     f"{100*r['RGEP']['RGEP_threshold']['exact_set_accuracy']:.2f} |")
    (run/'result/report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--template', default='run/CSAM_ResUNet_classification_20251119_002532/checkpoint/CSAM_ResUNet_classification_model.h5')
    p.add_argument('--raw-evaluation', default='run/ReferenceRawNNLS_online_seed19580920_20260925')
    p.add_argument('--denoised-evaluation', default='run/ReferenceDenoisedNNLS_online_seed19580920_20260925')
    p.add_argument('--batch-size', type=int, default=None,
                   help='Override training mixture batch; default uses the RGEP anchor batch')
    p.add_argument('--evaluation-batch-size', type=int, default=None,
                   help='Inference batch; default keeps the original anchor batch')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Verify all 50 data hashes; no model training or real evaluation')
    p.add_argument('--smoke', action='store_true', help='Two epochs/two tiny batches; validation subset only, no test evaluation')
    a = p.parse_args()
    if any(value is not None and value < 1 for value in (a.batch_size, a.evaluation_batch_size)):
        p.error('Batch sizes must be positive')
    if a.check_only and a.smoke:
        p.error('Choose at most one diagnostic mode')
    anchor, run = Path(a.anchor), Path(a.output)
    run.mkdir(parents=True, exist_ok=False)
    (run/'checkpoint').mkdir(); (run/'result').mkdir()
    started = time.perf_counter()
    write_json(run/'status.json', dict(state='preflight', started=datetime.now().isoformat()))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        cfg, anchor_state, generator, refs, held, audit = prepare(anchor)
        settings = cfg['args']
        for epoch in range(settings['epochs']):
            assert_epoch(generator.generate(epoch), epoch, audit, settings['seed'])
        write_json(run/'result/preflight.json', dict(all_epoch_hashes_match=True, epochs=settings['epochs'],
            validation_sha256=cfg['validation_array_sha256'], test_sha256=cfg['test_array_sha256'],
            source_manifest_sha256=cfg['source_manifest_sha256'], reference_sha256=file_sha256(anchor/'reference.npz')))
        print('PREFLIGHT PASS: all epoch arrays and held-out data match RGEP', flush=True)
        if a.check_only:
            write_json(run/'status.json', dict(state='preflight_completed'))
            return
        tf.keras.utils.set_random_seed(settings['seed'])
        epochs, batch = (2, 2) if a.smoke else (settings['epochs'], a.batch_size or settings['batch_size'])
        evaluation_batch = 2 if a.smoke else (a.evaluation_batch_size or settings['batch_size'])
        sequence = AlignedMixtures(generator, audit, batch, epochs, a.smoke)
        validation = held['validation']
        if a.smoke:
            validation = {k:v[:14] for k,v in validation.items()}
        shutil.copy2(anchor/'reference.npz', run/'reference.npz')
        shutil.copy2(settings['manifest'], run/'pure_manifest.json')
        for name in [__file__, 'reference_online_data.py', 'rssnet_data.py', 'reference_checkpoint.py',
                     'train_matched_csam_classifier.py', 'matched_separators.py', 'models/ReferenceGuidedExtractor.py']:
            shutil.copy2(name, run/Path(name).name)
        print('Initializing fresh CSAM classifier; template weights are discarded', flush=True)
        write_json(run/'status.json', dict(state='initializing', smoke=a.smoke))
        model = fresh_classifier(a.template)
        model.compile(optimizer=tf.keras.optimizers.Adam(settings['learning_rate'], clipnorm=5.),
                      loss='binary_crossentropy')
        (run/'architecture.json').write_text(model.to_json(), encoding='utf-8')
        config = dict(args=vars(a), anchor=str(anchor), anchor_config_sha256=file_sha256(anchor/'config.json'),
            template_sha256=file_sha256(a.template), initialization='random; no template trained weights reused',
            initial_weights_sha256=array_digest({str(i):w for i,w in enumerate(model.get_weights())}),
            parameters=model.count_params(), epochs=epochs, batch_size=batch, evaluation_batch_size=evaluation_batch,
            loss='standard seven-label BCE',
            selection=SELECTION_RULE, training_recipe=cfg['training_recipe'], learning_rate=settings['learning_rate'],
            schedule='halve at epoch21 and epoch36', budget_scope='same unique mixtures/epoch; NOT matched updates or FLOPs',
            updates_per_epoch=len(sequence), rgep_updates_per_epoch=(generator.per_pair*21*7+settings['batch_size']-1)//settings['batch_size'],
            real_data_usage='evaluation only after selection and validation threshold lock', smoke=a.smoke,
            tensorflow_version=tf.__version__, python_version=platform.python_version(),
            gpu_devices=[tf.config.experimental.get_device_details(d).get('device_name',d.name)
                         for d in tf.config.list_physical_devices('GPU')])
        write_json(run/'config.json', config)
        callback = ClassificationCheckpoint(validation, refs, evaluation_batch, run, True, direct_predict)
        class Audit(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                if sequence.epoch != epoch:
                    raise ValueError('Sequence epoch out of sync')
                assert_epoch(sequence.data, epoch, audit, settings['seed'])
                self.tick = time.perf_counter()
                write_json(run/'status.json', dict(state='training', epoch=epoch+1, epochs=epochs, smoke=a.smoke))
                with (run/'result/data_audit.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(epoch=epoch, training_array_sha256=array_digest(sequence.data),
                        mixture_order_sha256=array_digest(dict(order=sequence.order)), matches_anchor=True))+'\n')
            def on_epoch_end(self, epoch, logs=None):
                with (run/'result/budget_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(epoch=epoch+1, optimizer_updates=int(model.optimizer.iterations),
                        training_seconds_before_full_validation=time.perf_counter()-self.tick))+'\n')
        tick = time.perf_counter()
        hist = model.fit(sequence, epochs=epochs, shuffle=False, workers=1, max_queue_size=2, verbose=2,
            callbacks=[Audit(), tf.keras.callbacks.LearningRateScheduler(lambda e:matched_learning_rate(e,settings['learning_rate'])),
                       callback, tf.keras.callbacks.CSVLogger(str(run/'result/history.csv')), tf.keras.callbacks.TerminateOnNaN()])
        fit_seconds = time.perf_counter()-tick
        if len(hist.history['loss']) != epochs or not np.isfinite(hist.history['loss']).all():
            raise FloatingPointError('Incomplete or nonfinite training')
        if int(model.optimizer.iterations) != epochs*len(sequence):
            raise ValueError('Unexpected optimizer update budget')
        model.save_weights(str(run/'checkpoint/final.weights.h5'))
        model.load_weights(str(run/'checkpoint/best.weights.h5'))
        write_json(run/'result/budget.json', dict(epochs=epochs, updates_per_epoch=len(sequence),
            optimizer_updates=int(model.optimizer.iterations), selected_epoch=callback.best_epoch,
            selected_checkpoint_updates=callback.best_epoch*len(sequence), fit_seconds=fit_seconds,
            fit_includes_full_validation_and_checkpoint_io=True, elapsed_seconds=time.perf_counter()-started,
            unique_mixtures_per_epoch=generator.per_pair*21, rgep_total_updates=settings['epochs']*config['rgep_updates_per_epoch'],
            rgep_selected_updates=anchor_state['best_epoch']*config['rgep_updates_per_epoch'],
            rgep_parameters=cfg['parameters'], csam_parameters=model.count_params(),
            caveat='Different output/task/loss and query expansion; not equal compute or independent physical sample counts'))
        if a.smoke:
            write_json(run/'status.json', dict(state='smoke_completed', best_epoch=callback.best_epoch))
            return
        with np.load(run/'result/selected_validation_predictions.npz') as f:
            threshold = validation_threshold(f['truth'], f['probabilities'])
        write_json(run/'result/threshold.json', threshold)
        test = held['test']; y = truth_labels(test); probs = direct_predict(model,test['x'],evaluation_batch)
        write_json(run/'result/synthetic_metrics.json', decisions(y,probs,threshold['threshold']))
        np.savez_compressed(run/'result/synthetic_predictions.npz', truth=y,probabilities=probs,**test)
        real_evaluation(model,run,threshold['threshold'],evaluation_batch,
                        {'raw':Path(a.raw_evaluation),'paper-denoised':Path(a.denoised_evaluation)},anchor_state['checkpoint_sha256'])
        write_json(run/'status.json', dict(state='completed', epochs=epochs, best_epoch=callback.best_epoch,
            checkpoint_sha256=file_sha256(run/'checkpoint/best.weights.h5'), elapsed_seconds=time.perf_counter()-started))
        print('COMPLETED:',run,flush=True)
    except Exception as exc:
        write_json(run/'status.json',dict(state='failed',error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
