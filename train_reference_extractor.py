"""Train reference-conditioned extraction on P data; no real-mixture fitting."""
import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from models.ReferenceGuidedExtractor import ReferenceGuidedExtractor, target_spectral_loss, balanced_target_loss, balanced_presence_loss
from rssnet_data import pure_partitions, simulate, write_json
from reference_extraction_data import query_targets, training_references, query_all, score_presence, health_metrics
from matched_separators import array_digest, matched_learning_rate
from fixed_splits import file_sha256
from decoupled_data import DEFAULT_CATEGORIES
from reference_checkpoint import ClassificationCheckpoint, SELECTION_RULE
from reference_online_data import EpochMixtures, synthesis_seed

class Queries(tf.keras.utils.Sequence):
    def __init__(self, data, pure, references, batch, seed, training, generator=None, epochs=None):
        self.data, self.pure, self.references = data, pure, references
        self.batch, self.seed, self.training = batch, seed, training
        self.generator, self.max_epochs = generator, epochs
        self.epoch = -1
        self.on_epoch_end()

    def on_epoch_end(self):
        self.epoch += 1
        if self.generator is not None and self.epoch > 0 and (
                self.max_epochs is None or self.epoch < self.max_epochs):
            self.data = self.generator.generate(self.epoch)
        self.order = np.arange(len(self.data['x']) * len(self.references))
        if self.training:
            self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.order)

    def __len__(self):
        return (len(self.order) + self.batch - 1) // self.batch

    def __getitem__(self, index):
        flat = self.order[index * self.batch:(index + 1) * self.batch]
        ids, qs = flat // len(self.references), flat % len(self.references)
        target = query_targets(self.data, ids, qs)
        refs = (training_references(self.pure, target['spectrum'], qs,
                np.random.default_rng(self.seed + self.epoch * 1000003 + index)) if self.training
                else self.references[qs, :, None])
        return dict(mixture=self.data['x'][ids], reference=refs), target

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='dataset/dataset_train_mix_33600_20251116.mat')
    p.add_argument('--audit', default='run/Original_Source_Audit_20260921')
    p.add_argument('--manifest', default='dataset/splits/rssnet_pure_19580920.json')
    p.add_argument('--output-root', default='run')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16, help='query examples, not mixture count')
    p.add_argument('--seed', type=int, default=19580920)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--smoke', action='store_true', help='21 dynamically synthesized training samples; NOT P experiment')
    p.add_argument('--short-check', action='store_true', help='32 P rows per pair, synthetic validation only; no test selection')
    p.add_argument('--version', type=int, choices=(1,2), default=2)
    p.add_argument('--presence-only', action='store_true', help='Keep identical network and queries; spectral loss weight zero')
    p.add_argument('--compare-run', help='Require data/model/settings to match a completed joint V2 run')
    p.add_argument('--refresh-control-run', help='Completed online run for a fixed-augmentation refresh-only ablation')
    p.add_argument('--checkpoint-selection', choices=('legacy_loss', 'val_top2'), default='legacy_loss')
    p.add_argument('--save-every-epoch', action='store_true', help='Save each epoch for auditable selection')
    p.add_argument('--training-data', choices=('original_p','fixed_simulated','online_simulated'),
                   default='original_p', help='Original fixed rows, frozen augmented synthesis, or fresh synthesis every epoch')
    p.add_argument('--train-per-pair', type=int, default=512)
    p.add_argument('--evaluation-anchor', help='Prior run config/reference used to enforce unchanged validation/test and pure split')
    a = p.parse_args()
    if a.refresh_control_run and (a.compare_run or a.smoke or a.short_check or a.training_data != 'fixed_simulated'):
        p.error('Refresh control requires formal fixed_simulated training, without --compare-run')
    checkpoint_class = ClassificationCheckpoint
    if a.refresh_control_run:
        # Execute the anchor's saved callback, not a later generalized implementation.
        import importlib.util
        callback_path = Path(a.refresh_control_run) / 'reference_checkpoint.py'
        spec = importlib.util.spec_from_file_location('refresh_anchor_checkpoint', callback_path)
        callback_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(callback_module)
        if callback_module.SELECTION_RULE != SELECTION_RULE:
            raise ValueError('Anchor checkpoint rule differs')
        checkpoint_class = callback_module.ClassificationCheckpoint
    if a.smoke and a.short_check:
        p.error('Choose smoke or short-check, not both')
    if a.presence_only and (a.version != 2 or a.short_check):
        p.error('Presence-only requires V2 and is not a spectral recovery short-check')
    if a.checkpoint_selection == 'val_top2' and a.version != 2:
        p.error('Aligned classification selection requires V2')
    if a.training_data != 'original_p' and (a.version != 2 or a.checkpoint_selection != 'val_top2'):
        p.error('New synthesis experiments require --version 2 --checkpoint-selection val_top2')
    if a.train_per_pair < 1 or a.seed < 0:
        p.error('Positive train-per-pair and nonnegative seed required')
    if a.training_data != 'original_p' and not (a.smoke or a.short_check or a.evaluation_anchor):
        p.error('Formal synthesis experiments require --evaluation-anchor RUN to preserve prior evaluation arrays')
    if min(a.epochs, a.batch_size) < 1 or not np.isfinite(a.learning_rate) or a.learning_rate <= 0:
        p.error('Positive finite training settings required')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(a.seed)
    pure, refs, meta = pure_partitions(a.source, a.manifest, seed=19580920, create=False)
    generator = None
    if a.training_data != 'original_p':
        generator = EpochMixtures(pure['train'], 1 if a.smoke else 32 if a.short_check else a.train_per_pair,
                                  a.seed, refresh=a.training_data == 'online_simulated')
        train = generator.generate(0)
    elif a.smoke:
        train = simulate(pure['train'], 1, a.seed)
    else:
        from original_separator_data import load_original
        train = load_original(a.source, a.manifest, a.audit)
        if a.short_check:
            pair_ids = train['pair'][:,0] * 7 + train['pair'][:,1]
            keep = np.concatenate([np.flatnonzero(pair_ids == k)[:32] for k in np.unique(pair_ids)])
            train = {k:v[keep] for k,v in train.items()}
    validation = simulate(pure['validation'], 1 if a.smoke else 64, 19580920 + 100)
    test = simulate(pure['test'], 1 if a.smoke else 64, 19580920 + 200)
    run = Path(a.output_root) / ('ReferenceExtractor_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    (run / 'checkpoint').mkdir(parents=True, exist_ok=False)
    (run / 'result').mkdir()
    arch = dict(channels=a.channels, blocks=a.blocks, heads=4, depth=2, chunk_size=32, version=a.version)
    config = dict(args=vars(a), architecture=arch, width=1024, source=meta,
        source_manifest_sha256=file_sha256(a.manifest), training_array_sha256=array_digest(train),
        validation_array_sha256=array_digest(validation), test_array_sha256=array_digest(test),
        protocol='reference-conditioned extraction; seven queries per mixture; NOT matched R0 optimizer budget',
        selection='minimum synthetic validation spectrum loss + 0.1 BCE; no real selection',
        losses='relative spectral + .2 cosine + .1 derivative; zero-target energy floor .01; presence BCE weight .1',
        reference_policy='training pure pool; scaled exact target duplicates excluded; validation/test fixed training median references',
        inference='query all seven classes; presence Top2 and fixed .5; no true label conditioning',
        limitations='single seed; physical source independence and upstream denoiser exposure unverified')
    if a.version == 2:
        config['losses'] = 'balanced positive spectral loss; absent .5*energy; balanced presence BCE weight .5; ungated softplus spectrum'
        config['selection'] = 'minimum synthetic validation spectrum loss + 0.5 balanced BCE; no real selection'
    if a.presence_only:
        config['losses'] = 'spectral weight ZERO; balanced presence BCE weight .5; identical full network and reference queries'
        config['selection'] = 'minimum synthetic validation balanced presence BCE; no real selection'
        config['spectral_outputs_trained'] = False
    if a.checkpoint_selection == 'val_top2':
        config['selection'] = SELECTION_RULE
        config['selection_samples'] = len(validation['x'])
    config['training_recipe'] = generator.protocol() if generator is not None else dict(mode='original_p')
    config['training_generator_sha256'] = file_sha256('rssnet_data.py')
    if a.evaluation_anchor and not (a.smoke or a.short_check):
        anchor = Path(a.evaluation_anchor)
        anchor_cfg = json.loads((anchor / 'config.json').read_text())
        for key in ['source_manifest_sha256','validation_array_sha256','test_array_sha256']:
            if config[key] != anchor_cfg[key]:
                raise ValueError('Evaluation anchor mismatch: '+key+
                    '; use the original container/runtime, never bypass a fingerprint mismatch')
        with np.load(anchor / 'reference.npz') as saved_refs:
            np.testing.assert_array_equal(refs,saved_refs['reference'])
        config['evaluation_anchor'] = dict(run=str(anchor), passed=True,
                                           config_sha256=file_sha256(anchor/'config.json'))
    if generator is not None:
        config['protocol'] = 'train-pure-only augmented mixtures; fixed validation/test and references; no real-mixture fitting'
        config['training_array_scope'] = 'epoch zero; see per-epoch query_audit.jsonl for refreshed data fingerprints'
    if not a.smoke and generator is None:
        config['original_selected_rows_sha256'] = file_sha256(Path(a.audit) / 'P_selected_rows.json')
        shutil.copy2(Path(a.audit) / 'P_selected_rows.json', run / 'original_selected_rows.json')
    np.savez_compressed(run / 'reference.npz', reference=refs, names=np.asarray(DEFAULT_CATEGORIES))
    shutil.copy2(a.manifest, run / 'pure_manifest.json')
    # Persist exactly the fixed arrays used for selection; evaluators must verify their digest.
    np.savez_compressed(run / 'result/validation_data.npz', **validation)
    for path in [__file__, 'models/ReferenceGuidedExtractor.py', 'models/RSSNetInspired.py',
                 'reference_extraction_data.py', 'reference_checkpoint.py', 'reference_online_data.py',
                 'rssnet_data.py', 'original_separator_data.py']:
        if a.refresh_control_run and path == 'reference_checkpoint.py':
            path = str(Path(a.refresh_control_run) / 'reference_checkpoint.py')
        shutil.copy2(path, run / Path(path).name)
    if a.refresh_control_run:
        shutil.copy2('reference_refresh_audit.py', run / 'reference_refresh_audit.py')
        config['checkpoint_implementation_sha256'] = file_sha256(run / 'reference_checkpoint.py')
    write_json(run / 'config.json', config)
    write_json(run / 'status.json', dict(state='initializing', phase='model_build'))
    print('Initializing model:', run, '(first GPU kernel initialization may take time)', flush=True)
    model = ReferenceGuidedExtractor(**arch)
    model(dict(mixture=train['x'][:1], reference=refs[:1, :, None]))
    config['parameters'] = model.count_params()
    config['initial_weights_sha256'] = array_digest({str(i): w for i, w in enumerate(model.get_weights())})
    refresh_epochs = None
    if a.refresh_control_run:
        from reference_refresh_audit import load_anchor, validate_config
        online_config, refresh_epochs = load_anchor(a.refresh_control_run)
        validate_config(config, online_config)
        with np.load(Path(a.refresh_control_run) / 'reference.npz') as reference_file:
            np.testing.assert_array_equal(refs, reference_file['reference'])
        config['refresh_control'] = dict(run=a.refresh_control_run, audit_passed=True,
            intervention='only augmented-mixture refresh: epoch-zero arrays reused',
            budget='equal full training updates; validation-selected epochs can differ')
    if a.compare_run:
        base = Path(a.compare_run)
        bc = json.loads((base / 'config.json').read_text())
        assert json.loads((base / 'status.json').read_text())['state'] == 'completed'
        for key in ['source_manifest_sha256','training_array_sha256','validation_array_sha256',
                    'test_array_sha256','original_selected_rows_sha256','architecture','parameters']:
            if config.get(key) != bc.get(key):
                raise ValueError('Ablation mismatch: '+key)
        if config['training_recipe'] != bc.get('training_recipe',dict(mode='original_p')):
            raise ValueError('Ablation training recipe mismatch')
        if generator is not None and config['training_generator_sha256'] != bc.get('training_generator_sha256'):
            raise ValueError('Synthesis implementation changed')
        for key in ['seed','batch_size','epochs','learning_rate','channels','blocks','version']:
            if vars(a)[key] != bc['args'][key]:
                raise ValueError('Ablation settings mismatch: '+key)
        if file_sha256('models/ReferenceGuidedExtractor.py') != file_sha256(base/'ReferenceGuidedExtractor.py'):
            raise ValueError('Model implementation changed relative to joint baseline')
        with np.load(base/'reference.npz') as reference_file:
            np.testing.assert_array_equal(refs,reference_file['reference'])
        if a.checkpoint_selection == 'val_top2':
            for key in ['selection', 'selection_samples', 'initial_weights_sha256']:
                if config[key] != bc.get(key):
                    raise ValueError('Aligned ablation mismatch: '+key)
            for key in ['checkpoint_selection', 'save_every_epoch']:
                if vars(a)[key] != bc['args'].get(key):
                    raise ValueError('Aligned ablation settings mismatch: '+key)
        config['comparison'] = dict(run=str(base), audit_passed=True,
            caveat=('Common full-validation selection; single seed; same update count not equal wall time'
                    if a.checkpoint_selection == 'val_top2' else
                    'Different validation selection objectives: joint loss vs presence BCE; single seed, same update count not equal wall time'))
    write_json(run / 'config.json', config)
    write_json(run / 'status.json', dict(state='training'))
    print('Run:', run, 'parameters:', model.count_params(), 'queries/epoch:', len(train['x']) * 7, flush=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(a.learning_rate, clipnorm=5),
        loss=dict(spectrum=target_spectral_loss if a.version == 1 else balanced_target_loss,
                  presence='binary_crossentropy' if a.version == 1 else balanced_presence_loss),
        loss_weights=dict(spectrum=0. if a.presence_only else 1., presence=.1 if a.version == 1 else .5))
    sequence = Queries(train, pure['train'], refs, a.batch_size, a.seed, True,
                       generator=generator, epochs=a.epochs)
    del train
    valid = Queries(validation, None, refs, a.batch_size, a.seed, False)
    comparison_epochs = None
    if a.compare_run:
        audit_file = Path(a.compare_run) / 'result/query_audit.jsonl'
        comparison_epochs = {row['epoch']: row for row in
                             (json.loads(line) for line in audit_file.read_text().splitlines() if line.strip())}
        if set(comparison_epochs) != set(range(a.epochs)):
            raise ValueError('Comparison run lacks complete epoch/query audits')
    class Audit(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if sequence.epoch != epoch:
                raise RuntimeError('Unexpected query epoch')
            with (run / 'result/query_audit.jsonl').open('a') as stream:
                row = dict(epoch=epoch, order_sha256=array_digest({'order':sequence.order}))
                if generator is not None:
                    row.update(training_array_sha256=array_digest(sequence.data),
                               synthesis_seed=synthesis_seed(a.seed,epoch if generator.refresh else 0),
                               mixtures=len(sequence.data['x']),
                               alpha_min=float(sequence.data['alpha'].min()),
                               alpha_max=float(sequence.data['alpha'].max()))
                if comparison_epochs is not None and row != comparison_epochs.get(epoch):
                    raise ValueError('Paired ablation query/data mismatch at epoch '+str(epoch))
                if refresh_epochs is not None:
                    from reference_refresh_audit import validate_epoch
                    validate_epoch(row, refresh_epochs)
                stream.write(json.dumps(row) + '\n')
    best = run / 'checkpoint/best.weights.h5'
    # Fixed balanced validation probe; no training or test rows used for health gate.
    probe_ids = np.concatenate([np.arange(i * (1 if a.smoke else 64), i * (1 if a.smoke else 64) + (1 if a.smoke else 8)) for i in range(21)])
    probe = {k:v[probe_ids] for k,v in validation.items()}
    class Health(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch+1) % 2 and epoch+1 != a.epochs:
                return
            sp, pr = query_all(model, probe['x'], refs, a.batch_size)
            result = dict(epoch=epoch+1, **health_metrics(probe, sp, pr))
            if a.presence_only:
                result = {k:v for k,v in result.items() if k in ('epoch','samples','top2_pair_accuracy',
                    'fixed05_subset_accuracy','fixed05_macro_f1','presence_auc','positive_presence_mean','negative_presence_mean')}
            with (run / 'result/health_history.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            print('Validation health:', result, flush=True)
            write_json(run / 'status.json', dict(state='training', epoch=epoch+1, validation_health=result))
    aligned_checkpoint = (checkpoint_class(validation, refs, a.batch_size, run, a.save_every_epoch)
                          if a.checkpoint_selection == 'val_top2' else None)
    checkpoint_callbacks = ([aligned_checkpoint,
        tf.keras.callbacks.ModelCheckpoint(str(run / 'checkpoint/best_loss.weights.h5'),
            monitor='val_loss', save_best_only=True, save_weights_only=True)] if aligned_checkpoint is not None else
        [tf.keras.callbacks.ModelCheckpoint(str(best), save_best_only=True, save_weights_only=True)])
    if a.save_every_epoch and aligned_checkpoint is None:
        checkpoint_callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(run / 'checkpoint/epoch_{epoch:03d}.weights.h5'), save_weights_only=True))
    hist = model.fit(sequence, validation_data=valid, epochs=a.epochs, verbose=2, shuffle=False, workers=1, max_queue_size=2,
        callbacks=checkpoint_callbacks + [tf.keras.callbacks.CSVLogger(str(run / 'result/history.csv')),
            tf.keras.callbacks.LearningRateScheduler(lambda e: matched_learning_rate(e, a.learning_rate)),
            Audit(), Health(), tf.keras.callbacks.TerminateOnNaN()])
    if (len(hist.history['loss']) != a.epochs or not np.isfinite(hist.history['loss']).all()
            or not np.isfinite(hist.history['val_loss']).all()):
        raise FloatingPointError('Incomplete or non-finite training')
    model.save_weights(str(run / 'checkpoint/final.weights.h5'))
    if aligned_checkpoint is not None:
        loss_epoch = int(np.argmin(hist.history['val_loss']))+1
        write_json(run / 'result/loss_checkpoint_selection.json',dict(
            epoch=loss_epoch, val_loss=float(hist.history['val_loss'][loss_epoch-1]),
            checkpoint='checkpoint/best_loss.weights.h5',
            checkpoint_sha256=file_sha256(run / 'checkpoint/best_loss.weights.h5'),
            role='prespecified secondary checkpoint; not selected using real test performance'))
    selected_epoch = (aligned_checkpoint.best_epoch if aligned_checkpoint is not None
                      else int(np.argmin(hist.history['val_loss']))+1)
    if selected_epoch is None:
        raise RuntimeError('No valid checkpoint selected')
    model.load_weights(str(best))
    sp, pr = query_all(model, probe['x'], refs, a.batch_size)
    health = health_metrics(probe, sp, pr)
    if a.presence_only:
        health = {k:v for k,v in health.items() if k in ('samples','top2_pair_accuracy','fixed05_subset_accuracy',
            'fixed05_macro_f1','presence_auc','positive_presence_mean','negative_presence_mean')}
    write_json(run / 'result/validation_health.json', health)
    if a.short_check:
        write_json(run / 'status.json', dict(state='short_check_completed', epochs=a.epochs,
            health_passed=health['health_passed'], best_epoch=selected_epoch,
            checkpoint_sha256=file_sha256(best)))
        print('SHORT CHECK', health, '\nSaved:', run, flush=True)
        return
    spectra, probabilities = query_all(model, test['x'], refs, a.batch_size)
    truth = np.zeros((len(spectra), 7), dtype='int32')
    np.put_along_axis(truth, test['pair'], 1, axis=1)
    metrics, labels = score_presence(truth, probabilities)
    target = np.zeros_like(spectra)
    for i in range(2):
        target[np.arange(len(target)), :, test['pair'][:, i]] = test['y'][..., i]
    metrics.update(positive_query_rmse=float(np.sqrt(np.sum((spectra-target)**2 * truth[:, None])/(truth.sum()*1024))),
        absent_query_mean_energy=float(np.sum(spectra**2 * (1-truth[:, None]))/(1-truth).sum()),
        reconstruction_rmse=float(np.sqrt(np.mean((spectra.sum(axis=-1)-test['y'].sum(axis=-1))**2))),
        warning='positive/absent diagnostics use truth ONLY after all seven queries; synthetic data, no mass-fraction claim')
    if a.presence_only:
        for key in ['positive_query_rmse','absent_query_mean_energy','reconstruction_rmse']:
            metrics.pop(key)
        metrics['warning'] = 'Presence-only ablation; spectral head not supervised; do not interpret spectra as trained extraction'
    write_json(run / 'result/synthetic_metrics.json', metrics)
    np.savez_compressed(run / 'result/synthetic_predictions.npz', **test, extracted=spectra, probabilities=probabilities, labels=labels)
    write_json(run / 'status.json', dict(state='completed', epochs=a.epochs, best_epoch=selected_epoch,
        checkpoint_selection=a.checkpoint_selection,
        checkpoint_sha256=file_sha256(best)))
    print(metrics, '\nSaved:', run, flush=True)

if __name__ == '__main__':
    main()
