"""Common full-synthetic-validation checkpoint selection for RGEP ablations."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.ReferenceGuidedExtractor import balanced_presence_loss
from reference_extraction_data import query_all, score_presence

SELECTION_RULE = ('maximum full synthetic validation Top-2 strict accuracy; '
                  'tie: minimum balanced presence BCE; exact tie: earliest epoch')


def selection_key(correct, presence_loss):
    if not np.isfinite(presence_loss):
        raise FloatingPointError('Non-finite checkpoint selection loss')
    return int(correct), -float(presence_loss)


class ClassificationCheckpoint(tf.keras.callbacks.Callback):
    """No test tensors accepted; all seven fixed training references are queried."""
    def __init__(self, validation, references, batch_size, run, save_every_epoch=False,
                 predict_probabilities=None):
        super().__init__()
        self.validation = validation
        self.references = references
        self.batch_size = batch_size
        self.run = Path(run)
        self.save_every_epoch = save_every_epoch
        self.predict_probabilities = predict_probabilities
        self.best_key = None
        self.best_epoch = None
        self.truth = np.zeros((len(validation['x']), len(references)), dtype=np.int32)
        np.put_along_axis(self.truth, validation['pair'], 1, axis=1)
        if not np.all(self.truth.sum(axis=1) == 2):
            raise ValueError('Expected two distinct true classes per validation sample')

    def on_epoch_end(self, epoch, logs=None):
        if self.predict_probabilities is None:
            _, probabilities = query_all(self.model, self.validation['x'], self.references, self.batch_size)
        else:
            probabilities = self.predict_probabilities(self.model, self.validation['x'], self.batch_size)
        if probabilities.shape != self.truth.shape or not np.isfinite(probabilities).all():
            raise ValueError('Invalid full-validation probabilities')
        measured, labels = score_presence(self.truth, probabilities)
        # Same class-balanced per-query BCE as training; no spectral loss here.
        loss = float(tf.reduce_mean(balanced_presence_loss(
            tf.convert_to_tensor(self.truth.reshape(-1, 1), dtype=tf.float32),
            tf.convert_to_tensor(probabilities.reshape(-1, 1), dtype=tf.float32))))
        correct = int(np.all(labels == self.truth, axis=1).sum())
        key = selection_key(correct, loss)
        selected = self.best_key is None or key > self.best_key
        epoch_number = epoch + 1
        row = dict(epoch=epoch_number, **measured, correct_samples=correct,
                   balanced_presence_bce=loss, selected=selected, selection_rule=SELECTION_RULE)
        checkpoint = self.run / 'checkpoint'
        if self.save_every_epoch:
            self.model.save_weights(str(checkpoint / f'epoch_{epoch_number:03d}.weights.h5'))
        if selected:
            self.model.save_weights(str(checkpoint / 'best.weights.h5'))
            self.best_key, self.best_epoch = key, epoch_number
            np.savez_compressed(self.run / 'result/selected_validation_predictions.npz',
                                truth=self.truth, probabilities=probabilities, labels=labels)
            (self.run / 'result/checkpoint_selection.json').write_text(
                json.dumps(row, indent=2, allow_nan=False), encoding='utf-8')
        with (self.run / 'result/checkpoint_history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if logs is not None:
            logs.update(val_full_top2_accuracy=measured['top2_pair_accuracy'],
                        val_full_fixed05_macro_f1=measured['fixed05_macro_f1'],
                        val_full_balanced_presence_bce=loss)
        print(f'Checkpoint alignment: epoch={epoch_number} full_validation={len(labels)} '
              f'Top2={correct}/{len(labels)} BCE={loss:.8g} selected={selected}', flush=True)
