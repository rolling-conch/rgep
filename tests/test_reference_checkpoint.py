import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from reference_checkpoint import ClassificationCheckpoint, selection_key


class CheckpointTests(unittest.TestCase):
    def test_primary_tie_and_nan(self):
        self.assertGreater(selection_key(5,8),selection_key(4,.01))
        self.assertGreater(selection_key(5,.1),selection_key(5,.2))
        self.assertEqual(selection_key(5,.1),selection_key(5,.1))
        with self.assertRaises(FloatingPointError):
            selection_key(5,float('nan'))

    def test_full_validation_and_earliest_exact_tie(self):
        data=dict(x=np.ones((3,16,1),dtype=np.float32),pair=np.array([[0,1],[2,3],[4,5]]))
        refs=np.ones((7,16),dtype=np.float32)
        p=np.ones((3,7),dtype=np.float32)*.1
        np.put_along_axis(p,data['pair'],.9,axis=1)
        lower=p.copy(); lower[0,[0,1]]=.01
        spectra=np.zeros((3,16,7),dtype=np.float32)
        model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1)])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'result').mkdir(); (root/'checkpoint').mkdir()
            cb=ClassificationCheckpoint(data,refs,2,root,True)
            cb.set_model(model)
            with patch('reference_checkpoint.query_all',side_effect=[(spectra,lower),(spectra,p),(spectra,p)]) as query:
                for i in range(3):
                    cb.on_epoch_end(i,{})
                self.assertEqual(query.call_count,3)
                for call in query.call_args_list:
                    self.assertIs(call.args[1],data['x'])
                    self.assertIs(call.args[2],refs)
            self.assertEqual(cb.best_epoch,2)
            selected=json.loads((root/'result/checkpoint_selection.json').read_text())
            self.assertEqual(selected['samples'],3)
            self.assertEqual(selected['correct_samples'],3)
            self.assertEqual(len(list((root/'checkpoint').glob('epoch_*.weights.h5'))),3)
            self.assertTrue((root/'checkpoint/best.weights.h5').exists())

    def test_validation_requires_two_distinct_classes(self):
        with self.assertRaises(ValueError):
            ClassificationCheckpoint(dict(x=np.zeros((1,16,1)),pair=np.array([[1,1]])),
                                     np.ones((7,16)),1,Path('.'))


if __name__=='__main__':
    unittest.main()
