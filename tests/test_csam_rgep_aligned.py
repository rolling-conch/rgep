import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from reference_online_data import EpochMixtures, synthesis_seed
from matched_separators import array_digest
from train_csam_rgep_aligned import AlignedMixtures, assert_epoch, direct_predict
from reference_checkpoint import ClassificationCheckpoint


class AlignedTests(unittest.TestCase):
    def test_epoch_hashes_and_each_mixture_once(self):
        rng = np.random.default_rng(3)
        pure = [rng.uniform(.1,1,(4,16)).astype('float32') for _ in range(7)]
        generator = EpochMixtures(pure, 2, 31)
        audit = [dict(epoch=e, synthesis_seed=synthesis_seed(31,e), mixtures=42,
                      training_array_sha256=array_digest(generator.generate(e))) for e in range(2)]
        seq = AlignedMixtures(generator, audit, 5, 2)
        for epoch in range(2):
            self.assertEqual(seq.epoch, epoch)
            np.testing.assert_array_equal(np.sort(seq.order), np.arange(42))
            for index in range(len(seq)):
                x,y = seq[index]
                ids=seq.order[index*5:(index+1)*5]
                np.testing.assert_array_equal(x, generator.generate(epoch)['x'][ids])
                np.testing.assert_array_equal(y.sum(1), 2)
            seq.on_epoch_end()
        wrong = generator.generate(0)
        wrong['x'][0,0,0] += .1
        with self.assertRaises(ValueError):
            assert_epoch(wrong, 0, audit, 31)

    def test_direct_and_query_selection_identical(self):
        data=dict(x=np.ones((3,16,1),dtype='float32'),pair=np.array([[0,1],[2,3],[4,5]]))
        refs=np.ones((7,16),dtype='float32')
        probabilities=np.full((3,7),.1,dtype='float32')
        np.put_along_axis(probabilities,data['pair'],.9,axis=1)
        model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1)])
        with tempfile.TemporaryDirectory() as folder:
            callbacks=[]
            for name,fn in [('direct',lambda m,x,b:probabilities),('query',None)]:
                root=Path(folder)/name
                (root/'checkpoint').mkdir(parents=True); (root/'result').mkdir()
                cb=ClassificationCheckpoint(data,refs,2,root,True,fn); cb.set_model(model)
                with patch('reference_checkpoint.query_all',return_value=(None,probabilities)):
                    cb.on_epoch_end(0,{})
                    cb.on_epoch_end(1,{})
                self.assertEqual(cb.best_epoch,1)
                callbacks.append(cb)
            self.assertEqual(callbacks[0].best_key,callbacks[1].best_key)

    def test_direct_inference_covers_final_partial_batch(self):
        model=tf.keras.Sequential([tf.keras.layers.Input((4,1)),tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(7,activation='sigmoid')])
        x=np.ones((5,4,1),dtype='float32')
        np.testing.assert_allclose(direct_predict(model,x,2),model(x,training=False),atol=1e-7)


if __name__=='__main__':
    unittest.main()
