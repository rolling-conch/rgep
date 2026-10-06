import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tempfile
import unittest
import numpy as np
import tensorflow as tf
from models.ReferenceGuidedExtractor import ReferenceGuidedExtractor, target_spectral_loss, balanced_target_loss, balanced_presence_loss
from reference_extraction_data import query_targets, training_references, query_all

class ReferenceTests(unittest.TestCase):
    def test_targets_and_exclusion(self):
        d = dict(pair=np.array([[0, 1]]), y=np.array([[[.3,.7],[.0,.0]]],dtype='float32'))
        t = query_targets(d, np.array([0,0,0]), np.array([0,1,2]))
        np.testing.assert_array_equal(t['presence'][:,0],[1,1,0])
        self.assertEqual(float(t['spectrum'][2].sum()),0)
        pool = [np.array([[1,0],[0,1]],dtype='float32')]*3
        r = training_references(pool,t['spectrum'],np.array([0,1,2]),np.random.default_rng(1))
        np.testing.assert_array_equal(r[:2,:,0],[[0,1],[0,1]])

    def test_gradients_serialization_and_zero_target(self):
        tf.keras.utils.set_random_seed(42)
        model = ReferenceGuidedExtractor(channels=8,blocks=1,heads=2,depth=1,chunk_size=16)
        x = tf.random.uniform((2,65,1),.01,.1)
        inputs = dict(mixture=x,reference=tf.reverse(x,axis=[1]))
        truth = tf.concat([x[:1]*.4,tf.zeros_like(x[:1])],axis=0)
        with tf.GradientTape() as tape:
            out = model(inputs)
            loss = tf.reduce_mean(target_spectral_loss(truth,out['spectrum'])) + tf.reduce_mean(
                tf.keras.losses.binary_crossentropy([[1.],[0.]],out['presence']))
        grads = tape.gradient(loss,model.trainable_weights)
        self.assertTrue(all(g is not None and np.isfinite(g).all() for g in grads))
        tf.keras.optimizers.Adam().apply_gradients(zip(grads,model.trainable_weights))
        self.assertTrue(np.isfinite(target_spectral_loss(tf.zeros_like(truth),out['spectrum'])).all())
        with tempfile.TemporaryDirectory() as p:
            model.save_weights(p+'/model.h5')
            other = ReferenceGuidedExtractor.from_config(model.get_config())
            other(inputs); other.load_weights(p+'/model.h5')
            for k,v in model(inputs).items():
                np.testing.assert_allclose(v,other(inputs)[k],atol=1e-6)
        for length in [1024,1025]:
            z=tf.ones((1,length,1))
            self.assertEqual(model(dict(mixture=z,reference=z))['spectrum'].shape,(1,length,1))

    def test_all_classes_queried_without_labels(self):
        seen=[]
        def fake(inputs,training=False):
            seen.append(inputs['reference'][0,0,0])
            return dict(spectrum=inputs['mixture'],presence=np.ones((len(inputs['mixture']),1))*.5)
        s,p=query_all(fake,np.ones((3,9,1)),np.ones((7,9))*np.arange(7)[:,None],batch_size=2)
        self.assertEqual(s.shape,(3,9,7)); self.assertEqual(p.shape,(3,7))
        self.assertEqual(set(seen),set(range(7)))

    def test_v2_negative_decoder_recovers_and_presence_does_not_gate(self):
        model=ReferenceGuidedExtractor(channels=8,blocks=1,heads=2,depth=1,chunk_size=16,version=2)
        x=tf.ones((2,65,1))*.03
        inputs=dict(mixture=x,reference=x)
        model(inputs)
        model.decoder.bias.assign([-10.])
        with tf.GradientTape() as tape:
            out=model(inputs)
            loss=tf.reduce_mean(balanced_target_loss(x,out['spectrum']))
        grad=tape.gradient(loss,model.decoder.bias)
        self.assertTrue(np.isfinite(grad).all())
        self.assertGreater(float(tf.reduce_sum(tf.abs(grad))),0)
        spectrum=out['spectrum'].numpy()
        model.presence.bias.assign([-30.])
        np.testing.assert_allclose(spectrum,model(inputs)['spectrum'],atol=0,rtol=0)
        with tempfile.TemporaryDirectory() as p:
            model.save_weights(p+'/v2.h5')
            other=ReferenceGuidedExtractor.from_config(model.get_config())
            other(inputs); other.load_weights(p+'/v2.h5')
            np.testing.assert_allclose(model(inputs)['spectrum'],other(inputs)['spectrum'],atol=1e-7)

    def test_v2_prior_health_gate_fails(self):
        from reference_extraction_data import health_metrics
        from itertools import combinations
        pairs=np.asarray(list(combinations(range(7),2)))
        d=dict(pair=pairs,y=np.ones((21,16,2),dtype='float32')*.01)
        h=health_metrics(d,np.zeros((21,16,7)),np.full((21,7),2/7))
        self.assertFalse(h['health_passed'])
        self.assertAlmostEqual(h['presence_auc'],.5)

    def test_presence_only_zero_spectral_gradient(self):
        tf.keras.utils.set_random_seed(19)
        model=ReferenceGuidedExtractor(channels=8,blocks=1,heads=2,depth=1,chunk_size=16,version=2)
        x=tf.random.uniform((2,65,1),.01,.1)
        with tf.GradientTape() as tape:
            out=model(dict(mixture=x,reference=tf.reverse(x,[1])))
            loss=tf.reduce_mean(0.*balanced_target_loss(x,out['spectrum']) +
                .5*balanced_presence_loss(tf.constant([[1.],[0.]]),out['presence']))
        weights=model.trainable_weights
        grads=tape.gradient(loss,weights)
        decoder_ids={id(w) for w in model.decoder.trainable_weights+model.mask.trainable_weights}
        for w,g in zip(weights,grads):
            if id(w) in decoder_ids:
                self.assertTrue(g is None or np.all(g.numpy()==0),w.name)
        self.assertGreater(sum(float(tf.reduce_sum(tf.abs(g))) for w,g in zip(weights,grads)
                               if id(w) not in decoder_ids and g is not None),0)

if __name__ == '__main__':
    unittest.main()
