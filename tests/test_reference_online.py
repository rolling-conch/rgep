import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import unittest
import numpy as np
from scipy.optimize import lsq_linear
from reference_online_data import EpochMixtures, synthesis_seed
from reference_background_nnls import (BackgroundNNLS, select_background,
                                       validation_threshold, labels_top2)
from rssnet_data import simulate


class OnlineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.pure = []
        for _ in range(7):
            x = rng.uniform(.01,1,(6,64)).astype("float32")
            self.pure.append(x/np.linalg.norm(x,axis=1,keepdims=True))

    def test_refresh_reproducibility_and_fixed_control(self):
        online = EpochMixtures(self.pure,2,19580920)
        first, second = online.generate(0),online.generate(1)
        self.assertEqual(first["x"].shape,(42,64,1))
        self.assertFalse(np.array_equal(first["x"],second["x"]))
        for key in first:
            np.testing.assert_array_equal(first[key],online.generate(0)[key])
        direct=simulate(self.pure,2,synthesis_seed(19580920,0))
        for key in first:
            np.testing.assert_array_equal(first[key],direct[key])
        fixed=EpochMixtures(self.pure,2,19580920,refresh=False)
        np.testing.assert_array_equal(fixed.generate(0)["x"],fixed.generate(1)["x"])
        np.testing.assert_array_equal(first["x"],fixed.generate(0)["x"])
        self.assertTrue(np.all((first["alpha"]>=.1)&(first["alpha"]<=.9)))
        np.testing.assert_allclose(np.linalg.norm(first["x"][...,0],axis=1),1,atol=2e-7)

    def test_sequence_refresh_and_query_targets(self):
        from train_reference_extractor import Queries
        generator=EpochMixtures(self.pure,1,42)
        refs=np.stack([x[0] for x in self.pure])
        sequence=Queries(generator.generate(0),self.pure,refs,32,42,True,generator,2)
        x0=sequence.data["x"].copy()
        xb,y=sequence[0]
        self.assertEqual(xb["mixture"].shape,(32,64,1))
        self.assertTrue(np.all(y["spectrum"][y["presence"][:,0]==0]==0))
        sequence.on_epoch_end()
        self.assertEqual(sequence.epoch,1)
        self.assertFalse(np.array_equal(x0,sequence.data["x"]))
        np.testing.assert_array_equal(sequence.data["x"],generator.generate(1)["x"])

    def test_train_generator_does_not_change_validation(self):
        validation=simulate([x[::-1].copy() for x in self.pure],1,19581020)
        copy={k:v.copy() for k,v in validation.items()}
        generator=EpochMixtures(self.pure,2,42)
        generator.generate(0);generator.generate(1)
        for key in copy:
            np.testing.assert_array_equal(validation[key],copy[key])


class BackgroundTests(unittest.TestCase):
    def test_equivalence_to_joint_bounded_least_squares(self):
        rng=np.random.default_rng(19)
        refs=rng.uniform(size=(7,80))
        solver=BackgroundNNLS(refs,2)
        c=np.array([[.2,0,.5,0,0,0,.1],[0,.3,0,.1,0,0,0]])
        beta=np.array([[.4,-.2,.7],[.1,.3,-.2]])
        x=solver.reconstruct(c,beta)+rng.normal(0,.001,(2,80))
        pred,b=solver.predict(x)
        a=np.column_stack([refs.T,solver.background])
        for i in range(len(x)):
            joint=lsq_linear(a,x[i],bounds=(np.r_[np.zeros(7),np.full(3,-np.inf)],
                                            np.full(10,np.inf)),tol=1e-12)
            np.testing.assert_allclose(pred[i],joint.x[:7],atol=1e-6)
            np.testing.assert_allclose(solver.reconstruct(pred,b)[i],a@joint.x,atol=1e-6)
        self.assertTrue(np.all(pred>=0))

    def test_no_background_and_validation_selection(self):
        rng=np.random.default_rng(4)
        refs=rng.uniform(size=(7,64))
        truth=np.zeros((5,7),dtype=bool);truth[:,[0,3]]=True
        coefficients=truth.astype(float)*.4
        x=coefficients@refs
        degree,candidates,pred=select_background(refs,x,truth)
        self.assertEqual(degree,-1)  # ties keep plain NNLS, never force new method
        np.testing.assert_array_equal(labels_top2(pred[-1]),truth)
        selected=validation_threshold(truth,pred[-1])
        self.assertEqual(selected["validation"]["exact_set_accuracy"],1.)
        self.assertEqual(len(candidates),5)

    def test_rank_deficiency_rejected(self):
        with self.assertRaises(ValueError):
            BackgroundNNLS(np.ones((7,64)),2)


if __name__=="__main__":
    unittest.main()
