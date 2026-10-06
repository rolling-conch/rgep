import copy
import unittest

import numpy as np

from reference_online_data import EpochMixtures
from reference_refresh_audit import (ARG_FIELDS, CONFIG_FIELDS, array_digest, budget,
                                     validate_config, validate_epoch)
from run_reference_refresh_ablation import command_for
from pathlib import Path
import json


class RefreshAblationTests(unittest.TestCase):
    def configs(self):
        online = {key: "same" for key in CONFIG_FIELDS}
        online["args"] = {key: "same" for key in ARG_FIELDS}
        online["training_recipe"] = dict(mode="online_simulated", refresh="every epoch",
                                          alpha=[.1, .9], snr_db=[10, 30])
        fixed = copy.deepcopy(online)
        fixed["training_recipe"].update(mode="fixed_simulated", refresh="epoch zero reused")
        return fixed, online

    def test_only_refresh_difference_accepted(self):
        fixed, online = self.configs()
        validate_config(fixed, online)

    def test_control_mismatches_rejected(self):
        for key in CONFIG_FIELDS:
            fixed, online = self.configs()
            fixed[key] = "changed"
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_config(fixed, online)
        for key in ARG_FIELDS:
            fixed, online = self.configs()
            fixed["args"][key] = "changed"
            with self.subTest(argument=key), self.assertRaises(ValueError):
                validate_config(fixed, online)
        fixed, online = self.configs()
        fixed["training_recipe"]["snr_db"] = [5, 30]
        with self.assertRaises(ValueError):
            validate_config(fixed, online)

    def test_all_arrays_fixed_and_first_epoch_shared(self):
        rng = np.random.default_rng(9)
        pure = [rng.uniform(.01, 1, (5, 64)).astype("float32") for _ in range(7)]
        fixed, online = EpochMixtures(pure, 2, 42, False), EpochMixtures(pure, 2, 42, True)
        first = fixed.generate(0)
        for epoch in (1, 2, 49):
            for key in first:
                np.testing.assert_array_equal(first[key], fixed.generate(epoch)[key])
        self.assertEqual(array_digest(first), array_digest(online.generate(0)))
        self.assertNotEqual(array_digest(first), array_digest(online.generate(1)))

    def test_fixed_epoch_audit_rejects_refresh_and_order_changes(self):
        common = dict(training_array_sha256="first", synthesis_seed=42, mixtures=21,
                      alpha_min=.1, alpha_max=.9)
        rows = [dict(epoch=0, order_sha256="order0", **common),
                dict(epoch=1, order_sha256="order1", **{**common, "training_array_sha256": "fresh"})]
        row = dict(epoch=1, order_sha256="order1", **common)
        validate_epoch(row, rows)
        for key in common:
            wrong = dict(row, **{key: "changed"})
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_epoch(wrong, rows)
        with self.assertRaises(ValueError):
            validate_epoch(dict(row, order_sha256="wrong"), rows)

    def test_full_budget_equal_does_not_imply_selected_updates_equal(self):
        cfg = dict(training_recipe=dict(mixtures_per_epoch=10752), args=dict(batch_size=32, epochs=50))
        first, second = budget(cfg, 27), budget(cfg, 49)
        self.assertEqual(first["updates_per_epoch"], 2352)
        self.assertEqual(first["full_updates"], 117600)
        self.assertEqual(first["full_updates"], second["full_updates"])
        self.assertNotEqual(first["selected_updates"], second["selected_updates"])

    def test_command_inherits_anchor_optimizer_and_budget(self):
        args = dict(source="pure.mat", audit="audit", manifest="split.json", epochs=50,
                    batch_size=32, seed=42, learning_rate=.0003, channels=64, blocks=3,
                    version=2, train_per_pair=512, checkpoint_selection="val_top2",
                    save_every_epoch=True, presence_only=False)
        command = command_for(dict(args=args), "saved_online", Path("new_fixed"))
        for key in ("source", "manifest", "epochs", "batch_size", "learning_rate", "seed", "train_per_pair"):
            self.assertEqual(command[command.index("--" + key.replace("_", "-")) + 1], str(args[key]))
        self.assertEqual(command[command.index("--training-data") + 1], "fixed_simulated")
        self.assertIn("--refresh-control-run", command)
        self.assertIn("--save-every-epoch", command)
        self.assertNotIn("--presence-only", command)

    def test_historical_csam_numbers_not_mistaken_for_rgep_results(self):
        root = Path(__file__).resolve().parents[1]
        paths = [root / "run/CSAM_Recipe_Ablation_20260921_seed19580920/A/result/real_summary.json",
                 root / "run/CSAM_Direct_Matched_seed19580920/result/real_summary.json"]
        supplement_path = root / "manuscript/Reference_Guided_Raman_OnlineSynthesis_Supplement_20260925.md"
        if not all(p.is_file() for p in paths) or not supplement_path.is_file():
            self.skipTest("Historical results and manuscript are not bundled in the source-only release")
        scores = [json.loads(p.read_text(encoding="utf-8"))["condition_means"]["direct_accuracy"] for p in paths]
        self.assertEqual([f"{v*100:.2f}" for v in scores], ["98.01", "97.37"])
        self.assertAlmostEqual((scores[1]-scores[0])*100, -.64373897707232)
        supplement = (root / "manuscript/Reference_Guided_Raman_OnlineSynthesis_Supplement_20260925.md").read_text(encoding="utf-8")
        self.assertIn("历史CSAM对照（已完成）", supplement)
        self.assertIn("当前RGEP对照（两组50轮训练及两套九条件实测评估已完成）", supplement)


if __name__ == "__main__":
    unittest.main()
