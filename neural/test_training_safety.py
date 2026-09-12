"""Real CPU-training regressions for ancestry and optimizer recovery."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import distill, soup
from .model import PolicyValueNet
from .data_split import SPLIT_VERSION
from .training_provenance import training_provenance, soup_provenance
from .optimizer_recovery import checked_adamw_state, require_finite_model, restore_optimizer


def shard(rows=5, cols=5, connect=4):
    planes = torch.zeros(4, 7, 10, 10)
    planes[:, 2, :rows, :cols] = 1
    planes[:, 3] = connect / 10
    legal = torch.zeros(4, 13, dtype=torch.bool)
    legal[:, :cols] = True
    return dict(planes=planes, legal=legal, policy=legal.float() / cols,
                wdl=torch.ones(4, dtype=torch.long), q=torch.ones(4, 13, dtype=torch.long),
                config=(rows, cols, connect))


class TrainingSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def train(self, root, name, *, parent=None, holdouts="", sidecar=None, reset=False,
              data=None, corrupt_step=False, steps=2):
        output = root / name
        args = ["distill", "fixture", str(output), str(steps), "2"]
        env = dict(DISTILL_INIT=str(parent) if parent else "",
                   DISTILL_INIT_OPT=str(sidecar) if sidecar else "",
                   DISTILL_HOLDOUT_CONFIGS=holdouts, DISTILL_LR="0.001",
                   DISTILL_RESET_OPTIMIZER="1" if reset else "0",
                   DISTILL_PROFILE_STEPS="0", DISTILL_PERSIST_OPTIMIZER="1")
        log = io.StringIO()
        step = torch.optim.AdamW.step

        def update(optimizer, *args, **kwargs):
            result = step(optimizer, *args, **kwargs)
            if corrupt_step:
                with torch.no_grad():
                    optimizer.param_groups[0]["params"][0].fill_(float("nan"))
            return result

        with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", args), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(distill, "PolicyValueNet", side_effect=lambda *arch:
                             PolicyValueNet(*(arch or (4, 1, 4)))), \
                patch.object(distill, "load_shards", return_value=([copy.deepcopy(data or shard())], [])), \
                patch.object(torch.optim.AdamW, "step", update), redirect_stdout(log):
            distill.main()
        checkpoint = torch.load(output / "distilled.pt", weights_only=True)
        require_finite_model(checkpoint["model"])
        return checkpoint, log.getvalue()

    def test_scratch_and_clean_warmstart_preserve_only_lifetime_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _ = self.train(root, "first", holdouts="4x6c4chaos,4x4c3classic")
            parent = root / "first/distilled.pt"
            second, _ = self.train(root, "second", parent=parent,
                                  holdouts="6x4c4chaos,4x4c3classic,8x8c5classic")
            self.assertEqual(second["data_split_version"], SPLIT_VERSION)
            self.assertEqual(second["holdout_configs"], first["holdout_configs"])
            self.assertIn("8x8c5classic", second["training_provenance"]["run_holdout_configs"])
            dropped, _ = self.train(root, "dropped", parent=parent)
            self.assertEqual(dropped["holdout_configs"], "")
            again, _ = self.train(root, "again", parent=root / "dropped/distilled.pt",
                                 holdouts="4x4c3classic")
            self.assertEqual(again["holdout_configs"], "")

    def test_previously_trained_board_cannot_become_a_lifetime_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, _ = self.train(root, "parent", data=shard(4, 4, 3))
            child, _ = self.train(root, "child", parent=root / "parent/distilled.pt",
                                 holdouts="4x4c3classic", data=shard(5, 5, 4))
            self.assertEqual(child["holdout_configs"], "")
            self.assertEqual(child["training_provenance"]["run_holdout_configs"], "4x4c3classic")
            self.assertFalse(torch.equal(parent["model"]["stem.0.weight"], child["model"]["stem.0.weight"]))

    def test_legacy_unknown_and_unsupported_lineages_never_gain_clean_claims(self):
        variants = [{}, {"data_split_version": SPLIT_VERSION, "holdout_configs": "4x4c3classic"},
                    {"data_split_version": "future", "holdout_configs": "4x4c3classic"},
                    {**training_provenance(None, "4x4c3classic"), "data_split_version": "future"}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, metadata in enumerate(variants):
                with self.subTest(metadata=metadata):
                    path = root / f"legacy-{index}.pt"
                    torch.save(dict(model=PolicyValueNet(4, 1, 4).state_dict(), arch=(4, 1, 4), **metadata), path)
                    child, log = self.train(root, f"child-{index}", parent=path, holdouts="4x4c3classic")
                    self.assertEqual(child["data_split_version"], "")
                    self.assertEqual(child["holdout_configs"], "")
                    self.assertEqual(child["training_provenance"]["status"], "unknown")
                    self.assertIn("validation provenance: unknown", log)
                    grandchild = training_provenance(child, "4x4c3classic")
                    self.assertEqual(grandchild["training_provenance"]["status"], "unknown")

    def test_soup_cannot_launder_unknown_ancestry(self):
        clean = training_provenance(None, "4x4c3classic,4x6c4chaos")
        trusted = soup_provenance([clean, copy.deepcopy(clean)], "6x4c4chaos,4x4c3classic")
        self.assertEqual(trusted["holdout_configs"], clean["holdout_configs"])
        self.assertEqual(trusted["training_provenance"]["status"], "clean")
        for unknown in ({}, {"data_split_version": SPLIT_VERSION, "holdout_configs": clean["holdout_configs"]}):
            mixed = soup_provenance([clean, unknown], clean["holdout_configs"])
            self.assertEqual(mixed["data_split_version"], "")
            self.assertEqual(mixed["holdout_configs"], "")
            self.assertEqual(training_provenance(mixed, "4x4c3classic")["training_provenance"]["status"], "unknown")

    def test_soup_publication_keeps_clean_or_unknown_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = shard()
            # The test controls calibration positions; BatchNorm, averaging and
            # checkpoint IO are real, and no ONNX/Modal dependencies are needed.
            for clean in (False, True):
                metadata = training_provenance(None, "4x4c3classic") if clean else dict(
                    data_split_version=SPLIT_VERSION, holdout_configs="4x4c3classic")
                models = [root / "a.pt", root / "b.pt"]
                for path in models:
                    torch.save(dict(model=PolicyValueNet(4, 1, 4).state_dict(), arch=(4, 1, 4),
                                    **metadata), path)
                out = root / "soup.pt"
                with patch.object(sys, "argv", ["soup", str(out), "data", *map(str, models)]), \
                        patch.dict(os.environ, {"SOUP_BATCHES": "1"}), \
                        patch.object(torch.cuda, "is_available", return_value=False), \
                        patch.object(soup, "calibration_data", return_value=(data["planes"], data["legal"])), \
                        redirect_stdout(io.StringIO()):
                    soup.main()
                saved = torch.load(out, weights_only=True)
                self.assertEqual(saved["data_split_version"], SPLIT_VERSION if clean else "")
                self.assertEqual(saved["holdout_configs"], "4x4c3classic" if clean else "")
                self.assertEqual(saved["training_provenance"]["status"], "clean" if clean else "unknown")
                self.assertEqual(int(saved["model"]["stem.1.num_batches_tracked"]), 1)

    def test_valid_optimizer_restores_moments_and_current_execution_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.train(root, "parent")
            sidecar = root / "parent/optimizer.pt"
            saved = torch.load(sidecar, weights_only=True)
            # A sidecar from a GPU host must not force CPU capture/fused options.
            saved["optimizer"]["param_groups"][0].update(capturable=True, fused=True)
            torch.save(saved, sidecar)
            result, log = self.train(root, "child", parent=root / "parent/distilled.pt", sidecar=sidecar)
            self.assertIn("optimizer moments restored", log)
            state = torch.load(root / "child/optimizer.pt", weights_only=True)["optimizer"]
            self.assertEqual(float(next(iter(state["state"].values()))["step"]), 4)
            self.assertFalse(state["param_groups"][0]["capturable"])
            self.assertIsNotNone(result)

    def test_invalid_sidecars_are_discarded_before_real_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.train(root, "parent")
            parent = root / "parent/distilled.pt"
            baseline, _ = self.train(root, "baseline", parent=parent)
            valid = torch.load(root / "parent/optimizer.pt", weights_only=True)
            variants = {"missing": {}, "format": dict(valid, format=99)}
            for field in ("exp_avg", "exp_avg_sq", "step"):
                for value in (float("nan"), float("inf")):
                    bad = copy.deepcopy(valid)
                    next(iter(bad["optimizer"]["state"].values()))[field].fill_(value)
                    variants[f"{field}-{value}"] = bad
            bad = copy.deepcopy(valid)
            next(iter(bad["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(1)
            variants["shape"] = bad
            bad = copy.deepcopy(valid)
            del next(iter(bad["optimizer"]["state"].values()))["exp_avg_sq"]
            variants["missing-moment"] = bad
            bad = copy.deepcopy(valid)
            next(iter(bad["optimizer"]["state"].values()))["exp_avg_sq"].fill_(-1)
            variants["negative-variance"] = bad
            bad = copy.deepcopy(valid)
            bad["optimizer"]["param_groups"][0]["eps"] = float("nan")
            variants["nan-hyperparameter"] = bad
            for name, payload in variants.items():
                with self.subTest(name=name):
                    path = root / f"{name}.opt"
                    torch.save(payload, path)
                    child, log = self.train(root, name, parent=parent, sidecar=path)
                    self.assertIn("optimizer sidecar ignored", log)
                    for key in baseline["model"]:
                        self.assertTrue(torch.equal(baseline["model"][key], child["model"][key]), key)
            path = root / "truncated.opt"
            path.write_bytes(b"not a torch checkpoint")
            _, log = self.train(root, "truncated", parent=parent, sidecar=path)
            self.assertIn("optimizer sidecar ignored", log)

    def test_other_width_state_is_rejected_despite_same_parameter_list_length(self):
        net = PolicyValueNet(8, 1, 4)
        optimizer = distill.create_optimizer(net, 0.001, "cpu")
        sum(parameter.square().sum() for parameter in net.parameters()).backward()
        optimizer.step()
        small = distill.create_optimizer(PolicyValueNet(4, 1, 4), 0.001, "cpu")
        self.assertEqual(len(optimizer.param_groups[0]["params"]), len(small.param_groups[0]["params"]))
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            checked_adamw_state(small, optimizer.state_dict())

    def test_a_partially_failed_load_cannot_leak_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.train(root, "parent")
            made = []
            def factory():
                optimizer = distill.create_optimizer(PolicyValueNet(4, 1, 4), 0.001, "cpu")
                made.append(optimizer)
                return optimizer
            original = torch.optim.AdamW.load_state_dict
            def fail_after_load(optimizer, state):
                original(optimizer, state)
                raise RuntimeError("post-load failure")
            with patch.object(torch.optim.AdamW, "load_state_dict", fail_after_load):
                result = restore_optimizer(factory, root / "parent/optimizer.pt", device="cpu", log=lambda *a, **k: None)
            self.assertEqual(len(made), 2)
            self.assertIs(result, made[-1])
            self.assertTrue(made[0].state)
            self.assertFalse(result.state)

    def test_reset_skips_sidecar_and_nonfinite_weights_never_replace_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.train(root, "parent")
            _, log = self.train(root, "reset", parent=root / "parent/distilled.pt",
                                sidecar=root / "missing.opt", reset=True)
            self.assertNotIn("sidecar", log)
            output = root / "blocked"
            output.mkdir()
            checkpoint = output / "distilled.pt"
            checkpoint.write_bytes(b"previous completed model")
            with self.assertRaisesRegex(ValueError, "Non-finite model"):
                self.train(root, "blocked", parent=root / "parent/distilled.pt", corrupt_step=True, steps=1)
            self.assertEqual(checkpoint.read_bytes(), b"previous completed model")
            self.assertFalse((output / "optimizer.pt").exists())
            self.assertFalse(list(output.glob("*.partial")))

    def test_bad_model_buffers_are_checked_before_serialization(self):
        state = PolicyValueNet(4, 1, 4).state_dict()
        state["stem.1.running_var"].fill_(float("inf"))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "running_var"):
                distill.save_checkpoint({"model": state}, Path(temp) / "model.pt")
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
