"""CPU regressions for checkpoint recovery and model-soup holdouts."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from . import distill, soup
from .data_split import SPLIT_VERSION, validation_mask
from .model import PolicyValueNet
from .test_review import shard
from .test_rereview import data, positions

ROOT = Path(__file__).resolve().parents[1]


class TrainingRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_remote_learner_retains_completed_training_after_later_failures(self):
        # Run the real learner wrapper and a real two-step CPU trainer, replacing
        # only Modal's volume and subprocess boundary. Training failure must not
        # publish; evaluation and optional-sidecar failure must preserve weights.
        for failure in ("training", "checkpoint", "evaluation", "optimizer", None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                model_dir = root / "tables" / "models"
                model_dir.mkdir(parents=True)
                net = PolicyValueNet(4, 1, 4)
                initial = {key: value.clone() for key, value in net.state_dict().items()}
                torch.save({"model": initial, "arch": (4, 1, 4)}, model_dir / "init.pt")
                volume = SimpleNamespace(reload=Mock(), commit=Mock())

                class FailingNet(PolicyValueNet):
                    def forward(self, planes, legal):
                        if failure == "training" or (failure == "evaluation" and not self.training):
                            raise RuntimeError(f"{failure} failed")
                        return super().forward(planes, legal)

                def local_path(path):
                    path = str(path)
                    return root / path.lstrip("/") if path.startswith("/tmp/") else Path(path)

                def run(command, **kwargs):
                    output = io.StringIO()
                    save = torch.save

                    def save_with_failure(payload, path):
                        if ((failure == "optimizer" and "optimizer" in payload)
                                or (failure == "checkpoint" and "model" in payload)):
                            Path(path).write_bytes(b"interrupted serialization")
                            raise RuntimeError(f"{failure} failed")
                        save(payload, path)

                    train = shard([(5, 5, 4, False)] * 2)
                    held = shard([(5, 5, 4, False)] * 2)
                    env = dict(kwargs["env"], DISTILL_PERSIST_OPTIMIZER="1", DISTILL_INIT_OPT="",
                               DISTILL_HOLDOUT_CONFIGS="", DISTILL_PROFILE_STEPS="0")
                    with patch.object(sys, "argv", command[2:]), \
                            patch.dict(os.environ, env, clear=True), \
                            patch.object(distill, "PolicyValueNet", FailingNet), \
                            patch.object(distill, "load_shards", return_value=([train], [held])), \
                            patch.object(torch.cuda, "is_available", return_value=False), \
                            patch.object(torch, "save", side_effect=save_with_failure), redirect_stdout(output):
                        try:
                            distill.main()
                        except RuntimeError as exc:
                            return subprocess.CompletedProcess(command, 1, output.getvalue(), str(exc))
                    return subprocess.CompletedProcess(command, 0, output.getvalue(), "")

                tree = ast.parse((ROOT / "neural/modal_app.py").read_text())
                learn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "learn")
                learn.decorator_list = []
                context = dict(Path=local_path, os=os, time=time, TABLES=str(root / "tables"),
                               tables=volume, LEARNER_GPU="cpu", subprocess=SimpleNamespace(run=run))
                exec(compile(ast.Module(body=[learn], type_ignores=[]), "modal_app.py", "exec"), context)
                # Stale output from an earlier attempt cannot be mistaken for
                # the current generation's completed checkpoint.
                old_output = root / "tmp" / "learn-7"
                old_output.mkdir(parents=True)
                (old_output / "distilled.pt").write_bytes(b"stale")
                result = context["learn"](7, "init.pt", steps=2, batch=2)
                self.assertEqual(result["exit"], 1 if failure else 0)
                if failure:
                    self.assertIn(f"{failure} failed", result["err"])
                self.assertFalse(old_output.exists())
                if failure in ("training", "checkpoint"):
                    self.assertIsNone(result["model"])
                    volume.commit.assert_not_called()
                    self.assertEqual(list(model_dir.iterdir()), [model_dir / "init.pt"])
                    continue
                volume.commit.assert_called_once()
                checkpoint = torch.load(model_dir / result["model"], weights_only=True)
                restored = PolicyValueNet(*checkpoint["arch"])
                restored.load_state_dict(checkpoint["model"])
                self.assertEqual(checkpoint["steps"], 2)
                self.assertFalse(torch.equal(checkpoint["model"]["stem.0.weight"], initial["stem.0.weight"]))
                optimizer_path = model_dir / f"{result['model']}.opt"
                self.assertEqual(optimizer_path.exists(), failure != "optimizer")
                self.assertEqual(result["optimizer_state"], failure != "optimizer")
                if optimizer_path.exists():
                    optimizer = torch.optim.AdamW(restored.parameters())
                    optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True)["optimizer"])
                    self.assertTrue(optimizer.state)

    def test_interrupted_checkpoint_save_never_exposes_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            committed = root / "previous.pt"
            distill.save_checkpoint({"steps": 1}, committed)
            before = committed.read_bytes()

            def interrupted(_payload, path):
                path.write_bytes(b"partial")
                raise OSError("disk full")

            for path in (committed, root / "new.pt"):
                with self.subTest(path=path), patch.object(torch, "save", side_effect=interrupted):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        distill.save_checkpoint({"steps": 2}, path)
                self.assertEqual(committed.read_bytes(), before)
                self.assertFalse((root / "new.pt").exists())
                self.assertFalse(list(root.glob("*.partial")))

    def test_calibration_excludes_whole_board_and_rotated_replay_holdouts(self):
        for scaled in (False, True):
            with self.subTest(scaled=scaled), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                torch.save(shard([(4, 4, 3, False)]), root / "4x4c3classic-0001.pt")
                torch.save(shard([(5, 5, 4, False)]), root / "5x5c4classic-0001.pt")
                replay = shard([(4, 4, 3, False), (4, 6, 4, True), (6, 4, 4, True),
                                (5, 5, 4, False), (5, 5, 4, False)], True, scaled)
                replay.update(split_version=SPLIT_VERSION, validation=torch.tensor([False] * 4 + [True]))
                torch.save(replay, root / "gpu-sp-1.pt")
                planes, legal = soup.calibration_data(root, pool=20,
                    holdout_shapes=((4, 4, 3, False), (4, 6, 4, True)))
                expected = distill.quantize_planes(shard([(5, 5, 4, False)] * 2)["planes"])
                self.assertTrue(torch.equal(planes, expected))
                self.assertEqual(len(legal), 2)

    def test_all_filtered_calibration_fails_before_batchnorm_is_reset(self):
        for reserved in (False, True):
            with self.subTest(reserved=reserved), tempfile.TemporaryDirectory() as temp:
                payload = shard([(5, 5, 4, False)], True)
                payload.update(split_version=SPLIT_VERSION, validation=torch.tensor([reserved]))
                torch.save(payload, Path(temp) / "gpu-sp-1.pt")
                holdouts = () if reserved else ((5, 5, 4, False),)
                with self.assertRaisesRegex(SystemExit, "no positions available"):
                    soup.calibration_data(temp, holdout_shapes=holdouts)

    def test_calibration_rehashes_stale_validation_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = data(positions(100), True)
            mask = validation_mask(payload["planes"])
            self.assertTrue(bool(mask.any()) and bool((~mask).any()))
            payload.update(split_version="obsolete", validation=torch.zeros(100, dtype=torch.bool))
            torch.save(payload, Path(temp) / "gpu-sp-1.pt")
            planes, _ = soup.calibration_data(temp)
            self.assertEqual(len(planes), int((~mask).sum()))
            self.assertFalse(bool(validation_mask(planes).any()))

    def test_soup_preserves_holdouts_through_real_recalibration_and_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "data"
            dataset.mkdir()
            for tag, shape in (("4x4c3classic", (4, 4, 3, False)),
                               ("4x6c4chaos", (4, 6, 4, True)),
                               ("5x5c4classic", (5, 5, 4, False))):
                torch.save(shard([shape]), dataset / f"{tag}-0001.pt")
            models = [root / "a.pt", root / "b.pt"]
            holds = ["4x4c3classic,4x6c4chaos", " 6x4c4chaos,4x4c3classic,4x4c3classic "]
            for path, holdout in zip(models, holds):
                torch.save(dict(model=PolicyValueNet(4, 1, 4).state_dict(), arch=(4, 1, 4),
                                steps=2, data_split_version=SPLIT_VERSION, holdout_configs=holdout), path)
            recalibrate = soup.recalibrate
            observed = []

            def inspect_calibration(net, planes, legal, device, **_kwargs):
                observed.append(planes.clone())
                return recalibrate(net, planes, legal, device, batches=1, batch_size=2)

            output = root / "soup.pt"
            with patch.object(sys, "argv", ["soup", str(output), str(dataset), *map(str, models)]), \
                    patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(soup, "recalibrate", side_effect=inspect_calibration), redirect_stdout(io.StringIO()):
                soup.main()
            self.assertEqual(len(observed), 1)
            expected = distill.quantize_planes(shard([(5, 5, 4, False)])["planes"])
            self.assertTrue(torch.equal(observed[0], expected))
            saved = torch.load(output, weights_only=True)
            self.assertEqual(saved["holdout_configs"], holds[0])
            self.assertEqual(saved["data_split_version"], SPLIT_VERSION)
            self.assertEqual(int(saved["model"]["stem.1.num_batches_tracked"]), 1)

    def test_soup_rejects_conflicting_or_unsupported_source_partitions(self):
        baseline = dict(data_split_version=SPLIT_VERSION, holdout_configs="4x4c3classic")
        for other in (dict(baseline, holdout_configs=""), dict(baseline, data_split_version="")):
            with self.subTest(other=other), self.assertRaisesRegex(SystemExit, "different.*partitions"):
                soup.shared_partition([baseline, other])
        with self.assertRaisesRegex(SystemExit, "unsupported"):
            soup.shared_partition([dict(baseline, data_split_version="future")] * 2)
        with self.assertRaisesRegex(ValueError, "specific configurations"):
            soup.shared_partition([dict(baseline, holdout_configs="all")] * 2)
        metadata, holds = soup.shared_partition([{}, {}])
        self.assertEqual(metadata, dict(data_split_version="", holdout_configs=""))
        self.assertEqual(holds, ())


if __name__ == "__main__":
    unittest.main()
