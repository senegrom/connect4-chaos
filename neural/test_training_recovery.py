"""CPU regressions for checkpoint recovery."""
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

from . import distill
from .model import PolicyValueNet
from .test_review import shard

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
                (root / "tables" / "datasets-v3").mkdir()   # the exact corpus must exist
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
                    if path.startswith(("/tmp/learn-", "/tmp/replay-")):
                        return root / path.lstrip("/")
                    return Path(path)

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
                    self.assertFalse(result["adopted"])
                    volume.commit.assert_not_called()
                    self.assertEqual(list(model_dir.iterdir()), [model_dir / "init.pt"])
                    continue
                volume.commit.assert_called_once()
                # A run that saved everything and failed only its evaluation
                # is adopted with lineage rather than retrained; one that
                # failed saving its optimizer state is retained without it.
                self.assertEqual(result["adopted"], failure == "evaluation")
                self.assertEqual((model_dir / f"{result['model']}.lineage.json").exists(),
                                 failure in (None, "evaluation"))
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


if __name__ == "__main__":
    unittest.main()
