"""The learner's CUDA graph trains like the same steps run eagerly.

On CUDA the learner trains under bf16 autocast with channels_last and a
fused, capturable AdamW, and from its fourth step replays the whole training
step as one CUDA graph. Since #72 none of that can be switched off, and CI
has no GPU. Here one run is trained twice from the same checkpoint and rows:
as it runs, and with graph capture failing, which the learner answers by
training the remaining steps eagerly. A graph that replayed a stale batch,
skipped the optimizer or lost its learning-rate updates would move the
weights differently from the eager steps.

Run on a GPU: modal run neural/modal_app.py --task gpu-test
  --module test_distill_cuda --args=""
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import distill
from .model import PolicyValueNet

ARCH = (16, 2, 8)
STEPS = 12          # three warm-up steps, the capture, then replays


def shard(count, seed, selfplay):
    """Rows on a 5x5 Connect 4 board; the values need not be exact."""
    generator = torch.Generator().manual_seed(seed)
    cells = torch.randint(0, 3, (count, 5, 5), generator=generator)
    planes = torch.zeros(count, 7, 10, 10, dtype=torch.uint8)       # scaled by 10, as shards are
    planes[:, 0, :5, :5] = (cells == 1) * 10
    planes[:, 1, :5, :5] = (cells == 2) * 10
    planes[:, 2, :5, :5] = 10
    planes[:, 3] = 4
    legal = torch.zeros(count, 13, dtype=torch.bool)
    legal[:, :5] = True
    legal[:, 10:] = True
    logits = torch.randn(count, 13, generator=generator).masked_fill(~legal, float("-inf"))
    rows = dict(planes=planes, legal=legal, policy=torch.softmax(logits, dim=1),
                wdl=torch.randint(0, 3, (count,), generator=generator), config=(5, 5, 4))
    if selfplay:
        rows.update(source="selfplay", root_value=torch.rand(count, generator=generator) * 2 - 1)
    else:
        rows["q"] = torch.randint(0, 3, (count, 13), generator=generator).masked_fill(~legal, 3)
    return rows


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class CudaLearnerTests(unittest.TestCase):
    def train(self, root, name, *, capture):
        output = root / name
        env = dict(DISTILL_INIT=str(root / "init.pt"), DISTILL_INIT_OPT="", DISTILL_SEED="7",
                   DISTILL_LR="0.001", DISTILL_REPLAY_FRACTION="0.5", DISTILL_ROOT_VALUE_WEIGHT="0.5",
                   DISTILL_ENTROPY_BONUS="0", DISTILL_HOLDOUT_CONFIGS="", DISTILL_PROFILE_STEPS="0",
                   DISTILL_PERSIST_OPTIMIZER="0")
        data = [shard(512, 1, selfplay=False), shard(512, 2, selfplay=True)]
        log = io.StringIO()
        failing = patch.object(torch.cuda, "CUDAGraph", side_effect=RuntimeError("no capture in this run"))
        with patch.dict(os.environ, env), patch.object(sys, "argv", ["distill", "fixture", str(output),
                                                                     str(STEPS), "64"]), \
                patch.object(distill, "load_shards", return_value=(data, [])), redirect_stdout(log):
            if capture:
                distill.main()
            else:
                with failing:
                    distill.main()
        return torch.load(output / "distilled.pt", map_location="cpu", weights_only=True)["model"], log.getvalue()

    def test_graph_replays_train_like_eager_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.manual_seed(0)
            initial = PolicyValueNet(*ARCH).state_dict()
            torch.save({"model": initial, "arch": ARCH}, root / "init.pt")
            graphed, graph_log = self.train(root, "graph", capture=True)
            eager, eager_log = self.train(root, "eager", capture=False)
        self.assertIn("training step captured as a CUDA graph", graph_log)
        self.assertIn("CUDA graph capture failed, training eagerly", eager_log)
        moved = [(graphed[key].float() - start.float()).flatten() for key, start in initial.items()
                 if start.is_floating_point()]
        reference = [(eager[key].float() - start.float()).flatten() for key, start in initial.items()
                     if start.is_floating_point()]
        moved, reference = torch.cat(moved), torch.cat(reference)
        self.assertGreater(float(reference.norm()), 0.0, "the eager run trained nothing")
        cosine = float(torch.nn.functional.cosine_similarity(moved, reference, dim=0))
        ratio = float(moved.norm() / reference.norm())
        self.assertGreater(cosine, 0.999, f"update direction differs: cosine {cosine:.5f}")
        self.assertAlmostEqual(ratio, 1.0, delta=0.01, msg=f"update size differs: ratio {ratio:.4f}")


if __name__ == "__main__":
    unittest.main()
