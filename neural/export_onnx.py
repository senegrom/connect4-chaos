"""Exports a checkpoint to ONNX for the browser, and checks it matches.

The browser needs the raw heads: masking a policy to the legal actions is
trivial in JavaScript and keeps a bool input out of the graph. The batch
axis is dynamic so a search can evaluate several positions in one call.

Parity is checked against PyTorch on random positions rather than assumed,
because a silent mismatch here would be a player that quietly misplays.

Usage:
  python -m neural.export_onnx <model.pt> <out.onnx> [--half]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

from .model import ACTIONS, CANVAS, PLANES, PolicyValueNet


class Exported(nn.Module):
    """The network with the legality masking taken out."""

    def __init__(self, net: PolicyValueNet):
        super().__init__()
        self.net = net

    def forward(self, planes):
        legal = torch.ones((planes.shape[0], ACTIONS), dtype=torch.bool, device=planes.device)
        policy, value, q = self.net(planes, legal)
        return policy, value, q


def main() -> None:
    model_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    half = "--half" in sys.argv
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    arch = tuple(payload.get("arch", (192, 12, 48)))
    net = PolicyValueNet(*arch)
    net.load_state_dict(payload["model"])
    net.eval()
    exported = Exported(net).eval()

    sample = torch.rand((2, PLANES, CANVAS, CANVAS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exported, (sample,), str(out_path),
        input_names=["planes"], output_names=["policy", "value", "q"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"},
                      "value": {0: "batch"}, "q": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    if half:
        # Half precision halves the download; the graph keeps float32 at its
        # edges so the browser side needs no change.
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(str(out_path))
        onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), str(out_path))

    size = out_path.stat().st_size
    print(f"wrote {out_path.name}: {size / 1e6:.1f} MB, architecture {arch[0]}x{arch[1]}")

    # Parity: the exported graph must agree with the network it came from.
    import onnxruntime

    session = onnxruntime.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    probe = torch.rand((8, PLANES, CANVAS, CANVAS))
    with torch.no_grad():
        want = exported(probe)
    got = session.run(None, {"planes": probe.numpy()})
    worst = 0.0
    for name, reference, actual in zip(("policy", "value", "q"), want, got):
        gap = float((reference - torch.from_numpy(actual)).abs().max())
        worst = max(worst, gap)
        print(f"  {name:6s} largest difference {gap:.2e}")
    limit = 2e-2 if half else 1e-4
    if worst > limit:
        raise SystemExit(f"ONNX output differs from PyTorch by {worst:.2e}")

    meta = {
        "source": model_path.name,
        "architecture": {"channels": arch[0], "blocks": arch[1], "headChannels": arch[2]},
        "parameters": sum(p.numel() for p in net.parameters()),
        "planes": PLANES,
        "canvas": CANVAS,
        "actions": ACTIONS,
        "precision": "float16" if half else "float32",
        "bytes": size,
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {meta_path.name}")


if __name__ == "__main__":
    main()
