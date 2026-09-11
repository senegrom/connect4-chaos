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

import hashlib
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

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


def validate_parity(reference, actual, *, batch: int, half: bool = False) -> None:
    """Reject malformed/non-finite heads before comparing their distributions.

    NaN > tolerance is false, so finiteness is a prerequisite, not a byproduct
    of the tolerance test. Shape checks also prevent accidental broadcasting.
    """
    names = ("policy", "value", "q")
    shapes = ((batch, ACTIONS), (batch, 3), (batch, ACTIONS, 3))
    if len(reference) != len(names) or len(actual) != len(names):
        raise ValueError("ONNX parity requires exactly policy, value and Q outputs")
    limits = {"policy": 1e-2, "value": 1e-2, "q": 5e-2} if half else dict.fromkeys(names, 1e-4)
    worst = []
    for name, expected_shape, want, got in zip(names, shapes, reference, actual):
        for label, tensor in (("PyTorch", want), ("ONNX", got)):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{label} {name} shape {tuple(tensor.shape)} != {expected_shape}")
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{label} {name} must contain finite floating-point logits")
        dimension = 2 if name == "q" else 1
        ref_prob, act_prob = torch.softmax(want, dim=dimension), torch.softmax(got, dim=dimension)
        if not bool(torch.isfinite(ref_prob).all()) or not bool(torch.isfinite(act_prob).all()):
            raise ValueError(f"{name} probabilities are not finite")
        logit_gap = float((want - got).abs().max())
        prob_gap = float((ref_prob - act_prob).abs().max())
        if not math.isfinite(logit_gap) or not math.isfinite(prob_gap):
            raise ValueError(f"{name} parity differences are not finite")
        print(f"  {name:6s} largest difference: logits {logit_gap:.2e}, probabilities {prob_gap:.2e}")
        if prob_gap > limits[name]:
            worst.append(f"{name} by {prob_gap:.2e} (limit {limits[name]:.0e})")
    if worst:
        raise ValueError("ONNX probabilities differ from PyTorch: " + ", ".join(worst))


def main() -> None:
    model_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if out_path.suffix.lower() != ".onnx":
        raise ValueError("The export path must end in .onnx")
    half = "--half" in sys.argv
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    arch = tuple(payload.get("arch", (192, 12, 48)))
    net = PolicyValueNet(*arch)
    net.load_state_dict(payload["model"])
    net.eval()
    exported = Exported(net).eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep an earlier usable release intact when export or validation fails.
    with TemporaryDirectory(prefix=".onnx-export-", dir=out_path.parent) as temporary:
        staged = Path(temporary) / out_path.name
        sample = torch.rand((2, PLANES, CANVAS, CANVAS))
        torch.onnx.export(
            exported, (sample,), str(staged),
            input_names=["planes"], output_names=["policy", "value", "q"],
            dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"},
                          "value": {0: "batch"}, "q": {0: "batch"}},
            opset_version=17, dynamo=False, external_data=False,
        )

        if half:
            # Half precision halves the download; the graph keeps float32 at its
            # edges so the browser side needs no change.
            import onnx
            from onnxconverter_common import float16

            model = onnx.load(str(staged))
            onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), str(staged))

        size = staged.stat().st_size
        print(f"wrote {out_path.name}: {size / 1e6:.1f} MB, architecture {arch[0]}x{arch[1]}")

        # Parity: the exported graph must agree with the network it came from.
        import onnxruntime

        session = onnxruntime.InferenceSession(str(staged), providers=["CPUExecutionProvider"])
        probe = torch.rand((8, PLANES, CANVAS, CANVAS))
        with torch.no_grad():
            want = exported(probe)
        got = [torch.from_numpy(array) for array in session.run(None, {"planes": probe.numpy()})]
        validate_parity(want, got, batch=len(probe), half=half)

        meta = {
            "source": model_path.name,
            "architecture": {"channels": arch[0], "blocks": arch[1], "headChannels": arch[2]},
            "parameters": sum(p.numel() for p in net.parameters()),
            "planes": PLANES,
            "canvas": CANVAS,
            "actions": ACTIONS,
            "precision": "float16" if half else "float32",
            "bytes": size,
            "sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
        }
        # The export stays one file and stays out of the repository: it is served
        # from R2 (scripts/publish-model-r2.mjs), which is what removed both the
        # 100 MB file ceiling and the habit of adding a network's full size to git
        # history at every generation. This writes a manifest describing the file
        # on disk; publishing fills in `origin` and `object`.
        meta_path = out_path.with_suffix(".json")
        existing = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if "origin" in existing:
            meta["origin"] = existing["origin"]
        # A new export cannot inherit the URL of different immutable bytes.
        if all(existing.get(key) == meta[key] for key in ("source", "bytes", "sha256")):
            for key in ("object", "storedBytes"):
                if key in existing:
                    meta[key] = existing[key]
        staged_meta = staged.with_suffix(".json")
        staged_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
        del session  # Release the staged file before rename on Windows too.
        # Neither destination is touched until export, parity and metadata writing
        # succeed. Each replacement is atomic; readers still verify the pair's hash.
        staged.replace(out_path)
        staged_meta.replace(meta_path)
        print(f"wrote {meta_path.name}; publish with "
              f"`node scripts/publish-model-r2.mjs {out_path} --manifest {meta_path}`")


if __name__ == "__main__":
    main()
