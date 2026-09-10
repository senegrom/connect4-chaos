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


# Comfortably inside GitHub's 100 MB ceiling, with room for a network to
# grow before the part count changes.
MAX_PART_BYTES = 90_000_000


def split_file(path: Path, limit: int = MAX_PART_BYTES) -> list[Path]:
    """Splits a file into equal parts of at most `limit` bytes.

    Returns the file itself when it already fits, so the common case ships
    one plain `model.onnx`. Parts are `<name>.part1`, `.part2`, ...; they are
    raw slices, so `cat` reassembles them exactly.
    """
    size = path.stat().st_size
    if size <= limit:
        return [path]
    count = -(-size // limit)
    chunk = -(-size // count)
    parts = []
    with path.open("rb") as source:
        for index in range(count):
            part = path.with_name(f"{path.name}.part{index + 1}")
            part.write_bytes(source.read(chunk))
            parts.append(part)
    written = sum(part.stat().st_size for part in parts)
    if written != size:
        raise SystemExit(f"split lost bytes: {written} written, {size} expected")
    return parts


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
    got = [torch.from_numpy(array) for array in session.run(None, {"planes": probe.numpy()})]
    # Half precision moves large logits by a few hundredths, which is nothing
    # once they pass through softmax, so the criterion is on the
    # distributions the player actually uses: the policy over actions, the
    # W/D/L of the position, and the W/D/L of each action.
    def distributions(policy, value, q):
        return torch.softmax(policy, dim=1), torch.softmax(value, dim=1), torch.softmax(q, dim=2)

    # The policy and W/D/L heads decide the move and the score shown, so they
    # are held tight. The per-action Q head only seeds children the search has
    # not visited yet - a few simulations overwrite it, and it is the head
    # whose logits spread widest, so it gets a looser bound rather than a
    # tolerance loose enough to hide a real fault in the other two.
    limits = {"policy": 1e-2, "value": 1e-2, "q": 5e-2} if half else dict.fromkeys(
        ("policy", "value", "q"), 1e-4)
    worst = []
    for name, reference, actual, ref_prob, act_prob in zip(
            ("policy", "value", "q"), want, got, distributions(*want), distributions(*got)):
        logit_gap = float((reference - actual).abs().max())
        prob_gap = float((ref_prob - act_prob).abs().max())
        print(f"  {name:6s} largest difference: logits {logit_gap:.2e}, probabilities {prob_gap:.2e}")
        if prob_gap > limits[name]:
            worst.append(f"{name} by {prob_gap:.2e} (limit {limits[name]:.0e})")
    if worst:
        raise SystemExit("ONNX probabilities differ from PyTorch: " + ", ".join(worst))

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
    # GitHub refuses a file over 100 MB and Pages cannot serve Git LFS, so a
    # network too large for one file ships as equal parts that the browser
    # streams into a single buffer (src/neural-runtime.js). One part stays
    # one file, under its own name, so nothing changes for smaller networks.
    parts = split_file(out_path)
    meta["parts"] = [path.name for path in parts]
    meta["partBytes"] = [path.stat().st_size for path in parts]
    if len(parts) > 1:
        out_path.unlink()
        print(f"split into {len(parts)} parts: "
              + ", ".join(f"{path.name} {path.stat().st_size / 1e6:.1f} MB" for path in parts))

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {meta_path.name}")


if __name__ == "__main__":
    main()
