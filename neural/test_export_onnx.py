"""CPU regressions for export validation/publication; ONNX boundaries are stubbed.

The real exporter, model, checkpoint IO, tensor operations and filesystem run.
Only ONNX serialization/runtime are substituted to inject impossible outputs.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import export_onnx as export
from .model import PolicyValueNet


def heads():
    return [torch.zeros(8, 13), torch.zeros(8, 3), torch.zeros(8, 13, 3)]


class ExportValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_finite_parity_passes_in_both_precisions(self):
        for half in (False, True):
            with redirect_stdout(io.StringIO()):
                export.validate_parity(heads(), heads(), batch=8, half=half)

    def test_nonfinite_logits_fail_in_every_head_on_either_side(self):
        for half in (False, True):
            for side in (0, 1):
                for head in range(3):
                    for value in (float("nan"), float("inf"), -float("inf")):
                        with self.subTest(half=half, side=side, head=head, value=value):
                            pair = [heads(), heads()]
                            pair[side][head].flatten()[0] = value
                            with self.assertRaisesRegex(ValueError, "finite"), redirect_stdout(io.StringIO()):
                                export.validate_parity(*pair, batch=8, half=half)

    def test_output_count_and_shapes_cannot_be_hidden_by_zip_or_broadcasting(self):
        for actual in (heads()[:2], heads() + [torch.zeros(1)]):
            with self.assertRaisesRegex(ValueError, "exactly"):
                export.validate_parity(heads(), actual, batch=8)
        for head in range(3):
            actual = heads()
            actual[head] = actual[head][:1]  # would broadcast without an explicit shape check
            with self.assertRaisesRegex(ValueError, "shape"), redirect_stdout(io.StringIO()):
                export.validate_parity(heads(), actual, batch=8)

    def test_large_finite_errors_still_fail_in_every_head(self):
        for half in (False, True):
            for head in range(3):
                actual = heads()
                actual[head].flatten()[0] = 100
                with self.assertRaisesRegex(ValueError, "differ"), redirect_stdout(io.StringIO()):
                    export.validate_parity(heads(), actual, batch=8, half=half)

    def test_finite_logits_with_overflowing_difference_are_rejected(self):
        reference, actual = heads(), heads()
        reference[0].fill_(torch.finfo(torch.float32).max)
        actual[0].fill_(-torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(ValueError, "differences are not finite"):
            export.validate_parity(reference, actual, batch=8)

    def run_export(self, root, *, head=None, value=None, half=False, fail_export=False):
        initial, output = root / "tiny.pt", root / "model.onnx"
        net = PolicyValueNet(4, 1, 4)
        torch.save({"model": net.state_dict(), "arch": (4, 1, 4)}, initial)
        produced = b"staged fixture for export control-flow tests"
        context = {}

        def serialize(model, args, destination, **kwargs):
            path = Path(destination)
            self.assertNotEqual(path, output)
            self.assertEqual(path.parent.parent, root)
            self.assertFalse(kwargs["external_data"])
            path.write_bytes(produced)
            context["model"] = model
            if fail_export:
                raise RuntimeError("serialization interrupted")

        class Session:
            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, _names, inputs):
                with torch.no_grad():
                    arrays = [x.numpy().copy() for x in context["model"](torch.from_numpy(inputs["planes"]))]
                if head is not None:
                    arrays[head].flat[0] = value
                return arrays

        modules = {"onnxruntime": SimpleNamespace(InferenceSession=Session),
                   "onnx": SimpleNamespace(load=lambda _path: object(), save=lambda _model, _path: None),
                   "onnxconverter_common": SimpleNamespace(float16=SimpleNamespace(
                       convert_float_to_float16=lambda model, **_: model))}
        argv = ["export", str(initial), str(output)] + (["--half"] if half else [])
        with patch.object(torch.onnx, "export", side_effect=serialize), \
                patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            export.main()
        return produced

    def test_bad_exports_do_not_replace_previous_artifacts_or_publish_first_release(self):
        for existing in (False, True):
            for half in (False, True):
                for head in range(3):
                    for value in (float("nan"), float("inf"), -float("inf"), 100.0):
                        with self.subTest(existing=existing, half=half, head=head, value=value), \
                                tempfile.TemporaryDirectory() as directory:
                            root = Path(directory)
                            output, metadata = root / "model.onnx", root / "model.json"
                            if existing:
                                output.write_bytes(b"previous usable model")
                                metadata.write_text('{"source":"previous.pt"}')
                            with self.assertRaises(ValueError):
                                self.run_export(root, head=head, value=value, half=half)
                            self.assertEqual(output.exists(), existing)
                            self.assertEqual(metadata.exists(), existing)
                            if existing:
                                self.assertEqual(output.read_bytes(), b"previous usable model")
                                self.assertEqual(metadata.read_text(), '{"source":"previous.pt"}')
                            self.assertFalse(list(root.glob(".onnx-export-*")))

    def test_interrupted_serialization_cleans_staging_and_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.onnx").write_bytes(b"previous")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_export(root, fail_export=True)
            self.assertEqual((root / "model.onnx").read_bytes(), b"previous")
            self.assertFalse((root / "model.json").exists())
            self.assertFalse(list(root.glob(".onnx-export-*")))

    def test_metadata_write_failure_preserves_both_previous_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, metadata = root / "model.onnx", root / "model.json"
            output.write_bytes(b"previous")
            metadata.write_text('{"source":"previous.pt"}')
            original = Path.write_text

            def interrupted(path, *args, **kwargs):
                if path.parent.name.startswith(".onnx-export-") and path.suffix == ".json":
                    raise OSError("metadata disk full")
                return original(path, *args, **kwargs)

            with patch.object(Path, "write_text", new=interrupted), self.assertRaisesRegex(OSError, "disk full"):
                self.run_export(root)
            self.assertEqual(output.read_bytes(), b"previous")
            self.assertEqual(metadata.read_text(), '{"source":"previous.pt"}')
            self.assertFalse(list(root.glob(".onnx-export-*")))

    def test_success_promotes_matching_identity_without_inheriting_a_stale_url(self):
        for half in (False, True):
            with self.subTest(half=half), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                meta_path = root / "model.json"
                meta_path.write_text(json.dumps({"origin": "https://model.invalid", "object": "models/old/model.onnx",
                                                 "source": "old.pt", "sha256": "0" * 64, "storedBytes": 99}))
                data = self.run_export(root, half=half)
                meta = json.loads(meta_path.read_text())
                self.assertEqual((root / "model.onnx").read_bytes(), data)
                self.assertEqual(meta["bytes"], len(data))
                self.assertEqual(meta["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(meta["source"], "tiny.pt")
                self.assertEqual(meta["origin"], "https://model.invalid")
                self.assertNotIn("object", meta)
                self.assertNotIn("storedBytes", meta)
                self.assertFalse(list(root.glob(".onnx-export-*")))
                # An identical re-export may keep the already-published pointer.
                meta.update(object="models/tiny/same-digest", storedBytes=23)
                meta_path.write_text(json.dumps(meta))
                self.run_export(root, half=half)
                saved = json.loads(meta_path.read_text())
                self.assertEqual(saved["object"], meta["object"])
                self.assertEqual(saved["storedBytes"], 23)

    def test_bad_output_extension_fails_before_overwriting_a_sidecar(self):
        with patch.object(sys, "argv", ["export", "never-read.pt", "model.json"]):
            with self.assertRaisesRegex(ValueError, "end in .onnx"):
                export.main()


if __name__ == "__main__":
    unittest.main()
