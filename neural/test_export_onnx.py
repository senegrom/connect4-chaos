"""CPU regressions for export validation/publication, and the import round trip.

The real exporter, model, checkpoint IO, tensor operations and filesystem run.
The publication tests substitute ONNX serialization/runtime to inject
impossible outputs; the round-trip tests export a real graph with the real
exporter and read it back with neural/import_onnx.py (onnx, no onnxruntime).
"""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
from torch import nn

from . import export_onnx as export
from . import import_onnx
from .gpu_selfplay import _prepare_network
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


def half_precision(model):
    """What onnxconverter_common's float16 conversion does to an export, in
    miniature: float initializers stored as float16, and Casts where the
    graph meets its float32 input and outputs (keep_io_types)."""
    half = copy.deepcopy(model)
    for tensor in half.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            tensor.CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(tensor).astype(numpy.float16), tensor.name))
    for graph_input in half.graph.input:
        cast = f"{graph_input.name}_cast"
        for node in half.graph.node:
            node.input[:] = [cast if name == graph_input.name else name for name in node.input]
        half.graph.node.insert(0, helper.make_node("Cast", [graph_input.name], [cast], to=TensorProto.FLOAT16))
    for graph_output in half.graph.output:
        inner = f"{graph_output.name}_half"
        for node in half.graph.node:
            node.output[:] = [inner if name == graph_output.name else name for name in node.output]
            node.input[:] = [inner if name == graph_output.name else name for name in node.input]
        half.graph.node.append(helper.make_node("Cast", [inner], [graph_output.name], to=TensorProto.FLOAT))
    return half


class ImportRoundTripTests(unittest.TestCase):
    """A tiny random network through the real exporter and back."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        net = PolicyValueNet(8, 2, 4)
        # Random statistics make every fold non-trivial.
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.uniform_(-1, 1)
                module.running_var.uniform_(0.5, 2)
                module.weight.data.uniform_(0.5, 1.5)
                module.bias.data.uniform_(-1, 1)
        cls.net = net.eval()
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name) / "tiny.onnx"
        export.export_graph(cls.net, path)
        cls.model = onnx.load(str(path))
        cls.planes, cls.legal = import_onnx.calibration_positions(160, seed=3, games=48)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def outputs(self, net, rows=slice(96, 160)):
        with torch.no_grad():
            return net.eval()(self.planes[rows], self.legal[rows])

    def assert_same_outputs(self, imported, *, atol):
        for name, got, want in zip(("policy", "value", "q"), self.outputs(imported), self.outputs(self.net)):
            finite = torch.isfinite(want)
            self.assertTrue(torch.equal(finite, torch.isfinite(got)), name)
            torch.testing.assert_close(got[finite], want[finite], rtol=0, atol=atol, msg=name)

    def test_the_export_is_folded_and_imports_exactly_in_eval_mode(self):
        self.assertFalse([node for node in self.model.graph.node if node.op_type == "BatchNormalization"])
        imported, arch = import_onnx.import_network(self.model, self.planes[:96], self.legal[:96])
        self.assertEqual(arch, (8, 2, 4))
        self.assertEqual({key: value.shape for key, value in imported.state_dict().items()},
                         {key: value.shape for key, value in self.net.state_dict().items()})
        self.assert_same_outputs(imported, atol=1e-4)

    def test_calibration_makes_a_training_batch_normalise_like_eval_mode(self):
        # Calibrated on exactly this batch, train mode's batch statistics are
        # the running statistics layer by layer, so the two modes agree.
        batch = slice(0, 64)
        imported, _ = import_onnx.import_network(self.model, self.planes[batch], self.legal[batch])
        with torch.no_grad():
            want = imported.eval()(self.planes[batch], self.legal[batch])
            got = copy.deepcopy(imported).train()(self.planes[batch], self.legal[batch])
        for name, a, b in zip(("policy", "value", "q"), got, want):
            finite = torch.isfinite(b)
            torch.testing.assert_close(a[finite], b[finite], rtol=0, atol=1e-4, msg=name)
        deviation = import_onnx.train_mode_deviation(imported, self.planes[batch], self.legal[batch])
        self.assertLess(max(deviation[name][0] for name in ("policy", "value", "q")), 1e-4)
        # Uncalibrated (running statistics 0 and 1) train mode is far off.
        arch, folded, heads = import_onnx.folded_weights(self.model)
        raw, _norms = import_onnx.build_network(arch, folded, heads)
        self.assert_same_outputs(raw, atol=1e-4)
        far = import_onnx.train_mode_deviation(raw, self.planes[batch], self.legal[batch])
        self.assertGreater(far["value"][0], 1e-2)

    def test_half_precision_exports_with_casts_import_their_rounded_weights(self):
        half = half_precision(self.model)
        imported, _ = import_onnx.import_network(half, self.planes[:96], self.legal[:96])
        _, folded, heads = import_onnx.folded_weights(self.model)
        rounded = lambda weight: weight.half().float()
        self.assertTrue(torch.equal(imported.stem[0].weight, rounded(folded[0][0])))
        self.assertTrue(torch.equal(imported.tower[1].conv2.weight, rounded(folded[4][0])))
        # The drop heads were MatMul constants, stored transposed.
        self.assertTrue(torch.equal(imported.drop_q.weight, rounded(self.net.drop_q.weight.detach())))
        self.assertTrue(torch.equal(imported.value.weight, rounded(self.net.value.weight.detach())))
        # The exported heads (unmasked) agree within the exporter's own fp16 tolerance.
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            export.validate_parity(export.Exported(self.net)(self.planes[96:]),
                                   export.Exported(imported)(self.planes[96:]), batch=64, half=True)

    def test_parity_check_reads_onnxruntime_outputs_and_rejects_a_mismatch(self):
        imported, _ = import_onnx.import_network(self.model, self.planes[:96], self.legal[:96])
        probe = self.planes[96:104]
        reference = export.Exported(self.net).eval()

        class Session:
            def __init__(self, drift=0.0):
                self.drift = drift

            def run(self, _names, feeds):
                with torch.no_grad():
                    outputs = [x.numpy().copy() for x in reference(torch.from_numpy(feeds["planes"]))]
                outputs[2][:, 0, 0] += self.drift
                return outputs

        with redirect_stdout(io.StringIO()):
            gaps = import_onnx.onnx_parity(imported, Session(), probe, half=False)
            self.assertLess(max(logit for logit, _ in gaps.values()), 1e-4)
            with self.assertRaisesRegex(ValueError, "q by"):
                import_onnx.onnx_parity(imported, Session(drift=5.0), probe, half=False)

    def test_other_graphs_are_refused(self):
        unfolded = copy.deepcopy(self.model)
        unfolded.graph.node[-1].op_type = "BatchNormalization"
        broken = copy.deepcopy(self.model)
        add = next(node for node in broken.graph.node if node.op_type == "Add")
        add.input[1] = add.input[0]                      # no skip connection
        headless = copy.deepcopy(self.model)
        gemm = next(node for node in headless.graph.node if node.op_type == "Gemm")
        headless.graph.node.remove(gemm)
        for name, model, message in (("unfolded", unfolded, "BatchNormalization"),
                                     ("broken", broken, "skip connection|expected a single Add"),
                                     ("headless", headless, "missing heads|expected a single")):
            with self.subTest(graph=name), self.assertRaisesRegex(ValueError, message):
                import_onnx.folded_weights(model)

    def test_the_checkpoint_loads_wherever_checkpoints_are_loaded(self):
        imported, arch = import_onnx.import_network(self.model, self.planes[:96], self.legal[:96])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imported.pt"
            torch.save({"model": imported.state_dict(), "arch": arch}, path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
        restored = PolicyValueNet(*payload["arch"])
        restored.load_state_dict(payload["model"])            # strict: every name and shape
        self.assert_same_outputs(restored, atol=1e-4)
        inference = _prepare_network(payload, "cpu")          # what actors, arena and measure load
        self.assert_same_outputs(inference, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
