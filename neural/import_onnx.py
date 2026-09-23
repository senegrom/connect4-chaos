"""Imports an ONNX export back into a trainable PolicyValueNet checkpoint.

The exports are inference graphs. neural/export_onnx.py exports the network
in eval mode and the TorchScript exporter folds every BatchNorm into the
convolution before it, so an export holds one weight and one bias per
convolution - 84 anonymous onnx::Conv_N constants for the 42 convolutions of
20 blocks - and no normalisation at all; the drop heads' weights are transposed MatMul
constants, and the fp16 exports keep every weight in half precision. That is
exact for inference, but not a network the learner can train: training
normalises each batch by its own statistics, and the running statistics and
affine parameters that do so are gone.

The import walks the graph in execution order, checks that it is the network
in neural/model.py - stem, residual tower, column convolution, the linear
heads found by which outputs they feed - and rebuilds each BatchNorm around
its folded convolution. The convolution keeps the folded weight without its
bias, and the normalisation after it is chosen so that, whatever running
statistics it holds, eval mode is the identity plus that bias:

    running_mean = m, running_var = v, weight = sqrt(v + eps), bias = m + folded bias

With m and v the per-channel mean and (batch) variance of the convolution's
output over realistic positions, a training batch normalises by statistics
close to the running ones, so train mode stays close to eval mode as well.
The calibration positions come from cheap tactical playouts over every board
shape and rule set self-play uses: take a win, avoid a move that loses at once
or hands the opponent an immediate win, otherwise play at random.

The imported checkpoint has no optimizer state, so its first generation starts
AdamW from scratch; distill ramps the learning rate up when a warm start has
no moments to restore (DISTILL_WARMUP_STEPS).

Usage:
  python -m neural.import_onnx <model.onnx> <out.pt> [positions=2048] [seed=1]
"""

from __future__ import annotations

import copy
import hashlib
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from .export_onnx import Exported, validate_parity
from .gpu_env import ACTIONS, FLIP, LOSS, NOT_TERMINAL, WIN, BoardBatch, step
from .model import CANVAS, PLANES, PolicyValueNet

# The linear heads, named as in PolicyValueNet, and the graph outputs each
# one's result reaches: that, and its width, is how each is recognised.
HEADS = {
    "global_features.0": frozenset({"policy", "value", "q"}),
    "drop_logit": frozenset({"policy"}),
    "transform_logit": frozenset({"policy"}),
    "drop_q": frozenset({"q"}),
    "transform_q": frozenset({"q"}),
    "value": frozenset({"value"}),
}
WIDTHS = {"drop_logit": 1, "transform_logit": 3, "drop_q": 3, "transform_q": 9, "value": 3}
PASSTHROUGH = ("Cast", "Identity")    # fp16 exports cast at their inputs and outputs
TRANSFORM_WEIGHT = 0.5                # a transform is drawn half as often as a drop
MAX_PLIES = 120


def _attributes(node):
    from onnx import helper
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def folded_weights(model):
    """((channels, blocks, head_channels), convs, heads) from an exported
    ModelProto: `convs` lists every (weight, bias) in PolicyValueNet order -
    stem, conv1 and conv2 of each block, column - as they are folded, and
    `heads` maps each linear head's name to its (weight, bias) in the layout
    nn.Linear stores. Everything is float32. Raises ValueError when the graph
    is not an export of PolicyValueNet."""
    from onnx import numpy_helper

    graph = model.graph
    if any(node.op_type == "BatchNormalization" for node in graph.node):
        raise ValueError("the export keeps its BatchNormalization nodes; only folded exports are supported")
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            value = _attributes(node).get("value")
            if value is not None:
                constants[node.output[0]] = numpy_helper.to_array(value)
    producer = {name: node for node in graph.node for name in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)
    inputs = {tensor.name for tensor in graph.input} - set(constants)
    outputs = {tensor.name for tensor in graph.output}

    def tensor(name):
        if name not in constants:
            raise ValueError(f"{name} is not a constant of the graph")
        return torch.from_numpy(constants[name].astype("float32"))

    def source(name):
        while name in producer and producer[name].op_type in PASSTHROUGH:
            name = producer[name].input[0]
        return name

    def users(name):
        for node in consumers.get(name, []):
            if node.op_type in PASSTHROUGH:
                yield from users(node.output[0])
            else:
                yield node

    def only_user(name, op_type):
        found = list(users(name))
        if len(found) != 1 or found[0].op_type != op_type:
            raise ValueError(f"{name} feeds {[node.op_type for node in found]}, expected a single {op_type}")
        return found[0]

    def reached(name):
        found, pending, seen = set(), [name], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in outputs:
                found.add(current)
            for node in consumers.get(current, []):
                pending.extend(node.output)
        return frozenset(found)

    def convolution(node, kernel):
        attributes = _attributes(node)
        if (list(attributes.get("kernel_shape", [])) != [kernel, kernel] or attributes.get("group", 1) != 1
                or list(attributes.get("strides", [1, 1])) != [1, 1]
                or list(attributes.get("dilations", [1, 1])) != [1, 1]
                or list(attributes.get("pads", [0] * 4)) != [kernel // 2] * 4):
            raise ValueError(f"{node.name}: not a {kernel}x{kernel} same-padding convolution")
        weight = tensor(node.input[1])
        bias = tensor(node.input[2]) if len(node.input) > 2 and node.input[2] else torch.zeros(weight.shape[0])
        return weight, bias

    # Convolutions in execution order, checked link by link: a stem on the
    # input, each block conv-relu-conv-add(skip)-relu, then the column conv.
    convs = [node for node in graph.node if node.op_type == "Conv"]
    blocks, odd = divmod(len(convs) - 2, 2)
    if blocks < 1 or odd:
        raise ValueError(f"{len(convs)} convolutions: expected a stem, two per block and a column")
    if source(convs[0].input[0]) not in inputs:
        raise ValueError("the first convolution does not read the planes")
    trunk = only_user(convs[0].output[0], "Relu").output[0]
    for index in range(blocks):
        first, second = convs[1 + 2 * index], convs[2 + 2 * index]
        if source(first.input[0]) != trunk:
            raise ValueError(f"tower block {index} does not read the previous block")
        middle = only_user(first.output[0], "Relu").output[0]
        if source(second.input[0]) != middle:
            raise ValueError(f"tower block {index}: second convolution does not follow the first")
        add = only_user(second.output[0], "Add")
        if {source(name) for name in add.input} != {second.output[0], trunk}:
            raise ValueError(f"tower block {index} has no skip connection")
        trunk = only_user(add.output[0], "Relu").output[0]
    if source(convs[-1].input[0]) != trunk:
        raise ValueError("the column convolution does not read the tower")
    folded = [convolution(convs[0], 3)] + [convolution(node, 3) for node in convs[1:-1]]
    folded.append(convolution(convs[-1], 1))

    # Linear heads: a Gemm, or a MatMul by a constant plus a bias Add.
    heads = {}
    for node in graph.node:
        if node.op_type == "Gemm":
            attributes = _attributes(node)
            if attributes.get("transA", 0) or attributes.get("alpha", 1.0) != 1.0 or attributes.get("beta", 1.0) != 1.0:
                raise ValueError(f"{node.name}: not a plain linear layer")
            weight = tensor(node.input[1])
            weight = weight if attributes.get("transB", 0) else weight.T
            bias = tensor(node.input[2]) if len(node.input) > 2 and node.input[2] else torch.zeros(weight.shape[0])
            result = node.output[0]
        elif node.op_type == "MatMul" and source(node.input[1]) in constants:
            weight = tensor(source(node.input[1])).T      # x @ W stores W as (in, out)
            add = only_user(node.output[0], "Add")
            others = [source(name) for name in add.input if source(name) != node.output[0]]
            if len(others) != 1:
                raise ValueError(f"{node.name}: no bias after the MatMul")
            bias = tensor(others[0])
            result = add.output[0]
        else:
            continue
        reach, width = reached(result), weight.shape[0]
        names = [name for name, heads_reached in HEADS.items()
                 if heads_reached == reach and WIDTHS.get(name, width) == width]
        if len(names) != 1 or names[0] in heads:
            raise ValueError(f"{node.name}: cannot tell which head a {tuple(weight.shape)} layer "
                             f"reaching {sorted(reach)} is")
        heads[names[0]] = (weight.contiguous(), bias)
    if set(heads) != set(HEADS):
        raise ValueError(f"missing heads: {sorted(set(HEADS) - set(heads))}")
    arch = (folded[0][0].shape[0], blocks, folded[-1][0].shape[0])
    if folded[0][0].shape[1] != PLANES:
        raise ValueError(f"the stem reads {folded[0][0].shape[1]} planes, not {PLANES}")
    return arch, folded, heads


def _set_normalisation(norm, folded_bias, mean, var):
    """Eval mode computes x + folded_bias for these running statistics."""
    with torch.no_grad():
        norm.running_mean.copy_(mean)
        norm.running_var.copy_(var)
        norm.weight.copy_(torch.sqrt(var + norm.eps))
        norm.bias.copy_(mean + folded_bias)


def build_network(arch, folded, heads):
    """PolicyValueNet holding the folded weights, and (norm, folded bias) for
    each BatchNorm. The norms start as the identity plus their bias for
    running statistics (0, 1): exact in eval mode before calibration too."""
    net = PolicyValueNet(*arch)
    pairs = [(net.stem[0], net.stem[1])]
    for block in net.tower:
        pairs += [(block.conv1, block.norm1), (block.conv2, block.norm2)]
    pairs.append((net.column_features[0], net.column_features[1]))
    norms = []
    with torch.no_grad():
        for (conv, norm), (weight, bias) in zip(pairs, folded, strict=True):
            if weight.shape != conv.weight.shape:
                raise ValueError(f"a {tuple(weight.shape)} convolution where the network has "
                                 f"{tuple(conv.weight.shape)}")
            conv.weight.copy_(weight)
            _set_normalisation(norm, bias, torch.zeros_like(bias), torch.ones_like(bias))
            norms.append((norm, bias))
        for name, (weight, bias) in heads.items():
            linear = net.get_submodule(name)
            if weight.shape != linear.weight.shape or bias.shape != linear.bias.shape:
                raise ValueError(f"{name}: {tuple(weight.shape)} where the network has {tuple(linear.weight.shape)}")
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
    return net.eval(), norms


@torch.no_grad()
def calibrate(net, norms, planes, legal, batch_size=64):
    """Sets every BatchNorm's running statistics to the per-channel mean and
    batch variance of its input over these positions, keeping eval mode
    exact (see the module docstring). Returns the number of positions."""
    totals = {id(norm): None for norm, _ in norms}

    def observe(module, inputs):
        x = inputs[0].detach().double()
        count = x.shape[0] * x.shape[2] * x.shape[3]
        mean = x.mean(dim=(0, 2, 3))
        m2 = ((x - mean[None, :, None, None]) ** 2).sum(dim=(0, 2, 3))
        previous = totals[id(module)]
        if previous is not None:            # Chan et al.: merge two partial moments
            seen, seen_mean, seen_m2 = previous
            delta, combined = mean - seen_mean, seen + count
            mean = seen_mean + delta * count / combined
            m2 = seen_m2 + m2 + delta ** 2 * seen * count / combined
            count = combined
        totals[id(module)] = (count, mean, m2)

    handles = [norm.register_forward_pre_hook(observe) for norm, _ in norms]
    net.eval()
    try:
        for start in range(0, len(planes), batch_size):
            net(planes[start:start + batch_size].float(), legal[start:start + batch_size])
    finally:
        for handle in handles:
            handle.remove()
    for norm, bias in norms:
        count, mean, m2 = totals[id(norm)]
        # The variance a training batch normalises by: divided by the count.
        _set_normalisation(norm, bias, mean.float(), (m2 / count).float())
    return len(planes)


@torch.no_grad()
def _tactical_moves(board, generator, chunk=64):
    """One move per game: an immediate win if there is one; otherwise a move
    that neither loses at once (a transform can) nor hands the opponent an
    immediate win; otherwise any legal move. Random among those, drops
    preferred to transforms. Every action and every reply to it is stepped
    as one batch, `chunk` games at a time to bound the scratch memory."""
    legal = board.legal()
    wins = torch.zeros_like(legal)
    losing = torch.zeros_like(legal)
    for start in range(0, len(board), chunk):
        games = torch.arange(start, min(start + chunk, len(board)))
        n = len(games)
        child, outcome = step(board.select(games.repeat_interleave(ACTIONS)),
                              torch.arange(ACTIONS).repeat(n))
        outcome = outcome.view(n, ACTIONS)
        _grandchild, answer = step(child.select(torch.arange(n * ACTIONS).repeat_interleave(ACTIONS)),
                                   torch.arange(ACTIONS).repeat(n * ACTIONS))
        threat = (child.legal().view(-1) & (answer == WIN)).view(n, ACTIONS, ACTIONS).any(dim=2)
        wins[games] = outcome == WIN
        losing[games] = (outcome == LOSS) | ((outcome == NOT_TERMINAL) & threat)
    wins &= legal
    safe = legal & ~losing
    choice = torch.where(wins.any(dim=1, keepdim=True), wins,
                         torch.where(safe.any(dim=1, keepdim=True), safe, legal)).float()
    choice[:, FLIP:] *= TRANSFORM_WEIGHT
    return torch.multinomial(choice, 1, generator=generator).squeeze(1)


@torch.no_grad()
def calibration_positions(count, seed=1, games=None):
    """(planes, legal) for `count` positions sampled uniformly from every ply
    of tactical playouts (_tactical_moves), one game per position by default.
    The games cycle through gpu_selfplay.all_shapes() - every board, connect
    length and rule set - as the actors do: which boards a batch holds moves
    its normalisation statistics most, since off-board cells count too.
    Planes are rounded through the replay shards' uint8 encoding, so they are
    exactly what training would read."""
    from .gpu_selfplay import all_shapes

    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    shapes = all_shapes()
    games = games or count
    picks = [shapes[index % len(shapes)] for index in range(max(games, len(shapes)))]
    rng.shuffle(picks)
    picks = picks[:games]         # fewer games than boards: a random subset of boards
    board = BoardBatch([p[0] for p in picks], [p[1] for p in picks], [p[2] for p in picks],
                       [p[3] for p in picks], "cpu")
    planes, legal = [], []
    for _ply in range(MAX_PLIES):
        if len(board) == 0:
            break
        zeros = torch.zeros(len(board), dtype=torch.bool)
        planes.append((board.planes(zeros, zeros) * 10).round().to(torch.uint8))
        legal.append(board.legal())
        child, outcome = step(board, _tactical_moves(board, generator), check=True)
        board = child.select((outcome == NOT_TERMINAL).nonzero().squeeze(1))
    planes, legal = torch.cat(planes), torch.cat(legal)
    if len(planes) < count:
        raise ValueError(f"{len(planes)} positions from {len(picks)} games, fewer than {count}")
    pick = torch.randperm(len(planes), generator=generator)[:count]
    return planes[pick].float() / 10, legal[pick]


def import_network(model, planes, legal):
    """The trainable network for an exported ModelProto, calibrated on these
    positions, and its architecture."""
    arch, folded, heads = folded_weights(model)
    net, norms = build_network(arch, folded, heads)
    calibrate(net, norms, planes, legal)
    return net, arch


@torch.no_grad()
def onnx_parity(net, session, planes, *, half):
    """Runs the imported network (unmasked, as exported) and the export under
    onnxruntime on `planes` and compares every head with the exporter's own
    check (export_onnx.validate_parity), which raises past its tolerance.
    Returns {head: (largest logit gap, largest probability gap)}."""
    want = Exported(net).eval()(planes)
    got = [torch.from_numpy(array) for array in session.run(None, {"planes": planes.numpy()})]
    gaps = {}
    for name, dimension, reference, actual in zip(("policy", "value", "q"), (1, 1, 2), want, got):
        gaps[name] = (float((reference - actual).abs().max()),
                      float((torch.softmax(reference, dimension) - torch.softmax(actual, dimension)).abs().max()))
    validate_parity(want, got, batch=len(planes), half=half)
    return gaps


@torch.no_grad()
def train_mode_deviation(net, planes, legal):
    """How far train mode - normalising by this batch's own statistics -
    moves the outputs from eval mode: per head the largest and mean absolute
    probability differences (legal moves only for the policy and Q), and the
    share of positions whose top policy move changes."""
    want = net.eval()(planes, legal)
    got = copy.deepcopy(net).train()(planes, legal)      # a copy: train mode moves the running statistics
    mask = legal.float()
    policy = [torch.softmax(logits, dim=1) for logits in (want[0], got[0])]
    value = [torch.softmax(logits, dim=1) for logits in (want[1], got[1])]
    q = [torch.softmax(logits, dim=2) * mask[:, :, None] for logits in (want[2], got[2])]
    deviation = {}
    for name, (a, b), rows in (("policy", policy, mask), ("value", value, None), ("q", q, mask[:, :, None])):
        gap = (a - b).abs()
        mean = float(gap.sum() / (rows.expand_as(gap).sum() if rows is not None else gap.numel()))
        deviation[name] = (float(gap.max()), mean)
    deviation["top_move_changed"] = float((policy[0].argmax(dim=1) != policy[1].argmax(dim=1)).float().mean())
    return deviation


def main():
    import onnx
    import onnxruntime

    from .distill import save_checkpoint

    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    positions = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    if destination.suffix != ".pt":
        raise ValueError("the checkpoint path must end in .pt")
    started = time.time()
    raw = source.read_bytes()
    model = onnx.load_from_string(raw)
    half = any(tensor.data_type == onnx.TensorProto.FLOAT16 for tensor in model.graph.initializer)
    arch, folded, heads = folded_weights(model)
    del model
    net, norms = build_network(arch, folded, heads)
    del folded, heads
    print(f"{source.name}: {arch[0]} channels x {arch[1]} blocks, {arch[2]} head channels, "
          f"{'float16' if half else 'float32'} weights; {len(norms)} BatchNorms to rebuild", flush=True)

    # Disjoint positions: calibration, the train-mode check, the parity probe.
    planes, legal = calibration_positions(positions + 256 + 64, seed)
    print(f"{len(planes)} positions from tactical playouts in {time.time() - started:.0f}s", flush=True)
    calibrate(net, norms, planes[:positions], legal[:positions])
    print(f"calibrated on {positions} positions, {time.time() - started:.0f}s", flush=True)

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()     # the same CPU budget as torch
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    probe = torch.cat([planes[positions + 256:], torch.rand((8, PLANES, CANVAS, CANVAS))])
    print(f"parity with onnxruntime on {len(probe)} positions ({len(probe) - 8} played, 8 random):", flush=True)
    onnx_parity(net, session, probe, half=half)
    del session
    deviation = train_mode_deviation(net, planes[positions:positions + 256], legal[positions:positions + 256])
    print("train mode against eval mode on a fresh batch of 256: " + ", ".join(
        f"{name} max {worst:.2e} mean {mean:.2e}" for name, (worst, mean)
        in ((key, deviation[key]) for key in ("policy", "value", "q")))
        + f"; top move changed on {deviation['top_move_changed']:.1%}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint({"model": {key: value.detach().contiguous() for key, value in net.state_dict().items()},
                     "arch": arch,
                     "imported_from": {"onnx": source.name, "onnx_sha256": hashlib.sha256(raw).hexdigest(),
                                       "precision": "float16" if half else "float32",
                                       "calibration_positions": positions, "calibration_seed": seed}},
                    destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB) sha256 {digest}; "
          f"{time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
