"""Perfect distillation: trains the policy/value/Q net on exact shards.

Losses: cross-entropy of the masked policy against the exactly-optimal
action distribution, cross-entropy of the W/D/L head against the exact
value, and cross-entropy of the per-action Q head against the exact
value of every legal action. Reports, per held-out shard, value accuracy
and two blunder rates: the policy argmax's and the Q-argmax's (choosing
the action whose predicted outcome distribution has the best expectation).

Usage: python -m neural.distill <shard_dir> <out_dir> [steps] [batch]
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

from .model import PolicyValueNet
from .training_config import parse_shape_spec
from .data_split import SPLIT_CHUNK, SPLIT_VERSION, select_samples, validation_mask

def mirror_batch(planes, legal, policy, q):
    """Mirror each board about its own centre.

    Boards sit left-aligned on the 10-wide canvas, so flipping the canvas
    would slide a narrow board to the right edge and hand the network an
    input it never sees anywhere else. The reflection is therefore
    board-relative: column c swaps with cols-1-c, drops follow the same
    permutation, the two rotations swap because a mirror conjugates them,
    and the flip is self-conjugate."""
    width = planes.shape[3]
    device = planes.device
    columns = planes[:, 2, 0, :].sum(dim=1).long()          # region width per row
    positions = torch.arange(width, device=device)[None, :]
    mirrored = (columns[:, None] - 1 - positions).clamp(min=0)
    source = torch.where(positions < columns[:, None], mirrored, positions)
    planes = planes.gather(3, source[:, None, None, :].expand_as(planes))
    transforms = torch.tensor([10, 12, 11], device=device)[None, :].expand(len(planes), 3)
    order = torch.cat([source, transforms], dim=1)
    return planes, legal.gather(1, order), policy.gather(1, order), q.gather(1, order)


def decode_planes(planes):
    """Shard planes are float32 (exact shards) or uint8 scaled by 10
    (self-play shards); either way float16 is what training uses."""
    if planes.dtype == torch.uint8:
        return planes.half() / 10
    return planes.half()


def without_heldout_positions(shard, holdout_shapes):
    """Filter by encoded rules, not filenames or a game's starting orientation."""
    planes = shard["planes"]
    scale = float(shard.get("planes_scale", 10 if planes.dtype == torch.uint8 else 1))
    rows = (planes[:, 2, :, 0] > 0).sum(dim=1)
    cols = (planes[:, 2, 0, :] > 0).sum(dim=1)
    connects = (planes[:, 3, 0, 0].float() * (10 / scale)).round().long()
    chaos = planes[:, 4, 0, 0] > 0
    keep = torch.ones(len(planes), dtype=torch.bool)
    for r, c, k, mode in holdout_shapes:
        shape = (rows == r) & (cols == c)
        if mode:
            shape |= (rows == c) & (cols == r)
        keep &= ~(shape & (connects == k) & (chaos == mode))
    if bool(keep.all()):
        return shard
    if not bool(keep.any()):
        return None
    return select_samples(shard, keep)


def filtered_chunks(shard, holdout_shapes, *, validation=False, whole_board_held=False,
                    limit=None, newest_first=False, trusted_partition=False):
    """Yield bounded, aligned chunks; the last replay chunk obeys the exact cap.

    The same position partition is enforced on legacy exact shards and replay.
    The explicit whole-board holdout supersedes the default 10% partition.
    """
    count = len(shard["planes"])
    required = ("legal", "policy", "wdl")
    if any(len(shard[key]) != count for key in required):
        raise ValueError("Misaligned training tensors in shard")
    if "q" in shard and len(shard["q"]) != count:
        raise ValueError("Misaligned Q targets in shard")
    if "q" not in shard and not (shard.get("source") == "selfplay" and shard.get("q_default") == 3):
        raise ValueError("Shard has no Q targets or self-play Q default")
    remaining = count if limit is None else limit
    if newest_first:
        ranges = ((max(0, stop - SPLIT_CHUNK), stop) for stop in range(count, 0, -SPLIT_CHUNK))
    else:
        ranges = ((start, min(start + SPLIT_CHUNK, count)) for start in range(0, count, SPLIT_CHUNK))
    for start, stop in ranges:
        if remaining <= 0:
            break
        chunk = select_samples(shard, slice(start, stop))
        if not validation and holdout_shapes:
            chunk = without_heldout_positions(chunk, holdout_shapes)
            if chunk is None:
                continue
        if not whole_board_held and not trusted_partition:
            if chunk.get("split_version") == SPLIT_VERSION and "validation" in chunk:
                reserved = chunk["validation"].bool()
            else:
                reserved = validation_mask(chunk["planes"], chunk.get("planes_scale"))
            keep = reserved if validation else ~reserved
            if not bool(keep.any()):
                continue
            if not bool(keep.all()):
                chunk = select_samples(chunk, keep)
        size = len(chunk["planes"])
        if size > remaining:
            chunk = select_samples(chunk, slice(-remaining, None) if newest_first else slice(0, remaining))
        remaining -= len(chunk["planes"])
        yield chunk


def load_shards(shard_dirs):
    """Load exact validation shards and position-disjoint exact/replay training.

    Shard 0000 supplies validation candidates, but the stable position hash,
    not the filename or sampling seed, determines the default split. Existing
    legacy shards are filtered too. Directories may be separated by ';'.
    """
    holdout = {tag.strip() for tag in os.environ.get("DISTILL_HOLDOUT_CONFIGS", "").split(",") if tag.strip()}
    if "all" in holdout:
        raise ValueError("A training holdout must name specific configurations")
    holdout_shapes = [shape for tag in sorted(holdout) for shape in (parse_shape_spec(tag) or [])]
    window = int(os.environ.get("DISTILL_REPLAY_WINDOW", "4000000"))
    if window < 0:
        raise ValueError("DISTILL_REPLAY_WINDOW must be non-negative")
    train, held, replay_shards = [], [], []
    for shard_dir in str(shard_dirs).split(";"):
        for path in sorted(Path(shard_dir).glob("*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if "q" not in shard and not (shard.get("source") == "selfplay" and shard.get("q_default") == 3):
                raise SystemExit(f"{path} predates the Q head; rebuild the dataset")
            shard["mtime"] = path.stat().st_mtime
            if shard.get("source") == "selfplay":
                replay_shards.append(shard)
                continue
            tag = path.stem.rsplit("-", 1)[0]
            whole_board_held = tag in holdout
            current_split = shard.get("split_version") == SPLIT_VERSION
            declared = shard.get("split")
            if path.stem.endswith("0000"):
                if current_split and declared != "validation":
                    raise ValueError(f"{path} declares {declared!r}, expected validation")
                held.extend(filtered_chunks(shard, holdout_shapes, validation=True,
                                            whole_board_held=whole_board_held,
                                            trusted_partition=current_split))
            elif not whole_board_held:
                if current_split and declared != "train":
                    raise ValueError(f"{path} declares {declared!r}, expected train")
                train.extend(filtered_chunks(shard, holdout_shapes,
                                             trusted_partition=current_split))
    # Visit newest replay first, and only decode/filter chunks needed to fill
    # the position budget. No whole-shard overshoot or full-archive copies.
    replay_shards.sort(key=lambda s: s["mtime"], reverse=True)
    total = 0
    for shard in replay_shards:
        if total >= window:
            break
        for chunk in filtered_chunks(shard, holdout_shapes, limit=window - total, newest_first=True):
            train.append(chunk)
            total += len(chunk["planes"])
    if replay_shards:
        print(f"replay window {window}: keeping {total} newest eligible positions")
    if not train:
        raise ValueError("No training positions remain; held-out data will not be used for training")
    return train, held


def q_choice(q_logits, legal):
    """Action with the best expected outcome under the Q head."""
    distribution = torch.softmax(q_logits, dim=2)
    expectation = distribution[:, :, 2] - distribution[:, :, 0]
    return expectation.masked_fill(~legal, float('-inf')).argmax(dim=1)


def quantize_planes(planes, scale=None):
    """Canonical compact representation used by exact and replay shards."""
    if planes.dtype == torch.uint8:
        actual = int(scale if scale is not None else 10)
        if actual != 10:
            raise ValueError(f"Unsupported uint8 plane scale {actual}")
        return planes
    return (planes.float() * 10).round().clamp_(0, 10).to(torch.uint8)


def _tensor_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def stage_training_tensors(planes, legal, policy, wdl, q, replay_idx, exact_idx, device):
    """Prefer one H2D copy for the whole compact dataset when memory allows."""
    if device != "cuda":
        return (planes, legal, policy, wdl, q, replay_idx, exact_idx), False
    tensors = (planes, legal, policy, wdl, q, replay_idx, exact_idx)
    need = _tensor_bytes(*tensors)
    free, _total = torch.cuda.mem_get_info()
    reserve = int(float(os.environ.get("DISTILL_GPU_RESERVE_GB", "24")) * (1024 ** 3))
    setting = os.environ.get("DISTILL_GPU_DATA", "auto").lower()
    use_gpu = setting not in {"0", "false", "off"} and need + reserve < free
    if setting in {"1", "true", "on"} and not use_gpu:
        raise MemoryError(f"Training data needs {need / 1e9:.1f} GB plus {reserve / 1e9:.1f} GB reserve")
    if use_gpu:
        print(f"training data: {need / 1e9:.2f} GB resident on GPU", flush=True)
        return tuple(t.to(device) for t in tensors), True
    pin_limit = int(float(os.environ.get("DISTILL_PIN_MAX_GB", "8")) * (1024 ** 3))
    pin = os.environ.get("DISTILL_PIN_MEMORY", "1") != "0" and need <= pin_limit
    if pin:
        try:
            tensors = tuple(t.pin_memory() for t in tensors)
            print(f"training data: {need / 1e9:.2f} GB pinned on CPU", flush=True)
        except RuntimeError:
            pin = False
    if not pin:
        print(f"training data: {need / 1e9:.2f} GB pageable CPU fallback", flush=True)
    return tensors, False


def create_optimizer(net, lr, device, capturable=False):
    """Fused AdamW on CUDA, with a portable eager fallback. `capturable`
    keeps the step counters and the learning rate on the device, which a
    CUDA graph of the training step needs."""
    kwargs = dict(lr=torch.tensor(float(lr), device=device) if capturable else lr, weight_decay=1e-4)
    if device == "cuda" and os.environ.get("DISTILL_FUSED_ADAMW", "1") != "0":
        kwargs["fused"] = True
    if capturable:
        kwargs["capturable"] = True
    try:
        return torch.optim.AdamW(net.parameters(), **kwargs)
    except (TypeError, RuntimeError, ValueError):
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)


def save_checkpoint(payload, path):
    """Expose a checkpoint only after serialization finishes successfully."""
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    shard_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 512

    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, held = load_shards(shard_dir)
    # Keep the host copy in the same compact dtypes as the shards. This cuts
    # planes from float16 to uint8 and WDL/Q labels from int64 to uint8.
    total = sum(len(s["planes"]) for s in train)
    planes = torch.empty((total, 7, 10, 10), dtype=torch.uint8)
    legal = torch.empty((total, 13), dtype=torch.bool)
    policy = torch.empty((total, 13), dtype=torch.float32)
    wdl = torch.empty((total,), dtype=torch.uint8)
    q = torch.full((total, 13), 3, dtype=torch.uint8)
    root = torch.full((total,), 2.0, dtype=torch.float32)      # 2 = no search value recorded
    cursor = 0
    replay_flags = []
    for shard in train:
        count = len(shard["planes"])
        shard["count"] = count
        planes[cursor:cursor + count] = quantize_planes(
            shard["planes"], shard.get("planes_scale"))
        legal[cursor:cursor + count] = shard["legal"]
        policy[cursor:cursor + count] = shard["policy"]
        wdl[cursor:cursor + count] = shard["wdl"].to(torch.uint8)
        if "q" in shard:
            q[cursor:cursor + count] = shard["q"].to(torch.uint8)
        if "root_value" in shard:
            root[cursor:cursor + count] = shard["root_value"].float()
        replay_flags.append(torch.full((count,), shard.get("source") == "selfplay"))
        cursor += count
        for key in ("planes", "legal", "policy", "wdl", "q", "validation", "root_value"):
            if key in shard:
                shard[key] = None

    is_replay = torch.cat(replay_flags)
    replay_idx = is_replay.nonzero().squeeze(1)
    exact_idx = (~is_replay).nonzero().squeeze(1)
    replay_fraction = float(os.environ.get("DISTILL_REPLAY_FRACTION", "0.75"))
    if not 0 <= replay_fraction <= 1:
        raise ValueError("DISTILL_REPLAY_FRACTION must be between 0 and 1")
    if len(replay_idx) == 0 or len(exact_idx) == 0:
        replay_fraction = 1.0 if len(exact_idx) == 0 else 0.0
    print(f"train samples: {len(planes)} (exact {len(exact_idx)}, replay {len(replay_idx)}, "
          f"replay fraction {replay_fraction:.2f}), held shards: {len(held)}, device: {device}")

    init = os.environ.get("DISTILL_INIT")
    payload = torch.load(init, map_location=device, weights_only=True) if init else None
    if payload:
        net = PolicyValueNet(*payload.get("arch", (192, 12, 48))).to(device)
        net.load_state_dict(payload["model"])
        print(f"warm start from {init} arch={payload.get('arch', (192, 12, 48))}")
    else:
        net = PolicyValueNet().to(device)
    channels_last = device == "cuda" and os.environ.get("DISTILL_CHANNELS_LAST", "1") != "0"
    if channels_last:
        net.to(memory_format=torch.channels_last)
    print(f"architecture: {net.channels} channels x {net.blocks} blocks, "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M params")

    lr = float(os.environ.get("DISTILL_LR", "1e-3"))
    # Optional bonus on the policy's entropy over legal moves. Zero (the
    # default) leaves the loss exactly as before; a small weight keeps the
    # policy from collapsing onto its favourite move, which starves the
    # search of alternatives even when that move is usually right.
    entropy_bonus = float(os.environ.get("DISTILL_ENTROPY_BONUS", "0"))
    if entropy_bonus < 0:
        raise ValueError("DISTILL_ENTROPY_BONUS must not be negative")
    if entropy_bonus:
        print(f"entropy bonus {entropy_bonus:g} on the policy's legal-move entropy", flush=True)
    # Weight of the squared error between the value head's expectation
    # (win minus loss probability) and the search's own value of the
    # position, on the self-play rows that recorded one. Zero = off.
    root_value_weight = float(os.environ.get("DISTILL_ROOT_VALUE_WEIGHT", "0"))
    if root_value_weight < 0:
        raise ValueError("DISTILL_ROOT_VALUE_WEIGHT must not be negative")
    if root_value_weight:
        print(f"root value weight {root_value_weight:g} on the search value of self-play rows", flush=True)
    use_amp = device == "cuda" and os.environ.get("DISTILL_FP32", "") != "1"
    # The whole training step - forward, loss, backward, AdamW - replays as
    # one CUDA graph. Profiled eagerly, a step was about 30 ms of GPU work
    # and about as much CPU launch work, serialised by per-step host reads;
    # the graph leaves only the GPU work. DISTILL_GRAPH=0 runs the same
    # step eagerly, and any capture failure falls back to that.
    use_graph = device == "cuda" and os.environ.get("DISTILL_GRAPH", "1") != "0"
    optimizer = create_optimizer(net, lr, device, capturable=use_graph)
    capturable = bool(optimizer.defaults.get("capturable", False))
    use_graph = use_graph and capturable
    init_opt = os.environ.get("DISTILL_INIT_OPT")
    if init_opt and os.path.exists(init_opt) and os.environ.get("DISTILL_RESET_OPTIMIZER", "") != "1":
        try:
            state = torch.load(init_opt, map_location=device, weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            print(f"optimizer moments restored from {init_opt}", flush=True)
        except Exception as exc:  # a sidecar must never make its model unusable
            print(f"optimizer sidecar ignored: {type(exc).__name__}: {exc}", flush=True)
    lr_value = torch.tensor(lr, device=device) if capturable else lr
    for group in optimizer.param_groups:
        group["lr"] = lr_value

    def set_lr(step):
        # CosineAnnealingLR(T_max=steps) in closed form; `step` is 1-based and
        # the value is what that scheduler had set before this step.
        value = 0.5 * lr * (1.0 + math.cos(math.pi * (step - 1) / steps))
        for group in optimizer.param_groups:
            if torch.is_tensor(group["lr"]):
                group["lr"].fill_(value)
            else:
                group["lr"] = value

    torch.backends.cudnn.benchmark = True
    (planes, legal, policy, wdl, q, replay_idx, exact_idx), resident = stage_training_tensors(
        planes, legal, policy, wdl, q, replay_idx, exact_idx, device)
    root = root.to(device) if resident else root
    sample_device = device if resident else "cpu"
    generator = torch.Generator(device=sample_device).manual_seed(20260901)
    n_replay = int(round(batch * replay_fraction))
    n_exact = batch - n_replay

    # Static batch buffers: the graph reads these, the sampler fills them.
    static_planes = torch.zeros((batch, 7, 10, 10), device=device)
    if channels_last:
        static_planes = static_planes.contiguous(memory_format=torch.channels_last)
    static_legal = torch.zeros((batch, 13), dtype=torch.bool, device=device)
    static_policy = torch.zeros((batch, 13), device=device)
    static_wdl = torch.zeros((batch,), dtype=torch.int64, device=device)
    static_q = torch.full((batch, 13), 3, dtype=torch.int64, device=device)
    static_root = torch.full((batch,), 2.0, device=device)
    totals = torch.zeros(6, device=device)          # summed losses (+ entropy, root) since the last report

    def losses():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp,
                            cache_enabled=False):
            logits, values, q_logits = net(static_planes, static_legal)
        logits, values, q_logits = logits.float(), values.float(), q_logits.float()
        log_probs = torch.log_softmax(logits, dim=1)
        # Self-play rows from a shallow ply carry an all-zero policy target:
        # their outcome still teaches the value head, but they must not drag
        # the policy towards a distribution no search produced.
        per_row = -(static_policy * log_probs.masked_fill(~static_legal, 0.0)).sum(dim=1)
        taught = static_policy.sum(dim=1) > 0
        policy_loss = (per_row * taught).sum() / taught.sum().clamp(min=1)
        # Mean policy entropy over legal moves of the taught rows: reported
        # every window, and subtracted from the loss when a bonus is set.
        entropy_rows = -(log_probs.exp() * log_probs.masked_fill(~static_legal, 0.0)).sum(dim=1)
        entropy = (entropy_rows * taught).sum() / taught.sum().clamp(min=1)
        value_loss = nn.functional.cross_entropy(values, static_wdl)
        # Sum over the supervised action targets and divide by their count:
        # a batch without any (no exact rows) yields zero, never NaN, and
        # there is no data-dependent branch to break the graph.
        q_targets = static_q.reshape(-1)
        q_loss = (nn.functional.cross_entropy(q_logits.reshape(-1, 3), q_targets,
                                               ignore_index=3, reduction="sum")
                  / (q_targets != 3).sum().clamp(min=1))
        # Self-play rows also carry the search's value of the position (2 =
        # none recorded): a lower-variance target than the game's outcome,
        # matched by the value head's expectation win - loss.
        probabilities = torch.softmax(values, dim=1)
        expected = probabilities[:, 2] - probabilities[:, 0]
        has_root = static_root.abs() <= 1.0
        root_loss = ((expected - static_root) ** 2 * has_root).sum() / has_root.sum().clamp(min=1)
        total = policy_loss + value_loss + q_loss
        if root_value_weight:
            total = total + root_value_weight * root_loss
        if entropy_bonus:
            total = total - entropy_bonus * entropy
        return total, policy_loss, value_loss, q_loss, entropy, root_loss

    def train_step():
        loss, policy_loss, value_loss, q_loss, entropy, root_loss = losses()
        loss.backward()
        optimizer.step()
        totals.add_(torch.stack([loss.detach(), policy_loss.detach(), value_loss.detach(),
                                 q_loss.detach(), entropy.detach(), root_loss.detach()]))

    def load_batch(step):
        picks = torch.cat([
            replay_idx[torch.randint(0, max(1, len(replay_idx)), (n_replay,),
                                     generator=generator, device=sample_device)]
            if n_replay else torch.empty(0, dtype=torch.int64, device=sample_device),
            exact_idx[torch.randint(0, max(1, len(exact_idx)), (n_exact,),
                                    generator=generator, device=sample_device)]
            if n_exact else torch.empty(0, dtype=torch.int64, device=sample_device),
        ])
        b_planes, b_legal = planes[picks], legal[picks]
        b_policy, b_wdl, b_q = policy[picks], wdl[picks], q[picks]
        b_root = root[picks]
        if not resident:
            non_blocking = device == "cuda" and b_planes.is_pinned()
            b_planes = b_planes.to(device, non_blocking=non_blocking)
            b_legal = b_legal.to(device, non_blocking=non_blocking)
            b_policy = b_policy.to(device, non_blocking=non_blocking)
            b_wdl = b_wdl.to(device, non_blocking=non_blocking)
            b_q = b_q.to(device, non_blocking=non_blocking)
            b_root = b_root.to(device, non_blocking=non_blocking)
        b_planes = b_planes.float().mul_(0.1)
        b_wdl, b_q = b_wdl.long(), b_q.long()
        if step % 2 == 0:
            b_planes, b_legal, b_policy, b_q = mirror_batch(b_planes, b_legal, b_policy, b_q)
        static_planes.copy_(b_planes)
        static_legal.copy_(b_legal)
        static_policy.copy_(b_policy)
        static_wdl.copy_(b_wdl)
        static_q.copy_(b_q)
        static_root.copy_(b_root)

    graph = None
    side = torch.cuda.Stream() if use_graph else None
    started = time.time()
    # DISTILL_PROFILE_STEPS=N profiles steps 11..10+N and prints the kernel
    # table, so a slow learner can be read rather than guessed at.
    profile_steps = int(os.environ.get("DISTILL_PROFILE_STEPS", "0"))
    profiler = None
    for step in range(1, steps + 1):
        if profile_steps and step == 11:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
        if profiler is not None and step == 11 + profile_steps:
            profiler.__exit__(None, None, None)
            print("profile:" + profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30),
                  flush=True)
            profiler = None
        set_lr(step)
        load_batch(step)
        if use_graph and step <= 3:
            # Warm-up on a side stream, as graph capture requires; real steps.
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                optimizer.zero_grad(set_to_none=True)
                train_step()
            torch.cuda.current_stream().wait_stream(side)
        elif use_graph and graph is None:
            try:
                optimizer.zero_grad(set_to_none=True)
                candidate = torch.cuda.CUDAGraph()
                with torch.cuda.graph(candidate):
                    train_step()
                graph = candidate
                graph.replay()                    # capture ran nothing; this step's data still trains
                print("training step captured as a CUDA graph", flush=True)
            except Exception as exc:
                print(f"CUDA graph capture failed, training eagerly: {type(exc).__name__}: {exc}",
                      flush=True)
                use_graph = False
                optimizer.zero_grad(set_to_none=True)
                train_step()
        elif use_graph:
            graph.replay()
        else:
            optimizer.zero_grad(set_to_none=True)
            train_step()

        if step % 500 == 0 or step == steps:
            window = 500 if step % 500 == 0 else step % 500
            mean = (totals / window).tolist()
            totals.zero_()
            print(f"step {step}/{steps} loss={mean[0]:.4f} (policy {mean[1]:.4f}, value {mean[2]:.4f}, "
                  f"q {mean[3]:.4f}, root {mean[5]:.4f}, H {mean[4]:.3f}) "
                  f"{(time.time() - started):.0f}s", flush=True)

    # Save before evaluating: the checkpoint must never depend on the
    # evaluation surviving a crowded GPU.
    save_checkpoint({"model": net.state_dict(), "steps": steps,
                     "arch": (net.channels, net.blocks, net.head_channels),
                     "data_split_version": SPLIT_VERSION,
                     "holdout_configs": os.environ.get("DISTILL_HOLDOUT_CONFIGS", "")}, out_dir / "distilled.pt")
    if os.environ.get("DISTILL_PERSIST_OPTIMIZER", "1") != "0":
        save_checkpoint({"optimizer": optimizer.state_dict(), "format": 1}, out_dir / "optimizer.pt")
    print(f"saved {out_dir / 'distilled.pt'}", flush=True)

    net.eval()
    # The held-out data arrives in chunks (the loader filters it by
    # position); pool every chunk of a board before printing, so a board is
    # one line and the step lines above survive the driver's log window.
    pooled = {}
    with torch.no_grad():
        for shard in held:
            value_hits = policy_hits = q_hits = 0
            optimal = shard["policy"] > 0
            for start in range(0, len(shard["planes"]), 4096):
                h_planes = decode_planes(shard["planes"][start:start + 4096]).to(device).float()
                h_legal = shard["legal"][start:start + 4096].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits, values, q_logits = net(h_planes, h_legal)
                logits, values, q_logits = logits.float(), values.float(), q_logits.float()
                value_hits += (values.argmax(dim=1).cpu() == shard["wdl"][start:start + 4096].long()).sum().item()
                policy_pick = logits.argmax(dim=1).cpu()
                q_pick = q_choice(q_logits, h_legal).cpu()
                chunk_optimal = optimal[start:start + 4096]
                policy_hits += chunk_optimal.gather(1, policy_pick.unsqueeze(1)).sum().item()
                q_hits += chunk_optimal.gather(1, q_pick.unsqueeze(1)).sum().item()
            rows, columns, connect = shard["config"]
            chaos = bool(shard["planes"][0, 4].flatten()[0] > 0) if len(shard["planes"]) else False
            key = (rows, columns, connect, chaos)
            totals = pooled.setdefault(key, [0, 0, 0, 0])
            totals[0] += value_hits
            totals[1] += policy_hits
            totals[2] += q_hits
            totals[3] += len(shard["planes"])
    for (rows, columns, connect, chaos), (value_hits, policy_hits, q_hits, count) in sorted(pooled.items()):
        print(f"[held {rows}x{columns} c{connect} {'chaos' if chaos else 'classic'}] "
              f"value accuracy {value_hits / count:.4f}, "
              f"blunder rate policy {1.0 - policy_hits / count:.4f} / q {1.0 - q_hits / count:.4f} "
              f"({count} positions)", flush=True)


if __name__ == "__main__":
    main()
