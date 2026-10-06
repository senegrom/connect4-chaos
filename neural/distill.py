"""The learner: trains the policy/value/Q net on exact shards and self-play replay.

Each batch mixes exact rows with the newest replay window
(DISTILL_REPLAY_FRACTION of it replay). Losses: cross-entropy of the
masked policy against its target - the exactly-optimal actions, or the
search's policy on a self-play row that recorded one - cross-entropy of
the W/D/L head against the exact value or the game's outcome, and
cross-entropy of the per-action Q head against the exact value of every
legal action (exact rows only). Optional terms: the squared error between
the value head's expectation and the search's root value on self-play rows
(DISTILL_ROOT_VALUE_WEIGHT), and a bonus on the policy's entropy
(DISTILL_ENTROPY_BONUS). Reports, per held-out board, pooled over its
shards, value accuracy and two blunder rates: the policy argmax's and the
Q-argmax's (choosing the action whose predicted outcome distribution has
the best expectation).

Usage: python -m neural.distill <shard_dir>[;<shard_dir>...] <out_dir> [steps] [batch]
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

from .model import PolicyValueNet, checkpoint_arch
from .training_provenance import training_provenance
from .optimizer_recovery import require_finite_model, restore_optimizer
from .training_config import parse_shape_spec
from .data_split import SPLIT_CHUNK, SPLIT_VERSION, select_samples

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


def require_current_format(shard, where):
    """The one shard format there is: uint8 planes scaled by 10, a split of
    this SPLIT_VERSION (declared per exact shard, a row mask in self-play),
    and Q targets or self-play's Q default. Every shard of an older format
    went with the Volume's 2026-09 wipes; one that turns up is rebuilt."""
    planes = shard.get("planes")
    selfplay = shard.get("source") == "selfplay"
    problem = ("planes that are not uint8 scaled by 10"
               if planes is None or planes.dtype != torch.uint8 or int(shard.get("planes_scale", 10)) != 10
               else "no split of this version" if shard.get("split_version") != SPLIT_VERSION
               else "no validation mask" if selfplay and "validation" not in shard
               else "no Q targets" if "q" not in shard and not (selfplay and shard.get("q_default") == 3)
               else None)
    if problem:
        raise ValueError(f"{where} predates the current shard format ({problem}); rebuild it")


def decode_planes(planes):
    """Shard planes are uint8 scaled by 10; training reads float16. Planes in
    any other encoding are refused: divided by 10 they read as empty boards."""
    if planes.dtype != torch.uint8:
        raise ValueError("shard planes are not uint8 scaled by 10; rebuild the shard")
    return planes.half() / 10


def without_heldout_positions(shard, holdout_shapes):
    """Filter by encoded rules, not filenames or a game's starting orientation."""
    planes = shard["planes"]
    rows = (planes[:, 2, :, 0] > 0).sum(dim=1)
    cols = (planes[:, 2, 0, :] > 0).sum(dim=1)
    connects = planes[:, 3, 0, 0].long()          # plane 3 holds connect / 10
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


def filtered_chunks(shard, holdout_shapes, *, validation=False, limit=None, seed=None,
                    trusted_partition=False):
    """Yield bounded, aligned chunks of the rows this split may use.

    Replay rows carry their position partition; an exact shard declares its
    split as a whole, and its readers pass `trusted_partition`. Rows of the
    boards in holdout_shapes are dropped from every training read, whatever
    their partition; validation reads keep them.
    A `limit` below the eligible count keeps the first `limit` rows or, given
    a `seed`, a seeded uniform subset of them. Replay takes the subset: a
    self-play shard stores its rows ply by ply, so its tail - what the replay
    window used to keep of its oldest, partly used shard - is the late game
    alone.
    """
    chunks = _eligible_chunks(shard, holdout_shapes, validation=validation,
                              trusted_partition=trusted_partition)
    if limit is None:
        yield from chunks
        return
    if seed is None:
        remaining = limit
        for chunk in chunks:
            if remaining <= 0:
                break
            if len(chunk["planes"]) > remaining:
                chunk = select_samples(chunk, slice(0, remaining))
            remaining -= len(chunk["planes"])
            yield chunk
        return
    chunks = list(chunks)
    total = sum(len(chunk["planes"]) for chunk in chunks)
    if total <= limit:
        yield from chunks
        return
    if limit <= 0:
        return
    keep = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:limit].sort().values
    offset = 0
    for chunk in chunks:
        size = len(chunk["planes"])
        rows = keep[(keep >= offset) & (keep < offset + size)] - offset
        offset += size
        if len(rows):
            yield select_samples(chunk, rows)


def _eligible_chunks(shard, holdout_shapes, *, validation, trusted_partition):
    """Every row of the shard this split may use, in bounded aligned chunks,
    filtered one chunk at a time as the caller asks for them."""
    count = len(shard["planes"])
    required = ("legal", "policy", "wdl")
    if any(len(shard[key]) != count for key in required):
        raise ValueError("Misaligned training tensors in shard")
    if "q" in shard and len(shard["q"]) != count:
        raise ValueError("Misaligned Q targets in shard")
    require_current_format(shard, "shard")
    for start in range(0, count, SPLIT_CHUNK):
        chunk = select_samples(shard, slice(start, min(start + SPLIT_CHUNK, count)))
        if not validation and holdout_shapes:
            chunk = without_heldout_positions(chunk, holdout_shapes)
            if chunk is None:
                continue
        if not trusted_partition:
            # Self-play's rows carry their split; an exact shard declares its
            # own, and its readers trust that (trusted_partition).
            if "validation" not in chunk:
                raise ValueError("An exact shard's split is its own: read it with trusted_partition")
            reserved = chunk["validation"].bool()
            keep = reserved if validation else ~reserved
            if not bool(keep.any()):
                continue
            if not bool(keep.all()):
                chunk = select_samples(chunk, keep)
        yield chunk


def training_holdouts(spec=None):
    """One holdout parser for both replay staging and the training loader."""
    if spec is None:
        spec = os.environ.get("DISTILL_HOLDOUT_CONFIGS", "")
    holdout = {tag.strip() for tag in spec.split(",") if tag.strip()}
    if "all" in holdout:
        raise ValueError("A training holdout must name specific configurations")
    shapes = [shape for tag in sorted(holdout) for shape in (parse_shape_spec(tag) or [])]
    return holdout, shapes


def load_shards(shard_dirs, seed=0):
    """Load exact validation shards and position-disjoint exact/replay training.

    Shard 0000 holds a board's validation positions and the others its
    training ones, split by the stable position hash when the corpus was
    built; replay rows carry the same partition. Directories may be
    separated by ';'.
    `seed` picks the rows of the one replay shard the window cuts through.
    """
    holdout, holdout_shapes = training_holdouts()
    window = int(os.environ.get("DISTILL_REPLAY_WINDOW", "4000000"))
    if window < 0:
        raise ValueError("DISTILL_REPLAY_WINDOW must be non-negative")
    train, held, replay_shards = [], [], []
    for shard_dir in str(shard_dirs).split(";"):
        # A glob of a missing directory is empty, not an error: a misnamed
        # exact corpus used to train on replay alone with the Q loss at zero.
        if not shard_dir.strip():
            raise ValueError(f"empty shard directory in {shard_dirs!r}")
        if not Path(shard_dir).is_dir():
            raise FileNotFoundError(f"shard directory {shard_dir} does not exist")
        for path in sorted(Path(shard_dir).glob("*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            require_current_format(shard, path)
            shard["mtime"] = path.stat().st_mtime
            if shard.get("source") == "selfplay":
                replay_shards.append(shard)
                continue
            tag = path.stem.rsplit("-", 1)[0]
            whole_board_held = tag in holdout
            declared = shard.get("split")
            if path.stem.endswith("0000"):
                if declared != "validation":
                    raise ValueError(f"{path} declares {declared!r}, expected validation")
                held.extend(filtered_chunks(shard, holdout_shapes, validation=True, trusted_partition=True))
            elif not whole_board_held:
                if declared != "train":
                    raise ValueError(f"{path} declares {declared!r}, expected train")
                train.extend(filtered_chunks(shard, holdout_shapes, trusted_partition=True))
    # Visit newest replay first and stop at the position budget: no whole-shard
    # overshoot or full-archive copies. The one shard the budget cuts through
    # is filtered in full and contributes a seeded sample of its rows.
    replay_shards.sort(key=lambda s: s["mtime"], reverse=True)
    total = 0
    for shard in replay_shards:
        if total >= window:
            break
        for chunk in filtered_chunks(shard, holdout_shapes, limit=window - total, seed=seed):
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


def _tensor_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


# What stays free on the GPU for the network, its activations and the
# optimizer when the dataset moves there. The learner's environment is built
# by modal_app.learn, so the variables that used to set this, force the
# choice or turn off pinning, fused AdamW, channels-last and the step graph
# could never be set.
GPU_RESERVE_BYTES = 24 * 1024 ** 3


def stage_training_tensors(planes, legal, policy, wdl, q, replay_idx, exact_idx, device):
    """One host-to-device copy for the whole compact dataset when it fits."""
    tensors = (planes, legal, policy, wdl, q, replay_idx, exact_idx)
    if device != "cuda":
        return tensors, False
    need = _tensor_bytes(*tensors)
    free, _total = torch.cuda.mem_get_info()
    if need + GPU_RESERVE_BYTES < free:
        print(f"training data: {need / 1e9:.2f} GB resident on GPU", flush=True)
        return tuple(t.to(device) for t in tensors), True
    # Not pinned: each batch is gathered by advanced indexing, which returns
    # a new, pageable tensor, so pinning cost up to 8 GB of host memory for
    # a transfer that was never asynchronous.
    print(f"training data: {need / 1e9:.2f} GB in host memory", flush=True)
    return tensors, False


def sampler_seed(environ=os.environ):
    """(seed, source) for this run's row sampling.

    DISTILL_SEED when the caller names one - the Modal learner passes its
    generation, so a generation is reproducible and the next one draws other
    rows - and fresh entropy otherwise. It used to be the constant 20260901:
    with the exact corpus unchanged, every generation drew the same exact
    rows in the same order, and the Q head, which only exact rows supervise,
    refit the same subset every time.
    """
    value = environ.get("DISTILL_SEED", "").strip()
    if value:
        seed = int(value)
        if not 0 <= seed < 2 ** 63:
            raise ValueError("DISTILL_SEED must be an integer in [0, 2**63)")
        return seed, "DISTILL_SEED"
    return int.from_bytes(os.urandom(8), "little") >> 1, "os.urandom"


def warmup_length(steps, *, warm_start, resumed, environ=os.environ):
    """Steps of linear learning-rate warm-up for this run, and why.

    A warm start with no optimizer moments to resume - the first generation
    after an ONNX import (neural/import_onnx.py), DISTILL_RESET_OPTIMIZER, or
    a sidecar that failed validation - would take full-size Adam steps from
    the first batch, before the moment estimates mean anything. Such a run
    ramps up over a fifth of its steps, at most 1000; any other run does not.
    DISTILL_WARMUP_STEPS overrides both (0 turns it off), whether or not the
    moments were restored, so the reason names it then.
    """
    value = environ.get("DISTILL_WARMUP_STEPS", "").strip()
    if value:
        warmup = int(value)
        if warmup < 0:
            raise ValueError("DISTILL_WARMUP_STEPS must not be negative")
        return min(warmup, steps), "set by DISTILL_WARMUP_STEPS"
    if warm_start and not resumed:
        return min(1000, steps // 5), "no optimizer moments to resume"
    return 0, None


def held_board(chunk):
    """The board a held-out chunk measures: its shard's (rows, columns,
    connect), which leaves out the rule set, and whether it is Chaos."""
    rows, columns, connect = chunk["config"]
    chaos = bool(chunk["planes"][0, 4].flatten()[0] > 0) if len(chunk["planes"]) else False
    return rows, columns, connect, chaos


def draw_rows(generator, replay_idx, exact_idx, n_replay, n_exact, device):
    """One batch's corpus rows: n_replay drawn from replay, then n_exact from
    the exact tables, uniformly with replacement."""
    return torch.cat([
        replay_idx[torch.randint(0, max(1, len(replay_idx)), (n_replay,),
                                 generator=generator, device=device)]
        if n_replay else torch.empty(0, dtype=torch.int64, device=device),
        exact_idx[torch.randint(0, max(1, len(exact_idx)), (n_exact,),
                                generator=generator, device=device)]
        if n_exact else torch.empty(0, dtype=torch.int64, device=device),
    ])


def create_optimizer(net, lr, device, capturable=False):
    """Fused AdamW on CUDA, with a portable eager fallback. `capturable`
    keeps the step counters and the learning rate on the device, which a
    CUDA graph of the training step needs."""
    kwargs = dict(lr=torch.tensor(float(lr), device=device) if capturable else lr, weight_decay=1e-4)
    if device == "cuda":
        kwargs["fused"] = True
    if capturable:
        kwargs["capturable"] = True
    try:
        return torch.optim.AdamW(net.parameters(), **kwargs)
    except (TypeError, RuntimeError, ValueError):
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)


def save_checkpoint(payload, path):
    """Expose a checkpoint only after serialization finishes successfully."""
    if "model" in payload:
        require_finite_model(payload["model"])
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
    # The Modal wrapper reports this line as the learner's GPU.
    print(f"gpu: {torch.cuda.get_device_name() if device == 'cuda' else 'cpu'}", flush=True)
    seed, seed_source = sampler_seed()
    print(f"sampler seed {seed} ({seed_source})", flush=True)
    train, held = load_shards(shard_dir, seed=seed)
    # Exact rows are the only supervision the Q head gets. Without them the
    # run still trains, logs and publishes normally with a Q loss of zero,
    # so it takes an explicit opt-in.
    if (not any(shard.get("source") != "selfplay" for shard in train)
            and os.environ.get("DISTILL_ALLOW_NO_EXACT", "") != "1"):
        raise ValueError(f"no exact-table training rows in {shard_dir}; point the learner at the "
                         "exact corpus, or set DISTILL_ALLOW_NO_EXACT=1 to train on replay alone")
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
        planes[cursor:cursor + count] = shard["planes"]
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
    # The loader cuts each held-out shard into chunks of SPLIT_CHUNK rows, so
    # counting the chunks said nothing: a board's 25,000 rows are seven.
    held_boards = {held_board(chunk) for chunk in held}
    print(f"train samples: {len(planes)} (exact {len(exact_idx)}, replay {len(replay_idx)}, "
          f"replay fraction {replay_fraction:.2f}), held-out boards: {len(held_boards)} "
          f"({sum(len(chunk['planes']) for chunk in held)} positions), device: {device}")

    init = os.environ.get("DISTILL_INIT")
    payload = torch.load(init, map_location=device, weights_only=True) if init else None
    provenance = training_provenance(payload, os.environ.get("DISTILL_HOLDOUT_CONFIGS", ""))
    print(f"validation provenance: {provenance['training_provenance']['status']}; "
          f"lifetime holdouts: {provenance['holdout_configs'] or '(none)'}. "
          "This run's exclusions alone do not certify the parent weights.", flush=True)
    if payload is not None:
        require_finite_model(payload["model"])
        arch = checkpoint_arch(payload)
        net = PolicyValueNet(*arch).to(device)
        net.load_state_dict(payload["model"])
        print(f"warm start from {init} arch={arch}")
    else:
        net = PolicyValueNet().to(device)
    if device == "cuda":
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
    use_amp = device == "cuda"
    # The whole training step - forward, loss, backward, AdamW - replays as
    # one CUDA graph. Profiled eagerly, a step was about 30 ms of GPU work
    # and about as much CPU launch work, serialised by per-step host reads;
    # the graph leaves only the GPU work. A capture failure runs the same
    # step eagerly.
    use_graph = device == "cuda"
    init_opt = os.environ.get("DISTILL_INIT_OPT")
    if os.environ.get("DISTILL_RESET_OPTIMIZER", "") == "1":
        init_opt = None
    optimizer = restore_optimizer(
        lambda: create_optimizer(net, lr, device, capturable=use_graph), init_opt)
    capturable = bool(optimizer.defaults.get("capturable", False))
    use_graph = use_graph and capturable
    lr_value = torch.tensor(lr, device=device) if capturable else lr
    for group in optimizer.param_groups:
        group["lr"] = lr_value
    warmup, reason = warmup_length(steps, warm_start=payload is not None, resumed=bool(optimizer.state))
    if warmup:
        print(f"learning-rate warm-up over {warmup} steps: {reason}", flush=True)

    def set_lr(step):
        # CosineAnnealingLR(T_max=steps) in closed form; `step` is 1-based and
        # the value is what that scheduler had set before this step. A linear
        # warm-up scales the first `warmup` steps.
        value = 0.5 * lr * (1.0 + math.cos(math.pi * (step - 1) / steps))
        if step <= warmup:
            value *= step / warmup
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
    generator = torch.Generator(device=sample_device).manual_seed(seed)
    n_replay = int(round(batch * replay_fraction))
    n_exact = batch - n_replay

    # Static batch buffers: the graph reads these, the sampler fills them.
    static_planes = torch.zeros((batch, 7, 10, 10), device=device)
    if device == "cuda":
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
        picks = draw_rows(generator, replay_idx, exact_idx, n_replay, n_exact, sample_device)
        b_planes, b_legal = planes[picks], legal[picks]
        b_policy, b_wdl, b_q = policy[picks], wdl[picks], q[picks]
        b_root = root[picks]
        if not resident:
            b_planes, b_legal, b_policy = b_planes.to(device), b_legal.to(device), b_policy.to(device)
            b_wdl, b_q, b_root = b_wdl.to(device), b_q.to(device), b_root.to(device)
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
                     **provenance}, out_dir / "distilled.pt")
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
            totals = pooled.setdefault(held_board(shard), [0, 0, 0, 0])
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
