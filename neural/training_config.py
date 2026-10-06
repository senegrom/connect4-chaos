"""Settings the Modal app and the loop driver share, and the CPU-only
validation run before allocating remote work."""
import math
import re

DEFAULT_SIMS = 128
# The loop's arena: the boards (every one, so a model cannot look stronger by
# trading one shape against another), games per board, simulations per move,
# and the seed that alone decides its openings. The manual `--task arena` and
# the arena function take the same unless told otherwise, so they replay what
# the loop played for the same two checkpoints; `python -m neural.arena`
# defaults to them too.
ARENA_SHAPES = "all"
ARENA_GAMES = 6
ARENA_SIMS = 32
ARENA_SEED = 7
# How long one call of each Modal function the driver polls may run, in
# hours, by the driver's name for its role: the timeouts of selfplay_gpu,
# learn and arena in modal_app.py. The driver gives up on a call only after
# two attempts at it (CEILING_SECONDS in neural/modal_loop.py), so both read
# these: a timeout raised in modal_app.py alone would have the driver cancel
# calls still within it.
FUNCTION_TIMEOUT_HOURS = {"actor": 2, "learner": 3, "arena": 2}


def parse_shape_spec(spec):
    if spec.strip() == "all":
        return None
    shapes = []
    for item in spec.split(","):
        match = re.fullmatch(r"(10|[1-9])x(10|[1-9])c(10|[1-9])(chaos|classic)", item.strip())
        if not match:
            raise ValueError(f"Invalid self-play shape: {item!r}")
        rows, cols, connect = map(int, match.groups()[:3])
        if connect > max(rows, cols):
            raise ValueError("Connect length does not fit the board")
        shapes.append((rows, cols, connect, match[4] == "chaos"))
    return shapes


def validate_selfplay(games, sims, shapes="all", target_sims=0, target_share=0.25,
                      random_share=0.5, random_plies=4):
    for name, value, minimum in (("games", games, 1), ("sims", sims, 1),
                                 ("target_sims", target_sims, 0), ("random_plies", random_plies, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("target_share", target_share), ("random_share", random_share)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    parse_shape_spec(shapes)
