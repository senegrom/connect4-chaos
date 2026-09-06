"""CPU-only configuration validation; run before allocating remote work."""
import math
import re

DEFAULT_SIMS = 128


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


def validate_selfplay(games, sims, shapes="all", target_sims=0, target_share=0.25):
    for name, value, minimum in (("games", games, 1), ("sims", sims, 1),
                                 ("target_sims", target_sims, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not math.isfinite(target_share) or not 0 <= target_share <= 1:
        raise ValueError("target_share must be between 0 and 1")
    parse_shape_spec(shapes)
