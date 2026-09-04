"""Builds rank sidecars for C4PAIR2 block bitsets.

For every pair-<k>-<j>.bits in the given directory, writes
pair-<k>-<j>.ranks: raw little-endian u64s, entry g holding the number of
set bits before word g*2048 (16 KB of bitset per entry). With the sidecar,
a remote reader resolves any slot's rank from one bounded range read
instead of scanning the file - see scripts/perfect-chaos-remote-lookup.mjs.

Usage: python build-pair-rank-sidecars.py <directory> [<directory> ...]
"""
import struct
import sys
from pathlib import Path

GROUP_WORDS = 2048
HEADER = struct.Struct("<8s4BHHQ")


def build(bits_path: Path) -> bool:
    """Builds the sidecar unless it is already current; True when it wrote one."""
    ranks_path = bits_path.with_suffix(".ranks")
    if ranks_path.exists() and ranks_path.stat().st_mtime >= bits_path.stat().st_mtime:
        return False
    temporary = ranks_path.with_suffix(".ranks.tmp")
    with bits_path.open("rb") as bits, temporary.open("wb") as ranks:
        header = bits.read(HEADER.size)
        magic, _rows, _columns, _connect, kind, _layer, _pair, words = HEADER.unpack(header)
        if magic != b"C4PAIR2\x00" or kind not in (0, 2):   # 0 chaos, 2 classic bits
            raise SystemExit(f"{bits_path} is not a C4PAIR2 bits file")
        running = 0
        remaining = words
        while remaining > 0:
            take = min(GROUP_WORDS, remaining)
            ranks.write(struct.pack("<Q", running))
            chunk = bits.read(take * 8)
            if len(chunk) != take * 8:
                raise SystemExit(f"{bits_path} is truncated")
            running += int.from_bytes(chunk, "little").bit_count()
            remaining -= take
    temporary.replace(ranks_path)
    return True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    total = 0
    for directory in sys.argv[1:]:
        for bits_path in sorted(Path(directory).glob("pair-*.bits")):
            if build(bits_path):
                total += 1
    print(f"sidecars built: {total}")


if __name__ == "__main__":
    main()
