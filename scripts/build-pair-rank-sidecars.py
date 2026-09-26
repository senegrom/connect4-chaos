"""Builds rank sidecars for C4PAIR3 block bitsets.

For every pair-<k>-<j>.bits in the given directory, writes
pair-<k>-<j>.ranks: the magic C4RANK1, a verbatim copy of the bits file's
32-byte header, then raw little-endian u64s, entry g holding the number of
set bits before word g*2048 (16 KB of bitset per entry). With the sidecar,
a reader resolves any slot's rank from one bounded read instead of scanning
the file, as neural/pair_tables.py does.

The copied header names the block and its word count and carries the
CRC-32 of its bits, so it identifies exactly the bitset the ranks were
counted from: a sidecar is current when its copy equals the bits file's
header, whatever the files' timestamps say, and a reader refuses one that
does not match.

Usage: python build-pair-rank-sidecars.py <directory> [<directory> ...]
"""
import struct
import sys
import zlib
from pathlib import Path

GROUP_WORDS = 2048
# magic, rows, columns, connect, kind, layer, pair, words, version, crc32
HEADER = struct.Struct("<8s4BHHQII")
SIDECAR_MAGIC = b"C4RANK1\x00"


def build(bits_path: Path) -> bool:
    """Builds the sidecar unless it is already current; True when it wrote one."""
    ranks_path = bits_path.with_suffix(".ranks")
    with bits_path.open("rb") as bits:
        header = bits.read(HEADER.size)
    if len(header) != HEADER.size:
        raise SystemExit(f"{bits_path} is truncated")
    if ranks_path.exists():
        with ranks_path.open("rb") as existing:
            if existing.read(len(SIDECAR_MAGIC) + HEADER.size) == SIDECAR_MAGIC + header:
                return False
    temporary = ranks_path.with_suffix(".ranks.tmp")
    with bits_path.open("rb") as bits, temporary.open("wb") as ranks:
        bits.seek(HEADER.size)
        magic, _rows, _columns, _connect, kind, _layer, _pair, words, _version, crc = (
            HEADER.unpack(header)
        )
        if magic != b"C4PAIR3\x00" or kind not in (0, 2):   # 0 chaos, 2 classic bits
            raise SystemExit(f"{bits_path} is not a C4PAIR3 bits file")
        ranks.write(SIDECAR_MAGIC + header)
        running = 0
        remaining = words
        checksum = 0
        while remaining > 0:
            take = min(GROUP_WORDS, remaining)
            ranks.write(struct.pack("<Q", running))
            chunk = bits.read(take * 8)
            if len(chunk) != take * 8:
                raise SystemExit(f"{bits_path} is truncated")
            checksum = zlib.crc32(chunk, checksum)
            running += int.from_bytes(chunk, "little").bit_count()
            remaining -= take
        # The whole bitset passes through here anyway, so a damaged block is
        # refused before any rank derived from it is published.
        if checksum != crc:
            raise SystemExit(f"{bits_path} fails its checksum")
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
