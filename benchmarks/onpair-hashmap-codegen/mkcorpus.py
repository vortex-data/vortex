#!/usr/bin/env python3
"""Generate a synthetic ONPAIR01 corpus.

The real benchmark reads corpora produced from FineWeb, ClickBench and friends.
This generator makes a stand-in with a comparable shape: rows of English-like
text drawn from a Zipf-distributed vocabulary, so OnPair training produces a
dictionary with both short (<= 8 byte) and long tokens.

usage: mkcorpus.py OUTPUT [PAYLOAD_BYTES] [SEED]
"""

import random
import struct
import sys

MAGIC = b"ONPAIR01"

SYLLABLES = [
    "ing", "tion", "er", "the", "and", "re", "ent", "com", "pro", "con",
    "ate", "al", "an", "in", "on", "st", "ar", "or", "it", "is",
]


def vocabulary(rng, size):
    words = []
    for _ in range(size):
        parts = rng.randint(1, 4)
        words.append("".join(rng.choice(SYLLABLES) for _ in range(parts)))
    return words


def zipf_weights(n, skew=1.1):
    return [1.0 / ((i + 1) ** skew) for i in range(n)]


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    output = sys.argv[1]
    payload_target = int(sys.argv[2]) if len(sys.argv) > 2 else 4 * 1024 * 1024
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    rng = random.Random(seed)
    words = vocabulary(rng, 20000)
    weights = zipf_weights(len(words))

    rows = []
    payload = 0
    while payload < payload_target:
        count = rng.randint(8, 60)
        row = " ".join(rng.choices(words, weights=weights, k=count)).encode()
        rows.append(row)
        payload += len(row)

    with open(output, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<QQ", payload, len(rows)))
        for row in rows:
            handle.write(struct.pack("<I", len(row)))
            handle.write(row)

    print(f"corpus,rows={len(rows)},payload={payload}")


if __name__ == "__main__":
    main()
