# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Generate a deterministic Omnigraph JSONL load file for the `schema.pg` social graph.

The schema mirrors Omnigraph's own `e2e_query_shapes` benchmark fixture
(`Person { name @key, age, team, bio, embedding: Vector(32) }`, `Knows: Person -> Person`),
scaled up so that file size and scan runtime are measurable.
"""

import argparse
import json
import random

WORDS = (
    "data graph agent memory vector search query branch merge commit schema table column "
    "storage engine index retrieval context claim source evidence research team project "
    "rust python cloud object lake format compression encoding scan filter join traverse "
    "latency throughput cache shard replica cluster policy review deploy release model"
).split()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", type=int, default=200_000)
    parser.add_argument("--avg-degree", type=int, default=10)
    parser.add_argument("--teams", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n = args.people
    with open(args.out, "w") as f:
        for i in range(n):
            bio = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 24)))
            emb = [round(rng.gauss(0.0, 1.0), 6) for _ in range(32)]
            row = {
                "name": f"p{i}",
                "age": rng.randint(18, 99),
                "team": f"team-{rng.randrange(args.teams):03d}",
                "bio": bio,
                "embedding": emb,
            }
            f.write(json.dumps({"type": "Person", "data": row}, separators=(",", ":")) + "\n")
        # Skewed out-degree: most people know a few others, some know many.
        for i in range(n):
            degree = max(1, min(int(rng.expovariate(1.0 / args.avg_degree)), 20 * args.avg_degree))
            for _ in range(degree):
                j = rng.randrange(n)
                if j != i:
                    f.write(
                        json.dumps({"edge": "Knows", "from": f"p{i}", "to": f"p{j}"}, separators=(",", ":"))
                        + "\n"
                    )


if __name__ == "__main__":
    main()
