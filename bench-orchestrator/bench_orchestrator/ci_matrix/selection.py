# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Parse a selection of benchmarks, and optionally their queries, from one CI input string."""

import re

_QUERIES = re.compile(r"q?(\d+)(?:-q?(\d+))?", re.IGNORECASE)


def _parse_queries(token: str, benchmark_id: str) -> list[int]:
    match = _QUERIES.fullmatch(token)
    if match is None:
        raise ValueError(f"Invalid query {token!r} for benchmark {benchmark_id!r}; expected e.g. q3 or q1-4")
    start = int(match[1])
    end = int(match[2]) if match[2] is not None else start
    if start < 1 or end < start:
        raise ValueError(f"Invalid query range {token!r} for benchmark {benchmark_id!r}")
    return list(range(start, end + 1))


def parse_selection(value: str | None) -> dict[str, tuple[int, ...]]:
    """Map each selected benchmark ID to the queries to run, where an empty tuple means all of them.

    Benchmarks are separated by whitespace, `;` or `,`. A benchmark may be followed by `:` and a
    comma-separated list of queries or ranges, each with an optional `q`: `tpch-nvme:q1,2`,
    `tpch-nvme:q1-4` or `polarsignals tpch-nvme:3,5-7`. A comma-separated token that is a query
    adds to the previous benchmark's queries, because benchmark IDs never look like queries.
    """
    selection: dict[str, set[int] | None] = {}
    # The queries of the previous benchmark when it named some, which a bare query token extends.
    current: tuple[str, set[int]] | None = None
    tokens: list[str] = [token for token in re.split(r"[\s;,]+", value or "") if token]
    for token in tokens:
        benchmark_id, colon, queries = token.partition(":")
        if not colon and current is not None and _QUERIES.fullmatch(token):
            current[1].update(_parse_queries(token, current[0]))
            continue
        if not benchmark_id or (colon and not queries):
            raise ValueError(f"Invalid benchmark selection {token!r}; expected e.g. tpch-nvme or tpch-nvme:q1-4")
        new = set(_parse_queries(queries, benchmark_id)) if colon else None
        if benchmark_id in selection:
            # Selecting a benchmark twice runs the union of its queries; all queries wins.
            previous = selection[benchmark_id]
            new = None if previous is None or new is None else previous | new
        selection[benchmark_id] = new
        current = None if new is None else (benchmark_id, new)
    return {benchmark_id: tuple(sorted(queries or ())) for benchmark_id, queries in selection.items()}
