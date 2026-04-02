# Vortex FSST

A Vortex Encoding for Binary and Utf8 data that utilizes the [Fast Static Symbol Table](https://github.com/spiraldb/fsst)
compression algorithm.

## LIKE Pushdown

The FSST encoding has a specialized LIKE fast path for a narrow subset of
patterns:

- `prefix%`
- `%needle%`

Unsupported shapes, including `_`, `%suffix`, or patterns with interior
wildcards, fall back to ordinary decompression-based LIKE evaluation.

There are also two implementation limits on the pushdown path, both measured in
pattern bytes:

- `prefix%` supports up to 253 bytes.
- `%needle%` supports up to 254 bytes.

Patterns beyond those limits are still evaluated correctly, but they do so via
the fallback path instead of the DFA matcher.
