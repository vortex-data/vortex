# vortex-row

`vortex-row` provides an experimental row-oriented byte encoder for Vortex arrays. It is
**not ready for production use**. It produces byte strings that can be compared
lexicographically to sort rows according to the configured column ordering.

Only supported Vortex logical types are accepted. Extension types are rejected until their
logical sort semantics are defined.

## Experimental Format

The row encoding byte layout is experimental and not ready for production. Its exact bytes,
supported type set, and edge-case semantics may change between Vortex releases.

Do not persist row-encoded bytes or use them as a stable interchange format. They are intended
for internal sort-key and row-key operations where the encoder version, schema, and sort
options are controlled together.
