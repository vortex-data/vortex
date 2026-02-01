# Encodings

:::{warning}
This page is under construction.
:::

Planned content:

- What is an encoding and how it differs from a dtype
- Canonical encodings (Arrow-compatible): Null, Bool, Primitive, VarBin, VarBinView, Struct, List, Extension
- Utility encodings: Constant, Chunked, ByteBool
- Compressed encodings: ALP, FastLanes, FSST, RunEnd, Sparse, ZigZag, PCO, Zstd, BtrBlocks
- Cascading compression: how encodings compose into trees
- The vtable system and how encodings register compute kernels
- Writing your own encoding (link to extending guide)
