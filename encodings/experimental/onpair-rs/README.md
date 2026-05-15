# onpair-lib

Pure-Rust port of the training + encoding parts of
[`onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp).

Scope is limited to what `vortex-onpair` actually consumes from
`vortex-onpair-sys`: `Column::compress` (BPE-style dictionary training plus
LSB-first bit-packed token encoding) and raw access to the resulting parts
(dictionary bytes/offsets, packed token stream, per-row boundaries). Decode,
LIKE, and EQ predicates are already pure Rust in `vortex-onpair` and reuse the
same `parts()` layout.
