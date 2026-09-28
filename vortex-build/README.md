# vortex-build

Build script helpers used by the Vortex crates that own Protocol Buffers (`.proto`) schema
definitions. FlatBuffers (`.fbs`) bindings are checked in instead and regenerated with
`cargo run -p xtask -- generate-flatbuffers`.

Each schema lives in the crate that owns the types it describes, and is compiled into `OUT_DIR` by
that crate's `build.rs`. Schemas that `include`/`import` schemas from another crate declare that
crate explicitly:

```rust,ignore
fn main() {
    vortex_build::proto()
        .depends_on("vortex-array")
        .compile(&["scalar.proto"]);
}
```

`depends_on` resolves the dependency's schema directory through Cargo's `links` metadata
(`DEP_<LINKS>_PROTO`), so it works identically for path dependencies in
the workspace and for packages unpacked from a registry. The exporting crate only needs a `links`
key in its manifest; `vortex-build` emits the metadata automatically.

## Requirements

`.proto` compilation is pure Rust (via [`protox`](https://docs.rs/protox)) and needs no external
tooling.
