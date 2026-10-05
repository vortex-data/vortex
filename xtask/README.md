# xtask - swiss army knife builder

This crate is not published and is only used by developers.

It automates a number of tasks that a project maintainer might need to do.

You can run `cargo xtask -h` to get a list of supported commands.

## Current commands

### `generate-editions`

Regenerates the edition records under `vortex/editions`.

### `generate-flatbuffers`

Regenerates the checked-in FlatBuffers bindings under `<crate>/src/flatbuffers/generated/`.
Requires the pinned `flatc` release on `PATH`, or at the location in `FLATC`.

### `generate-proto`

Regenerates the checked-in Protocol Buffers bindings under `<crate>/src/proto/generated/`. Pure
Rust, so no `protoc` is needed.

### `check-editions`

Checks that frozen edition records never change, comparing against `--base` (default
`origin/develop`).
