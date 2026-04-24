# xtask - swiss army knife builder

This crate is not published and is only used by developers.

It automates a number of tasks that a project maintainer might need to do, for example
code generation.

You can run `cargo xtask -h` to get a list of supported commands.

## Current commands

### `generate-fbs`

This will generate the `src/generated` Rust files in the `vortex-flatbuffers` crate. This
must be run every time changes are made to one of the .fbs files, or if any are added/deleted.

### `generate-proto`

This will generate the `src/generated` Rust files in the `vortex-proto` crate. This must
be run every time changes are made to one of the .fbs files, or if any are added/deleted.



