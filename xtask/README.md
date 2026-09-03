# xtask - swiss army knife builder

This crate is not published and is only used by developers.

It automates a number of tasks that a project maintainer might need to do.

You can run `cargo xtask -h` to get a list of supported commands.

## Current commands

### `generate-editions`

Regenerates the edition records under `vortex/editions`.

### `check-editions`

Checks that frozen edition records never change, comparing against `--base` (default
`origin/develop`).
