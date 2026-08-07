# Vortex Documentation

Run documentation commands from the repository root through `docs/Makefile`.

## Validate the documentation

```bash
uv run --all-packages make -C docs check
```

This is the canonical validation flow. It creates a clean Sphinx build, builds the HTML docs, and
runs the doctests. `uv` installs the locked dependencies and builds the local `vortex-data` Python
package through Maturin, so a separate `maturin develop` step is not required.

## Live development

```bash
make -C docs serve
```

The `serve` target starts a live-reloading Sphinx server. Use focused targets such as `html` or
`doctest` while iterating; run `make -C docs help` to list all targets, and finish with `check`.
