# Vortex Documentation

## Building

First, you must compile the vortex-python Rust code into a native library because the Python package
inherits some of its doc strings from Rust docstrings:

```
cd ../vortex-python && uv run maturin develop
```

Build the Vortex docs:

```
uv run make html
```

## Development

Live-reloading (ish) build of the docs:

```
uv run make serve
```

## Python Doctests

```
uv run make doctest
```
