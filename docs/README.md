# Vortex Documentation

## Building

First, you must compile the vortex-python Rust code into a native library because the Python package
inherits some of its doc strings from Rust docstrings:

```
cd ../vortex-python && uv run maturin develop
```

The docs also require the [`doxygen`](https://www.doxygen.nl/) tool, which can be installed with:

```
brew install doxygen
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
