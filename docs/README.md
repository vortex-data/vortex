# Vortex Documentation

## Building

First, you must compile the pyvortex Rust code into a native library because the Python package
inherits some of its doc strings from Rust docstrings:

```
cd ../pyvortex && uv run maturin develop
```

Build just the Python docs:

```
uv run make html
```

Build the Python and Rust docs and place the rust docs at `_build/rust/html`:

```
uv run make full-html
```

## Viewing

After building:

```
open pyvortex/_build/html/index.html
```

## Python Doctests

```
uv run make doctest
```
