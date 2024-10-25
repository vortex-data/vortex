# Vortex Documentation

## Building

```
rye run build-docs
```

or

```
rye run sphinx-build -M html . _build --fail-on-warning --keep-going
```

## Viewing

After building:

```
open pyvortex/_build/html/index.html
```
