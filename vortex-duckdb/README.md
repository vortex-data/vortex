# Vortex DuckDB

Rust bindings for DuckDB. Supports DuckDB precompiled libraries for fast builds and from source builds for debugging.

## Prerequisites

- **Ninja**: `brew install ninja` (macOS) | `apt-get install ninja-build` (Ubuntu)
- **CMake**: `brew install cmake` (macOS) | `apt-get install cmake` (Ubuntu)
- **C++17 compatible compiler**: GCC or Clang

## Build Modes

### Default (Release)

Link against the precompiled DuckDB release build.

```bash
cargo build -p vortex-duckdb
```

### Debug Build

Opt into DuckDB debug build: `VX_DUCKDB_DEBUG=1`.

```bash
VX_DUCKDB_DEBUG=1 cargo build -p vortex-duckdb
```

### AddressSanitizer

Enable ASAN: `VX_DUCKDB_ASAN=1`.

```bash
VX_DUCKDB_DEBUG=1 VX_DUCKDB_ASAN=1 cargo build -p vortex-duckdb
```

## Environment Variables

| Variable          | Effect                          |
| ----------------- | ------------------------------- |
| `VX_DUCKDB_DEBUG` | Build from source in debug mode |
| `VX_DUCKDB_ASAN`  | Enable AddressSanitizer         |

## Running Tests

```bash
# By default, link against the precompiled DuckDB release build.
cargo test -p vortex-duckdb

# Link against the DuckDB debug build from source.
VX_DUCKDB_DEBUG=1 cargo test -p vortex-duckdb

# Link against the DuckDB debug build from source with ASAN.
ASAN_OPTIONS=detect_container_overflow=0 VX_DUCKDB_DEBUG=1 VX_DUCKDB_ASAN=1 cargo test -p vortex-duckdb
```
