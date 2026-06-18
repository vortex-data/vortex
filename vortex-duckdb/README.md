# Vortex DuckDB

Rust bindings for DuckDB. Supports DuckDB precompiled libraries for fast builds and from source builds for debugging.

## Prerequisites

- **Ninja**: `brew install ninja` (macOS) | `apt-get install ninja-build` (Ubuntu)
- **CMake**: `brew install cmake` (macOS) | `apt-get install cmake` (Ubuntu)
- **C++20 compatible compiler**: GCC or Clang

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

### AddressSanitizer & ThreadSanitizer

Enable both ASAN & TSAN: `VX_DUCKDB_SAN=1`.

```bash
VX_DUCKDB_DEBUG=1 VX_DUCKDB_SAN=1 cargo build -p vortex-duckdb
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

# Link against the DuckDB debug build from source with ASAN & TSAN.
ASAN_OPTIONS=detect_container_overflow=0 VX_DUCKDB_DEBUG=1 VX_DUCKDB_SAN=1 cargo test -p vortex-duckdb
```

## Testing the extension with DuckDB

By default, our tests use a precompiled build which means you don't get an
.extension which you can load in DuckDB. If you want to test a full setup,

1. Clone [duckdb-vortex](https://github.com/vortex-data/duckdb-vortex)
   repository.

2. If there is an api difference between duckdb-vortex's duckdb submodule and
   vortex's vortex-duckdb/duckdb submodule, checkout duckdb-vortex to previous
   commit. For example, if duckdb-vortex's HEAD uses 1.6 API but vortex's HEAD
   uses 1.5.2, checkout duckdb-vortex at 8a41ee6ebd9.

3. Update duckdb-vortex's submodules. Replace vortex/ submodule by a softlink to
   your local vortex repository.
4. Inside duckdb-vortex, run make -j.

./target/release/duckdb will be a duckdb instance with vortex-duckdb already
loaded.

## Testing a custom DuckDB tag

Change `DUCKDB_VERSION` environment variable value to a preferred hash or commit
(local build), or change build.rs (for testing in CI). If you use a commit,
DuckDB needs to link httpfs statically so you also need to install CURL
development headers (e.g. `libcurl4-openssl-dev`).
