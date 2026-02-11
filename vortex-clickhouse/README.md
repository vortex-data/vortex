# Vortex ClickHouse

ClickHouse format plugin for reading and writing [Vortex](https://github.com/spiraldb/vortex) files. Implemented as a Rust static library linked into ClickHouse via C++ FFI wrappers.

## Features

- Read Vortex files: `SELECT * FROM file('data.vortex', 'Vortex')`
- Write Vortex files: `INSERT ... TO 'output.vortex' FORMAT Vortex`
- Automatic schema inference
- Predicate & projection pushdown

## Prerequisites

- **Ninja**: `brew install ninja` (macOS) | `apt-get install ninja-build` (Ubuntu)
- **CMake 3.20+**: `brew install cmake` (macOS) | `apt-get install cmake` (Ubuntu)
- **Rust 1.89+**
- **C++17 compatible compiler**: GCC or Clang

## Build Modes

### Default (Release)

```bash
cargo build -p vortex-clickhouse
```

### Debug Build

Opt into ClickHouse debug build: `VX_CLICKHOUSE_DEBUG=1`.

```bash
VX_CLICKHOUSE_DEBUG=1 cargo build -p vortex-clickhouse
```

## Environment Variables

| Variable               | Effect                                                        |
| ---------------------- | ------------------------------------------------------------- |
| `VX_CLICKHOUSE_DEBUG`  | Build ClickHouse in debug mode                                |
| `CLICKHOUSE_VERSION`   | ClickHouse version to build against (default: latest release) |
| `CLICKHOUSE_SOURCE_DIR`| Path to ClickHouse source directory                           |

## Running Tests

```bash
# Default release build
cargo test -p vortex-clickhouse

# Debug build
VX_CLICKHOUSE_DEBUG=1 cargo test -p vortex-clickhouse
```

## Usage

```sql
-- Read from a Vortex file
SELECT * FROM file('data.vortex', 'Vortex');

-- Read with predicate pushdown
SELECT * FROM file('data.vortex', 'Vortex') WHERE id > 100;

-- Write query results to Vortex
INSERT INTO FUNCTION file('output.vortex', 'Vortex')
SELECT * FROM my_table;
```
