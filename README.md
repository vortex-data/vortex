# 🌪️ Vortex

[![Build Status](https://github.com/vortex-data/vortex/actions/workflows/ci.yml/badge.svg)](https://github.com/vortex-data/vortex/actions)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10567/badge)](https://www.bestpractices.dev/projects/10567)
[![Documentation](https://docs.rs/vortex/badge.svg)](https://docs.vortex.dev)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/vortex-data/vortex)
[![Crates.io](https://img.shields.io/crates/v/vortex.svg)](https://crates.io/crates/vortex)
[![PyPI - Version](https://img.shields.io/pypi/v/vortex-data)](https://pypi.org/project/vortex-data/)
[![Maven - Version](https://img.shields.io/maven-central/v/dev.vortex/vortex-spark)](https://central.sonatype.com/artifact/dev.vortex/vortex-spark)
[![codecov](https://codecov.io/github/vortex-data/vortex/graph/badge.svg)](https://codecov.io/github/vortex-data/vortex)

[Join the community on Slack!](https://vortex.dev/slack) | [Documentation](https://docs.vortex.dev/) | [Performance Benchmarks](https://bench.vortex.dev)

## Overview

Vortex is a next-generation columnar file format and toolkit designed for high-performance data processing.
It is the fastest and most extensible format for building data systems backed by object storage. It provides:

- **Blazing Fast Performance**
  - 100x faster random access reads (vs. modern Apache Parquet)
  - 10-20x faster scans
  - 5x faster writes
  - Similar compression ratios
  - Efficient support for wide tables with zero-copy/zero-parse metadata

- **Extensible Architecture**
  - Modeled after Apache DataFusion's extensible approach
  - Pluggable encoding system, type system, compression strategy, & layout strategy
  - Zero-copy compatibility with Apache Arrow

- **Open Source, Neutral Governance**
  - A Linux Foundation (LF AI & Data) Project
  - Apache-2.0 Licensed

- **Integrations**
  - Arrow, DataFusion, DuckDB, Spark, Pandas, Polars, & more
  - Apache Iceberg (coming soon)

> 🟢 **Development Status**: Library APIs may change from version to version, but we now consider
> the file format <ins>_stable_</ins>. From release 0.36.0, all future releases of Vortex should
> maintain backwards compatibility of the file format (i.e., be able to read files written by
> any earlier version >= 0.36.0).

## Key Features

### Core Capabilities

- **Logical Types** - Clean separation between logical schema and physical layout
- **Zero-Copy Arrow Integration** - Seamless conversion to/from Apache Arrow arrays
- **Extensible Encodings** - Pluggable physical layouts with built-in optimizations
- **Cascading Compression** - Support for nested encoding schemes
- **High-Performance Computing** - Optimized compute kernels for encoded data
- **Rich Statistics** - Lazy-loaded summary statistics for optimization

### Technical Architecture

#### Logical vs Physical Design

Vortex strictly separates logical and physical concerns:

- **Logical Layer**: Defines data types and schema
- **Physical Layer**: Handles encoding and storage implementation
- **Built-in Encodings**: Compatible with Apache Arrow's memory format
- **Extension Encodings**: Optimized compression schemes (RLE, dictionary, etc.)

## Quick Start

### Installation

#### Rust Crate

All features are exported through the main `vortex` crate.

```bash
cargo add vortex
```

#### Python Package

```bash
uv add vortex-data
```

#### Command Line UI (vx)

For browsing the structure of Vortex files, you can use the `vx` command-line tool.

```bash
# Install latest release
cargo install vortex-tui --locked

# Or build from source
cargo install --path vortex-tui --locked

# Usage
vx browse <file>
```

### Development Setup

#### Prerequisites (macOS)

```bash
# Optional but recommended dependencies
brew install flatbuffers protobuf  # For .fbs and .proto files
brew install duckdb               # For benchmarks

# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# or
brew install rustup

# Initialize submodules
git submodule update --init --recursive

# Setup dependencies with uv
uv sync --all-packages
```

### Benchmarking

Use `vx-bench` to run benchmarks comparing engines (DataFusion, DuckDB) and formats (Parquet, Vortex):

```bash
# Install the benchmark orchestrator
uv tool install "bench_orchestrator @ ./bench-orchestrator/"

# Run TPC-H benchmarks
vx-bench run tpch --engine datafusion,duckdb --format parquet,vortex

# Compare results
vx-bench compare --run latest
```

See [bench-orchestrator/README.md](bench-orchestrator/README.md) for full documentation.

### Performance Optimization

For optimal performance, we suggest using [MiMalloc](https://github.com/microsoft/mimalloc):

```rust,ignore
#[global_allocator]
static GLOBAL_ALLOC: MiMalloc = MiMalloc;
```

## Project Information

### License

Licensed under the Apache License, Version 2.0.

### Governance

Vortex is an independent open-source project and not controlled by any single company. The Vortex Project is a
sub-project of the Linux Foundation Projects. The governance model is documented in
[CONTRIBUTING.md](CONTRIBUTING.md) and is subject to the terms of
the [Technical Charter](https://vortex.dev/charter.pdf).

### Contributing

Please **do** read [CONTRIBUTING.md](CONTRIBUTING.md) before you contribute.

### Reporting Vulnerabilities

If you discover a security vulnerability, please email <vuln-report@vortex.dev>.

### Trademarks

Copyright © Vortex a Series of LF Projects, LLC.
For terms of use, trademark policy, and other project policies please see <https://lfprojects.org>

## Acknowledgments

The Vortex project benefits enormously from groundbreaking work from the academic & open-source communities.

### Research in Vortex

- [BtrBlocks](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf) - Efficient columnar compression
- [FastLanes](https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf) & [FastLanes on GPU](https://dbdbd2023.ugent.be/abstracts/felius_fastlanes.pdf) - High-performance integer compression
- [FSST](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf) - Fast random access string compression
- [ALP](https://ir.cwi.nl/pub/33334/33334.pdf) & [G-ALP](https://dl.acm.org/doi/pdf/10.1145/3736227.3736242) - Adaptive lossless floating-point compression
- [Procella](https://dl.acm.org/citation.cfm?id=3360438) - YouTube's unified data system
- [Anyblob](https://www.durner.dev/app/media/papers/anyblob-vldb23.pdf) - High-performance access to object storage
- [ClickHouse](https://www.vldb.org/pvldb/vol17/p3731-schulze.pdf) - Fast analytics for everyone
- [MonetDB/X100](https://www.cidrdb.org/cidr2005/papers/P19.pdf) - Hyper-Pipelining Query Execution
- [Morsel-Driven Parallelism](https://db.in.tum.de/~leis/papers/morsels.pdf): A NUMA-Aware Query Evaluation Format for the Many-Core Age
- [The FastLanes File Format](https://github.com/cwida/FastLanes/blob/dev/docs/specification.pdf) - Expression Operators

### Vortex in Research

- [Anyblox](https://gienieczko.com/anyblox-paper) - A Framework for Self-Decoding Datasets
- [F3](https://dl.acm.org/doi/pdf/10.1145/3749163) - Open-Source Data File Format for the Future

### Open Source Inspiration

- [Apache Arrow](https://arrow.apache.org)
- [Apache DataFusion](https://github.com/apache/datafusion)
- [parquet2](https://github.com/jorgecarleitao/parquet2) by Jorge Leitao
- [DuckDB](https://github.com/duckdb/duckdb)
- [Velox](https://github.com/facebookincubator/velox) & [Nimble](https://github.com/facebookincubator/nimble)

#### Thanks to all contributors who have shared their knowledge and code with the community! 🚀
