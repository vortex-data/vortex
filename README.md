🌪️ Vortex
=========

[![Build Status](https://github.com/vortex-data/vortex/actions/workflows/ci.yml/badge.svg)](https://github.com/vortex-data/vortex/actions)
[![Documentation](https://docs.rs/vortex-array/badge.svg)](https://docs.vortex.dev)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/vortex-data/vortex)
[![Crates.io](https://img.shields.io/crates/v/vortex-array.svg)](https://crates.io/crates/vortex-array)
[![PyPI - Version](https://img.shields.io/pypi/v/vortex-array)](https://pypi.org/project/vortex-array/)
[![Maven - Version](https://img.shields.io/maven-central/v/dev.vortex/vortex-spark)](https://central.sonatype.com/artifact/dev.vortex/vortex-spark)

📚 [Documentation](https://docs.vortex.dev/) | 📊 [Performance Benchmarks](https://bench.vortex.dev)

## Overview

Vortex is a next-generation columnar file format and toolkit designed for high-performance data analytics. It provides:

- **⚡️ Blazing Fast Performance**
    - 100-200x faster random access reads than Apache Parquet
    - 2-10x faster scans with similar compression ratios and write throughput
    - Efficient support for wide tables with zero-copy/zero-parse metadata

- **🔧 Extensible Architecture**
    - Modeled after Apache DataFusion's extensible approach
    - Pluggable encoding system
    - Zero-copy compatibility with Apache Arrow

> 🚧 **Development Status**: This project is under active development. APIs and file formats may change, and some
> features are still being implemented.

## Key Features

### Core Capabilities

- ✨ **Logical Types** - Clean separation between logical schema and physical layout
- 🔄 **Zero-Copy Arrow Integration** - Seamless conversion to/from Apache Arrow arrays
- 🧩 **Extensible Encodings** - Pluggable physical layouts with built-in optimizations
- 📦 **Cascading Compression** - Support for nested encoding schemes
- 🚀 **High-Performance Computing** - Optimized compute kernels for encoded data
- 📊 **Rich Statistics** - Lazy-loaded summary statistics for optimization

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
uv add vortex-array
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

### Performance Optimization

For optimal performance, use [MiMalloc](https://github.com/microsoft/mimalloc):

```rust,ignore
#[global_allocator]
static GLOBAL_ALLOC: MiMalloc = MiMalloc;
```

## Project Information

### License

Licensed under the Apache License, Version 2.0

### Governance

Vortex is committed to remaining open-source, following governance models inspired by
the [Substrait project](https://substrait.io/governance/) and Apache Software Foundation.

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments 🏆

This project builds upon groundbreaking work from the academic and open-source communities:

### Key Research Papers

- [BtrBlocks](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf) - Efficient columnar compression
- [FastLanes](https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf) - High-performance integer compression
- [FSST](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf) - Fast random access string compression
- [ALP](https://ir.cwi.nl/pub/33334/33334.pdf) - Adaptive lossless floating-point compression
- [Procella](https://dl.acm.org/citation.cfm?id=3360438) - YouTube's unified data system
- [Cloud Object Storage Analytics](https://www.durner.dev/app/media/papers/anyblob-vldb23.pdf) - High-performance
  analytics
- [ClickHouse](https://www.vldb.org/pvldb/vol17/p3731-schulze.pdf) - Fast analytics for everyone

### Open Source Inspiration

- [Apache Arrow](https://arrow.apache.org) & [Apache DataFusion](https://github.com/apache/datafusion)
- [parquet2](https://github.com/jorgecarleitao/parquet2) by Jorge Leitao
- [DuckDB](https://github.com/duckdb/duckdb)
- [Velox](https://github.com/facebookincubator/velox) & [Nimble](https://github.com/facebookincubator/nimble)

---
*Thanks to all contributors who have shared their knowledge and code with the community! 🚀*
