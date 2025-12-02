<!--
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Hack Week

## Goals

_Approximately in order of priority:_

- Get faster load times of Vortex benchmarks on the web by using Vortex itself to store benchmarks measurements instead of JSON
- Make Vortex work reliably on the web via WASM
- Allow addition/removal of different benchmark measurements with schema evolution on Vortex
- Make the benchmarks website easier to read / more understandable
- Rewrite the entire benchmarks website to a WASM framework like Dioxus
- (Stretch) Make benchmarks website more dynamic
- (Stretch) Add Vortex demo in the browser
- (Stretch) Add Vortex vs. Parquet demo in the browser
- (Stretch) Add wasm-bindgen bindings for Vortex?

## Plan of Attack

- Design (at a high level) a better benchmarks website (figure out what components and pages it needs, plus general layout)
- Figure out the minimal API for the current benchmark website
- If the current JavaScript code conflicts than the new design, refactor the architecture of the website so that it is easy to switch out the implementations
- Determine the schema of each of the current benchmark (and the evolution of each over time)
- Figure out if the current schemas make sense or if they need to change
- Design extensible(?) Vortex schemas for benchmarking
- Migrate all existing data to Vortex files
- Design writer (append-only) interface for adding benchmark measurements that can evolve its schema
- Design reader interface for loading specific columns of Vortex from S3 and parsing data to a format easily read by JavaScript (should probably be streaming over chunks?)
- Implement the reader and writer interfaces with wasm-bindgen
- Migrate the JavaScript code to use the Rust bindings
- Test


### Ideas

```rust
/// The 20 byte SHA-1 Git commit ID.
pub struct CommitId([u8; 20]);

/// String ID lookup so that we don't have to store the string every time.
pub struct NameId(u32);

/// A benchmark entry, grouped by benchmark group, then chart name, then series name.
pub struct BenchmarkEntry {     // `StructArray`
    commit_id: CommitId,        // fixed size list of `u8`?
    benchmark_group: NameId,    // `u16` array
    chart_name: NameId,         // `u16` array
    series_name: NameId,        // `u16` array
    value: u64,                 // `u64` array
}

fn main() {
    println!("{}", size_of::<BenchmarkEntry>()); // 64
    println!("{}", align_of::<BenchmarkEntry>()); // 8
}
```

### Findings

- There is an insane amount of wasted space in `data.json`
- The amount of actual benchmarking data is actually very small, and it can easily fit in memory of
  the CI runners
- We can simply read the entire file of all benchmarking data into memory, decompress in memory, add
  a new entry, compress, and then write back to S3


### Things to update

Start with just the random access benchmark


generate a bunch of fake data and upload it to S3

- Add bindings to read and write `BenchmarkEntry` vortex arrays to and from S3
USE IPC FORMAT INSTEAD
- `query_bench` to post directly to S3
- `random_access` and `compress` to also post directly to S3
