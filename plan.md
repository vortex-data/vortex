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

/// Maps [`CommitId`] to benchmark value.
pub type CommitValueMap<'a> = HashMap<&'a CommitId, u64, PassthroughBuildHasher>;

/// Maps series name to commit values.
pub type SeriesMap<'a> = HashMap<&'a str, CommitValueMap<'a>>;

/// Series in a chart mapped to their data.
pub type ChartMap<'a> = HashMap<&'a str, SeriesMap<'a>>;

/// Chart names in a group mapped to their data.
pub type GroupedEntries<'a> = HashMap<&'a str, ChartMap<'a>>;
```

A benchmark group should be defined by 1 or more charts and 1 or more series (that always appear on
every chart in the group).

### Findings

- There is an insane amount of wasted space in `data.json`&
- The amount of actual benchmarking data is actually very small, and it can easily fit in memory of
  the CI runners
- We can simply read the entire file of all benchmarking data into memory, decompress in memory, add
  a new entry, compress, and then write back to S3

### 1 file vs many files

With 1 file, we have to stuff every different kind of benchmark into the same place, which isnt great
for compression and it means we have to do more work on read time to group data correctly (by benchmark group, chart, then series).

The seemingly obvious alternative here is to have a different file per "same" data. But what exactly would these be grouped by? We definitely do not want to group by series as that makes it pretty
difficult to add a new series to a chart (maybe it's not terrible with some more engineering). It
also would mean that we would start to approach 1000+ files.

We could also do a file per chart, as that maps much closer to how we generate these chart. One
program is generating all the data for one chart, but that program might also generate data for
multiple charts. This is definitely something we should look into later, but for now having a single
file that has all the data (all with the same schema) is the most flexible.

### Things to update

Start with just the random access benchmark

generate a bunch of fake data and upload it to S3

- Add bindings to read and write `BenchmarkEntry` vortex arrays to and from S3
- `query_bench` to post directly to S3
- `random_access` and `compress` to also post directly to S3
