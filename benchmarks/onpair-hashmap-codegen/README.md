# OnPair hash lookup codegen benchmark

This directory is deliberately independent of the repository's normal benchmark suite. It
replays lookup traces extracted from OnPair training and parsing, without timing parsing,
allocation, insertion, or I/O. It contains only the implementations needed for the codegen
comparison and frozen-table experiments:

- hashbrown 0.16 `HashTable` with its default foldhash;
- a Rust port of Boost.Unordered's 15-slot FOA group, both scalar and four-lookups-at-once
  (`get4`);
- immutable 15- and 16-slot split-key/value Group15 layouts and a packed long-key layout;
- ordinary, blocked, home-group, and short/long-tuned Bloom-filter front ends;
- an immutable Xor8-filter front end;
- exact and approximate four-byte-prefix length directories for skipping impossible descending
  OnPair probes;
- the actual `boost::unordered_flat_map`, built from the same C++ source with GCC and Clang.

The Vortex branch starts at `1fe8dda03` and changes no production code. The standalone crate pins
OnPair at `bac803e`, before the experimental production hashmap and parser changes.

## Prerequisites

- Rust 1.91 or newer;
- GCC, Clang, GNU Make, and binutils;
- Boost 1.89 headers. The benchmark was measured with an unpacked tree at
  `/tmp/boost_1_89_0`; pass `BOOST_ROOT=/another/path` if yours is elsewhere;
- an `ONPAIR01` corpus such as `/tmp/onpair-corpora/fineweb-32mib.onpair`.

Boost is header-only for this benchmark. One way to obtain the exact version is:

```bash
curl -LO https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.bz2
tar -xf boost_1_89_0.tar.bz2 -C /tmp
```

## Reproduce one result

Run commands from this directory. `make run` builds Rust, GCC, and Clang with an explicit native
CPU target, creates an `OPHASH01` trace using the unmodified OnPair library, checks Rust lookup
results against hashbrown, prints probe statistics, and reports the median of 15 iterations.
Every timed Rust iteration also checks its aggregate result against a checksum derived directly
from the trace dictionaries. A false-negative filter therefore fails the run instead of producing
an attractive but invalid timing.

```bash
make run \
  CORPUS=/tmp/onpair-corpora/fineweb-32mib.onpair \
  BITS=16 \
  BOOST_ROOT=/tmp/boost_1_89_0
```

The trace construction is outside the timed region. `BITS=12` and `BITS=16` select the OnPair
dictionary size. To cap trace generation to the first 4 MiB of corpus payload, run the two stages
explicitly:

```bash
TRACE_PAYLOAD_LIMIT=4194304 make trace CORPUS=/tmp/onpair-corpora/fineweb-32mib.onpair BITS=16
make run CORPUS=/tmp/onpair-corpora/fineweb-32mib.onpair BITS=16
```

The second command regenerates an uncapped trace, so for a capped run invoke the three binaries
directly after `make all`:

```bash
HASH_PROBE_STATS=1 HASH_WARMUPS=3 HASH_ITERATIONS=15 \
  target/native/release/bench /tmp/onpair-hashmap-codegen.oph
HASH_WARMUPS=3 HASH_ITERATIONS=15 \
  target/native/boost-gcc /tmp/onpair-hashmap-codegen.oph
HASH_WARMUPS=3 HASH_ITERATIONS=15 \
  target/native/boost-clang /tmp/onpair-hashmap-codegen.oph
```

The trace generator also supports `TRACE_INTERLEAVE_WIDTH=4`. That changes probe order to model
four-row parsing. Leave it unset for the scalar row-major results in `CODEGEN.md`. Grouped-prefix
reports require contiguous scalar token searches and are automatically skipped for an interleaved
trace. Their boundaries are recovered from actual dictionary hits; adjacent descending lengths
alone are not a valid boundary signal.

Set `HASH_FINAL_ONLY=1` to limit a run to hashbrown, scalar Group15, and the strongest compact
blocked-Bloom candidates. The default intentionally retains the wider algorithm set for ablation
and handoff work.

## Benchmark contract

The Rust and C++ flat-table reports replay every short and long probe in the same `OPHASH01` file.
Table construction, filter construction, trace loading, allocation, and correctness validation are
outside the timed sections. All maps contain the same entries and return zero for a miss.

The prefix reports measure a different algorithmic contract: they preserve the scalar lookup
result while avoiding exact probes that an immutable prefix index proves cannot hit. Their output
uses `moriginal_probes_s`, meaning original trace probes divided by elapsed time, and separately
reports the number of exact and skipped probes. Do not compare that column as if it were raw exact
hash-table probe throughput.

This isolated lookup benchmark is not an end-to-end compressor benchmark. In particular, the Rust
and C++ parsers used during earlier investigation emitted different physical representations
(`u16` codes versus timed bit-packing), so those parser timings are not reproduced or claimed here.

## Inspect assembly

Stable exported symbols keep the relevant loops easy to find:

- `asm_hashbrown_long`;
- `asm_rust_group15_long`;
- `asm_rust_group15_get4_long`;
- `asm_boost_long` in each C++ binary.

```bash
make asm BOOST_ROOT=/tmp/boost_1_89_0
```

For a file suitable for diffing:

```bash
objdump -d --no-show-raw-insn -Mintel target/native/release/bench > /tmp/rust.asm
objdump -d --no-show-raw-insn -Mintel target/native/boost-gcc > /tmp/boost-gcc.asm
objdump -d --no-show-raw-insn -Mintel target/native/boost-clang > /tmp/boost-clang.asm
```

The native build currently makes LLVM vectorize the four long-key hash calculations into
AVX-512 `vpmullq` instructions and performs much worse. Build an x86-64-v3 control in a separate
target directory to disable AVX-512 while retaining AVX2 and BMI2:

```bash
make rust TARGET_DIR=target/v3 RUST_CPU=x86-64-v3
HASH_WARMUPS=3 HASH_ITERATIONS=15 \
  target/v3/release/bench /tmp/onpair-hashmap-codegen.oph
```

See [CODEGEN.md](CODEGEN.md) for the recorded comparison, assembly observations, and suggested
LLVM experiments.
