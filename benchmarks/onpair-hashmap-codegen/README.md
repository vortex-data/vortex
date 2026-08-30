# OnPair hash lookup codegen benchmark

This directory is deliberately independent of the repository's normal benchmark suite. It
replays lookup traces extracted from OnPair training and parsing, without timing parsing,
allocation, insertion, or I/O. It contains only the implementations needed for the codegen
comparison:

- hashbrown 0.16 `HashTable` with its default foldhash;
- a Rust port of Boost.Unordered's 15-slot FOA group, both scalar and four-lookups-at-once
  (`get4`);
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
four-row parsing. Leave it unset for the scalar row-major results in `CODEGEN.md`.

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
