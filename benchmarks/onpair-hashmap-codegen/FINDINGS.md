# The GCC-versus-LLVM gap is a benchmark bug, not codegen

## Bottom line

There is no GCC-versus-LLVM code generation gap on this workload. The 1.32x-1.44x reported in
[CODEGEN.md](CODEGEN.md) came from `load_trace` in `boost.cpp` relying on unspecified argument
evaluation order, which made **the GCC build populate its long map with garbage keys**. Every long
lookup in the GCC binary missed, so GCC was timing the cheap miss path while Clang timed real
lookups.

One line was at fault:

```cpp
trace.long_entries.emplace_back(read_u64(input), read_u16(input));
```

The order in which those two `read_*` calls run is unspecified in C++. Clang evaluates left to
right and reads the 8-byte key then the 2-byte value; GCC evaluates right to left and reads the
value first, so its key is assembled from the wrong 8 bytes of the stream. Both orders consume the
same 10 bytes per entry, so the stream stays in sync, the trailing-bytes check passes, and nothing
reports an error. The C++ benchmark never checked its own results — only the Rust binary verifies
against hashbrown.

Instrumenting both binaries to report what they actually find, before the fix:

| Build | `checksum_long` | hits (of 502,053) |
|---|---:|---:|
| GCC 13 | 0 | 0 |
| Clang 18 | 1,893,020,570 | 502,053 |

After sequencing the two reads explicitly, both report `checksum_long=1095757159, hits=283294` on
the real trace, and the performance difference disappears:

| Build | Mprobe/s | short_ms | long_ms |
|---|---:|---:|---:|
| Boost, GCC 13 | 126.2 | 6.86 | 4.92 |
| Boost, GCC 14 | 124.8 | 7.06 | 4.86 |
| Boost, Clang 18 | 121.2 | 7.30 | 4.96 |
| Boost, Clang 20 | 119.9 | 7.47 | 4.92 |

The long map — where the entire gap lived — is now identical across all four compilers (4.86-4.96
ms). The residual 4% total-throughput spread is in the short map and is within run-to-run noise.

Only the long map was affected. The short-entry loop reads its key into a local first, so it has
just one function call in the `emplace_back` argument list and no ordering hazard, which is exactly
why `CODEGEN.md`'s short-key numbers were always a tie while the long-key numbers were not.

## Corrected comparison

Long-map `get`, ns per probe, hit rate varied by rewriting only the long-probe section of the
trace (`predictability_sweep.py`):

| Hit % | Boost GCC | Boost Clang | Rust Group15 | hashbrown |
|---:|---:|---:|---:|---:|
| 0 | 4.81 | 4.81 | 5.10 | 9.50 |
| 10 | 6.27 | 6.51 | 6.84 | 11.93 |
| 25 | 7.55 | 8.03 | 8.34 | 11.40 |
| 50 | 10.22 | 10.61 | 11.12 | 11.85 |
| 75 | 7.38 | 8.67 | 8.47 | 7.28 |
| 90 | 5.49 | 5.71 | 6.67 | 4.85 |
| 100 | 3.76 | 4.03 | 4.08 | 2.46 |

GCC and Clang now track each other within noise at every hit rate. The Rust Group15 port is within
3-9% of Boost throughout — it was already at parity, and the "1.32x behind GCC" figure it was being
held to did not exist.

What remains real is the shape of the curve: cost peaks at 50% hit for every Group15
implementation, because that is where the fingerprint-match and key-compare branches are least
predictable. hashbrown has a different shape — worse on misses, best of all on hits — because its
empty-byte metadata scheme pays on the miss path. For OnPair's low hit rates Group15 remains the
right design.

## How it was found

The route matters, because the assembly-level story in `CODEGEN.md` looked convincing and was
measuring the wrong thing.

1. Splitting the trace by hit rate showed GCC completely flat (4.1-4.5 ns at every hit rate) while
   Clang traced a textbook misprediction curve. A loop doing real work cannot be indifferent to
   whether its branches are predictable.
2. A standalone 130-line reproducer of the same probe shape (`probe_repro.cpp`) showed the
   misprediction curve *equally* under both compilers — no compiler difference at all. That was the
   first hard evidence the difference was not in codegen.
3. Callgrind's branch simulator (`--branch-sim=yes`) put numbers on it: at 50% hit, Clang's
   fingerprint-match branch mispredicted 48.1% of 566,608 executions, while the GCC binary had no
   comparably unpredictable branch anywhere in its long-probe loop, and its total mispredict count
   did not move between 0%, 50% and 100% hit.
4. The instruction-level profile showed the GCC loop taking its "no fingerprint match" exit on
   essentially every probe. Printing the checksum confirmed zero hits.

Note that `asm_boost_long`, the exported symbol `CODEGEN.md` documents and `make asm` dumps, is
**not** the code being timed — the timed loops are lambdas inlined into `main`. The assembly
comparison in `CODEGEN.md` (constant hoisting, `blsr` versus `lea`+`and`, stack-reference counts)
was made against a function that never runs.

## What this means for CODEGEN.md

Its scalar conclusions should be withdrawn:

- "GCC's `boost::unordered_flat_map` remains 1.32x faster by geometric mean" — artifact.
- "Compiling that same Boost source with Clang loses 1.44x to GCC" — artifact; this was the
  GCC-versus-Clang control, and it was measuring GCC not doing the work.
- "GCC/hashbrown 1.86x" — same artifact in the denominator.
- The "hot-loop assembly differences" section — disassembles an unused symbol.
- The scalar half of "LLVM investigation order" (steps 5-9) — no scalar gap to investigate.

The `get4` AVX-512 finding is unaffected: it is Rust-only, does not involve the C++ trace loader,
and reproduces independently (a single `vpmullq` costs 4.8x; disabling either AVX-512 or SLP
recovers it).

Probe statistics in `CODEGEN.md` are also unaffected — they come from the Rust binary.

## Reproducing

```bash
python3 mkcorpus.py /tmp/synth-4mib.onpair 4194304 42     # stand-in ONPAIR01 corpus
make all BOOST_ROOT=/tmp/boost_1_89_0
ONPAIR_BITS=16 target/native/release/trace /tmp/synth-4mib.onpair /tmp/synth16.oph
python3 predictability_sweep.py /tmp/synth16.oph \
  target/native/boost-gcc target/native/boost-clang
```

To see the original bug, revert the `long_entries` loop in `boost.cpp` to the single
`emplace_back(read_u64(input), read_u16(input))` call and compare `checksum_long` between a GCC and
a Clang build.

Measured on an Intel Xeon @ 2.10GHz (4 vCPU, AVX-512), GCC 13.3/14, Clang 18.1.3/20.1.2, Boost
1.89.0, `-std=c++20 -O3 -DNDEBUG -march=native`. No ONPAIR01 corpus was available, so `mkcorpus.py`
generates a Zipf-distributed stand-in; the conclusion is a correctness bug and does not depend on
the corpus.

## Suggested follow-ups

1. Have the C++ benchmark verify its own checksums against the trace, the way the Rust binary
   verifies against hashbrown. This class of bug is otherwise invisible.
2. Time the exported `asm_*` symbols, or drop them, so `make asm` and the timed code are the same
   loop.
3. Re-measure the six real corpora with the fix before drawing any Rust-versus-Boost conclusion.
4. Since the remaining cost is branch misprediction and it is compiler-neutral, a branch-free probe
   is the actual optimization opportunity here.

## Confirmation on the paper's own C++ implementation

The corrected microbenchmark says there is no compiler gap. To check that end to end rather than on
a probe loop, the same comparison was run on the OnPair authors' C++ library over the paper's own
datasets — real training and real parsing, not a synthetic trace.

- Library: [onpair_cpp](https://github.com/gargiulofrancesco/onpair_cpp) (the standalone
  implementation referenced by [compression_benchmark_cpp](https://github.com/gargiulofrancesco/compression_benchmark_cpp),
  the paper's evaluation framework for [arXiv:2508.02280](https://arxiv.org/abs/2508.02280)).
- Relevance: `include/onpair/encoding/lpm.h` builds its longest-prefix matcher on
  `boost::unordered_flat_map`, so the probe path under investigation is exactly the hot path of
  parsing.
- Driver: `onpair_cpp_bench.cpp` here, timing `encoding::train()` and `encoding::parse()`
  separately.
- Datasets: MS MARCO queries (28 MiB, 808,731 rows, full), MS MARCO URLs (25 MiB, 400,000 rows) and
  DBpedia abstracts (40 MiB, 135,918 rows) — three of the paper's four, each a prefix of the real
  download where the full file is tens of GiB.

Throughput in MiB/s, median of 5 iterations, median of 3 interleaved repetitions per compiler.
`g/c` above 1 means GCC is faster.

| Dataset | Bits | Train GCC | Train Clang | g/c | Parse GCC | Parse Clang | g/c |
|---|---:|---:|---:|---:|---:|---:|---:|
| MS MARCO queries | 12 | 308.0 | 307.6 | 1.00 | 121.1 | 116.8 | 1.04 |
| MS MARCO queries | 16 | 191.3 | 190.9 | 1.00 | 106.4 | 104.4 | 1.02 |
| MS MARCO URLs | 12 | 319.1 | 300.0 | 1.06 | 114.7 | 107.1 | 1.07 |
| MS MARCO URLs | 16 | 197.5 | 204.9 | 0.96 | 102.6 | 99.6 | 1.03 |
| DBpedia abstracts | 12 | 732.7 | 713.2 | 1.03 | 147.0 | 138.4 | 1.06 |
| DBpedia abstracts | 16 | 336.7 | 344.8 | 0.98 | 135.6 | 136.8 | 0.99 |

Geometric means: training 1.005, parsing 1.035. Clang wins two of the twelve cells outright, and
the largest GCC margin is 7%. Every configuration produced identical `dict_tokens` and total token
counts under both compilers, so the two builds are doing the same work — the check that the
original benchmark lacked.

GCC 14 and Clang 20 were also measured on MS MARCO queries and land in the same band.

There is no 1.3-1.4x GCC advantage in OnPair training or parsing. Whatever remains is a few percent,
dataset-dependent in sign, and within the run-to-run spread of a 4-vCPU shared machine.

### Reproducing

```bash
git clone https://github.com/gargiulofrancesco/onpair_cpp /tmp/onpair_cpp
SRC="/tmp/onpair_cpp/src/onpair/encoding/training/trainer.cpp
     /tmp/onpair_cpp/src/onpair/encoding/parsing/parser.cpp
     /tmp/onpair_cpp/src/onpair/core/dictionary_view.cpp"
for cxx in g++ clang++; do
  $cxx -std=c++20 -O3 -DNDEBUG -march=native \
    -I/tmp/onpair_cpp/include -I/tmp/boost_1_89_0 \
    onpair_cpp_bench.cpp $SRC -o /tmp/onpair-${cxx%%+*}
done
/tmp/onpair-g /tmp/data/msmarco_queries.txt 16 5
/tmp/onpair-clang /tmp/data/msmarco_queries.txt 16 5
```

Datasets come from `scripts/process_datasets.py` in the paper's benchmark repository; the runs above
used streamed prefixes of the same sources, one string per line.

Interleave the two binaries and compare medians — a sequential all-GCC-then-all-Clang run on a
shared vCPU drifts by more than the effect being measured. An earlier run of exactly this
comparison reported spurious differences because a background download was still appending to the
dataset file between runs; check the input is stable first.

## Decompression: three implementations compared

Since compression and parsing show no compiler effect, the remaining question is the decode path.
Three implementations of the same algorithm were compared on the same datasets:

- **onpair_cpp** — the paper's C++ library, built with both GCC 13 and Clang 18.
- **spiraldb/onpair** — the Rust port Vortex depends on (`bac803e`), a direct counterpart of the
  C++ design (bit-packed store, compact dictionary, 16-byte over-copy).
- **onpair_rs** — the paper's own Rust implementation, in both `OnPair16` and `OnPair` variants.

Two access patterns per implementation: bulk decode of the whole column, and random-access decode of
every row in a shuffled order (same permutation for all implementations). Every decode is verified
byte-for-byte against the original input; all 72 runs passed.

`decompress_into`/`decompress_all` and `decompress_row_into`/`decompress_string`/`decompress` are
each library's own public API, used as documented.

| Dataset | Bits | Implementation | Bulk MiB/s | Random ns/row | Compressed MiB |
|---|---:|---|---:|---:|---:|
| MS MARCO queries | 12 | onpair_cpp (GCC) | 6148 | 133.8 | 13.39 |
| | 12 | onpair_cpp (Clang) | 5974 | 135.9 | 13.39 |
| | 12 | spiraldb/onpair | 3297 | 85.6 | 13.37 |
| MS MARCO queries | 16 | onpair_cpp (GCC) | 4266 | 104.6 | 12.13 |
| | 16 | onpair_cpp (Clang) | 4384 | 109.5 | 12.13 |
| | 16 | spiraldb/onpair | 2889 | 66.4 | 12.15 |
| | 16 | onpair_rs OnPair16 | 4303 | 67.5 | 8.76 |
| | 16 | onpair_rs OnPair | 3113 | 73.0 | 8.62 |
| MS MARCO URLs | 12 | onpair_cpp (GCC) | 5100 | 162.8 | 11.48 |
| | 12 | spiraldb/onpair | 3136 | 118.5 | 11.51 |
| MS MARCO URLs | 16 | onpair_cpp (GCC) | 4147 | 152.5 | 10.51 |
| | 16 | onpair_cpp (Clang) | 4399 | 148.3 | 10.51 |
| | 16 | spiraldb/onpair | 2660 | 106.9 | 10.51 |
| | 16 | onpair_rs OnPair16 | 4119 | 105.4 | 8.70 |
| | 16 | onpair_rs OnPair | 2523 | 113.6 | 8.16 |
| DBpedia abstracts | 12 | onpair_cpp (GCC) | 4740 | 233.7 | 14.49 |
| | 12 | spiraldb/onpair | 3038 | 192.7 | 14.71 |
| DBpedia abstracts | 16 | onpair_cpp (GCC) | 5265 | 229.4 | 11.61 |
| | 16 | onpair_cpp (Clang) | 5030 | 246.9 | 11.61 |
| | 16 | spiraldb/onpair | 3548 | 171.3 | 11.69 |
| | 16 | onpair_rs OnPair16 | 5087 | 152.3 | 11.00 |
| | 16 | onpair_rs OnPair | 2480 | 241.7 | 10.05 |

Medians of 3 interleaved repetitions, each the median of 5 iterations.

### Bulk decode: spiraldb/onpair is 1.3-1.6x behind

`onpair_cpp` and `onpair_rs OnPair16` are level (4100-6100 MiB/s); GCC and Clang are within 4% of
each other, consistent with the rest of this investigation. **The Rust port Vortex uses is the
slowest bulk decoder in every configuration** — 2660-3548 MiB/s, a 1.3-1.6x deficit against both
the C++ and the paper's Rust.

This is not a compression-quality artifact. `spiraldb/onpair` and `onpair_cpp` produce essentially
identical compressed sizes (12.15 vs 12.13 MiB on queries at 16 bits, 10.51 vs 10.51 on URLs,
11.69 vs 11.61 on DBpedia) — same algorithm, same dictionary, same token count. The gap is decode
throughput on the same bytes, and it is the one real optimization opportunity this whole
investigation has turned up.

`onpair_rs`'s smaller output is a training-configuration difference, not a decode one: it takes a
fixed frequency threshold (5) where the other two take a dynamic 0.15, producing a different
dictionary. Its `space_used()` also omits the row-boundary array the other two include.

### Random access: the C++ API leaves 1.7x on the table

On the as-shipped APIs, the Rust implementations lead: `spiraldb/onpair` at 66.4 ns/row and
`onpair_rs OnPair16` at 67.5 versus `onpair_cpp` at 104.6 (queries, 16 bits).

That deficit is an API artifact, not codegen. `onpair::decoding::decompress()` calls
`dispatch_bits()` — the runtime-to-compile-time bit-width switch — on *every row*. The library's own
`TokenCursor` documentation says to resolve the width once and reuse the monomorphised cursor, which
the per-row entry point cannot do for a caller looping over rows. Hoisting the dispatch outside the
loop (`onpair_cpp_random_access_probe.cpp`) and changing nothing else:

| Bits | Per-row API | Dispatch hoisted | Speedup |
|---:|---:|---:|---:|
| 12 | 131.6 ns | 85.8 ns | 1.53x |
| 16 | 105.8 ns | 61.5 ns | 1.72x |

At 61.5 ns/row the C++ becomes the fastest of the three. So the honest reading is that C++ and
`onpair_rs OnPair16` are the strongest decoders on both axes, and a batched random-access API — one
that resolves the bit width once per call rather than once per row — is worth 1.5-1.7x to any C++
caller decoding more than one row.

### Takeaways for Vortex

1. `spiraldb/onpair`'s bulk decode is 1.3-1.6x off both reference implementations at matching
   compression. Worth profiling against `decode_all<Bits>` in the C++, which is a branch-free
   maximally-unrolled loop, and against `onpair_rs`'s flat `decompress_all`.
2. Its random-access path is already competitive — best of the three as APIs ship today.
3. Neither result has anything to do with GCC versus LLVM.

### Reproducing

```bash
# C++ decoder (both compilers)
SRC="…trainer.cpp …parser.cpp …dictionary_view.cpp …column.cpp"   # from onpair_cpp/src
g++     -std=c++20 -O3 -DNDEBUG -march=native -I/tmp/onpair_cpp/include -I/tmp/boost_1_89_0 \
  onpair_cpp_decode.cpp $SRC -o /tmp/decode-gcc
clang++ … onpair_cpp_decode.cpp $SRC -o /tmp/decode-clang

# Both Rust implementations in one binary
cd onpair_decode_rs && RUSTFLAGS="-C target-cpu=native" cargo build --release

/tmp/decode-gcc                              /tmp/data/msmarco_queries.txt 16 5
onpair_decode_rs/target/release/decode-rs    /tmp/data/msmarco_queries.txt 16 5
```

Both drivers print `bulk_ok` and `random_ok`; treat any run without both true as void.

## Compression time across the three implementations

The same three implementations, timed on compression instead of decompression. `onpair_cpp` and
`spiraldb/onpair` expose training and parsing separately; `onpair_rs` compresses in one call, so
only its total is available.

Throughput in MiB/s of input, medians of 3 interleaved repetitions of the median of 5 iterations.

| Dataset | Bits | Implementation | Train | Parse | Total |
|---|---:|---|---:|---:|---:|
| MS MARCO queries | 12 | onpair_cpp (GCC) | 306.7 | 124.2 | 88.4 |
| | 12 | onpair_cpp (Clang) | 293.6 | 110.7 | 80.4 |
| | 12 | spiraldb/onpair | 196.0 | 110.8 | 70.8 |
| MS MARCO queries | 16 | onpair_cpp (GCC) | 185.3 | 106.1 | 67.5 |
| | 16 | onpair_cpp (Clang) | 195.7 | 105.7 | 68.6 |
| | 16 | spiraldb/onpair | 164.6 | 90.5 | 58.4 |
| | 16 | onpair_rs OnPair16 | - | - | 40.5 |
| | 16 | onpair_rs OnPair | - | - | 25.5 |
| MS MARCO URLs | 12 | onpair_cpp (GCC) | 311.1 | 109.2 | 80.8 |
| | 12 | onpair_cpp (Clang) | 292.9 | 104.6 | 77.1 |
| | 12 | spiraldb/onpair | 186.4 | 89.5 | 60.5 |
| MS MARCO URLs | 16 | onpair_cpp (GCC) | 197.9 | 99.6 | 66.3 |
| | 16 | onpair_cpp (Clang) | 200.1 | 105.5 | 69.1 |
| | 16 | spiraldb/onpair | 163.3 | 75.8 | 51.8 |
| | 16 | onpair_rs OnPair16 | - | - | 40.6 |
| | 16 | onpair_rs OnPair | - | - | 29.3 |
| DBpedia abstracts | 12 | onpair_cpp (GCC) | 665.5 | 137.6 | 114.1 |
| | 12 | onpair_cpp (Clang) | 627.0 | 126.2 | 105.0 |
| | 12 | spiraldb/onpair | 376.4 | 45.2 | 40.4 |
| DBpedia abstracts | 16 | onpair_cpp (GCC) | 329.6 | 133.4 | 95.0 |
| | 16 | onpair_cpp (Clang) | 339.1 | 139.9 | 99.0 |
| | 16 | spiraldb/onpair | 292.1 | 124.1 | 87.1 |
| | 16 | onpair_rs OnPair16 | - | - | 80.8 |
| | 16 | onpair_rs OnPair | - | - | 34.1 |

`onpair_cpp` is the fastest compressor in every configuration. `spiraldb/onpair` is 1.1-2.8x
behind it, and `onpair_rs` is the slowest at 16 bits except on DBpedia.

The worst case is **DBpedia at 12 bits, where `spiraldb/onpair` parses at 45.2 MiB/s against
onpair_cpp's 137.6 — a 3.0x deficit** that drags its total to 40.4 versus 114.1 MiB/s. The same
pair at 16 bits is only 1.1x apart (124.1 vs 133.4), so this is specific to the narrow-dictionary,
long-string combination rather than a uniform slowdown. Training is 1.1-1.8x behind throughout.

Since parsing is where the longest-prefix-match hashmap lives, this — not any compiler
difference — is where the Rust port's probe path actually costs something. It is the same
conclusion the decode comparison reached: the gaps worth chasing are between implementations, not
between GCC and LLVM, which stay within a few percent of each other in every row above.

`onpair_rs` numbers are not strictly comparable: it takes a fixed frequency threshold (5) where the
other two take a dynamic 0.15, so it builds a different dictionary and does a different amount of
work. It is included for reference, not as a like-for-like control.

```bash
onpair_decode_rs/target/release/compress /tmp/data/msmarco_queries.txt 16 5   # both Rust impls
/tmp/onpair-gcc /tmp/data/msmarco_queries.txt 16 5                            # onpair_cpp
```
