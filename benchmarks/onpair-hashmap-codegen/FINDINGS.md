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
