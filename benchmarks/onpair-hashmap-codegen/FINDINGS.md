# Reproducing the GCC-versus-LLVM gap

This records an independent reproduction of the results in [CODEGEN.md](CODEGEN.md) on different
hardware, plus the experiments that narrow down *why* LLVM loses. It changes no benchmark code;
it only adds a synthetic corpus generator, a predictability sweep, and a standalone reproducer.

## Setup

Different machine from `CODEGEN.md`, so absolute numbers are not comparable with it; the ratios
are what carry over.

- CPU: Intel Xeon @ 2.10GHz, 4 vCPU, AVX-512 available (`avx512f/bw/dq/vl/vbmi/vnni`).
- GCC 13.3.0 and GCC 14; Clang 18.1.3 and Clang 20.1.2; rustc 1.94 with its bundled LLVM.
- Boost 1.89.0, `-std=c++20 -O3 -DNDEBUG -march=native`.
- Corpus: no ONPAIR01 corpus was available, so `mkcorpus.py` generates a 4 MiB stand-in of
  Zipf-distributed English-like words. OnPair training on it yields 7,652 tokens, 2,649 short and
  3,306 long dictionary entries, and 1.49M probes at `ONPAIR_BITS=16`.

```bash
python3 mkcorpus.py /tmp/synth-4mib.onpair 4194304 42
make run CORPUS=/tmp/synth-4mib.onpair BITS=16 BOOST_ROOT=/tmp/boost_1_89_0
```

## Both reported effects reproduce

Median of 15 iterations, three warmups, Mprobe/s over the full trace.

| Implementation | Mprobe/s | short_ms | long_ms |
|---|---:|---:|---:|
| hashbrown 0.16 | 111.3 | 8.29 | 5.07 |
| Rust Group15 scalar | 116.3 | 7.60 | 5.19 |
| Rust Group15 `get4` (`target-cpu=native`) | 24.9 | 6.33 | 53.38 |
| Boost, GCC 13 | 179.4 | 6.33 | 1.96 |
| Boost, Clang 18 | 127.5 | 6.65 | 5.01 |

GCC beats Clang on identical Boost source by 1.41x, matching the 1.44x geometric mean recorded in
`CODEGEN.md`. The AVX-512 `get4` pathology reproduces at 4.7x.

The gap is not spread evenly: **the short-key map is a tie (6.33 vs 6.65 ms) and the entire gap is
in the long, `u64`-keyed map (1.96 vs 5.01 ms, 2.6x).** That narrowing was the useful lead.

## `get4`: confirmed as an AVX-512 SLP cost-model problem

`asm_rust_group15_get4_long` contains exactly one `vpmullq` in the native build. Rebuilding the
same source with AVX-512 unavailable removes it and the regression:

| Rust flags | Group15 scalar | Group15 `get4` |
|---|---:|---:|
| `-C target-cpu=native` | 116.3 | 24.9 |
| `-C target-cpu=x86-64-v3` | 125.1 | 144.5 |
| `-C target-cpu=native -C target-feature=-avx512f,-avx512vl,-avx512dq,-avx512bw` | 114.9 | 141.5 |
| `-C target-cpu=native -C llvm-args=-vectorize-slp=false` | 116.6 | 139.9 |

This is steps 1-3 of the `CODEGEN.md` investigation order, and they all agree: SLP-vectorizing the
four `long_hash` multiplies into `vpmullq` is a 5.7x loss, `vpmullq` is the only difference, and
turning off either AVX-512 or SLP alone recovers it. `vpmullq` is a 3-uop, ~15-cycle-throughput
sequence on this core, so four scalar `imul`s are strictly cheaper — a plain cost-model bug, and
the cleanest thing to report upstream.

Workaround available today: build this code path at `x86-64-v3`, or disable SLP for the crate.

## Scalar: the gap is branch-misprediction sensitivity, not instruction count

The `get4` result did not explain the scalar gap, and neither did any of the usual suspects. Two
things ruled out immediately:

- **Not AVX-512 mask registers.** Clang's long loop uses `vpcmpeqb k0` + `kmovd` where GCC uses
  `vpcmpeqb` + `vpmovmskb`. Rebuilding Clang with `-mno-avx512f` produces the GCC-style sequence
  and is *no faster* (4.90 vs 4.69 ms).
- **Not block layout.** Clang 20 with full PGO on the exact benchmark trace: 4.92 ms versus 5.11 ms
  without. Profile data buys 4%; the gap is 2.5x.

What does explain it is the *predictability of the probe outcomes*. Rewriting only the long-probe
keys of the trace to a chosen hit rate (`predictability_sweep.py`), ns per long probe:

| Hit % | GCC 13 | GCC 14 | Clang 18 | Clang 20 |
|---:|---:|---:|---:|---:|
| 0 | 4.27 | 4.49 | 4.44 | 4.92 |
| 10 | 4.17 | 4.27 | 6.63 | 6.18 |
| 25 | 4.29 | 4.33 | 9.00 | 7.78 |
| 50 | 4.40 | 4.53 | **10.39** | **10.79** |
| 75 | 4.41 | 4.40 | 7.96 | 8.08 |
| 90 | 4.10 | 4.55 | 5.81 | 5.39 |
| 100 | 4.12 | 4.24 | 4.03 | 3.64 |

GCC is flat within noise at every hit rate. Clang traces a textbook misprediction curve peaking at
50%, where it is 2.4x slower — and at 100% hit it is *faster* than GCC. The Rust Group15 port shows
the same curve (2.58 / 5.26 / 1.85 ms at 0 / 50 / 100%), so this is LLVM-wide and not a C++ or
Boost-specific artifact.

This also explains why the gap looked like a constant 1.3-1.4x across the `CODEGEN.md` corpora:
their long-map hit rates are 2-73%, i.e. squarely inside the unpredictable region.

## What is not yet explained

GCC's flatness is the surprising half, and it is not simply "GCC emits fewer branches". A
standalone 130-line reproducer of the same probe shape (`probe_repro.cpp`: one metadata group,
`vpcmpeqb` fingerprint match, `tzcnt`/`blsr` candidate walk, full key compare) shows the
misprediction curve *equally* under both compilers:

| Hit % | GCC 13 | Clang 18 | Clang 20 |
|---:|---:|---:|---:|
| 0 | 2.48 | 2.29 | 2.45 |
| 50 | 8.67 | 8.59 | 8.97 |
| 100 | 2.67 | 2.90 | 2.87 |

So the mispredict sensitivity is inherent to the algorithm, and GCC's real Boost `find` is somehow
avoiding or hiding it in a way this reduction does not capture. Static differences seen in the
disassembly (GCC folds the metadata load into `vpcmpeqb`, keeps the overflow byte as a plain byte
load where Clang uses `vpextrb`, uses no stack slots) are all plausible contributors but none of
them is obviously worth 6 ns per probe.

Nailing this needs hardware counters, and **this VM exposes no PMU** — `perf stat` reports
`<not supported>` for cycles, instructions, and branch-misses. The next step is to run the sweep on
a bare-metal machine and measure `branch-misses` and `br_misp_retired.all_branches` for the two
binaries at 50% hit. If GCC genuinely mispredicts less, the question becomes which of its
scheduling choices resolves the fingerprint branch earlier; if it mispredicts equally and is still
flat, the cost is in recovery, and the fix is to shorten the dependency chain feeding the branch.

## Suggested next steps

1. Report the `vpmullq` SLP cost-model bug upstream with the `get4` reduction. It is
   self-contained, target-dependent, and worth 4.8-5.7x.
2. Ship `x86-64-v3` (or SLP off) for the OnPair parse path until that is fixed.
3. Re-run `predictability_sweep.py` on PMU-capable hardware to settle the scalar mechanism before
   attempting source-level fixes; steps 5-8 of the `CODEGEN.md` investigation order are all
   plausible but currently unfalsifiable here.
4. Independently of the compilers: since the cost is misprediction, a branch-free probe (compare
   all 15 candidates and select arithmetically, or defer the value load) should flatten the curve
   for both compilers. That is a bigger win than closing a codegen gap, and it is portable.

## Reproducing

```bash
python3 mkcorpus.py /tmp/synth-4mib.onpair 4194304 42
make all BOOST_ROOT=/tmp/boost_1_89_0
ONPAIR_BITS=16 target/native/release/trace /tmp/synth-4mib.onpair /tmp/synth16.oph
python3 predictability_sweep.py /tmp/synth16.oph \
  target/native/boost-gcc target/native/boost-clang

g++     -std=c++20 -O3 -DNDEBUG -march=native probe_repro.cpp -o /tmp/probe-gcc
clang++ -std=c++20 -O3 -DNDEBUG -march=native probe_repro.cpp -o /tmp/probe-clang
/tmp/probe-gcc && /tmp/probe-clang
```

Note that `predictability_sweep.py` reads the first `long_ms=` in a binary's output, which for the
Rust `bench` binary is hashbrown's; pass the C++ binaries for a compiler comparison.
