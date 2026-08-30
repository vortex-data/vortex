# Boost Group15: GCC versus Rust/LLVM codegen

## Later immutable-filter results

The broader 32 MiB, 12-case matrix (FineWeb, ClickBench, Stack v3, OnPair titles, TPC-H
`l_comment`, and Apache access at 12 and 16 dictionary bits) produced these robust aggregate
results with scalar row-major traces, a pinned core, three warmups, and median-of-nine timing:

- Rust Group15 improved lookup throughput over hashbrown by 37.8% geometric mean.
- A blocked three-probe Bloom filter at 16 bits per dictionary key improved over Rust Group15 by
  another 14.1% geometric mean.
- GCC Boost remained approximately 16% faster than that blocked-Bloom Rust implementation across
  all 12 cases.
- The Rust blocked-Bloom implementation did beat GCC Boost on FineWeb-12 by 2.6% and Stack-v3-12
  by 6.6%; both were confirmed with median-of-15 reruns.

These are flat lookup comparisons: every implementation consumes the same short and long probe
sequences and returns the same aggregate checksum. Bloom filters may avoid an exact table access,
but do not change the logical query stream.

An audit found that an earlier version of the grouped-prefix experiment inferred token boundaries
only from descending adjacent lengths. That can merge two searches, most often on TPC-H and Apache
access. The benchmark now derives boundaries from exact dictionary hits and checks every timed
checksum. The earlier grouped-prefix performance claims are withdrawn; only a correctness smoke
run has been performed since the repair, so rerun the complete matrix before selecting a prefix
design.

The C++ lookup benchmark does not contain a timed `emplace`: reserve and insertion happen before
the measured lookup loops. The paper's full-compression benchmark does time dictionary insertion
as part of training, intentionally. It should not be compared directly with this isolated lookup
benchmark.

## Bottom line

On the scalar lookup trace, the Rust Group15 port is much better than hashbrown for the skewed
OnPair workload, but GCC's actual `boost::unordered_flat_map` remains 1.32x faster by geometric
mean. Compiling that same Boost source with Clang loses 1.44x to GCC. The GCC-versus-Clang control
is the strongest evidence that most of the remaining scalar gap is backend code generation, not
Rust semantics or probe behavior.

The four-at-a-time path exposes a second, larger LLVM problem. With `target-cpu=native`, LLVM 21
SLP-vectorizes four 64-bit hash multiplies to AVX-512 `vpmullq`, creates a 1,900-byte hot function
with an 0x118-byte stack frame and 84 stack references, and falls to 52 Mprobe/s on FineWeb-16.
With `target-cpu=x86-64-v3`, the same Rust source reaches 253 Mprobe/s. This is the first issue to
fix or report upstream.

## Measurement setup

- CPU: Intel Xeon 6975P-C, 16 cores / 32 threads, AVX2 and AVX-512 available.
- Rust: rustc 1.91.0, LLVM 21.1.2.
- GCC: 11.5.0 20240719 (Red Hat).
- Clang: 15.0.7 (AWS).
- Boost: 1.89.0.
- C++ flags: `-std=c++20 -O3 -DNDEBUG -march=native`.
- Rust flags: `-C target-cpu=native`, one codegen unit, fat LTO, aborting panics.
- Input: parsing-derived trace from the first 4 MiB of each real corpus.
- Timing: three warmups, median of 15 iterations; insertion, parsing, allocation, trace loading,
  and correctness checks are outside the timed region.
- Units below: million lookups per second (Mprobe/s). Higher is better.

The trace contains both OnPair lookup shapes: a packed 9-byte `(u64, u8)` short key and a `u64`
long-prefix key. Both maps store `u16` token IDs. A missing key returns zero. Every Rust result is
checked against hashbrown before timing.

## Scalar results and native `get4` pathology

| Bits | Dataset | hashbrown | Rust G15 | Rust G15 `get4` | Boost GCC | Boost Clang | GCC / Rust | GCC / Clang |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 12 | FineWeb | 228.8 | 265.1 | 62.8 | 320.9 | 241.3 | 1.21x | 1.33x |
| 12 | ClickBench | 165.1 | 303.1 | 55.4 | 405.6 | 267.7 | 1.34x | 1.52x |
| 12 | Stack v3 | 113.3 | 323.1 | 68.4 | 361.4 | 281.3 | 1.12x | 1.29x |
| 12 | OnPair titles | 216.0 | 285.4 | 46.3 | 378.1 | 253.8 | 1.32x | 1.49x |
| 12 | TPC-H `l_comment` | 152.6 | 156.1 | 32.3 | 222.1 | 156.4 | 1.42x | 1.42x |
| 12 | Apache access | 122.0 | 271.2 | 44.1 | 379.2 | 244.4 | 1.40x | 1.55x |
| 16 | FineWeb | 193.6 | 221.1 | 52.2 | 277.6 | 204.2 | 1.26x | 1.36x |
| 16 | ClickBench | 236.9 | 258.3 | 40.7 | 357.1 | 233.6 | 1.38x | 1.53x |
| 16 | Stack v3 | 134.9 | 290.0 | 59.4 | 367.0 | 257.3 | 1.27x | 1.43x |
| 16 | OnPair titles | 249.3 | 271.4 | 35.0 | 392.8 | 245.8 | 1.45x | 1.60x |
| 16 | TPC-H `l_comment` | 134.3 | 144.6 | 28.2 | 195.7 | 146.5 | 1.35x | 1.34x |
| 16 | Apache access | 191.5 | 202.9 | 32.7 | 278.0 | 193.0 | 1.37x | 1.44x |

Geometric means: GCC/Rust-G15 1.32x, GCC/Clang-Boost 1.44x, and GCC/hashbrown 1.86x. The native
`get4` column should not be interpreted as an algorithm result; it is the AVX-512 regression.

FineWeb-16 is a compact reproducer for the target-dependent `get4` behavior:

| Rust target | hashbrown | Rust G15 scalar | Rust G15 `get4` |
|---|---:|---:|---:|
| `native` | 194.2 | 221.2 | 52.2 |
| `x86-64-v3` | 203.0 | 219.1 | 253.3 |

## Probe behavior

Probe counts cannot explain the GCC gap. The Rust port normally reads one 16-byte metadata group;
the worst average is 1.057 groups per long lookup. `Cmp/probe` counts full-key comparisons after
the one-byte fingerprint match. `Hit` is the long-map hit rate.

| Bits | Dataset | Groups/probe | Cmp/probe | Hit |
|---:|---|---:|---:|---:|
| 12 | FineWeb | 1.040 | 0.065 | 2.1% |
| 12 | ClickBench | 1.037 | 0.380 | 34.2% |
| 12 | Stack v3 | 1.037 | 0.119 | 7.1% |
| 12 | OnPair titles | 1.045 | 0.365 | 32.0% |
| 12 | TPC-H `l_comment` | 1.000 | 0.674 | 64.8% |
| 12 | Apache access | 1.000 | 0.464 | 44.8% |
| 16 | FineWeb | 1.009 | 0.183 | 14.8% |
| 16 | ClickBench | 1.000 | 0.572 | 55.3% |
| 16 | Stack v3 | 1.000 | 0.199 | 17.1% |
| 16 | OnPair titles | 1.001 | 0.599 | 57.6% |
| 16 | TPC-H `l_comment` | 1.001 | 0.750 | 73.4% |
| 16 | Apache access | 1.057 | 0.638 | 59.5% |

## What is held equivalent

The Rust Group15 table copies the non-concurrent Boost FOA lookup design:

- 16 aligned metadata bytes: 15 reduced-hash fingerprints plus one overflow byte;
- the same short and long hash formulas and constants;
- the same high hash bits for the initial group;
- SIMD byte comparison followed by a candidate bit mask;
- quadratic overflow probing and one overflow bit selected by `hash & 7`;
- no insertions or deletions in the timed section.

One layout difference remains: Boost packs 15 entries per group (240 bytes for these 16-byte
entries), while the Rust experiment reserves 16 entries (256 bytes) so group addressing is a
shift. The Rust layout has simpler addressing but is 6.7% less dense. It should be tested after
the codegen issues, but it does not explain GCC beating Clang on the identical Boost layout.

Hashbrown is not a compiler control. It uses SwissTable metadata, a different capacity policy,
foldhash, empty-byte detection, and different group walking. It is included only as the current
Rust baseline.

## Hot-loop assembly differences

Native symbol summary:

| Function | Bytes | Instructions | Stack references | Calls |
|---|---:|---:|---:|---:|
| Rust hashbrown scalar | 290 | 85 | 4 | 0 |
| Rust Group15 scalar | 343 | 92 | 5 | 1 |
| Rust Group15 `get4` | 1,900 | 410 | 84 | 5 |
| Boost GCC scalar | 307 | 85 | 0 | 0 |
| Boost Clang scalar | 276 | 76 | 2 | 0 |

The counts are static disassembly counts, not executed-instruction counts.

GCC's scalar Boost loop does several things the Rust/LLVM loop does not:

1. It hoists the `0xd6e8feb86659fd93` hash constant before the loop. Rust/LLVM and Clang load the
   constant inside the outer loop.
2. It uses BMI2 `shrx`, avoiding the legacy variable-shift dependency on `cl` seen in non-native
   Rust builds.
3. It broadcasts the wanted fingerprint directly from the lookup table and folds the metadata
   memory operand into `vpcmpeqb`. Rust commonly materializes metadata and the wanted value in
   separate registers.
4. It keeps overflow probing inline. Rust calls a cold helper, spilling loop state around the ABI
   boundary. Clang keeps Boost's overflow path inline and still loses to GCC, so this is only part
   of the scalar gap.
5. It uses `blsr` to clear the lowest candidate bit. LLVM often emits `lea` plus `and`.
6. It needs no stack references in the exported loop. Rust scalar uses five; Clang Boost uses two.

Clang Boost is smaller than GCC Boost but slower on every dataset. Code size alone is therefore
not the objective; dependency chains, load folding, invariant motion, and spill behavior matter.

For native Rust `get4`, LLVM vectorizes the four hashes with `vpmullq`, then retains four metadata
vectors, masks, keys, positions, and overflow states at once. Every possible cold overflow call
forces a large live set to the stack. The emitted function has an 0x118-byte frame and spills YMM
and XMM values around those calls. The x86-64-v3 build cannot select AVX-512 `vpmullq` and is 4.8x
faster on FineWeb-16.

## LLVM investigation order

Use FineWeb-16 first; a 4 MiB input gives millions of probes while remaining quick to rebuild.

1. Reproduce `native` versus `x86-64-v3` exactly as shown in `README.md`. Confirm `vpmullq` exists
   only in the slow native `asm_rust_group15_get4_long`.
2. Build native with AVX-512 disabled while preserving the rest of the native CPU:

   ```bash
   CARGO_TARGET_DIR=target/no-avx512 \
   RUSTFLAGS='-C target-cpu=native -C target-feature=-avx512f,-avx512vl,-avx512dq' \
     cargo build --release --manifest-path Cargo.toml
   ```

   If this matches x86-64-v3, the regression is isolated to the SLP/AVX-512 decision rather than
   unrelated target tuning.
3. Try `-C llvm-args=-vectorize-slp=false`. If that fixes `get4`, reduce the reproducer to the four
   `long_hash` calls plus four independent loads and file an LLVM missed-cost-model issue.
4. Break only the hash SLP pack: move the scalar hash into an out-of-line helper, add an optimizer
   barrier between lanes, or compute hashes in a small separate loop. These are diagnostic changes;
   keep one only if it preserves the scalar path and wins across all corpora.
5. Then address scalar codegen. Manually keep the hash multiplier, metadata pointer, entry pointer,
   shift, mask, match-word base, and overflow-bit base in locals outside the probe loop. Check
   whether LLVM actually hoists them; source-level hoisting is not enough.
6. Compare an inline overflow loop against the current cold helper. The goal is to eliminate the
   call and spill/reload sequence without bloating the overwhelmingly common one-group path.
7. Test replacing the match-word table load with `_mm_set1_epi8(reduced_hash as i8)`, and test a
   direct metadata-byte load for overflow. Keep each change isolated.
8. Test 15-entry versus 16-entry storage only after codegen parity. This separates cache density
   from compiler effects.
9. Try PGO last. It may place overflow code better, but it should not be needed to hoist constants
   or avoid a catastrophically unprofitable vectorization.

Generate inspectable compiler output with:

```bash
CARGO_TARGET_DIR=target/inspect \
RUSTFLAGS='-C target-cpu=native -C save-temps' \
  cargo rustc --release --manifest-path Cargo.toml --bin bench -- --emit=asm,llvm-ir
```

Use `make asm`, `perf stat`, and `llvm-mca` on the exported symbols. Useful counters include cycles,
instructions, branches, branch misses, L1 data misses, and stalled frontend/backend cycles. Always
record the checksum-equivalent correctness pass and the exact compiler flags with a result.

## Success criteria

- Scalar Rust Group15 is within 5% of Boost GCC by geometric mean across all 12 cases.
- `get4` never regresses below scalar on any corpus and retains its x86-64-v3 gain under `native`.
- Probe counts, lookup results, hash formulas, and table capacity remain unchanged while evaluating
  compiler-only fixes.
- Any table-layout change is reported separately from a codegen change.
