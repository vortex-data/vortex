# OnPair production and isolated compression results

Recorded 2026-08-31 on an Intel Xeon 6975P-C, pinned to CPU 2 with
`RAYON_NUM_THREADS=1`. All compression processes ran sequentially. Decimal GB/s is used
throughout. Quantiles are Hyndman-Fan Type 7. `p99 high` is the 99th percentile of throughput;
`p99 slow` is input bytes divided by the 99th percentile of latency.

The production comparison is the recorded `origin/develop` baseline (`eff83d41f`) versus the optimized Rust
snapshot wired through the workspace `onpair` dependency. The candidate retains develop's exact
training-row order. Hashed warmups and every measured output had the same Vortex byte count; all
six hashed output pairs were byte-for-byte identical.

These production tables predate the later medium-row tail-load experiment documented below; they
measure the larger branch-unification win and do not include that additional roughly 1% isolated
ClickBench improvement.

## Whole-file `vx convert`

Each cell has 15 measured paired AB/BA conversions after one warmup. Source GB/s is physical
Parquet bytes divided by complete conversion wall time. Uncompressed GB/s uses the sum of Parquet
footer encoded-uncompressed column bytes; it is not exact Arrow buffer size. Ratio is
Vortex bytes / Parquet encoded-uncompressed bytes.

| Dataset | Develop source GB/s median [IQR] | Snapshot source GB/s median [IQR] | Snapshot p99 high / slow | Uncompressed GB/s develop -> snapshot | Paired speedup | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Book reviews | 0.02942 [0.00012] | 0.03772 [0.00030] | 0.03786 / 0.03175 | 0.04663 -> 0.05979 | 1.280x | 0.41074 |
| Book titles | 0.03116 [0.00377] | 0.03877 [0.00012] | 0.03895 / 0.03867 | 0.05174 -> 0.06437 | 1.247x | 0.44515 |
| ClickBench `hits_0` | 0.01990 [0.00028] | 0.02356 [0.00021] | 0.02385 / 0.02138 | 0.07857 -> 0.09306 | 1.184x | 0.19466 |
| FineWeb 001 | 0.03227 [0.00010] | 0.04369 [0.00006] | 0.04382 / 0.04353 | 0.05379 -> 0.07282 | 1.354x | 0.42538 |
| FineWeb sample | 0.03230 [0.00009] | 0.04311 [0.00007] | 0.04317 / 0.04301 | 0.05207 -> 0.06950 | 1.333x | 0.44070 |
| Stack v3 | 0.01406 [0.00130] | 0.02300 [0.00674] | 0.02942 / 0.02255 | 0.04384 -> 0.07172 | 1.647x | 0.56972 |
| **Byte-weighted six-file aggregate** | **0.02928 [0.00038]** | **0.03996 [0.00067]** | **0.04087 / —** | **0.05199 -> 0.07095** | **1.372x** | — |

The aggregate speedup is the median of 15 paired total-wall ratios. It therefore need not equal
the ratio of the two independently computed median-throughput cells.

The 15-sample whole-file p99 is an exploratory tail estimate, effectively near the observed
maximum. The isolated results below use 100 samples.

## Full ClickBench

All 100 tracked Parquet partitions were converted once by each binary, with AB/BA order
alternating by file. Output sizes matched for all 100 pairs.

| Variant | Total wall | Byte-weighted source GB/s | Parquet-uncompressed GB/s | Across-file median [IQR] | Across-file p99 high | Total Vortex bytes | Vortex/uncompressed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Develop | 760.76 s | 0.01905 | 0.06558 | 0.02196 [0.00428] | 0.03411 | 11,531,859,192 | 0.23114 |
| Snapshot-greedy | 617.16 s | 0.02348 | 0.08084 | 0.02562 [0.00315] | 0.03586 | 11,531,859,192 | 0.23114 |

The byte-weighted full-dataset speedup is **1.233x**. Across-file quantiles describe variation
between partitions, not repeated-run noise.

## Production OnPair phase timing

A temporary env-gated diagnostic (`ONPAIR_PHASE_TIMING=1`) timed dictionary training and parsing inside
every production `onpair::compress` call made by `vx convert`. The process was pinned to CPU 2
with `RAYON_NUM_THREADS=1`. These are single diagnostic conversions; their purpose is attribution,
not a replacement for the repeated end-to-end matrix above.

| Dataset | Calls | OnPair input | Train | Parse | Total OnPair | Parse share | OnPair / wall | OnPair GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ClickBench `hits_0` | 710 | 0.170 GB | 0.541 s | 0.960 s | 1.501 s | 63.9% | 28.7% | 0.1135 |
| FineWeb sample | 883 | 3.781 GB | 7.385 s | 28.108 s | 35.493 s | 79.2% | 64.5% | 0.1065 |

Both diagnostic `.vortex` files retained the exact byte size and SHA-256 recorded by the earlier
candidate warmup. The call count and median input size (119 KiB for ClickBench and 129 KiB for
FineWeb) also show that production conversion invokes OnPair hundreds of times on relatively small
arrays, making allocation reuse a relevant next experiment.

### Rejected: flatten-buffer reuse

A thread-local pool reused the temporary flattened bytes and offsets between production OnPair
calls. Five balanced AB/BA ClickBench `hits_0` conversions measured 5.3158 s without reuse and
5.3096 s with reuse, a **1.001x** difference inside run noise. The output SHA-256 remained
identical. This experiment was removed; pooling only these two buffers does not justify retaining
thread-local state.

### Rejected: cheaper long-prefix filter fold

The frozen long-prefix membership filter was changed to select its bit with an XOR fold over the
four 16-bit words, deferring the full integer hash until a possible filter hit. Six balanced fresh
processes on FineWeb 32 MiB measured native parsing at 205.87 ms for the retained hashed filter and
211.25 ms for the XOR filter (**2.61% slower**). Full native compression regressed from 256.43 ms
to 260.22 ms (**1.48% slower**). Every run round-tripped. The cheaper fold was removed because its
additional filter positives cost more than the saved multiply.

### Rejected: masked-key SIMD comparison

Because short-token keys are zero above their logical length, the AVX2 comparison can replace
`(input ^ key) & mask == 0` with `(input & mask) == key`, removing one vector XOR. Six balanced
FineWeb processes measured a 1.0017x native-parse change, while full native compression moved
0.9982x. All outputs round-tripped. The result is a tie and the original expression was retained.

### Rejected: long-row hash dispatch

Completed dictionaries for inputs averaging more than 256 bytes per row were routed around the
read-only short-token buckets to the hash-map matcher. Three balanced FineWeb 32 MiB pilots measured
205.58 ms with buckets and 273.15 ms with the hash path, a **32.9% parsing regression**. The gate
was removed; the SIMD loop is hot because it avoids substantially more expensive repeated hash
probes even on long rows.

### Rejected: Group15 frozen long-prefix map

The scalar Rust Group15 table from the isolated hashmap experiment was integrated into the
snapshot's completed long-prefix map, first behind the existing 8 KiB membership filter and then
with the experiment's three-probe blocked Bloom filter at 16 bits per key. The comparison covered
FineWeb, ClickBench, Stack v3, OnPair titles, TPC-H comments, and Apache access at both 12 and 16
dictionary bits. Every cell used a 32 MiB block, one warmup, and ten sequential timed iterations
on CPU 2.

Group15 alone achieved a **0.979x** geometric-mean full-compression speedup, and Group15 plus the
blocked Bloom filter achieved **0.985x**. The 12-bit cells generally regressed by 3–7%; the 16-bit
cells were approximately neutral. All 12 cells round-tripped with exactly the same payload ratio.
The temporary integration was removed: its 37.8% isolated lookup advantage does not offset table
construction and the production snapshot's already effective short-bucket and long-miss paths.

### Retained: packed Group15 short map with blocked Bloom filter

Completed dictionaries that are too large for the direct `ShortBuckets` index now freeze their
short-token map into the workload that produced Group15's isolated win: an exactly nine-byte packed
`(u64 bytes, u8 length)` key mapped to a `u16` token. The complete entry is 12 bytes; an earlier
16-byte entry approximation regressed full compression by 2.2% geometric mean and was discarded.
The parser selects the Group15 path outside the token loop. A three-probe blocked Bloom filter at
16 bits per key rejects definite misses with one 64-bit word load before the exact table lookup.

Two order-balanced, same-binary comparisons covered six 32 MiB corpora at 16 dictionary bits, with
one warmup and ten sequential iterations per cell on CPU 2. Packed Group15 without the Bloom filter
improved full compression by **1.012x** geometric mean. Adding the blocked Bloom filter raised the
speedup to **1.040x**: FineWeb improved **1.105x**, Stack v3 **1.121x**, and ClickBench **1.023x**;
TPC-H and Apache were neutral, while OnPair titles was 0.4% slower. All payload ratios were exactly
unchanged. The 12-bit control remained on `ShortBuckets` and measured within run noise at 1.005x.

Raw results are in `target-onpair-final-results/group15-short-packed9-*.jsonl`.

### Retained: release the completed short map

When the 12-bit parser builds `ShortBuckets`, parsing no longer needs the mutable short-token hash
map. The completed map is now released and reconstructed from the buckets only if the crate-private
mutation path is subsequently used. This removes roughly 0.1--0.2 MiB per trained parser without
changing its lookup path. Six sequential 32 MiB corpus comparisons were neutral in aggregate and
produced identical dictionaries, code counts, and round trips.

A lower-memory sparse bitmap/rank directory was also tested on the same six 12-bit cells. Although
it reduced the fixed bucket-directory metadata from 256--384 KiB to about 10 KiB, the extra rank and
population-count dependency made full compression **2.1% slower** geometrically. Since the retained
speed parser uses less than a few MiB, the sparse implementation was removed.

### Retained: medium-row bounded tail loads

Partial prefix windows previously used a zero-filled eight-byte temporary. A specialized parser
path instead assembles tails with bounded 4/2/1-byte unaligned loads. Matcher selection is hoisted
outside the token loop and the two paths are const-generic specializations, so there is no
per-token mode branch. The fast path is limited to inputs averaging more than 32 and at most 256
bytes per row: shorter TPC-H rows and long FineWeb/Stack rows keep the original loader.

Eight balanced AB/BA ClickBench 32 MiB pairs measured native parsing at 93.31 ms before and
92.18 ms after, a **1.013x** paired-median speedup with 8/8 wins. Full native compression improved
from 122.68 ms to 121.48 ms (**1.010x**, 8/8 wins); full compression including 12-bit packing
improved **1.009x** with 7/8 wins. A final sequential 13-corpus screen had median speedups of
**1.005x** for native parsing and **1.003x** for full native compression. Every run round-tripped
with identical dictionary size and code count. Safe slice-to-array tail loads were also tested but
were about 0.8% slower on ClickBench than the retained bounded unaligned loads.
The commands and retained raw samples are in
[`TAIL_LOAD_RESULTS.md`](TAIL_LOAD_RESULTS.md).

OnPair itself remains strictly single-threaded. `RAYON_NUM_THREADS=1` was also used around these
benchmarks; production concurrency belongs at Vortex's higher array/block layer.

## Isolated Rust snapshot-greedy versus native C++/Boost

The isolated benchmark includes dictionary training, matcher construction, parsing, and packing.
Each cell has one warmup and 100 timed compressions in one fresh process. Every retained output is
decoded and checked. C++ was built by GCC 11.5 and uses `boost::unordered_flat_map`.

This is a speed comparison at each algorithm's own compressed ratio, not a ratio-matched speedup.
Across the 32 MiB cells, Rust payload ratio ranges from 1.03% better to 13.82% worse than C++;
the per-cell CSV linked below reports both ratios.

| Block MiB | C++ median [IQR] | Rust median [IQR] | Rust p99 high / slow | Rust/C++ median speedup |
|---:|---:|---:|---:|---:|
| 2 | 0.08254 [0.00003] | 0.13083 [0.00015] | 0.13105 / 0.12794 | 1.585x |
| 4 | 0.08558 [0.00005] | 0.13525 [0.00020] | 0.13550 / 0.13211 | 1.580x |
| 8 | 0.08726 [0.00011] | 0.13791 [0.00064] | 0.13856 / 0.13472 | 1.581x |
| 16 | 0.08904 [0.00010] | 0.13976 [0.00030] | 0.14021 / 0.13799 | 1.570x |
| 32 | 0.09268 [0.00012] | 0.14408 [0.00049] | 0.14480 / 0.14251 | 1.555x |
| 128 (two FineWeb corpora) | 0.08203 [0.00005] | 0.14204 [0.00083] | 0.14311 / 0.14063 | 1.732x |

The complete 13-corpus x 5-block x 2-algorithm table, including median, IQR, both p99 views,
and payload ratio for every cell, is in
[`isolated-32mib-summary-per-dataset.csv`](/home/ec2-user/vortex/target-onpair-final-results/isolated-32mib-summary-per-dataset.csv).
The two 128 MiB FineWeb cells are in
[`isolated-128mib-summary-per-dataset.csv`](/home/ec2-user/vortex/target-onpair-final-results/isolated-128mib-summary-per-dataset.csv).

## Raw artifacts

- [`vx-repeated.jsonl`](/home/ec2-user/vortex/target-onpair-final-results/vx-repeated.jsonl)
- [`vx-repeated-summary-per-dataset.csv`](/home/ec2-user/vortex/target-onpair-final-results/vx-repeated-summary-per-dataset.csv)
- [`vx-clickbench-full.jsonl`](/home/ec2-user/vortex/target-onpair-final-results/vx-clickbench-full.jsonl)
- [`vx-clickbench-full-summary-per-dataset.csv`](/home/ec2-user/vortex/target-onpair-final-results/vx-clickbench-full-summary-per-dataset.csv)
- [`isolated-32mib-cell100.jsonl`](/home/ec2-user/vortex/target-onpair-final-results/isolated-32mib-cell100.jsonl)
- [`isolated-32mib-summary-aggregate.csv`](/home/ec2-user/vortex/target-onpair-final-results/isolated-32mib-summary-aggregate.csv)
- [`isolated-128mib-cell100.jsonl`](/home/ec2-user/vortex/target-onpair-final-results/isolated-128mib-cell100.jsonl)
