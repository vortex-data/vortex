# Morsel Prototype: Handoff

Everything needed to re-run the morsel-executor evaluation on other hardware and interpret what
comes back. All the code is on branch `claude/morsel-executor-prototype-vvrscx`.

The original numbers came from a 4-core host with memory-backed segments. The latest rerun used a
**16-core/32-thread Intel Xeon 6975P**, pinned to CPUs 0–15 (one hardware thread from each physical
core), with the same generated segment pack read from XFS. Both hot page-cache and advisory-cold
`POSIX_FADV_DONTNEED` results are recorded below.

## 1. Get it running

```bash
git fetch origin claude/morsel-executor-prototype-vvrscx
git checkout claude/morsel-executor-prototype-vvrscx
cargo build --release -p vortex-morsel --features _test-harness --bins
```

Needs nothing external: TPC-H data is generated in-process by `tpchgen`, which was already a
workspace dependency.

```bash
# Correctness. 24 tests, including differential tests against the V1 LayoutReader.
cargo test -p vortex-morsel

# Real TPC-H at SF=1. ~1 min including generation and write.
./target/release/tpch-eval 1

# Bigger. Memory scales roughly 1.5 GB per scale factor; SF=10 wants ~24 GB.
./target/release/tpch-eval 10

# Thread scaling, V1 concurrency tuning, morsel-size sweep.
TPCH_SWEEP=1 ./target/release/tpch-eval 1

# The synthetic workloads (string-heavy / wide-numeric / narrow-analytic).
MORSEL_EVAL_ROWS=1000000 ./target/release/morsel-eval

# Real on-disk reads, hot and advisory-cold.
TPCH_DISK_PATH=target/tpch-morsel-sf1.segments TPCH_CACHE_MODE=hot \
  taskset -c 0-15 ./target/release/tpch-eval 1
TPCH_DISK_PATH=target/tpch-morsel-sf1.segments TPCH_CACHE_MODE=cold \
  taskset -c 0-15 ./target/release/tpch-eval 1
```

Knobs: `TPCH_SCALE`, `TPCH_ROW_BLOCK` (default 8192, the write pipeline's repartition size),
`TPCH_BLOCK_BYTES` (default 1 MiB, the coalescing target — **this is what decides how many
natural splits the file has**, so it is the first thing to vary if you want more morsels),
`TPCH_DISK_PATH`, `TPCH_CACHE_MODE={hot,cold}`, `TPCH_QUERY`, `TPCH_ITERATIONS`,
`TPCH_MORSEL_ONLY=1`, and `MORSEL_EVAL_ROWS`.

The primary read-side morsel is **131,072 rows (128k rows)**. This does not rewrite or repartition
the file: the on-disk layout still uses the 8192-row write repartition and 1 MiB coalescing target
above. Each complete column stream passes through one strategy invocation, so the coalescer and
2 MiB `BufferedStrategy` see across the generated 65,536-row input batches. Morsels are only
row-range cuts made while reading the resulting layout.

The SF=1 XFS pack contains 1,789 logical segments and 174,410,852 payload bytes. Compressed segment
sizes are 3,780/102,396/393,492 bytes min/median/max; the 1 MiB target is based on uncompressed
input, so compressed segments are expected to be smaller. `BufferedStrategy` keeps several chunks
near one another while writing but does not merge them into one segment. The aligned raw benchmark
pack is one contiguous XFS extent; it is a segment payload pack, not a complete Vortex file with a
footer.

## 2. What the eval guarantees

`tpch-eval` validates **before it times anything**: every configuration's output is compared to
V1's on dtype, row count and ordered content, and a mismatch aborts the run rather than quietly
dropping a row from the table. If you see a timing table, the exactness check passed for every
row in it. Then five alternating iterations are run by default, with median and min/max reported.

The morsel executor rejects at build time anything outside its scope (nested structs, non-struct
roots, nullable root structs, non-flat/non-chunked columns), so an unsupported query can never be
timed as if it had run.

## 3. Current executor and measured results

The scheduler has one affinity-owned active morsel per worker. Its arena and partial operator state
never migrate. Planning registers keyed cells and divides them into required and speculative
batches. Speculative batches enter the shared normal-priority I/O queue immediately; required
batches remain dormant until execution asks for one of their tickets.

`ExecCx::ready` is the only inline I/O point. For a local file it calls
`preadv2(..., RWF_NOWAIT)`: a page-cache hit returns the segment synchronously, while `EAGAIN`
creates the normal segment futures, suspends the morsel on its exact ticket, and promotes the
whole required batch to the shared urgent queue. Other workers poll that queue while their own
morsels are suspended. Completion wakes only the ticket owner; stale generation/epoch wakes are
ignored. Filesystem/source lack of `RWF_NOWAIT` support is remembered scan-wide, after which
required reads take the background path directly. `execute` therefore makes an explicitly
non-waiting syscall, but never polls a background future, performs a blocking read, or parks the
worker.

One 128k-row morsel supplies substantial I/O concurrency: depending on the query it names about
5–16 logical segment uses, creates about 3.5–14 new requests, and groups them into one or two
batches. A cold miss on the first required ticket submits the complete required batch, not only
that ticket. In
the x16 cold runs, every 128k morsel blocked about 1.8–2.7 times. `POSIX_FADV_DONTNEED` is
advisory, so some queries retained pages and made more than one NOWAIT attempt before falling back.

Hot XFS results from two complete five-iteration runs, using one thread per physical core and the
128k-row primary morsel (median range across the two runs):

| query | best V1 x16 | morsel x16/128k | result |
|---|--:|--:|---|
| Q6 | 6.976–7.543 ms | 4.325–4.409 ms | morsel wins |
| Q1 | 5.736–6.199 ms | 6.406–6.606 ms | V1 wins by 3–15% |
| Q14 | 5.806–5.890 ms | 3.945–4.133 ms | morsel wins |
| Q15 | 4.918–5.154 ms | 3.886–3.928 ms | morsel wins |
| Q12 | 7.720–7.919 ms | 4.420–4.495 ms | morsel wins |
| Q19 | 15.855–17.348 ms | 5.380–5.423 ms | morsel wins by about 3x |
| scan-6col | 3.256–3.389 ms | 2.192–2.331 ms | morsel median wins, but is noisy |
| selective | 4.469–4.611 ms | 2.462 ms | morsel wins |

The corrected whole-column writer invocation matters independently of NOWAIT: for the six-column
scan it reduced stored segments from 552 to 276 and moved its hot median from 3.04–3.91 ms to
2.19–2.33 ms. Q1 is still the only repeatable hot loss because only its predicate reads hit inline;
its projection remains speculative background work. It is the next CPU-profile target.
`scan-6col` still has extreme iteration noise (about 1.4–11.2 ms), so its median advantage should
not be overinterpreted.

The earlier x32 SMT sweep used 64k morsels and predates the inline `RWF_NOWAIT` path, so those
numbers are no longer a like-for-like answer to the thread-count question. The all-core sweep must
be rerun with this implementation before making an SMT recommendation.

A representative final hot x16 run gives this read shape. “Physical bytes” means bytes returned
by successful file-reader requests, not block-device traffic: a successful inline page-cache
probe counts, while an `EAGAIN` probe does not. Background reads can coalesce and over-read;
inline hits are exact segment ranges.

| query | V1 / morsel physical bytes | NOWAIT hits / background pending polls | blocks per morsel | interpretation |
|---|--:|--:|--:|---|
| Q6 | 49.39 / 36.53 MB | 115 / 46 | 0.63 | predicates hit inline; speculative projection reads run in background |
| Q1 | 40.30 / 53.45 MB | 23 / 368 | 2.02 | projected columns remain fragmented background work |
| Q14 | 53.74 / 47.16 MB | 23 / 138 | 1.52 | inline predicate plus background projection saves bytes |
| Q15 | 51.13 / 43.11 MB | 23 / 138 | 1.50 | same two-stage shape as Q14 |
| Q12 | 42.05 / 40.46 MB | 69 / 136 | 0.63 | required predicates hit inline; projections run in background |
| Q19 | 43.04 / 53.68 MB | 46 / 596 | 1.85 | required hits plus a wide speculative projection |
| scan-6col | 59.93 / 59.93 MB | 276 / 0 | 0.00 | all six columns are hot inline hits; no futures or suspension |
| selective | 65.77 / 51.28 MB | 115 / 92 | 0.98 | inline predicates and background projections |

The advisory-cold runs are storage-bound and noisy. Across two five-iteration reruns, best V1 versus
x16/128k morsel median ranges were: Q6 264.58–264.68/262.76–262.99 ms,
Q1 304.42–305.47/302.48–303.52 ms, Q14 335.95–336.97/329.89–332.07 ms,
Q15 313.71–314.37/307.20–308.20 ms, Q12 302.55–305.93/302.88–303.48 ms,
Q19 329.05–329.16/327.20–327.45 ms, `scan-6col` 456.95–457.28/453.66–454.70 ms, and
`selective` 343.03–343.08/342.17–342.74 ms. That is effectively parity to a small morsel win, as
expected when storage dominates. `fadvise` is advisory, so occasional hits and hot outliers remain;
use medians and inspect min/max.

For the like-for-like all-core comparison, divide each V1 x16 median by the corresponding
x16/128k morsel median. The ranges across the two independent five-iteration runs are:

| query | hot V1-to-morsel speedup | cold V1-to-morsel speedup |
|---|--:|--:|
| Q6 | 1.61–1.71x | 1.007–1.008x |
| Q1 | 0.87–0.97x | 1.005–1.010x |
| Q14 | 1.40–1.49x | 1.016–1.018x |
| Q15 | 1.27–1.31x | 1.018–1.025x |
| Q12 | 1.75–1.76x | 1.013x |
| Q19 | 2.92–3.22x | 1.012–1.015x |
| scan-6col | 1.45–1.49x | 1.009–1.017x |
| selective | 1.82–1.87x | 1.009–1.011x |

Q1 remains the only hot regression. Every cold result is within 2.5% of parity.

This storage-bound conclusion was checked against raw bytes, not inferred only from V1 parity. The
XFS device reports as Amazon Elastic Block Store. After one short 528 MB/s cache/short-window
sample, four consecutive cache-bypassing sequential reads of 166 MiB stabilized at
1.32556–1.32627 seconds, or 125.22 MiB/s:

```bash
dd if=target/tpch-morsel-streamed-sf1.segments of=/dev/null \
  bs=1M count=166 iflag=direct status=none
```

Dividing each query's exact logical segment bytes by its x16/128k cold median gives:

| query | cold throughput | fraction of raw direct throughput |
|---|--:|--:|
| Q6 | 125.43 MiB/s | 1.002 |
| Q1 | 125.40 MiB/s | 1.001 |
| Q14 | 125.89 MiB/s | 1.005 |
| Q15 | 125.87 MiB/s | 1.005 |
| Q12 | 124.74 MiB/s | 0.996 |
| Q19 | 125.43 MiB/s | 1.002 |
| scan-6col | 125.70 MiB/s | 1.004 |
| selective | 124.99 MiB/s | 0.998 |

The single stream did not by itself prove the maximum aggregate bandwidth, so a second direct-I/O
probe issued synchronous `pread` calls from increasing numbers of threads. Each large-read point
transferred 2 GiB; the two long 128 KiB points transferred 4 GiB:

| direct-read shape | aggregate throughput |
|---|--:|
| 1 x 1 MiB | 133.02 MiB/s |
| 2 x 1 MiB | 125.06 MiB/s |
| 4 x 1 MiB | 124.89 MiB/s |
| 8 x 1 MiB | 125.06 MiB/s |
| 16 x 1 MiB | 125.08 MiB/s |
| 32 x 1 MiB | 125.08 MiB/s |
| 64 x 1 MiB | 125.06 MiB/s |
| 4 x 128 KiB, 4 GiB | 128.95 MiB/s |
| 256 x 128 KiB, 4 GiB | 128.75 MiB/s |

Short 1 GiB trials reached 142.44 MiB/s, but the rate returned to 128.75–128.95 MiB/s over 4 GiB.
More concurrency therefore does not unlock additional sustained bandwidth. The machine is an
[`m8i.8xlarge`](https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html), whose 1,250 MB/s EBS
attachment is rated far above this result; the plateau is at the attached volume or its provisioned
throughput, not the instance interface. The exact volume configuration could not be queried without
AWS credentials, but the measured 125 MiB/s plateau exactly matches the
[default gp3 baseline](https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html).

The query paths are within 0.5% of the stable 125.22 MiB/s single-stream rate and within 3% of the
long-window parallel maximum. CPU decode, scheduler, request fragmentation, and output assembly are
hidden under I/O on this volume. Cold wall time can only improve materially here by reading fewer
device bytes (for example, equal statistics pruning) or by provisioning faster storage; changing
executor scheduling cannot exceed this volume ceiling.

Cold time to first computed batch was 30.5–95.0 ms for the morsel path, still earlier than V1 but
not instant streaming. Output is collected and reordered at the end, so TTFB measures internal
readiness rather than delivery to a streaming consumer.

The stable accounting invariant remains: `decodes + reuses` with sharing equals the work without
sharing for the corresponding query. A dedicated test also proves 15 straddled stored segments
produce exactly 15 source requests across four workers when decoded sharing is disabled.

## 4. What is not covered

- **Statistics pruning is disabled for V1 until the morsel executor implements the same pruning.**
  Zone maps and dictionary layout are therefore disabled for both executors in these results. V1
  supports them and P1 does not; enabling them only for V1 would compare pruning capability rather
  than executor behavior. Keep them disabled on both paths for like-for-like measurements, then
  enable them on both in the same benchmark once morsel execution can consume the statistics. On
  selective queries, V1 with zone maps would otherwise skip blocks the prototype must read.
- **Local file I/O is covered; object storage is not.** The latest results use real positional
  reads from XFS, but the prototype plan's latency grid of {0,1,10,50} ms is not built.
- **Gate E1 as written cannot be evaluated in this repository.** It requires rows B and C — the
  self-paced graph/reactor and pipeline executors — and neither exists at any commit reachable
  here (`self_paced`, `morsel`, `vortex-scan-v2` all find nothing). If those exist on a branch
  elsewhere, running row C against these same fixtures is the highest-value next measurement.
- **`lineitem` only.** The joins in Q12/Q14/Q15/Q19 are above the scan.
- **Successful file-reader bytes are counted for both executors.** This includes hot page-cache
  hits and therefore is not a block-device byte counter. Background reads are counted after
  coalescing; successful inline NOWAIT reads are exact segment ranges and failed probes count only
  in the NOWAIT-miss column.
- ClickBench and FineWeb still need multi-gigabyte downloads and remain synthetic
  (`morsel-eval`). Their absolute times are not comparable to any published suite number.

## 5. Code map

| path | what |
|---|---|
| `vortex-morsel/src/node.rs` | The `ExecNode` contract, exact wait sets, and retry propagation |
| `vortex-morsel/src/nodes/` | FLAT, CHUNKED, STRUCT, CONJUNCT (cascade/parallel), FILTER |
| `vortex-morsel/src/io.rs` | Scan-wide raw cells plus each morsel's local ticket view |
| `vortex-morsel/src/cells.rs` | Leased shared decoded cells (lease counts from the morsel cut) |
| `vortex-morsel/src/build.rs` | `ExecPlan`: immutable blueprint, per-thread instantiation |
| `vortex-morsel/src/driver.rs` | Worker affinity, shared I/O queues, ticket wakeups, ordering |
| `vortex-morsel/src/tpch.rs` | Real TPC-H generation, queries, write strategy |
| `vortex-morsel/src/harness.rs` | Fair-comparison harness, V1 and morsel runners |
| `vortex-morsel/src/bin/tpch-eval.rs` | The TPC-H evaluation and sweep |
| `vortex-morsel/src/bin/morsel-eval.rs` | The synthetic evaluation |
| `vortex-io/src/std_file/read_at.rs` | Linux `preadv2(RWF_NOWAIT)` implementation |
| `vortex-file/src/segments/source.rs` | Exact segment-range inline probe adapter |

Design context: [morsel-based plan execution](morsel-based-plan-execution.md),
[graph model](scan-execution-graph-model.md). Results:
[TPC-H findings](morsel-prototype-tpch-findings.md),
[P1 findings](morsel-prototype-p1-findings.md).

## 6. If you are picking this up

In rough order of value:

1. **Profile Q1 hot execution on a quiet, profiler-capable host.** Its predicate reads now hit
   inline, but six speculative projection reads per natural unit still traverse background
   futures and coalescing; separate scheduler/assembly cost from that request shape.
2. **Run a full on-disk thread sweep.** `TPCH_SWEEP=1` still rejects `TPCH_DISK_PATH`; add the disk
   backend to the sweep and measure 1/2/4/8/16 physical cores plus SMT before claiming x16 is
   optimal.
3. **Bound scan-wide raw-cell retention by bytes.** The current service deduplicates correctly but
   retains completed raw buffers until scan teardown.
4. **Stream ordered output with a bounded reorder buffer.** Results are currently sorted after all
   workers finish, so measured TTFB is internal readiness rather than consumer-visible delivery.
5. **Add real zone-map pruning to morsel execution**, then enable statistics pruning for V1 and
   morsel together in the same benchmark. A pass-through node is useful for layout compatibility,
   but is not sufficient reason to enable V1's pruning in a performance comparison.
6. **Build the latency-injection segment source** for gate E2. The IO plane already carries
   `source_range`, `extent`, `producer` and `estimated_bytes`; nothing reads them yet, and the
   latency grid is what makes them earn their place.
7. **A wider schema than `lineitem`.** Decode sharing and morsel coalescing both looked neutral
   here specifically because the write pipeline aligns every column. Q19 shows what happens when
   it does not.
