# Compressor selection feasibility study

Checks whether a learned scheme-selection system (see `benchmarks/compressor-lab/README.md`) is
worth building before building it. Integers only.

## Method

`compressor-feasibility` compresses each integer chunk (65,536 rows) with the production
compressor (`BtrBlocksCompressorBuilder::from_session`) and with variants built by wrapping the
stock schemes. Nothing in the compressor itself is modified.

- **`forced/<scheme>`** puts that scheme at the root and lets production choose everything
  below. The best forced root is a **one-step oracle**: a lower bound on the headroom.
- **`nosample`** replaces every "sample me" estimate with a closed-form one computed from stats.
- **`capoff/<scheme>`** turns that scheme's heuristic skip (Dict > 50% distinct, Sparse < 90% top
  value, the RunEnd and RLE run thresholds) into "sample and let the sample decide".

For each variant the harness records the serialized size (buffers plus metadata), compression
time, decode-to-canonical time (median of 7) and the encoding tree. It also records per-chunk
features. `analyze.py` answers the four questions.

Corpus: 1,485 chunks (352.6 MB canonical), up to 16 chunks per column.

- ClickBench `hits_0.parquet`, 72 integer columns.
- NYC taxi 2023-11.
- TPC-H SF1: lineitem, orders, partsupp and customer.

```bash
cargo build --release -p compressor-feasibility
./target/release/compressor-feasibility --data-dir <parquet dir> --tpch-sf 1 \
    --max-chunks-per-column 16 --decode-reps 7 --out <out>
uv run --no-project --with pandas --with scikit-learn \
    python benchmarks/compressor-feasibility/analyze.py <out>
```

## Results

### 1. Is there much to gain over the current compressor?

**Some, and it's concentrated in a few columns.**

| | Size | Ratio |
|---|---|---|
| Production | 62.95 MB | 5.60× |
| One-step oracle (best root, production children) | 58.11 MB | 6.07× |
| **Headroom** | **7.7%** (a lower bound) | |

| Source | Headroom |
|---|---|
| ClickBench `hits_0` | 5.3% |
| NYC taxi | 2.6% |
| TPC-H lineitem | 15.4% |
| TPC-H partsupp | 33.2% |
| TPC-H orders and customer | 0% |

- On 76% of chunks production is already optimal (within 0.1%). 11% of chunks would shrink by
  more than 20%.
- The winners are almost all **RunEnd (168 chunks), Sparse (112) and Dict (35)**.
- Worst misses:
  - `ps_partkey`: production uses FOR at 1.4 MB; RunEnd with both children as Sequence is
    3.4 KB.
  - `l_orderkey`: −49%.
  - ClickBench `ConnectTiming` −36%, `RefererCategoryID` −24%, `FetchTiming` −21%.
  - Taxi `RatecodeID` −35%.
- Pco would save another ~40% of bytes but decodes **5.9× slower**. It stays excluded by
  default for that reason.

**The misses are estimation errors, not caps.** Where forced RunEnd wins by more than 5%, the
median average run length is 38, far above the RunEnd threshold. So RunEnd was sampled and lost
on the 1% sample, but it wins when the whole chunk is compressed. Removing the caps recovers only
11–20% of these chunks. The stratified sample misjudges schemes whose children cascade well
(RunEnd ends and values that become Sequence or FOR, the codes of a skewed Sparse).

A simple rule (trial-compress RunEnd, Sparse and Dict in full when `sorted_frac >= 0.99` or
`top1 >= 0.5`) captures 77% of the headroom (−6% bytes). But it costs **+204% compression
time**, so it isn't viable as is.

### 2. Can we choose faster without a compression estimate?

**Partly. Today's quality without sampling at the root is reachable, but it doesn't yet
generalise across datasets.**

- **Time:** sampling and deferred estimates are **19%** of compression time (411 → 509 MB/s
  single-threaded without them). That's the ceiling for "pick faster".
- **Closed-form estimates instead of sampling:** +3.2% bytes. They pick the same root on 61% of
  chunks.
- **A decision tree on features only:**

| | Bytes vs production | Regret vs oracle |
|---|---|---|
| Depth 4, 5-fold split grouped by column | **−0.3%** | 8.0% |
| Depth 3, leave-one-source-out | +79% | 94% |

  With a split grouped by column, the tree matches production without sampling at the root.
  With leave-one-source-out it fails badly: there are too few sources to generalise across
  datasets. The features carry the signal; the corpus is too small.

### 3. Does removing specific caps help?

**No. Keep the caps.**

| Caps removed | Bytes | Decode | Compress (1 rep) | Chunks changed |
|---|---|---|---|---|
| all | −1.7% | +3.6% | **+231%** | 7.1% |
| dict | +0.2% | −1.7% | +8% | 1.1% |
| sparse | **+1.2%** | +0.5% | +47% | 3.9% |
| runend | −0.1% | +3.9% | +16% | 3.4% |
| rle | −0.0% | −1.4% | +21% | 1.6% |

Removing the caps individually is neutral or worse, because without them the inaccurate sample
picks worse trees. Removing all of them buys 1.7% for 3.3× the compression time.

### 4. Can we combine decompression speed with compression ratio?

**Yes, and it matters a lot, but only when bandwidth is high.** Cost = bytes ÷ bandwidth +
decode time, picking per chunk among the production and forced trees:

| Bandwidth | Cost vs production | Bytes | Decode | Trees changed | Size-only oracle cost |
|---|---|---|---|---|---|
| S3 ~100 MB/s | −7.3% | −7.4% | −6.9% | 23% | −7.0% |
| NVMe ~2 GB/s | **−25.8%** | +14% | **−54%** | 43% | −1.8% |
| Memory ~20 GB/s | **−60.5%** | +81% | **−70%** | 55% | +1.7% |

- Within a chunk, size and decode time are almost uncorrelated (median r = −0.12). The smallest
  tree is rarely the fastest to decode.
- At object-store bandwidth, size-only is right. At NVMe or in-memory bandwidth, a
  decode-aware objective cuts scan cost by 26–60%, while a size-only oracle barely helps.

## Verdict

**Don't build the full system yet. Build two targeted pieces, which are justified by the data.**

1. **Fix estimation for cascading schemes.** About 6–8% bytes are available, almost entirely
   RunEnd, Sparse and Dict misranked by the 1% sample. Try contiguous or larger samples, or
   estimate the children's sizes when sampling. Full trials prove the gain but cost 2–3×
   compression time.
2. **Add a decode-aware objective** (presets per bandwidth). This is the largest effect found:
   −26% scan cost at NVMe bandwidth and −60% in memory. It needs reliable decode and pushdown
   timings, which is the measurement half of the lab design.
3. **Defer the learned selector.** It can match today without sampling at the root, worth at most
   ~19% of compression time, but it doesn't generalise across sources yet. Revisit after
   items 1 and 2, with a much larger and more diverse corpus.

## Follow-up: why the sample misranks schemes, and a fix

The harness's `est/*` and `model/*` variants replace each scheme's estimate with a custom one.
The custom estimator reproduces production: `est/prod` is within 0.16% of default and picks the
same tree on 87% of chunks.

| Estimator | Bytes vs production | Headroom captured | Compress time | Decode time |
|---|---|---|---|---|
| Production (16 slices × 64 values) | 0 | 0% | 1.00× | 1.00× |
| Zero-byte samples allowed to compete | +0.2% | −2% | 1.00× | 0.97× |
| Sizes include metadata (16 × 64) | +2.4% | −32% | 1.42× | 0.99× |
| One contiguous 4,096-value window | −3.1% | 41% | 1.30× | 1.07× |
| 4 slices × 1,024 values | −3.2% | 42% | 1.32× | 1.08× |
| **Size model for RunEnd and Sparse** | **−6.9%** | **89%** | **1.01×** | 1.37× |
| Size model + 4 × 1,024 sample for the rest | −7.3% | 95% | 1.15× | 1.59× |

Per source, the size model captures most of the available headroom:

| Source | Size model | Oracle |
|---|---|---|
| ClickBench | −4.1% | −5.3% |
| Taxi | −2.4% | −2.6% |
| TPC-H lineitem | −15.4% | −15.4% |
| TPC-H partsupp | −33.2% | −33.2% |

Findings:

- **The misses are closed-form estimates and caps that ignore value widths.**
  - RunEnd pays off at an average run of only ~2 when values are wide. ClickBench
    `ClientEventTime` and `HID` are 31-bit values with runs of 2, and RunEnd makes them 22%
    smaller. Its run-length threshold skips them.
  - Sparse pays off well below its 90% cap. `FetchTiming` has an 81% top value, and Sparse makes
    it 31% smaller.
  - The size model decides both from bits:
    - RunEnd: `runs × (value-range bits + position bits)`.
    - Sparse: `exceptions × (value-range bits + position bits)`.
    - Both are compared with `len × width`.
- **`IntegerStats::average_run_length` is a `u32`,** so an average run of 1.97 truncates to 1.
  The model needs the exact run count, which should become part of the stats.
- **Longer sample slices help sorted and run-heavy data,** like TPC-H keys, where 64-value slices
  break runs and sequences apart. They don't help ClickBench, and they cost about 30% compression
  time.
- **A sample that compresses to zero buffer bytes is disqualified** (`EstimateScore::ZeroBytes`
  fails `is_valid`). Allowing it alone changed nothing measurable here, but it's wrong: it
  rejects the best possible encoding.
- **The size model trades decode speed for size:** RunEnd and Sparse decode slower than
  bit-packing (+37% decode time). That's the size-vs-decode tradeoff from question 4. A
  decode-aware preset would weigh it.

## Caveats

- Integers only. The oracle is one step (root only), so the true headroom is at least this large.
- Timings come from a shared 4-core cloud VM, single-threaded: decode is a median of 7,
  compression a median of 3 for default and nosample, and a single run for the other variants.
  The direction of the effects is reliable; the exact percentages aren't.
- "Decode" means full decompression to canonical. Pushdown (compare, filter, take) wasn't
  measured.
- Only 6 sources. The TPC-H keys are synthetic and inflate the TPC-H headroom. The real-data
  headroom is 2.6–5.3%.
- `forced/sequence` fails on non-sequences (expected), and `forced/bitpacking` fails on 19
  chunks with negative values. These count as infeasible.
