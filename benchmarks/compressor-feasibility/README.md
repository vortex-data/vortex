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

## Training a selector that trades ratio against decode time

`train.py` learns, for each candidate, how big the result will be and how long it takes to
decode. The candidates are every root scheme (Pco included), the production choice and the size
model. Gradient-boosted trees predict `log2(bytes / canonical bytes)` and `log2(decode ns per
value)` from 28 features: the per-chunk stats plus closed-form size estimates per scheme. A
classifier predicts whether each scheme is feasible.

The choice is the lowest `bytes / bandwidth + decode time`, so **bandwidth is a knob at
inference time**: one model serves every preset.

Used directly, the model can lose to production when its size predictions are off. On held-out
datasets at S3 bandwidth it was up to +21% worse. So it is used as a **proposer**:

1. Compress with production, as today.
2. If the model predicts an alternative is at least `gate` cheaper, compress that too.
3. Keep whichever has the lower real cost.

The result is never worse than production. Results on datasets the model never trained on
(leave-one-source-out), with cost relative to production at the same bandwidth:

| Bandwidth | Gate | Cost | Best possible | Bytes | Decode | Chunks tried | Extra compression time |
|---|---|---|---|---|---|---|---|
| S3 ~100 MB/s | 0% | −13.6% | −21.7% | −20.2% | +75% | 71% | +118% |
| S3 ~100 MB/s | 10% | −13.4% | −21.7% | −20.0% | +75% | 34% | +92% |
| NVMe ~2 GB/s | 0% | −18.1% | −27.6% | +20.2% | −44% | 89% | +61% |
| NVMe ~2 GB/s | 10% | −16.3% | −27.6% | +18.6% | −40% | 47% | +35% |
| Memory ~20 GB/s | 0% | −52.4% | −61.9% | +64.4% | −60% | 94% | +64% |
| Memory ~20 GB/s | 10% | −48.5% | −61.9% | +52.6% | −55% | 61% | +47% |

- With a 5-fold split grouped by column, the same policy reaches −17% / −25% / −57%.
- **At S3, the model picks Pco wherever its 40% smaller output outweighs a slower decode.** Pco
  is excluded today; a cost model can admit it per chunk.
- **At NVMe and memory bandwidth, it trades bytes for decode speed:** roughly 40–55% less decode
  time.
- The extra compression time comes from verifying proposals (one more full compression per tried
  chunk). Pco's slow compression dominates it at S3. The gate trades it off: at 20%, NVMe costs
  −6.9% for +14% compression time.

To ship this, the decode-time measurement in step 3 should become a prediction (the decode model
itself, or per-encoding throughput), because timing at write time is noisy. The proposer can be
distilled to a shallow tree and code-generated.

## End to end: the model-driven compressor

`src/model.rs` is a model-driven compressor:

1. Compute the chunk's features from the Vortex array, over 8 contiguous windows of 512 values.
2. Evaluate the exported trees: per candidate, whether it is feasible, its size and its decode
   time.
3. Propose the cheapest candidate for the target bandwidth.
4. Compress with production. If the proposal is predicted at least 10% cheaper, compress it too.
5. Keep whichever has the lower *measured* cost: serialized bytes plus a timed decode.

`train.py --export` writes the gradient-boosted trees as JSON. It checks that a plain tree walk
(what the Rust side does) reproduces scikit-learn exactly. `models/int-v1.json` is a model
trained on all 1,540 chunks.

`run_e2e.sh <parquet-dir> <work-dir>` runs the whole loop:

1. Generate training data.
2. Fit one model per held-out source.
3. Compress every chunk with the model that never saw its source.
4. Report.

Corpus: 1,540 integer chunks (372 MB canonical) from eight sources: ClickBench `hits_0`, NYC
yellow, green and FHVHV taxi (November 2023), and TPC-H SF1 lineitem, orders, partsupp and
customer. Every number below is a real compression: serialized bytes, decode time (median of 7),
and compression time including features, inference and verification (median of 3).

| Bandwidth | Compressor | Cost | Bytes | Decode | Compress time |
|---|---|---|---|---|---|
| S3 ~100 MB/s | size-model thresholds | −3.5% | −6.4% | +39% | 1.06× |
| S3 ~100 MB/s | **model-driven** | **−15.7%** | −23.2% | +94% | 3.11× |
| NVMe ~2 GB/s | size-model thresholds | +20.0% | −6.4% | +39% | 1.06× |
| NVMe ~2 GB/s | **model-driven** | **−13.1%** | +12.3% | −32% | 1.98× |
| Memory ~20 GB/s | size-model thresholds | +36.2% | −6.4% | +39% | 1.06× |
| Memory ~20 GB/s | **model-driven** | **−43.5%** | +51.8% | −50% | 1.79× |

Costs are relative to the production compressor's thresholds at the same bandwidth.

- **No held-out source gets worse at any bandwidth.** The worst is −0.0% on the TPC-H tables at
  NVMe. The biggest wins:
  - ClickBench: −15% / −16% / −50% at S3 / NVMe / memory.
  - Green taxi: −16% at NVMe, −30% in memory.
  - TPC-H orders: −40% at S3.
- **What it keeps:**
  - At S3: Pco (191 chunks) and the size model (70).
  - At NVMe: FOR (203 chunks), which decodes faster than production's choices.
  - In memory: FOR and plain bit-packing.
- **The size-model thresholds alone only make sense for a size-only preset.** They save 6.4% of
  bytes but decode 39% slower, so they lose at NVMe and memory bandwidth.
- **Compression time:**
  - Features (0.10 ms per chunk) and inference (0.04 ms) add ~24% to production's 0.58 ms.
  - The rest is the verification compression, done on the 50–64% of chunks where the model
    predicts a 10%+ saving.
  - Raising the threshold cuts that time but loses most of the NVMe and memory gains (at 50%:
    −0.9% / −0.7%).

Getting the end-to-end result right took two fixes the offline evaluation missed:

- **Verifying with predicted decode times broke the "never worse" guarantee** (TPC-H lineitem
  +61% at NVMe). Measuring one decode of each candidate fixed it.
- **Features over the whole chunk cost 3–4× the compression.** Contiguous windows fixed it.

## What the model uses, and how it compares with the compressor's own estimates

`compare_estimators.py` reads a run with the `spy` variant, which logs the estimate each scheme
gives the selector at the root without changing the choice.

**Features that matter** (permutation importance on held-out columns, mean over candidates):

| Model | Top features | Barely used |
|---|---|---|
| Size | `step_mode_frac`, `est_runend`, `bits_p99`, `est_sparse`, `est_bp`, `est_for`, `avg_run`, `top1_frac` | `distinct_frac`, `null_frac`, `trailing_zeros`, `bits_delta`, `for_gain` |
| Decode time | `avg_run`, `est_runend`, `bits_for`, `ptype_bits`, `est_bp`, `top1_frac`, `bits_p99` | |

**Ratio accuracy** (median |log2(estimate / actual)|, per scheme at the root):

| Estimator | Median error | Within 10% |
|---|---|---|
| Compressor closed-form (Pass 1) | 47% | 31% |
| Compressor 1% sample (Pass 2) | 19% | 46% |
| Model, unseen columns | 22% | 39% |
| Model + compressor estimates as features, unseen columns | **11%** | **56%** |
| Model, unseen source | 92% | 14% |

- The compressor is exact for BitPacking and FOR. Its errors are Dict (62%), RunEnd (47%), RLE
  (40%) and Sparse (21%).
- **The compressor ranks schemes better** (rank correlation 0.70 vs 0.42).
- **Its failure is skipping, not ranking.** At the root it skips FOR on 65% of chunks, Sparse on
  61%, ZigZag on 75% and RunEnd on 28%. On 6% of chunks the best scheme is skipped outright.
- **Without verification, the model's pick is worse than the compressor's on size alone:** +15%
  regret against the compressor's +8%.

**Compression ratio** (real compressions, ratio-only bandwidth):

| Compressor | Ratio | Bytes |
|---|---|---|
| Production | 5.54× | — |
| Model-driven, production schemes | 5.75× | −3.6% |
| Size-model thresholds (earlier) | — | −6.4% |
| Best root, production schemes | 5.99× | −7.5% |
| Model-driven, Pco allowed | 8.03× | −31.0% |
| Best root, Pco allowed | 8.97× | −38.2% |

**Takeaway:** for ratio alone, the compressor's own estimates are the better predictor, and the
size-model fix beats the learned model. The learned model earns its place through decode time
(which nothing in today's compressor models) and through admitting Pco per chunk. The design to
build next is a hybrid:

- **Ratio:** the compressor's estimates with the size-model fix, plus the model as a correction
  (11% error with both).
- **Decode time:** the learned model.
- **Choice:** by `bytes / bandwidth + decode time`.

## Counting compression time too (default encodings only)

The objective per chunk is now `compress time / reads + bytes / bandwidth + decode time`, where
`reads` is reads per write. The model gains a third prediction per candidate: compression time,
trained on 3 timed repetitions. Pco is excluded, since it isn't in the default encodings.

The model is only run when it could plausibly pay for itself. Production is always compressed. An
alternative is tried only when `reads × predicted serving saving` covers twice its predicted
compression time, and its serving cost is at least 10% lower.

Measured total cost against production (all costs real, including features, inference and
verification):

| Bandwidth | Reads | Size-model thresholds | Model-driven | Bytes | Decode | Compress time |
|---|---|---|---|---|---|---|
| S3 | 1 | +1.6% | +21.4% | 0% | +1% | 1.39× |
| S3 | 10 | −2.3% | +8.5% | −2.6% | +5% | 1.99× |
| S3 | 100 | −3.1% | −1.1% | −3.1% | +2% | 2.39× |
| NVMe | 1 | +7.0% | +0.8% | 0% | +1% | 1.01× |
| NVMe | 10 | +13.8% | +18.3% | 0% | +2% | 1.34× |
| NVMe | 100 | +21.0% | **−2.9%** | +12.5% | −26% | 1.62× |
| Memory | 1 | +7.5% | −0.1% | 0% | 0% | 1.00× |
| Memory | 10 | +18.4% | +22.4% | 0% | +2% | 1.34× |
| Memory | 100 | +35.3% | **−11.8%** | +31.2% | −28% | 1.59× |

**Once compression time counts, the model's fixed overhead dominates.** Features (~0.07 ms per
chunk) plus inference (~0.04 ms) add ~25–35% to production's 0.58 ms. It's paid even when the
model keeps production's choice. That loses at 1–10 reads per write; only at ~100 reads does the
decode saving outweigh it.

- **The "worth it?" check only helps where serving is cheap** (NVMe and memory at 1 read). At S3,
  transfer cost makes the model look worthwhile even when it ends up trying nothing.
- **The size-model thresholds** only win at S3. Elsewhere their slower decode and compression
  cost more than the bytes they save.

To make this pay at low read counts, the overhead has to go. Computed inside the compressor, the
features could reuse `IntegerStats` (which already counts distinct values and runs) rather than
being recomputed. The ensembles (12 candidates × 3 targets × 40 trees) can be distilled to a few
small trees.

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
