// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ClassifiedMultiStep DFA prototype vs. existing FSST DFA vs. uncompressed
//! `memmem`, on the **real** ClickBench `hits_0.parquet` URL column.
//!
//! Variants per needle:
//!
//! | name                          | what it measures                                |
//! | ----------------------------- | ----------------------------------------------- |
//! | `dfa_old`                     | existing `FsstMatcher::matches` per string      |
//! | `dfa_new`                     | new `ClassifiedDfa::matches` per string         |
//! | `decompress_only`             | per-string FSST decode, no LIKE                 |
//! | `decompress_plus_memmem`      | per-string decode then `memmem::find` on text   |
//! | `memmem_pre_decompressed`     | `memmem` on already-decompressed corpus         |
//! | `construct_old`               | build the existing matcher                      |
//! | `construct_new`               | build the new matcher                           |
//!
//! Set `CLICKBENCH_HITS_0` to override the parquet path; defaults to
//! `/home/ec2-user/clickbench-data/hits_0.parquet`.

#![expect(clippy::unwrap_used)]

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::sync::LazyLock;

use arrow_array::Array as _;
use arrow_array::StringArray;
use divan::Bencher;
use divan::black_box;
use fsst::Decompressor;
use memchr::memmem::Finder;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_fsst::FSSTArray;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::dfa::FsstMatcher;
use vortex_fsst::dfa_compressed::ClassifiedDfa;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;

fn main() {
    divan::main();
}

fn hits_0_path() -> PathBuf {
    env::var_os("CLICKBENCH_HITS_0")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/ec2-user/clickbench-data/hits_0.parquet"))
}

fn load_real_urls() -> Vec<String> {
    let path = hits_0_path();
    let file = File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let schema = builder.schema().clone();
    let url_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "URL")
        .expect("hits parquet has a URL column");
    let mask = ProjectionMask::roots(builder.parquet_schema(), [url_idx]);
    let reader = builder.with_projection(mask).build().unwrap();
    let mut out: Vec<String> = Vec::with_capacity(1_000_000);
    for batch in reader {
        let batch = batch.unwrap();
        let col = batch.column(0);
        let strs = col
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("URL column is StringArray");
        for i in 0..strs.len() {
            if strs.is_null(i) {
                continue;
            }
            out.push(strs.value(i).to_string());
        }
    }
    out
}

static URLS: LazyLock<Vec<String>> = LazyLock::new(load_real_urls);

static FSST_URLS: LazyLock<FSSTArray> = LazyLock::new(|| {
    let urls: &Vec<String> = &URLS;
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor)
});

/// Compressed FSST stream with offsets (host buffers), so we can run the
/// matcher loop without going through canonicalization.
struct CompressedView {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
    n: usize,
}

static COMPRESSED: LazyLock<CompressedView> = LazyLock::new(|| {
    let fsst = &*FSST_URLS;
    let n = fsst.len();
    let view = fsst.as_view();
    let codes = view.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let offsets: Vec<u32> = offsets
        .as_slice::<i32>()
        .iter()
        .map(|&v| v as u32)
        .collect();
    let bytes = codes.bytes().as_slice().to_vec();
    CompressedView { bytes, offsets, n }
});

static DECOMPRESSED_PER_STRING: LazyLock<Vec<Vec<u8>>> =
    LazyLock::new(|| URLS.iter().map(|s| s.as_bytes().to_vec()).collect());

const NEEDLES: &[&str] = &[
    "google",
    "https",
    ".ru",
    "/catalog",
    "facebook",
    "moscow",
    "smartphone",
];

// ---------------------------------------------------------------------------
// Per-string matcher loops (DFA flavors)
// ---------------------------------------------------------------------------

#[divan::bench(args = NEEDLES)]
fn dfa_old(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let pattern = format!("%{needle}%");
    let matcher = FsstMatcher::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        pattern.as_bytes(),
    )
    .unwrap()
    .unwrap();
    bencher.bench_local(|| {
        let mut count: u64 = 0;
        let mut start = cv.offsets[0] as usize;
        for i in 0..cv.n {
            let end = cv.offsets[i + 1] as usize;
            if matcher.matches(&cv.bytes[start..end]) {
                count += 1;
            }
            start = end;
        }
        count
    });
}

#[divan::bench(args = NEEDLES)]
fn dfa_new(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dfa = ClassifiedDfa::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        needle.as_bytes(),
    )
    .expect("ClassifiedDfa builds for this needle");
    bencher.bench_local(|| {
        let mut count: u64 = 0;
        let mut start = cv.offsets[0] as usize;
        for i in 0..cv.n {
            let end = cv.offsets[i + 1] as usize;
            if dfa.matches(&cv.bytes[start..end]) {
                count += 1;
            }
            start = end;
        }
        count
    });
}

/// Single-pass corpus-wide SIMD scan: `memchr_iter` over the full
/// compressed buffer + per-candidate verify. The path that actually
/// combines SIMD-throughput byte-scanning with FSST's compression
/// dividend. See [`ClassifiedDfa::scan_corpus`].
#[divan::bench(args = NEEDLES)]
fn dfa_new_corpus(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dfa = ClassifiedDfa::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        needle.as_bytes(),
    )
    .expect("ClassifiedDfa builds for this needle");
    bencher.bench_local(|| {
        let bits = dfa.scan_corpus(&cv.bytes, &cv.offsets, cv.n);
        bits.iter().filter(|b| **b).count() as u64
    });
}

/// Strict-anchor scan: drop ESCAPE_CODE from the anchor set so the
/// 1-progressing case (e.g. `%google%`) stays on the fast `memchr1`
/// path (29 GB/s) instead of degrading to `memchr2` (~4 GB/s).
#[divan::bench(args = NEEDLES)]
fn dfa_new_corpus_strict(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dfa = ClassifiedDfa::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        needle.as_bytes(),
    )
    .expect("ClassifiedDfa builds for this needle");
    bencher.bench_local(|| {
        let bits = dfa.scan_corpus_strict(&cv.bytes, &cv.offsets, cv.n);
        bits.iter().filter(|b| **b).count() as u64
    });
}

/// Parallel multi-thread microbench that mimics DuckDB's per-chunk
/// dispatch model without DuckDB. Pre-splits the FSST corpus into ~10K
/// row chunks (the same shape Vortex layout chunks have on real ClickBench
/// shards), then for `T` threads, each thread loops over its share of
/// chunks and runs:
///   - `ClassifiedDfa::try_new` (mimics per-chunk symbol-table change),
///   - `scan_corpus_strict_to_bitbuf` over the chunk's bytes/offsets,
///   - aggregates a count of matching strings.
///
/// Sweeps `THREAD_COUNTS = [1, 2, 4, 8, 16]` so we can see the same
/// scaling shape we observed in `duckdb-bench` and confirm whether the
/// per-call cost growth is intrinsic to the kernel or a DuckDB artifact.
const PARALLEL_NEEDLE: &str = "google";
const PARALLEL_CHUNK_ROWS: usize = 10_000;
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

struct ParallelChunks {
    /// Pre-split chunks: `(start_offset_idx, end_offset_idx)`. Chunk `i`
    /// covers strings `[i*PARALLEL_CHUNK_ROWS, (i+1)*PARALLEL_CHUNK_ROWS)`.
    /// All chunks reference shared offsets/bytes via `ParallelData`.
    ranges: Vec<(usize, usize)>,
}

struct ParallelData {
    bytes: std::sync::Arc<Vec<u8>>,
    offsets: std::sync::Arc<Vec<u32>>,
    symbols: std::sync::Arc<Vec<fsst::Symbol>>,
    symbol_lengths: std::sync::Arc<Vec<u8>>,
    chunks: ParallelChunks,
}

static PARALLEL: LazyLock<ParallelData> = LazyLock::new(|| {
    let cv = &*COMPRESSED;
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < cv.n {
        let end = (start + PARALLEL_CHUNK_ROWS).min(cv.n);
        ranges.push((start, end));
        start = end;
    }
    let view = FSST_URLS.as_view();
    ParallelData {
        bytes: std::sync::Arc::new(cv.bytes.clone()),
        offsets: std::sync::Arc::new(cv.offsets.clone()),
        symbols: std::sync::Arc::new(view.symbols().as_slice().to_vec()),
        symbol_lengths: std::sync::Arc::new(view.symbol_lengths().as_slice().to_vec()),
        chunks: ParallelChunks { ranges },
    }
});

#[divan::bench(args = THREAD_COUNTS)]
fn parallel_scan(bencher: Bencher, threads: usize) {
    let pd = &*PARALLEL;
    let n_chunks = pd.chunks.ranges.len();
    let needle: &[u8] = PARALLEL_NEEDLE.as_bytes();

    bencher.bench_local(|| {
        let total_count = std::sync::atomic::AtomicU64::new(0);
        // Atomic chunk index — work-stealing semantics. Threads pull
        // chunk indices off this counter until exhausted.
        let next_chunk = std::sync::atomic::AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for _t in 0..threads {
                let bytes = pd.bytes.clone();
                let offsets = pd.offsets.clone();
                let symbols = pd.symbols.clone();
                let lengths = pd.symbol_lengths.clone();
                let ranges = &pd.chunks.ranges;
                let counter = &total_count;
                let next = &next_chunk;
                scope.spawn(move || {
                    let mut local_count: u64 = 0;
                    loop {
                        let idx =
                            next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if idx >= n_chunks {
                            break;
                        }
                        let (s_lo, s_hi) = ranges[idx];
                        let chunk_n = s_hi - s_lo;
                        // Per-chunk: rebuild ClassifiedDfa (mimics the
                        // unique symbol table per Vortex layout chunk) and
                        // run scan_corpus_strict_to_bitbuf over the chunk
                        // bytes only.
                        let dfa = ClassifiedDfa::try_new(
                            symbols.as_slice(),
                            lengths.as_slice(),
                            needle,
                        )
                        .expect("dfa builds");
                        let chunk_offsets = &offsets[s_lo..=s_hi];
                        let bits = dfa.scan_corpus_strict_to_bitbuf(
                            bytes.as_slice(),
                            chunk_offsets,
                            chunk_n,
                            false,
                        );
                        // Count matched bits.
                        for i in 0..chunk_n {
                            if bits.value(i) {
                                local_count += 1;
                            }
                        }
                    }
                    counter.fetch_add(local_count, std::sync::atomic::Ordering::Relaxed);
                });
            }
        });
        // Reset for next iteration.
        next_chunk.store(0, std::sync::atomic::Ordering::Relaxed);
        total_count.load(std::sync::atomic::Ordering::Relaxed)
    });
}

/// Same shape as `parallel_scan` but using the existing FsstMatcher
/// (the dfa_old path). Compares the two kernels under identical
/// threading/contention conditions.
#[divan::bench(args = THREAD_COUNTS)]
fn parallel_scan_old(bencher: Bencher, threads: usize) {
    let pd = &*PARALLEL;
    let n_chunks = pd.chunks.ranges.len();
    let pattern: &[u8] = b"%google%";

    bencher.bench_local(|| {
        let total_count = std::sync::atomic::AtomicU64::new(0);
        let next_chunk = std::sync::atomic::AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for _t in 0..threads {
                let bytes = pd.bytes.clone();
                let offsets = pd.offsets.clone();
                let symbols = pd.symbols.clone();
                let lengths = pd.symbol_lengths.clone();
                let ranges = &pd.chunks.ranges;
                let counter = &total_count;
                let next = &next_chunk;
                scope.spawn(move || {
                    let mut local_count: u64 = 0;
                    loop {
                        let idx =
                            next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if idx >= n_chunks {
                            break;
                        }
                        let (s_lo, s_hi) = ranges[idx];
                        let chunk_n = s_hi - s_lo;
                        let matcher = FsstMatcher::try_new(
                            symbols.as_slice(),
                            lengths.as_slice(),
                            pattern,
                        )
                        .unwrap()
                        .unwrap();
                        let chunk_offsets = &offsets[s_lo..=s_hi];
                        // Scan per-string within the chunk.
                        let mut start = chunk_offsets[0] as usize;
                        for i in 0..chunk_n {
                            let end = chunk_offsets[i + 1] as usize;
                            if matcher.matches(&bytes[start..end]) {
                                local_count += 1;
                            }
                            start = end;
                        }
                    }
                    counter.fetch_add(local_count, std::sync::atomic::Ordering::Relaxed);
                });
            }
        });
        next_chunk.store(0, std::sync::atomic::Ordering::Relaxed);
        total_count.load(std::sync::atomic::Ordering::Relaxed)
    });
}

/// Multi-pass scan: one `memchr1` pass per progressing anchor instead
/// of a single `memchr2_iter`. memchr2 collapses to ~4 GB/s on real
/// ClickBench because `ESCAPE_CODE` appears ~1% of bytes; multiple
/// memchr1 passes each run at ~29 GB/s.
#[divan::bench(args = NEEDLES)]
fn dfa_new_corpus_multipass(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dfa = ClassifiedDfa::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        needle.as_bytes(),
    )
    .expect("ClassifiedDfa builds for this needle");
    bencher.bench_local(|| {
        let bits = dfa.scan_corpus_multipass(&cv.bytes, &cv.offsets, cv.n);
        bits.iter().filter(|b| **b).count() as u64
    });
}

/// Inlined-everything variant of `dfa_new_corpus`: tighter merge loop
/// + 4-byte `multi_step` lookup directly from state 0 (folds the anchor
/// into the window). Falls through to `dfa_new_corpus` for needles whose
/// anchor set isn't size 1.
#[divan::bench(args = NEEDLES)]
fn dfa_new_corpus_inlined(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dfa = ClassifiedDfa::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        needle.as_bytes(),
    )
    .expect("ClassifiedDfa builds for this needle");
    bencher.bench_local(|| {
        let bits = match dfa.scan_corpus_memchr1_inlined(&cv.bytes, &cv.offsets, cv.n) {
            Some(b) => b,
            None => dfa.scan_corpus(&cv.bytes, &cv.offsets, cv.n),
        };
        bits.iter().filter(|b| **b).count() as u64
    });
}

/// Strip-down attribution bench: just `memchr_iter` over the corpus,
/// counting candidates. No verify, no merge. Isolates the SIMD scan cost.
#[divan::bench(args = NEEDLES)]
fn corpus_memchr_only(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dfa = ClassifiedDfa::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        needle.as_bytes(),
    )
    .unwrap();
    let prog = dfa.state0_progressing_codes();
    let scan_start = cv.offsets[0] as usize;
    let scan_end = cv.offsets[cv.n] as usize;
    let scan_slice = cv.bytes[scan_start..scan_end].to_vec();
    bencher.bench_local(|| match prog.as_slice() {
        [a] => memchr::memchr_iter(*a, &scan_slice).count() as u64,
        [a, b] => memchr::memchr2_iter(*a, *b, &scan_slice).count() as u64,
        [a, b, c] => memchr::memchr3_iter(*a, *b, *c, &scan_slice).count() as u64,
        _ => 0,
    });
}

/// Strip-down attribution bench: scan + two-pointer merge to set bits,
/// no verify. Pretends every candidate matches. This + `corpus_memchr_only`
/// brackets the verify cost.
#[divan::bench(args = NEEDLES)]
fn corpus_merge_no_verify(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dfa = ClassifiedDfa::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        needle.as_bytes(),
    )
    .unwrap();
    let prog = dfa.state0_progressing_codes();
    let scan_start = cv.offsets[0] as usize;
    let scan_end = cv.offsets[cv.n] as usize;
    let scan_slice_owned = cv.bytes[scan_start..scan_end].to_vec();
    bencher.bench_local(|| {
        let mut result = vec![false; cv.n];
        let mut s = 0usize;
        let mut s_end = cv.offsets[1] as usize;
        let iter = match prog.as_slice() {
            [a] => Box::new(memchr::memchr_iter(*a, &scan_slice_owned).map(move |i| scan_start + i))
                as Box<dyn Iterator<Item = usize>>,
            [a, b] => Box::new(
                memchr::memchr2_iter(*a, *b, &scan_slice_owned).map(move |i| scan_start + i),
            ),
            [a, b, c] => Box::new(
                memchr::memchr3_iter(*a, *b, *c, &scan_slice_owned).map(move |i| scan_start + i),
            ),
            _ => Box::new(std::iter::empty()),
        };
        for cand in iter {
            while cand >= s_end {
                s += 1;
                if s >= cv.n {
                    break;
                }
                s_end = cv.offsets[s + 1] as usize;
            }
            if s >= cv.n {
                break;
            }
            result[s] = true;
        }
        result.iter().filter(|b| **b).count() as u64
    });
}

// ---------------------------------------------------------------------------
// Decompression baselines
// ---------------------------------------------------------------------------

/// Decompress every FSST-compressed string into a scratch buffer.
/// No LIKE, just the raw decode cost.
#[divan::bench]
fn decompress_only(bencher: Bencher) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dec = Decompressor::new(view.symbols().as_slice(), view.symbol_lengths().as_slice());
    bencher.bench_local(|| {
        let mut total: u64 = 0;
        let mut start = cv.offsets[0] as usize;
        for i in 0..cv.n {
            let end = cv.offsets[i + 1] as usize;
            let out = dec.decompress(&cv.bytes[start..end]);
            total += out.len() as u64;
            black_box(out);
            start = end;
        }
        total
    });
}

/// Decompress + `memmem::find` on the decompressed text. Apples-to-apples
/// "no FSST pushdown" baseline.
#[divan::bench(args = NEEDLES)]
fn decompress_plus_memmem(bencher: Bencher, needle: &str) {
    let cv = &*COMPRESSED;
    let view = FSST_URLS.as_view();
    let dec = Decompressor::new(view.symbols().as_slice(), view.symbol_lengths().as_slice());
    let finder = Finder::new(needle.as_bytes());
    bencher.bench_local(|| {
        let mut count: u64 = 0;
        let mut start = cv.offsets[0] as usize;
        for i in 0..cv.n {
            let end = cv.offsets[i + 1] as usize;
            let out = dec.decompress(&cv.bytes[start..end]);
            if finder.find(&out).is_some() {
                count += 1;
            }
            start = end;
        }
        count
    });
}

/// `memmem::find` on the already-decompressed corpus. The "LIKE alone"
/// baseline (no decode cost).
#[divan::bench(args = NEEDLES)]
fn memmem_pre_decompressed(bencher: Bencher, needle: &str) {
    let strings = &*DECOMPRESSED_PER_STRING;
    let finder = Finder::new(needle.as_bytes());
    bencher.bench_local(|| {
        let mut count: u64 = 0;
        for s in strings.iter() {
            if finder.find(s).is_some() {
                count += 1;
            }
        }
        count
    });
}

// ---------------------------------------------------------------------------
// Matcher construction time
// ---------------------------------------------------------------------------

#[divan::bench(args = NEEDLES)]
fn construct_old(bencher: Bencher, needle: &str) {
    let view = FSST_URLS.as_view();
    let symbols = view.symbols().as_slice().to_vec();
    let lengths = view.symbol_lengths().as_slice().to_vec();
    let pattern = format!("%{needle}%");
    bencher.bench_local(|| {
        let m = FsstMatcher::try_new(&symbols, &lengths, pattern.as_bytes())
            .unwrap()
            .unwrap();
        black_box(m);
    });
}

#[divan::bench(args = NEEDLES)]
fn construct_new(bencher: Bencher, needle: &str) {
    let view = FSST_URLS.as_view();
    let symbols = view.symbols().as_slice().to_vec();
    let lengths = view.symbol_lengths().as_slice().to_vec();
    bencher.bench_local(|| {
        let m = ClassifiedDfa::try_new(&symbols, &lengths, needle.as_bytes()).unwrap();
        black_box(m);
    });
}

// ---------------------------------------------------------------------------
// Warm-cache diagnostics. The divan-driven benches above measure
// cold-cache throughput (each sample evicts the corpus); this one runs
// in a tight loop so the corpus stays L3-resident and we can compare
// to standalone memchr/memmem numbers.
// ---------------------------------------------------------------------------

#[divan::bench]
fn diagnose_warm(_bencher: Bencher) {
    use std::time::Instant;
    let cv = &*COMPRESSED;
    let scan_start = cv.offsets[0] as usize;
    let scan_end = cv.offsets[cv.n] as usize;
    let scan_slice = &cv.bytes[scan_start..scan_end];
    let view = FSST_URLS.as_view();

    let dfa = ClassifiedDfa::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        b"google",
    )
    .unwrap();

    eprintln!(
        "diagnose_warm: corpus is {} MB compressed",
        scan_slice.len() / (1024 * 1024)
    );

    let prog = dfa.state0_progressing_codes();
    eprintln!(
        "  state0_progressing_codes: {:?} (len={})",
        prog,
        prog.len()
    );
    let prog_for_memchr = prog.clone();

    let _warm1 = match prog_for_memchr.as_slice() {
        [a] => memchr::memchr_iter(*a, scan_slice).count(),
        [a, b] => memchr::memchr2_iter(*a, *b, scan_slice).count(),
        [a, b, c] => memchr::memchr3_iter(*a, *b, *c, scan_slice).count(),
        _ => 0,
    };
    drop(dfa.scan_corpus(&cv.bytes, &cv.offsets, cv.n));

    let iters = 20;
    let t = Instant::now();
    let mut count = 0u64;
    for _ in 0..iters {
        count += match prog_for_memchr.as_slice() {
            [a] => memchr::memchr_iter(*a, scan_slice).count(),
            [a, b] => memchr::memchr2_iter(*a, *b, scan_slice).count(),
            [a, b, c] => memchr::memchr3_iter(*a, *b, *c, scan_slice).count(),
            _ => 0,
        } as u64;
    }
    let dt = t.elapsed();
    eprintln!(
        "  WARM memchr*  ({iters}x): {:.2} ms/iter, {:.2} GB/s, hits/iter={}",
        dt.as_secs_f64() * 1000.0 / iters as f64,
        scan_slice.len() as f64 * iters as f64 / dt.as_secs_f64() / 1e9,
        count / iters as u64
    );

    // memchr1 alone (just c153 = 'g'), no ESCAPE.
    let t = Instant::now();
    let mut count = 0u64;
    let g_code = 153u8;
    for _ in 0..iters {
        count += memchr::memchr_iter(g_code, scan_slice).count() as u64;
    }
    let dt = t.elapsed();
    eprintln!(
        "  WARM memchr1  ({iters}x): {:.2} ms/iter, {:.2} GB/s, hits/iter={}",
        dt.as_secs_f64() * 1000.0 / iters as f64,
        scan_slice.len() as f64 * iters as f64 / dt.as_secs_f64() / 1e9,
        count / iters as u64
    );

    // Bare-minimum byte loop (no SIMD): tells us if compiler-level
    // inhibitions are preventing memchr from kicking in.
    let t = Instant::now();
    let mut count = 0u64;
    for _ in 0..iters {
        let mut c = 0u64;
        for &b in scan_slice {
            if b == g_code {
                c += 1;
            }
        }
        count += c;
    }
    let dt = t.elapsed();
    eprintln!(
        "  WARM scalar   ({iters}x): {:.2} ms/iter, {:.2} GB/s, hits/iter={}",
        dt.as_secs_f64() * 1000.0 / iters as f64,
        scan_slice.len() as f64 * iters as f64 / dt.as_secs_f64() / 1e9,
        count / iters as u64
    );

    let t = Instant::now();
    let mut total_matches = 0u64;
    for _ in 0..iters {
        let bits = dfa.scan_corpus(&cv.bytes, &cv.offsets, cv.n);
        total_matches += bits.iter().filter(|b| **b).count() as u64;
    }
    let dt = t.elapsed();
    eprintln!(
        "  WARM dfa_v2   ({iters}x): {:.2} ms/iter (matches/iter={})",
        dt.as_secs_f64() * 1000.0 / iters as f64,
        total_matches / iters as u64
    );

    let matcher = FsstMatcher::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        b"%google%",
    )
    .unwrap()
    .unwrap();
    let t = Instant::now();
    let mut total = 0u64;
    for _ in 0..iters {
        let mut count = 0u64;
        let mut start = cv.offsets[0] as usize;
        for i in 0..cv.n {
            let end = cv.offsets[i + 1] as usize;
            if matcher.matches(&cv.bytes[start..end]) {
                count += 1;
            }
            start = end;
        }
        total += count;
    }
    let dt = t.elapsed();
    eprintln!(
        "  WARM dfa_old  ({iters}x): {:.2} ms/iter (matches/iter={})",
        dt.as_secs_f64() * 1000.0 / iters as f64,
        total / iters as u64
    );

    let strings = &*DECOMPRESSED_PER_STRING;
    let finder = Finder::new(b"google");
    let t = Instant::now();
    let mut count = 0u64;
    for _ in 0..iters {
        let mut c = 0u64;
        for s in strings.iter() {
            if finder.find(s).is_some() {
                c += 1;
            }
        }
        count += c;
    }
    let dt = t.elapsed();
    eprintln!(
        "  WARM memmem   ({iters}x): {:.2} ms/iter (matches/iter={})",
        dt.as_secs_f64() * 1000.0 / iters as f64,
        count / iters as u64
    );
}

// ---------------------------------------------------------------------------
// One-shot diagnostics: print build stats per needle so we can attribute
// perf differences to k, K, table size.
// ---------------------------------------------------------------------------

#[divan::bench]
fn diagnose_classified(_bencher: Bencher) {
    let view = FSST_URLS.as_view();
    let symbols = view.symbols().as_slice();
    let lengths = view.symbol_lengths().as_slice();
    eprintln!(
        "real corpus: {} URLs, {} FSST symbols",
        URLS.len(),
        symbols.len()
    );
    for &needle in NEEDLES {
        let dfa = ClassifiedDfa::try_new(symbols, lengths, needle.as_bytes());
        match dfa {
            Some(d) => {
                let s = d.stats;
                eprintln!(
                    "  needle={needle:>14}  K={:>2}  k={}  states={}  multi_step={} bytes",
                    s.n_classes, s.k, s.n_states, s.multi_step_bytes,
                );
            }
            None => eprintln!("  needle={needle:>14}  (bailed: K > MAX_CLASSES)"),
        }
    }

    // Full-corpus correctness check for every variant: scan_corpus,
    // scan_corpus_strict, scan_corpus_multipass. Counts must agree
    // with `memmem` on the decompressed strings.
    let cv = &*COMPRESSED;
    eprintln!(
        "\n  full-corpus correctness ({} URLs each):",
        URLS.len()
    );
    for &needle in NEEDLES {
        let Some(dfa) = ClassifiedDfa::try_new(symbols, lengths, needle.as_bytes()) else {
            continue;
        };
        let finder = Finder::new(needle.as_bytes());
        let exp = URLS
            .iter()
            .filter(|s| finder.find(s.as_bytes()).is_some())
            .count() as u64;

        // Also check dfa_old (the existing FoldedContainsDfa via FsstMatcher)
        let pat = format!("%{needle}%");
        let old_matcher =
            FsstMatcher::try_new(symbols, lengths, pat.as_bytes()).unwrap().unwrap();
        let mut c_old: u64 = 0;
        let mut start = cv.offsets[0] as usize;
        for i in 0..cv.n {
            let end = cv.offsets[i + 1] as usize;
            if old_matcher.matches(&cv.bytes[start..end]) {
                c_old += 1;
            }
            start = end;
        }

        let r1 = dfa.scan_corpus(&cv.bytes, &cv.offsets, cv.n);
        let c1 = r1.iter().filter(|b| **b).count() as u64;

        let r2 = dfa.scan_corpus_strict(&cv.bytes, &cv.offsets, cv.n);
        let c2 = r2.iter().filter(|b| **b).count() as u64;

        let r3 = dfa.scan_corpus_multipass(&cv.bytes, &cv.offsets, cv.n);
        let c3 = r3.iter().filter(|b| **b).count() as u64;

        eprintln!(
            "  needle={needle:>14}  memmem={exp:>6}  dfa_old={c_old:>6}{}  scan_corpus={c1:>6}{}  strict={c2:>6}{}  multipass={c3:>6}{}",
            if c_old == exp { " OK " } else { " ERR" },
            if c1 == exp { " OK " } else { " ERR" },
            if c2 == exp { " OK " } else { " ERR" },
            if c3 == exp { " OK " } else { " ERR" },
        );
        if c2 != exp || c3 != exp {
            let mut shown = 0;
            for (i, url) in URLS.iter().enumerate() {
                let mm = url.contains(needle);
                let st = r2[i];
                if mm != st {
                    let s = cv.offsets[i] as usize;
                    let e = cv.offsets[i + 1] as usize;
                    let bytes_in_string = &cv.bytes[s..e];
                    let matches_says = dfa.matches(bytes_in_string);
                    eprintln!(
                        "    DISAGREE i={i} memmem={mm} strict={st} matches()={matches_says}\n      compressed_bytes={:?}\n      url={url:?}",
                        bytes_in_string
                    );
                    // Print prog set for context
                    let prog_strict = dfa.state0_progressing_codes_strict();
                    eprintln!("      progressing_strict={prog_strict:?}");
                    shown += 1;
                    if shown >= 2 {
                        break;
                    }
                }
            }
        }
    }
}
