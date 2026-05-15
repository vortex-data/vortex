// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! End-to-end skip-index evaluation on the **real ClickBench `hits`
//! dataset**.
//!
//! The test reads 8 192 rows of the `URL` column from one ClickBench
//! shard, OnPair-compresses them, splits the result into 8 chunks of
//! 1 024 rows (sharing the column dictionary), then for each chunk
//! builds three independent skip indexes ([`DictPresence`],
//! [`TrigramBloom`], [`SeamBloom`]).
//!
//! For a fixed query workload of `eq`, `LIKE 'p%'`, and `LIKE '%s%'`
//! patterns we report:
//!
//! * Storage size in bytes per structure (and ratios to raw + OnPair-
//!   compressed footprints).
//! * Per-query pruning power: `chunks_pruned / chunks_with_zero_matches`
//!   (recall of the prefilter against an oracle).
//! * `prefilter_prob = chunks_kept / chunks_total` (Pr that we still
//!   have to run the kernel).
//! * `false_pos_rate = chunks_kept_but_empty / chunks_kept_total` (the
//!   "wasted work" rate).
//!
//! The dataset shard is downloaded once to `/tmp/hits_0.parquet` (see
//! `download_data` in `vortex-bench`); override with the env var
//! `CLICKBENCH_HITS=<path>`. The test prints a structured ASCII report
//! to stdout — run with `--nocapture` (or via `cargo nextest run` which
//! captures by default; use `--success-output immediate` to see it).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::tests_outside_test_module,
    clippy::use_debug
)]

use std::fs::File;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Instant;

use arrow_array::Array as ArrowArray;
use arrow_array::cast::AsArray;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::decode::OwnedDecodeInputs;
use vortex_onpair::lpm::DictIndex;
use vortex_onpair::onpair_compress;
use vortex_onpair::skip::DictPresence;
use vortex_onpair::skip::SeamBloom;
use vortex_onpair::skip::TrigramBloom;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| LEGACY_SESSION.clone());

const N_ROWS: usize = 8 * 1024; // 8K
const CHUNK_SIZE: usize = 1024;
const NUM_CHUNKS: usize = N_ROWS / CHUNK_SIZE;
/// Trigram Bloom sizing. ≈ 32 bits per row → ~4 KB per chunk of 1024 rows.
/// Sized for ~3 K distinct trigrams per chunk at k=3 hashes.
const TRIGRAM_BITS_PER_ROW: usize = 32;
/// Seam Bloom sizing — much smaller because only boundary trigrams.
const SEAM_BITS_PER_ROW: usize = 8;

fn hits_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CLICKBENCH_HITS") {
        return Some(PathBuf::from(p));
    }
    let default = PathBuf::from("/tmp/hits_0.parquet");
    default.exists().then_some(default)
}

fn read_url_column(path: &PathBuf, n: usize) -> Vec<String> {
    let file = File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet builder");
    let schema = builder.schema().clone();
    let col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "URL")
        .expect("URL column");
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [col_idx]);
    let mut reader = builder
        .with_projection(mask)
        .with_batch_size(n.min(8192))
        .build()
        .expect("reader");
    let mut out: Vec<String> = Vec::with_capacity(n);
    while out.len() < n {
        let Some(batch) = reader.next() else { break };
        let batch = batch.expect("batch");
        let col = batch.column(0);
        let pushed = push_strings(col, n - out.len(), &mut out);
        if pushed == 0 {
            panic!("unexpected URL column type: {:?}", col.data_type());
        }
    }
    out
}

fn push_strings(col: &dyn ArrowArray, want: usize, out: &mut Vec<String>) -> usize {
    if let Some(s) = col.as_string_opt::<i32>() {
        for i in 0..s.len().min(want) {
            out.push(s.value(i).to_string());
        }
        return s.len().min(want);
    }
    if let Some(s) = col.as_string_opt::<i64>() {
        for i in 0..s.len().min(want) {
            out.push(s.value(i).to_string());
        }
        return s.len().min(want);
    }
    if let Some(s) = col.as_string_view_opt() {
        for i in 0..s.len().min(want) {
            out.push(s.value(i).to_string());
        }
        return s.len().min(want);
    }
    if let Some(b) = col.as_binary_opt::<i32>() {
        for i in 0..b.len().min(want) {
            out.push(String::from_utf8_lossy(b.value(i)).into_owned());
        }
        return b.len().min(want);
    }
    if let Some(b) = col.as_binary_opt::<i64>() {
        for i in 0..b.len().min(want) {
            out.push(String::from_utf8_lossy(b.value(i)).into_owned());
        }
        return b.len().min(want);
    }
    if let Some(b) = col.as_binary_view_opt() {
        for i in 0..b.len().min(want) {
            out.push(String::from_utf8_lossy(b.value(i)).into_owned());
        }
        return b.len().min(want);
    }
    0
}

#[derive(Clone, Debug)]
enum Pred {
    Eq(String),
    StartsWith(String),
    Contains(String),
}

impl Pred {
    fn name(&self) -> String {
        match self {
            Pred::Eq(s) => format!("col = {:?}", trunc(s, 40)),
            Pred::StartsWith(s) => format!("LIKE {:?}", format!("{}%", trunc(s, 40))),
            Pred::Contains(s) => format!("LIKE {:?}", format!("%{}%", trunc(s, 40))),
        }
    }

    /// Ground-truth: does this chunk's row range actually contain a match?
    fn truly_matches_chunk(&self, rows: &[String]) -> bool {
        match self {
            Pred::Eq(s) => rows.iter().any(|r| r == s),
            Pred::StartsWith(s) => rows.iter().any(|r| r.as_bytes().starts_with(s.as_bytes())),
            Pred::Contains(s) => rows.iter().any(|r| r.contains(s.as_str())),
        }
    }
}

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}...", &s[..n])
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn clickbench_url_skip_indexes() {
    let Some(path) = hits_path() else {
        eprintln!(
            "skipping clickbench_url_skip_indexes: {} not found; set CLICKBENCH_HITS=<path>",
            "/tmp/hits_0.parquet"
        );
        return;
    };

    // ---------------------------------------------------------------- load
    let t0 = Instant::now();
    let rows = read_url_column(&path, N_ROWS);
    assert_eq!(rows.len(), N_ROWS, "wanted {} rows", N_ROWS);
    let raw_bytes: usize = rows.iter().map(|s| s.len()).sum();
    let avg_len = raw_bytes as f64 / rows.len() as f64;
    eprintln!(
        "loaded {} URL rows from {:?} in {:?}; raw_bytes={} avg_len={:.1}",
        rows.len(),
        path,
        t0.elapsed(),
        raw_bytes,
        avg_len
    );

    // -------------------------------------------------------------- compress
    let varbin = VarBinArray::from_iter(
        rows.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );

    let t0 = Instant::now();
    let arr =
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();
    let bits = arr.bits();
    let mut ctx = SESSION.create_execution_ctx();
    let inputs =
        OwnedDecodeInputs::collect(arr.as_view(), &mut ctx).expect("decode inputs");
    let dv = inputs.view();
    let index = DictIndex::build(&dv);
    eprintln!(
        "OnPair-compressed in {:?}; bits/code={}; dict_size={}",
        t0.elapsed(),
        bits,
        dv.dict_table.len(),
    );

    // Approximate compressed footprint (the four numeric children +
    // dict bytes, no Vortex framing). Codes use `bits` bits each.
    let n_tokens = dv.codes.len();
    let dict_bytes_len = dv.dict_bytes.len();
    let dict_offsets_bytes = dv.dict_table.len() * size_of::<u32>(); // ~ what would be stored
    let codes_bytes = (n_tokens * bits as usize + 7) / 8;
    let codes_offsets_bytes = dv.codes_offsets.len() * size_of::<u32>();
    let compressed_bytes = dict_bytes_len + dict_offsets_bytes + codes_bytes + codes_offsets_bytes;
    eprintln!(
        "approx OnPair footprint: dict_bytes={} dict_offsets={} codes={} (12bpc) codes_offsets={} \
         total={}",
        dict_bytes_len, dict_offsets_bytes, codes_bytes, codes_offsets_bytes, compressed_bytes
    );

    // -------------------------------------------------------- build indexes
    let mut presence: Vec<DictPresence> = Vec::with_capacity(NUM_CHUNKS);
    let mut trigram: Vec<TrigramBloom> = Vec::with_capacity(NUM_CHUNKS);
    let mut seam: Vec<SeamBloom> = Vec::with_capacity(NUM_CHUNKS);
    let t0 = Instant::now();
    for c in 0..NUM_CHUNKS {
        let lo = c * CHUNK_SIZE;
        let hi = lo + CHUNK_SIZE;
        presence.push(DictPresence::build(&dv, lo, hi));
        trigram.push(TrigramBloom::build(&dv, lo, hi, TRIGRAM_BITS_PER_ROW));
        seam.push(SeamBloom::build(&dv, lo, hi, SEAM_BITS_PER_ROW));
    }
    eprintln!("built {} chunks of skip indexes in {:?}", NUM_CHUNKS, t0.elapsed());

    let presence_bytes: usize = presence.iter().map(DictPresence::byte_size).sum();
    let trigram_bytes: usize = trigram.iter().map(TrigramBloom::byte_size).sum();
    let seam_bytes: usize = seam.iter().map(SeamBloom::byte_size).sum();

    // -------------------------------------------------------------- queries
    // Tailored to URLs. Mix of:
    //  - eq on a real row's URL (must keep its chunk)
    //  - selective prefix
    //  - common prefix
    //  - "%google%" (the motivating case)
    //  - more substrings
    // Add a contains needle pulled from a real row so we *know* at least
    // one chunk truly contains it.
    let real_substr: String = {
        let s = rows[2345].clone();
        // pick a long-ish unique chunk of an actual URL
        if s.len() >= 24 { s[6..24].to_string() } else { s }
    };

    let queries: Vec<Pred> = vec![
        Pred::Eq(rows[1234].clone()),                                 // must hit
        Pred::Eq("https://no-such-domain.example/never".to_string()), // none
        Pred::StartsWith("http://".to_string()),
        Pred::StartsWith("https://www.google.".to_string()),
        Pred::StartsWith("http://m.kino.".to_string()),
        Pred::StartsWith("https://m.lenta.ru".to_string()),
        Pred::Contains("google".to_string()),
        Pred::Contains("youtube".to_string()),
        Pred::Contains("vkontakte".to_string()),
        Pred::Contains("/admin/".to_string()),
        Pred::Contains(".gif?".to_string()),
        Pred::Contains("status=500".to_string()),
        Pred::Contains("lol".to_string()),
        Pred::Contains("zzzzzzzz".to_string()),
        Pred::Contains(real_substr),
    ];

    // ---------------------------------------------------- compute ground truth
    // For each query, which chunks have ≥1 actual match?
    let mut truth: Vec<Vec<bool>> = Vec::with_capacity(queries.len());
    for q in &queries {
        let mut v = Vec::with_capacity(NUM_CHUNKS);
        for c in 0..NUM_CHUNKS {
            let lo = c * CHUNK_SIZE;
            let hi = lo + CHUNK_SIZE;
            v.push(q.truly_matches_chunk(&rows[lo..hi]));
        }
        truth.push(v);
    }

    // ---------------------------------------------------- run each predicate
    // We measure four prefilters per query:
    //   A      — DictPresence alone
    //   B      — TrigramBloom alone (only applicable to substring queries; for
    //            eq/prefix we degrade to "kept always")
    //   C      — SeamBloom + DictPresence (only for substring queries)
    //   A∧B    — combined (best per-query)
    // Plus a single-row "decompressed" oracle to verify NO false negatives.
    fn frac(a: usize, b: usize) -> f64 {
        if b == 0 { 0.0 } else { a as f64 / b as f64 }
    }

    let mut per_query_keep: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(queries.len());
    println!();
    println!("=== Per-query prefilter results (NUM_CHUNKS = {}) ===", NUM_CHUNKS);
    println!(
        "{:<48} {:>6} {:>6}  {:>6} {:>6} {:>6} {:>6}  {:>6} {:>6} {:>6} {:>6}",
        "query", "real", "empty",
        "A.keep", "B.keep", "C.keep", "AB.keep",
        "A.rec", "B.rec", "C.rec", "AB.rec",
    );
    println!("{}", "-".repeat(132));

    for (qi, q) in queries.iter().enumerate() {
        let actual_match: Vec<bool> = truth[qi].clone();
        let real = actual_match.iter().filter(|&&b| b).count();
        let empty = NUM_CHUNKS - real;

        let mut keep_a = 0usize;
        let mut keep_b = 0usize;
        let mut keep_c = 0usize;
        let mut keep_ab = 0usize;
        // Accumulate keep counts for the workload-level Pr[keep] line below.
        per_query_keep.push((0usize, 0usize, 0usize, 0usize));
        let keeps = per_query_keep.last_mut().unwrap();

        for c in 0..NUM_CHUNKS {
            let pa = match q {
                Pred::Eq(s) => presence[c].might_eq(&dv, &index, s.as_bytes()),
                Pred::StartsWith(s) => presence[c].might_starts_with(&dv, &index, s.as_bytes()),
                Pred::Contains(s) => presence[c].might_contain(&dv, &index, s.as_bytes()),
            };
            // Tier-B trigram Bloom: the needle is always a substring of any
            // matching row, so trigram bloom is sound for eq, prefix, and
            // contains alike.
            let needle_bytes: &[u8] = match q {
                Pred::Eq(s) => s.as_bytes(),
                Pred::StartsWith(s) => s.as_bytes(),
                Pred::Contains(s) => s.as_bytes(),
            };
            let pb = trigram[c].might_contain(needle_bytes);
            let pc = seam[c].might_contain(&dv, &presence[c], needle_bytes);

            // Combined (AND of all enabled).
            let pab = pa && pb;

            keep_a += pa as usize;
            keep_b += pb as usize;
            keep_c += pc as usize;
            keep_ab += pab as usize;
            keeps.0 += pa as usize;
            keeps.1 += pb as usize;
            keeps.2 += pc as usize;
            keeps.3 += pab as usize;

            // Soundness check — never prune a chunk that actually has a match.
            assert!(!actual_match[c] || pa, "A false-negative on chunk {c} for query {q:?}");
            assert!(!actual_match[c] || pb, "B false-negative on chunk {c} for query {q:?}");
            assert!(!actual_match[c] || pc, "C false-negative on chunk {c} for query {q:?}");
            assert!(!actual_match[c] || pab, "AB false-negative on chunk {c} for query {q:?}");
        }

        let pruned_a = NUM_CHUNKS - keep_a;
        let pruned_b = NUM_CHUNKS - keep_b;
        let pruned_c = NUM_CHUNKS - keep_c;
        let pruned_ab = NUM_CHUNKS - keep_ab;

        // Recall = pruned / empty. Closer to 1 is tighter.
        let rec_a = frac(pruned_a, empty);
        let rec_b = frac(pruned_b, empty);
        let rec_c = frac(pruned_c, empty);
        let rec_ab = frac(pruned_ab, empty);

        println!(
            "{:<48} {:>6} {:>6}  {:>6} {:>6} {:>6} {:>6}  {:>6.2} {:>6.2} {:>6.2} {:>6.2}",
            q.name(), real, empty,
            keep_a, keep_b, keep_c, keep_ab,
            rec_a, rec_b, rec_c, rec_ab,
        );
    }
    println!();
    println!("Columns:");
    println!("  real      = chunks (of {}) that actually contain a match", NUM_CHUNKS);
    println!("  empty     = chunks with zero matches (best-case pruneable)");
    println!("  X.keep    = chunks the X prefilter still keeps  (lower = better)");
    println!("  X.rec     = pruning recall = (empty - kept-but-empty) / empty  (1.00 = perfect)");
    println!("  A=DictPresence  B=TrigramBloom  C=SeamBloom+Presence  AB=A∧B");

    // Aggregate "prefilter probability" Pr[keep] across the workload.
    let total_chunks = NUM_CHUNKS as f64 * queries.len() as f64;
    let total_real: usize = truth.iter().flatten().filter(|&&b| b).count();
    let sum_a: usize = per_query_keep.iter().map(|t| t.0).sum();
    let sum_b: usize = per_query_keep.iter().map(|t| t.1).sum();
    let sum_c: usize = per_query_keep.iter().map(|t| t.2).sum();
    let sum_ab: usize = per_query_keep.iter().map(|t| t.3).sum();
    println!();
    println!(
        "  workload Pr[keep] (lower = more pruning; floor = real/total = {:.3}):",
        total_real as f64 / total_chunks,
    );
    println!(
        "    A=DictPresence            {:.3}",
        sum_a as f64 / total_chunks
    );
    println!(
        "    B=TrigramBloom            {:.3}",
        sum_b as f64 / total_chunks
    );
    println!(
        "    C=SeamBloom+Presence      {:.3}",
        sum_c as f64 / total_chunks
    );
    println!(
        "    AB=DictPresence AND Bloom {:.3}",
        sum_ab as f64 / total_chunks
    );
    println!();

    // -------------------------------------------------- size + ratio report
    println!("=== Size report ===");
    println!(
        "{:<28} {:>14} {:>14} {:>14}",
        "structure", "bytes_total", "bytes/chunk", "bytes/row",
    );
    println!("{}", "-".repeat(74));
    let raw_per_row = raw_bytes as f64 / N_ROWS as f64;
    let comp_per_row = compressed_bytes as f64 / N_ROWS as f64;
    println!(
        "{:<28} {:>14} {:>14} {:>14.3}",
        "raw_text",
        raw_bytes,
        format!("{}", raw_bytes / NUM_CHUNKS),
        raw_per_row,
    );
    println!(
        "{:<28} {:>14} {:>14} {:>14.3}",
        "onpair_compressed (≈)",
        compressed_bytes,
        format!("{}", compressed_bytes / NUM_CHUNKS),
        comp_per_row,
    );
    println!(
        "{:<28} {:>14} {:>14} {:>14.3}",
        "skip A: DictPresence",
        presence_bytes,
        format!("{}", presence_bytes / NUM_CHUNKS),
        presence_bytes as f64 / N_ROWS as f64,
    );
    println!(
        "{:<28} {:>14} {:>14} {:>14.3}",
        "skip B: TrigramBloom",
        trigram_bytes,
        format!("{}", trigram_bytes / NUM_CHUNKS),
        trigram_bytes as f64 / N_ROWS as f64,
    );
    println!(
        "{:<28} {:>14} {:>14} {:>14.3}",
        "skip C: SeamBloom",
        seam_bytes,
        format!("{}", seam_bytes / NUM_CHUNKS),
        seam_bytes as f64 / N_ROWS as f64,
    );
    println!();

    println!("=== Size ratios ===");
    println!(
        "  A / raw          = {:>7.4}%   A / compressed = {:>7.4}%",
        100.0 * presence_bytes as f64 / raw_bytes as f64,
        100.0 * presence_bytes as f64 / compressed_bytes as f64,
    );
    println!(
        "  B / raw          = {:>7.4}%   B / compressed = {:>7.4}%",
        100.0 * trigram_bytes as f64 / raw_bytes as f64,
        100.0 * trigram_bytes as f64 / compressed_bytes as f64,
    );
    println!(
        "  C / raw          = {:>7.4}%   C / compressed = {:>7.4}%",
        100.0 * seam_bytes as f64 / raw_bytes as f64,
        100.0 * seam_bytes as f64 / compressed_bytes as f64,
    );
    println!(
        "  (A+B) / raw      = {:>7.4}%   (A+B) / compressed = {:>7.4}%",
        100.0 * (presence_bytes + trigram_bytes) as f64 / raw_bytes as f64,
        100.0 * (presence_bytes + trigram_bytes) as f64 / compressed_bytes as f64,
    );
    println!(
        "  (A+C) / raw      = {:>7.4}%   (A+C) / compressed = {:>7.4}%",
        100.0 * (presence_bytes + seam_bytes) as f64 / raw_bytes as f64,
        100.0 * (presence_bytes + seam_bytes) as f64 / compressed_bytes as f64,
    );
    println!();
}
