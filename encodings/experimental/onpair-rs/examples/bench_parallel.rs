// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::clone_on_ref_ptr,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::print_stdout,
    clippy::unwrap_used
)]
//
// Wall-clock prototype: parse is bound by per-token find *latency* on one core
// (proven by bench_find — ~1.0x MLP headroom, no single-core room left). But
// parse over rows is embarrassingly parallel, and the crate runs it on one
// thread (Column::compress → parse). This splits the rows across threads, runs
// the find loop per chunk, then merges into a byte-IDENTICAL token stream, and
// A/Bs the wall-clock vs serial parse. Throughput = cores × (1/latency).
//
//   ONPAIR_BENCH_PARQUET=target/l_comment.parquet ONPAIR_BENCH_COLUMN=l_comment \
//     cargo run --release -p vortex-onpair-rs --example bench_parallel
//
// Env: ONPAIR_BENCH_MAX_BYTES (default 256 MiB), ONPAIR_BENCH_ITERS (default 5),
//      THREADS (default = available parallelism).

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_onpair_rs::OnPairTrainingConfig;
use vortex_onpair_rs::Store;
use vortex_onpair_rs::TrainingConfig;
use vortex_onpair_rs::bits::BitWriter;
use vortex_onpair_rs::lpm::LongestPrefixMatcher;
use vortex_onpair_rs::parse;
use vortex_onpair_rs::train;

const BITS: &[u8] = &[12, 16];

/// Parse rows `[lo, hi)` with `lpm`, returning the token codes in row order and
/// the per-row token count. Pure `find` work — no shared state, no locking.
fn parse_chunk(
    bytes: &[u8],
    off32: &[u32],
    lpm: &LongestPrefixMatcher,
    lo: usize,
    hi: usize,
) -> (Vec<u16>, Vec<u32>) {
    let mut codes: Vec<u16> = Vec::new();
    let mut counts: Vec<u32> = Vec::with_capacity(hi - lo);
    for i in lo..hi {
        let s = off32[i] as usize;
        let e = off32[i + 1] as usize;
        let mut pos = s;
        let mut c = 0u32;
        while pos < e {
            let (tok, mlen) = lpm.find_longest_match(&bytes[pos..e]);
            codes.push(tok);
            pos += mlen;
            c += 1;
        }
        counts.push(c);
    }
    (codes, counts)
}

/// Parallel parse: split rows across `nthreads`, find per chunk, merge into a
/// `Store` whose `packed`/`boundaries` are byte-identical to serial `parse`.
fn parse_parallel(
    bytes: &[u8],
    off32: &[u32],
    n: usize,
    lpm: &LongestPrefixMatcher,
    bits: u8,
    nthreads: usize,
) -> Store {
    let chunk = n.div_ceil(nthreads.max(1));
    let results: Vec<(Vec<u16>, Vec<u32>)> = thread::scope(|sc| {
        let handles: Vec<_> = (0..nthreads)
            .map(|t| {
                let lo = (t * chunk).min(n);
                let hi = ((t + 1) * chunk).min(n);
                sc.spawn(move || parse_chunk(bytes, off32, lpm, lo, hi))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut store = Store {
        bit_width: bits,
        ..Default::default()
    };
    let mut boundaries: Vec<u32> = Vec::with_capacity(n + 1);
    boundaries.push(0);
    let mut running = 0u32;
    {
        let mut w = BitWriter::new(&mut store);
        for (codes, counts) in &results {
            for &c in counts {
                running += c;
                boundaries.push(running);
            }
            for &code in codes {
                w.write(code);
            }
        }
    }
    store.boundaries = boundaries;
    store
}

fn main() {
    let max_bytes = env::var("ONPAIR_BENCH_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256 << 20);
    let iters = env::var("ONPAIR_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);
    // `std::thread::available_parallelism` is disallowed workspace-wide; this is
    // a standalone bench, so default to 4 and let the caller set `THREADS`.
    let nthreads = env::var("THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);

    let (bytes, offsets) = load_corpus(max_bytes).expect("set ONPAIR_BENCH_PARQUET to l_comment");
    let n = offsets.len() - 1;
    let off32: Vec<u32> = offsets.iter().map(|&o| o as u32).collect();
    println!(
        "corpus {:.1} MiB, {n} rows, threads = {nthreads}\n",
        bytes.len() as f64 / (1024.0 * 1024.0)
    );

    for &bits in BITS {
        let cfg = TrainingConfig::from(OnPairTrainingConfig {
            bits: bits as u32,
            threshold: 0.2,
            seed: 42,
        });
        let result = train(&bytes, &off32, n, &cfg);
        let lpm = &result.lpm;

        let mut serial = Store::default();
        let mut ser_s = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            parse(&bytes, &off32, n, lpm, bits, &mut serial);
            ser_s = ser_s.min(t.elapsed().as_secs_f64());
        }

        let mut par = Store::default();
        let mut par_s = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            par = parse_parallel(&bytes, &off32, n, lpm, bits, nthreads);
            par_s = par_s.min(t.elapsed().as_secs_f64());
        }

        let identical = par.packed == serial.packed && par.boundaries == serial.boundaries;
        let mib = bytes.len() as f64 / (1024.0 * 1024.0);
        println!("=== bits = {bits} ===  ({} tokens)", serial.num_tokens());
        println!("  serial   : {:.3}s  {:.1} MiB/s", ser_s, mib / ser_s);
        println!(
            "  parallel : {:.3}s  {:.1} MiB/s   ({:.2}x on {nthreads} threads)",
            par_s,
            mib / par_s,
            ser_s / par_s
        );
        println!(
            "  output-identical: {}",
            if identical { "YES" } else { "NO (BUG)" }
        );
        assert!(
            identical,
            "parallel parse diverged from serial at bits={bits}"
        );
        println!();
    }
}

fn load_corpus(max_bytes: usize) -> Option<(Vec<u8>, Vec<u64>)> {
    let path = env::var("ONPAIR_BENCH_PARQUET").ok()?;
    let file = File::open(PathBuf::from(&path)).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    let schema = builder.schema().clone();
    let col_name = env::var("ONPAIR_BENCH_COLUMN").ok();
    let picked = match col_name.as_deref() {
        Some(name) => schema.fields().iter().position(|f| f.name() == name)?,
        None => schema.fields().iter().position(|f| {
            use arrow_schema::DataType::*;
            matches!(f.data_type(), Utf8 | LargeUtf8 | Utf8View)
        })?,
    };
    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    let reader = builder.build().ok()?;
    'outer: for batch in reader.flatten() {
        let arr = batch.column(picked);
        use arrow_schema::DataType::*;
        macro_rules! push_iter {
            ($it:expr) => {
                for s in $it {
                    let b = s.unwrap_or("").as_bytes();
                    bytes.extend_from_slice(b);
                    offsets.push(bytes.len() as u64);
                    if bytes.len() >= max_bytes {
                        break 'outer;
                    }
                }
            };
        }
        match arr.data_type() {
            Utf8 => push_iter!(arr.as_string::<i32>().iter()),
            LargeUtf8 => push_iter!(arr.as_string::<i64>().iter()),
            Utf8View => push_iter!(arr.as_string_view().iter()),
            _ => return None,
        }
    }
    Some((bytes, offsets))
}
