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
// Lower-bound proof for parse speed. The parse is a *dependent chain*: within a
// string, each `find_longest_match` starts where the previous match ended, so
// consecutive finds cannot overlap — the loop is bounded below by per-find
// memory *latency*, not throughput. This harness quantifies exactly that floor
// and the only headroom that exists, on the real `l_comment` dictionary:
//
//   (1) full parse        — the shipping path (find + bit-pack + boundaries).
//   (2) dependent find     — the same greedy traversal, find only (no store).
//                            Isolates find; (1)-(2) is the bit-packing overhead.
//   (3) independent find   — the identical set of finds, but issued from a
//                            pre-collected position array so iterations are
//                            independent and the out-of-order core overlaps
//                            their misses. This is the *floor* find could reach
//                            if all memory-level parallelism were captured.
//
// (2) ≈ (1) proves the surrounding code is already free. (2) vs (3) is the MLP
// headroom — the same headroom idea B2 (AVX-512 gather) targets, which
// `bench_gather` shows vpgatherqq cannot realize on this hardware. If (2) ≈ (3),
// there is no headroom at all and the latency floor is absolute.
//
//   ONPAIR_BENCH_PARQUET=target/l_comment.parquet ONPAIR_BENCH_COLUMN=l_comment \
//     cargo run --release -p vortex-onpair-rs --example bench_find
//
// Env: ONPAIR_BENCH_MAX_BYTES (default 256 MiB), ONPAIR_BENCH_ITERS (default 5).

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_onpair_rs::OnPairTrainingConfig;
use vortex_onpair_rs::Store;
use vortex_onpair_rs::TrainingConfig;
use vortex_onpair_rs::parse;
use vortex_onpair_rs::train;

const BITS: &[u8] = &[12, 16];

fn main() {
    let max_bytes = env::var("ONPAIR_BENCH_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256 << 20);
    let iters = env::var("ONPAIR_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);

    let (bytes, offsets) = load_corpus(max_bytes).expect("set ONPAIR_BENCH_PARQUET to l_comment");
    let n = offsets.len() - 1;
    let off32: Vec<u32> = offsets.iter().map(|&o| o as u32).collect();
    println!(
        "corpus {:.1} MiB, {n} rows\n",
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

        // (1) full parse.
        let mut store = Store::default();
        let mut parse_s = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            parse(&bytes, &off32, n, lpm, bits, &mut store);
            parse_s = parse_s.min(t.elapsed().as_secs_f64());
        }
        let ntok = store.num_tokens();

        // (2) dependent find-chain (latency-bound): exact parse traversal, no store.
        let mut dep_s = f64::MAX;
        let mut dep_count = 0usize;
        for _ in 0..iters {
            let t = Instant::now();
            let mut acc = 0u64;
            let mut count = 0usize;
            for i in 0..n {
                let s = off32[i] as usize;
                let e = off32[i + 1] as usize;
                let mut pos = s;
                while pos < e {
                    let (tok, mlen) = lpm.find_longest_match(&bytes[pos..e]);
                    acc = acc.wrapping_add(tok as u64);
                    pos += mlen;
                    count += 1;
                }
            }
            black_box(acc);
            dep_s = dep_s.min(t.elapsed().as_secs_f64());
            dep_count = count;
        }

        // Collect the exact token-start positions produced by the greedy parse.
        let mut starts: Vec<(u32, u32)> = Vec::with_capacity(dep_count);
        for i in 0..n {
            let s = off32[i] as usize;
            let e = off32[i + 1] as usize;
            let mut pos = s;
            while pos < e {
                let (_, mlen) = lpm.find_longest_match(&bytes[pos..e]);
                starts.push((pos as u32, e as u32));
                pos += mlen;
            }
        }

        // (3) independent find-throughput: identical finds, but each iteration's
        // input comes from the array (not the previous result), so the OoO core
        // can overlap their memory accesses. This is the MLP-saturated floor.
        let mut indep_s = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let mut acc = 0u64;
            for &(p, e) in &starts {
                let (tok, _) = lpm.find_longest_match(&bytes[p as usize..e as usize]);
                acc = acc.wrapping_add(tok as u64);
            }
            black_box(acc);
            indep_s = indep_s.min(t.elapsed().as_secs_f64());
        }

        let ns = |secs: f64| secs / ntok as f64 * 1e9;
        println!("=== bits = {bits} ===  ({ntok} tokens)");
        println!("  (1) full parse        : {:6.2} ns/token", ns(parse_s));
        println!(
            "  (2) dependent find     : {:6.2} ns/token   (bit-pack overhead = {:.2} ns = {:.0}%)",
            ns(dep_s),
            ns(parse_s) - ns(dep_s),
            (ns(parse_s) - ns(dep_s)) / ns(parse_s) * 100.0
        );
        println!(
            "  (3) independent find   : {:6.2} ns/token   (MLP headroom = {:.2}x)",
            ns(indep_s),
            dep_s / indep_s
        );
        println!();
    }

    println!(
        "Reading: (2)≈(1) ⇒ bit-packing/bookkeeping is already free.\n\
         (3) is the floor only achievable by overlapping independent finds (MLP).\n\
         The parse is a dependent chain, so it is bounded below by (2); the (2)→(3)\n\
         gap is exactly the cross-string MLP that B2 targets — and `bench_gather`\n\
         shows AVX-512 gather runs at 0.5–0.7x scalar here, so it cannot close it.\n\
         The scalar multi-string interleave (PERFORMANCE.md) also cost more than\n\
         the gap. Hence no known output-preserving technique beats (2) on this HW."
    );
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
