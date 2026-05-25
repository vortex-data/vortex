// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::clone_on_ref_ptr,
    clippy::cognitive_complexity,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::print_stdout,
    clippy::unwrap_used
)]
//
// Single-core only. bench_find proved find is ~89% of parse and immovable; the
// ~11% residual is the bit-packer. The decode path monomorphizes on `const BITS`
// (TokenCursor / dispatch_bits!), but the encode BitWriter packs with a runtime
// `bits` (data-dependent shift/branch per token). This A/Bs the shipping serial
// `parse` against an otherwise-identical loop whose packing is monomorphized on
// `const BITS`, so the shift/mask/spill fold to literals. Output is asserted
// byte-identical. No threads.
//
//   ONPAIR_BENCH_PARQUET=target/l_comment.parquet ONPAIR_BENCH_COLUMN=l_comment \
//     cargo run --release -p vortex-onpair-rs --example bench_pack
//
// Env: ONPAIR_BENCH_MAX_BYTES (default 256 MiB), ONPAIR_BENCH_ITERS (default 6).

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_onpair_rs::OnPairTrainingConfig;
use vortex_onpair_rs::Store;
use vortex_onpair_rs::TrainingConfig;
use vortex_onpair_rs::dispatch_bits;
use vortex_onpair_rs::lpm::LongestPrefixMatcher;
use vortex_onpair_rs::parse;
use vortex_onpair_rs::train;

const BITS: &[u8] = &[12, 16];

/// Parse + monomorphized (`const BITS`) bit-packing in one pass. Produces the
/// same `packed`/`boundaries` as the runtime-width `BitWriter`.
fn parse_mono(
    bytes: &[u8],
    off32: &[u32],
    n: usize,
    lpm: &LongestPrefixMatcher,
    bits: u8,
) -> Store {
    dispatch_bits!(bits, |B| {
        let mut packed: Vec<u64> = Vec::with_capacity(256);
        let mut boundaries: Vec<u32> = Vec::with_capacity(n + 1);
        boundaries.push(0);
        let mut buf: u64 = 0;
        let mut shift: u32 = 0;
        let mut count: u32 = 0;
        for i in 0..n {
            let s = off32[i] as usize;
            let e = off32[i + 1] as usize;
            let mut pos = s;
            while pos < e {
                let (tok, mlen) = lpm.find_longest_match(&bytes[pos..e]);
                let v = (tok as u64) & ((1u64 << B) - 1);
                buf |= v << shift;
                shift += B;
                if shift >= 64 {
                    packed.push(buf);
                    shift -= 64;
                    buf = if shift == 0 { 0 } else { v >> (B - shift) };
                }
                count += 1;
                pos += mlen;
            }
            boundaries.push(count);
        }
        if shift > 0 {
            packed.push(buf);
        }
        if count > 0 {
            packed.push(0);
        }
        Store {
            bit_width: bits,
            packed,
            boundaries,
        }
    })
}

fn main() {
    let max_bytes = env::var("ONPAIR_BENCH_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256 << 20);
    let iters = env::var("ONPAIR_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(6);

    let (bytes, offsets) = load_corpus(max_bytes).expect("set ONPAIR_BENCH_PARQUET to l_comment");
    let n = offsets.len() - 1;
    let off32: Vec<u32> = offsets.iter().map(|&o| o as u32).collect();
    println!(
        "corpus {:.1} MiB, {n} rows (single core)\n",
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

        let mut mono = Store::default();
        let mut mono_s = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            mono = parse_mono(&bytes, &off32, n, lpm, bits);
            mono_s = mono_s.min(t.elapsed().as_secs_f64());
        }

        let identical = mono.packed == serial.packed && mono.boundaries == serial.boundaries;
        let ntok = serial.num_tokens();
        let ns = |secs: f64| secs / ntok as f64 * 1e9;
        println!("=== bits = {bits} ===  ({ntok} tokens)");
        println!(
            "  serial parse (runtime bits): {:.3}s  {:.2} ns/tok",
            ser_s,
            ns(ser_s)
        );
        println!(
            "  parse + const-BITS pack    : {:.3}s  {:.2} ns/tok   ({:.3}x)",
            mono_s,
            ns(mono_s),
            ser_s / mono_s
        );
        println!(
            "  output-identical: {}",
            if identical { "YES" } else { "NO (BUG)" }
        );
        assert!(identical, "mono pack diverged at bits={bits}");
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
