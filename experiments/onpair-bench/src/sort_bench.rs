// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort-throughput benchmark for `compare_fused` vs decode-then-byte-compare.

use std::time::Instant;

use anyhow::Result;
use rand::SeedableRng;
use rand::seq::SliceRandom;

use crate::compare_fused::compare_fused;
use crate::encoders::OnPairOut;
use crate::encoders::onpair_compress;

#[derive(Debug, Clone)]
pub struct SortRow {
    pub method: String,
    pub elapsed_ms: u128,
    pub mb_per_s: f64,
    pub ns_per_row: f64,
}

/// Run the sort benchmark on a single column. Shuffles the rows, encodes
/// once, then times three sort strategies:
///
/// 1. `compare_fused` over OnPair token sequences (the new primitive).
/// 2. Sorting the *pre-decoded* `Vec<Vec<u8>>` directly. Isolates the sort
///    cost when the bytes are already materialised.
/// 3. Decode-from-encoded + sort. End-to-end if you store OnPair-encoded.
pub fn run_sort_bench(name: &str, rows: Vec<Vec<u8>>) -> Result<(String, Vec<SortRow>)> {
    let n = rows.len();
    let raw_bytes: usize = rows.iter().map(|r| r.len()).sum();
    eprintln!(
        "[{name}] n={n} raw={:.2} MiB avg_len={:.1}",
        raw_bytes as f64 / 1024.0 / 1024.0,
        raw_bytes as f64 / n as f64,
    );

    // Shuffle so sort has actual work to do.
    let mut shuffled = rows;
    shuffled.shuffle(&mut rand::rngs::StdRng::seed_from_u64(0xCAFE_BABE));

    let out: OnPairOut = onpair_compress(&shuffled, 12)?;

    // Borrow dict parts ONCE for compare_fused.
    let parts = out
        .col
        .parts()
        .map_err(|e| anyhow::anyhow!("onpair parts: {e:?}"))?;
    let dict_bytes_vec = parts.dict_bytes.to_vec();
    let dict_offsets_vec = parts.dict_offsets.to_vec();
    let _ = parts;

    let dict_bytes = dict_bytes_vec.as_slice();
    let dict_offsets = dict_offsets_vec.as_slice();
    let tokens = &out.tokens;

    let mut results = Vec::new();
    let reference: Option<Vec<u32>>;

    // ── Method 1: compare_fused on tokens ───────────────────────────────
    {
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let t = Instant::now();
        indices.sort_by(|&i, &j| {
            compare_fused(
                &tokens[i as usize],
                &tokens[j as usize],
                dict_bytes,
                dict_offsets,
            )
        });
        let elapsed = t.elapsed();
        let ms = elapsed.as_millis();
        results.push(SortRow {
            method: "compare_fused (tokens)".into(),
            elapsed_ms: ms,
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });
        reference = Some(indices);
    }

    // ── Method 2: sort the pre-decoded bytes (sort-only timing) ─────────
    {
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let t = Instant::now();
        indices.sort_by(|&i, &j| shuffled[i as usize].cmp(&shuffled[j as usize]));
        let elapsed = t.elapsed();
        results.push(SortRow {
            method: "byte cmp (sort only, pre-decoded)".into(),
            elapsed_ms: elapsed.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });
        // Sanity-check that compare_fused produced the same permutation.
        if let Some(ref r) = reference {
            assert_eq!(*r, indices, "compare_fused order ≠ byte cmp order");
        }
    }

    // ── Method 3: decode-then-sort (end-to-end, includes decode cost) ───
    {
        let t = Instant::now();
        let (bytes, offsets) = out.col.decode_all();
        // Build Vec<Vec<u8>> from the (bytes, offsets) layout to match the
        // memory shape Method 2 sorts over. This is the realistic cost if
        // your storage form is OnPair-encoded and you want to sort.
        let mut decoded: Vec<Vec<u8>> = Vec::with_capacity(n);
        for w in offsets.windows(2) {
            decoded.push(bytes[w[0] as usize..w[1] as usize].to_vec());
        }
        let mut indices: Vec<u32> = (0..n as u32).collect();
        indices.sort_by(|&i, &j| decoded[i as usize].cmp(&decoded[j as usize]));
        let elapsed = t.elapsed();
        results.push(SortRow {
            method: "byte cmp (end-to-end: decode + sort)".into(),
            elapsed_ms: elapsed.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });
        if let Some(ref r) = reference {
            assert_eq!(*r, indices, "compare_fused order ≠ decode+sort order");
        }
    }

    Ok((name.to_string(), results))
}

pub fn print_sort_table(name: &str, rows: &[SortRow]) {
    println!("\n## sort_bench: {name}");
    println!();
    println!("| Method | Time (ms) | MB/s (raw) | ns/row |");
    println!("|---|---:|---:|---:|");
    for r in rows {
        println!(
            "| {} | {} | {:.1} | {:.0} |",
            r.method, r.elapsed_ms, r.mb_per_s, r.ns_per_row
        );
    }
}
