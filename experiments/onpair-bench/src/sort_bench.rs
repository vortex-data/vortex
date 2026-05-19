// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort-throughput benchmark for `compare_fused` vs decode-then-byte-compare.

use std::time::Instant;

use anyhow::Result;
use rand::SeedableRng;
use rand::seq::SliceRandom;

use crate::compare_fused::build_dict_table;
use crate::compare_fused::build_row_prefix;
use crate::compare_fused::build_token_prefix;
use crate::compare_fused::compare_fused;
use crate::compare_fused::compare_fused_v2;
use crate::compare_fused::compare_fused_v3;
use crate::encoders::OnPairOut;
use crate::encoders::onpair_compress;

#[derive(Debug, Clone)]
pub struct SortRow {
    pub method: String,
    pub elapsed_ms: u128,
    pub mb_per_s: f64,
    pub ns_per_row: f64,
}

/// Flatten `Vec<Vec<T>>` into `(flat: Vec<T>, boundaries: Vec<u32>)`.
/// `boundaries[i]..boundaries[i+1]` are row i's elements.
fn flatten<T: Copy + Default>(rows: &[Vec<T>]) -> (Vec<T>, Vec<u32>) {
    let total: usize = rows.iter().map(|r| r.len()).sum();
    let mut flat = Vec::with_capacity(total);
    let mut boundaries = Vec::with_capacity(rows.len() + 1);
    boundaries.push(0u32);
    for r in rows {
        flat.extend_from_slice(r);
        boundaries.push(flat.len() as u32);
    }
    (flat, boundaries)
}

#[inline(always)]
fn slice_at<'a, T>(flat: &'a [T], bd: &[u32], i: usize) -> &'a [T] {
    let s = bd[i] as usize;
    let e = bd[i + 1] as usize;
    // SAFETY: boundaries are produced by `flatten` and are in-range by construction.
    unsafe { flat.get_unchecked(s..e) }
}

#[allow(dead_code)]
pub fn run_sort_bench(name: &str, rows: Vec<Vec<u8>>) -> Result<(String, Vec<SortRow>)> {
    run_sort_bench_with(name, rows, Order::Shuffled)
}

#[derive(Copy, Clone, Debug)]
pub enum Order {
    Shuffled,
    AlmostSorted, // sorted, then a small fraction of pairs swapped
}

pub fn run_sort_bench_with(
    name: &str,
    rows: Vec<Vec<u8>>,
    order: Order,
) -> Result<(String, Vec<SortRow>)> {
    let n = rows.len();
    let raw_bytes: usize = rows.iter().map(|r| r.len()).sum();
    eprintln!(
        "[{name}, {order:?}] n={n} raw={:.2} MiB avg_len={:.1}",
        raw_bytes as f64 / 1024.0 / 1024.0,
        raw_bytes as f64 / n as f64,
    );

    let mut shuffled = rows;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFE_BABE);
    match order {
        Order::Shuffled => shuffled.shuffle(&mut rng),
        Order::AlmostSorted => {
            shuffled.sort();
            // Swap 1% of adjacent pairs to break perfect sortedness.
            for k in 0..(n / 100) {
                let i = k * 100;
                if i + 1 < n {
                    shuffled.swap(i, i + 1);
                }
            }
        }
    }

    let out: OnPairOut = onpair_compress(&shuffled, 12)?;

    // Borrow dict parts ONCE and copy out (no lifetime entanglement).
    let parts = out
        .col
        .parts()
        .map_err(|e| anyhow::anyhow!("onpair parts: {e:?}"))?;
    let dict_bytes_vec = parts.dict_bytes.to_vec();
    let dict_table = build_dict_table(parts.dict_offsets);
    let token_prefix = build_token_prefix(parts.dict_bytes, parts.dict_offsets);
    let _ = parts;

    // Flat layouts for both sides (kill per-row Vec allocs, give the
    // comparator one indexed load + slice construction per side).
    let (token_flat, token_bd) = flatten(&out.tokens);
    let (byte_flat, byte_bd) = flatten(&shuffled);

    let dict_bytes = dict_bytes_vec.as_slice();
    let dict_table_s = dict_table.as_slice();
    let token_prefix_s = token_prefix.as_slice();

    let mut results = Vec::new();
    let reference: Option<Vec<u32>>;

    // ── Method 1a: compare_fused v1 (slice-cmp Phase 2) ─────────────────
    {
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let t = Instant::now();
        indices.sort_unstable_by(|&i, &j| {
            let a = slice_at(&token_flat, &token_bd, i as usize);
            let b = slice_at(&token_flat, &token_bd, j as usize);
            compare_fused(a, b, dict_bytes, dict_table_s)
        });
        let elapsed = t.elapsed();
        results.push(SortRow {
            method: "compare_fused v1 (slice cmp Phase 2)".into(),
            elapsed_ms: elapsed.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });
        reference = Some(indices);
    }

    // ── Method 1c: compare_fused v3 (precomputed row prefix) ────────────
    // Precompute row prefixes once (charged as encode/sort prep, not inside
    // the timed loop — same treatment as flatten and dict_table prep).
    let row_prefix = build_row_prefix(&token_flat, &token_bd, dict_bytes, dict_table_s);
    // Row lengths in decoded bytes — needed to determine if the prefix
    // difference is in real content. We use the actual `shuffled` lengths
    // (cheap; same as what byte cmp baseline sees).
    let row_len: Vec<u32> = (0..n as u32)
        .map(|r| byte_bd[r as usize + 1] - byte_bd[r as usize])
        .collect();
    {
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let t = Instant::now();
        indices.sort_unstable_by(|&i, &j| {
            let a = slice_at(&token_flat, &token_bd, i as usize);
            let b = slice_at(&token_flat, &token_bd, j as usize);
            compare_fused_v3(
                a,
                b,
                row_prefix[i as usize],
                row_prefix[j as usize],
                row_len[i as usize] as usize,
                row_len[j as usize] as usize,
                dict_bytes,
                dict_table_s,
            )
        });
        let elapsed = t.elapsed();
        results.push(SortRow {
            method: "compare_fused v3 (row-prefix u64 fast path)".into(),
            elapsed_ms: elapsed.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });
        if let Some(ref r) = reference {
            for k in 0..n {
                let ra = slice_at(&token_flat, &token_bd, r[k] as usize);
                let rb = slice_at(&token_flat, &token_bd, indices[k] as usize);
                if ra != rb {
                    let ba = slice_at(&byte_flat, &byte_bd, r[k] as usize);
                    let bb = slice_at(&byte_flat, &byte_bd, indices[k] as usize);
                    assert_eq!(ba, bb, "v3 order ≠ v1 order at k={k}");
                }
            }
        }
    }

    // ── Method 1b: compare_fused v2 (u64 token prefix Phase 2) ──────────
    {
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let t = Instant::now();
        indices.sort_unstable_by(|&i, &j| {
            let a = slice_at(&token_flat, &token_bd, i as usize);
            let b = slice_at(&token_flat, &token_bd, j as usize);
            compare_fused_v2(a, b, dict_bytes, dict_table_s, token_prefix_s)
        });
        let elapsed = t.elapsed();
        results.push(SortRow {
            method: "compare_fused v2 (u64 prefix Phase 2)".into(),
            elapsed_ms: elapsed.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });
        if let Some(ref r) = reference {
            for k in 0..n {
                let ra = slice_at(&token_flat, &token_bd, r[k] as usize);
                let rb = slice_at(&token_flat, &token_bd, indices[k] as usize);
                if ra != rb {
                    let ba = slice_at(&byte_flat, &byte_bd, r[k] as usize);
                    let bb = slice_at(&byte_flat, &byte_bd, indices[k] as usize);
                    if ba != bb {
                        eprintln!("v2 mismatch k={k}:");
                        eprintln!("  v1 row idx {} = {:?}", r[k], std::str::from_utf8(ba).unwrap_or("<bin>"));
                        eprintln!("  v2 row idx {} = {:?}", indices[k], std::str::from_utf8(bb).unwrap_or("<bin>"));
                        eprintln!("  v1 tokens = {:?}", ra);
                        eprintln!("  v2 tokens = {:?}", rb);
                        for &t in ra {
                            let e = dict_table_s[t as usize];
                            let off = (e >> 16) as usize;
                            let len = (e & 0xffff) as usize;
                            eprintln!("    a tok {t} = {:?}", std::str::from_utf8(&dict_bytes[off..off+len]).unwrap_or("<bin>"));
                        }
                        for &t in rb {
                            let e = dict_table_s[t as usize];
                            let off = (e >> 16) as usize;
                            let len = (e & 0xffff) as usize;
                            eprintln!("    b tok {t} = {:?}", std::str::from_utf8(&dict_bytes[off..off+len]).unwrap_or("<bin>"));
                        }
                        panic!("v2 sort order differs from v1");
                    }
                }
            }
        }
    }

    // ── Method 2: sort the pre-decoded flat bytes (sort-only timing) ────
    {
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let t = Instant::now();
        indices.sort_unstable_by(|&i, &j| {
            let a = slice_at(&byte_flat, &byte_bd, i as usize);
            let b = slice_at(&byte_flat, &byte_bd, j as usize);
            a.cmp(b)
        });
        let elapsed = t.elapsed();
        results.push(SortRow {
            method: "byte cmp (flat bytes, sort only, unstable)".into(),
            elapsed_ms: elapsed.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });
        if let Some(ref r) = reference {
            // unstable sort may permute equal-keyed elements differently;
            // verify by mapping both index lists back to bytes and comparing
            // those byte sequences instead of the index permutations.
            for k in 0..n {
                let ra = slice_at(&byte_flat, &byte_bd, r[k] as usize);
                let rb = slice_at(&byte_flat, &byte_bd, indices[k] as usize);
                assert_eq!(ra, rb, "compare_fused order ≠ byte cmp order at k={k}");
            }
        }
    }

    // ── Method 3: decode-then-sort (end-to-end, includes decode cost) ───
    {
        let t = Instant::now();
        let (decoded_flat, decoded_bd) = out.col.decode_all();
        let mut indices: Vec<u32> = (0..n as u32).collect();
        indices.sort_unstable_by(|&i, &j| {
            let a = slice_at(&decoded_flat, &decoded_bd, i as usize);
            let b = slice_at(&decoded_flat, &decoded_bd, j as usize);
            a.cmp(b)
        });
        let elapsed = t.elapsed();
        results.push(SortRow {
            method: "byte cmp (decode + sort, end-to-end)".into(),
            elapsed_ms: elapsed.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });
        if let Some(ref r) = reference {
            for k in 0..n {
                let ra = slice_at(&byte_flat, &byte_bd, r[k] as usize);
                let rb = slice_at(&decoded_flat, &decoded_bd, indices[k] as usize);
                assert_eq!(ra, rb, "compare_fused order ≠ decode+sort at k={k}");
            }
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
