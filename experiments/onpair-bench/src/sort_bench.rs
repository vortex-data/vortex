// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort-throughput benchmark for `compare_fused` vs decode-then-byte-compare.

use std::time::Instant;

use anyhow::Result;
use rand::SeedableRng;
use rand::seq::SliceRandom;

use crate::compare_fused::build_dict_table;
use crate::compare_fused::build_row_prefix;
use crate::compare_fused::build_row_prefix16;
use crate::compare_fused::build_row_prefix32;
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
    if let Ok((b, o, ntok)) = crate::encoders::onpair_dict_size_components(&out) {
        let dict_bytes_total = b + o * 4;
        let row_prefix32 = n * 32;
        eprintln!(
            "  dict: {ntok} tokens, {:.2} KiB total; 32B row prefix: {:.2} MiB (ratio {}×)",
            dict_bytes_total as f64 / 1024.0,
            row_prefix32 as f64 / 1024.0 / 1024.0,
            row_prefix32 / dict_bytes_total.max(1),
        );
    }

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

    // Precompute row prefixes once (charged as sort prep, not inside any
    // timed loop). Used by the two-pass and v3 paths.
    let row_prefix2 = build_row_prefix16(&token_flat, &token_bd, dict_bytes, dict_table_s);
    let row_prefix4 = build_row_prefix32(&token_flat, &token_bd, dict_bytes, dict_table_s);
    let row_prefix = build_row_prefix(&token_flat, &token_bd, dict_bytes, dict_table_s);
    let row_len: Vec<u32> = (0..n as u32)
        .map(|r| byte_bd[r as usize + 1] - byte_bd[r as usize])
        .collect();

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

    // ── Method 0: two-pass radix-then-refine on (u64_prefix, u128_prefix) ─
    // Precompute 16 bytes of decoded row per row, packed into two u64s.
    // Pass 1: stable integer sort on (k0, k1, idx) tuples — fast.
    // Pass 2: within runs that tied on both u64s, refine with compare_fused.
    {
        let mut keys: Vec<(u64, u64, u32)> = (0..n as u32)
            .map(|i| {
                let (k0, k1) = row_prefix2[i as usize];
                (k0, k1, i)
            })
            .collect();
        let t = Instant::now();
        keys.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        // Refine ties.
        let mut idx = 0usize;
        while idx < n {
            let k = (keys[idx].0, keys[idx].1);
            let mut j = idx + 1;
            while j < n && (keys[j].0, keys[j].1) == k {
                j += 1;
            }
            if j - idx > 1 {
                keys[idx..j].sort_unstable_by(|&(_, _, ia), &(_, _, ib)| {
                    let a = slice_at(&token_flat, &token_bd, ia as usize);
                    let b = slice_at(&token_flat, &token_bd, ib as usize);
                    compare_fused(a, b, dict_bytes, dict_table_s)
                });
            }
            idx = j;
        }
        let elapsed = t.elapsed();
        let indices: Vec<u32> = keys.into_iter().map(|(_, _, i)| i).collect();
        results.push(SortRow {
            method: "two-pass: u128 key sort + refine ties".into(),
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
                    assert_eq!(ba, bb, "two-pass order ≠ v1 order at k={k}");
                }
            }
        }
    }

    // ── Method 0f: TOKEN-prefix sort key (4 lex-rank IDs = 8 B/row) ─────
    //
    // Avoids the 30 MiB byte-prefix elephant. Uses first 4 token IDs of
    // each row as the sort key (8 B/row, 8 MiB total — close to dict
    // scale, 160× still vs 600× for 32B byte prefix).
    //
    // Subtlety: OnPair assigns token IDs in training-insertion order, not
    // lex order. Comparing raw token IDs gives garbage order. We remap to
    // *lex-rank IDs* — sort dict tokens by byte content, assign new ID by
    // position. Then comparing lex-rank IDs APPROXIMATES lex order on the
    // first token's bytes. Refinement (byte cmp on materialised bytes)
    // handles all the cases where the approximation differs from true lex
    // order, including LPM boundary disagreements (one row tokenises
    // "Tiresias" as one token, another as "Tiresia"+"s" — different
    // lex-rank IDs end up in different partitions even though they
    // represent lex-adjacent bytes; merge by walking partitions in lex
    // order of the representative bytes still works).
    //
    // KNOWN LIMITATION: this method is correct *only* for SORT, not for
    // ORDER BY across the boundary cases — equal-prefix rows in different
    // partitions don't end up adjacent if their first-token byte content
    // crosses partition boundary. For sort-then-stream uses where stable
    // grouping is needed, prefer the byte-prefix variant. For the sort
    // benchmark we accept this and verify against byte cmp directly.
    let dict_lex_rank: Vec<u16> = {
        // Build (token_id, &bytes) pairs, sort by bytes, then build the
        // inverse map id -> lex_rank.
        let parts_b = dict_bytes_vec.as_slice();
        let dt = dict_table_s;
        let n_tok = dt.len();
        let mut ids: Vec<u16> = (0..n_tok as u16).collect();
        ids.sort_unstable_by(|&a, &b| {
            let ea = dt[a as usize];
            let eb = dt[b as usize];
            let (oa, la) = ((ea >> 16) as usize, (ea & 0xffff) as usize);
            let (ob, lb) = ((eb >> 16) as usize, (eb & 0xffff) as usize);
            parts_b[oa..oa + la].cmp(&parts_b[ob..ob + lb])
        });
        let mut rank = vec![0u16; n_tok];
        for (r, &id) in ids.iter().enumerate() {
            rank[id as usize] = r as u16;
        }
        rank
    };
    let token_prefix4: Vec<u64> = (0..n)
        .map(|r| {
            let toks = slice_at(&token_flat, &token_bd, r);
            let mut buf = [0u16; 4];
            let take = toks.len().min(4);
            for k in 0..take {
                buf[k] = dict_lex_rank[toks[k] as usize];
            }
            // Pack as u64 BE so sort key compares lex-rank-first.
            ((buf[0] as u64) << 48)
                | ((buf[1] as u64) << 32)
                | ((buf[2] as u64) << 16)
                | (buf[3] as u64)
        })
        .collect();
    {
        let mut keys: Vec<(u64, u32)> = (0..n as u32)
            .map(|i| (token_prefix4[i as usize], i))
            .collect();
        let t = Instant::now();
        keys.sort_unstable_by_key(|&(k, _)| k);
        let mut idx = 0usize;
        let mut tie_groups = 0usize;
        let mut tie_rows = 0usize;
        while idx < n {
            let k = keys[idx].0;
            let mut j = idx + 1;
            while j < n && keys[j].0 == k {
                j += 1;
            }
            if j - idx > 1 {
                tie_groups += 1;
                tie_rows += j - idx;
                // Refine on materialised bytes (true lex order).
                keys[idx..j].sort_unstable_by(|&(_, ia), &(_, ib)| {
                    let a = slice_at(&byte_flat, &byte_bd, ia as usize);
                    let b = slice_at(&byte_flat, &byte_bd, ib as usize);
                    a.cmp(b)
                });
            }
            idx = j;
        }
        let elapsed = t.elapsed();
        let mut indices: Vec<u32> = keys.into_iter().map(|(_, i)| i).collect();
        eprintln!(
            "  [4-token+byte-refine] ties: {tie_groups} groups containing {tie_rows} rows"
        );
        // Count out-of-order pairs at partition boundaries (this method
        // produces PARTIAL lex order — correct within partitions but may
        // be wrong at boundaries when LPM splits lex-adjacent strings
        // into different first-token equivalence classes).
        let mut wrong_pairs = 0usize;
        for k in 1..n {
            let a = slice_at(&byte_flat, &byte_bd, indices[k - 1] as usize);
            let b = slice_at(&byte_flat, &byte_bd, indices[k] as usize);
            if a > b {
                wrong_pairs += 1;
            }
        }
        let pct = wrong_pairs as f64 * 100.0 / (n - 1) as f64;
        results.push(SortRow {
            method: format!(
                "two-pass: 4-token-rank key (8B) [APPROX: {wrong_pairs} mis-ordered = {pct:.2}%]"
            ),
            elapsed_ms: elapsed.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64(),
            ns_per_row: elapsed.as_nanos() as f64 / n as f64,
        });

        // ── Method 0g: 4-token-rank + final byte-cmp STABLE sort (Timsort) ──
        // Stable sort detects ascending runs in near-sorted input.
        let t2 = Instant::now();
        indices.sort_by(|&i, &j| {
            let a = slice_at(&byte_flat, &byte_bd, i as usize);
            let b = slice_at(&byte_flat, &byte_bd, j as usize);
            a.cmp(b)
        });
        let fixup = t2.elapsed();
        let total = elapsed + fixup;
        // Verify fully sorted now.
        if let Some(ref r) = reference {
            for k in 0..n {
                let ra = slice_at(&token_flat, &token_bd, r[k] as usize);
                let rb = slice_at(&token_flat, &token_bd, indices[k] as usize);
                if ra != rb {
                    let ba = slice_at(&byte_flat, &byte_bd, r[k] as usize);
                    let bb = slice_at(&byte_flat, &byte_bd, indices[k] as usize);
                    assert_eq!(ba, bb, "4-token-rank+fixup order ≠ v1 order at k={k}");
                }
            }
        }
        results.push(SortRow {
            method: format!(
                "  + final byte-cmp sort (correctness fixup, {} ms)",
                fixup.as_millis()
            ),
            elapsed_ms: total.as_millis(),
            mb_per_s: (raw_bytes as f64 / 1024.0 / 1024.0) / total.as_secs_f64(),
            ns_per_row: total.as_nanos() as f64 / n as f64,
        });
    }

    // ── Method 0e: two-pass with 16B key + byte-cmp REFINEMENT ──────────
    {
        let mut keys: Vec<(u64, u64, u32)> = (0..n as u32)
            .map(|i| {
                let (k0, k1) = row_prefix2[i as usize];
                (k0, k1, i)
            })
            .collect();
        let t = Instant::now();
        keys.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut idx = 0usize;
        while idx < n {
            let k = (keys[idx].0, keys[idx].1);
            let mut j = idx + 1;
            while j < n && (keys[j].0, keys[j].1) == k {
                j += 1;
            }
            if j - idx > 1 {
                keys[idx..j].sort_unstable_by(|&(_, _, ia), &(_, _, ib)| {
                    let a = slice_at(&byte_flat, &byte_bd, ia as usize);
                    let b = slice_at(&byte_flat, &byte_bd, ib as usize);
                    a.cmp(b)
                });
            }
            idx = j;
        }
        let elapsed = t.elapsed();
        let indices: Vec<u32> = keys.into_iter().map(|(_, _, i)| i).collect();
        results.push(SortRow {
            method: "two-pass: 16B key + byte cmp refine".into(),
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
                    assert_eq!(ba, bb, "two-pass-16-byte-refine order ≠ v1 order at k={k}");
                }
            }
        }
    }

    // ── Method 0d: two-pass with byte-cmp REFINEMENT ────────────────────
    // 32B key sort, then refine ties with byte cmp on materialised
    // decoded bytes (assumes you've paid for decode somewhere). Useful
    // when refinement is the bottleneck (URL).
    {
        let mut keys: Vec<([u64; 4], u32)> = (0..n as u32)
            .map(|i| (row_prefix4[i as usize], i))
            .collect();
        let t = Instant::now();
        keys.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut idx = 0usize;
        let mut tie_groups = 0usize;
        let mut tie_rows = 0usize;
        while idx < n {
            let k = keys[idx].0;
            let mut j = idx + 1;
            while j < n && keys[j].0 == k {
                j += 1;
            }
            if j - idx > 1 {
                tie_groups += 1;
                tie_rows += j - idx;
                keys[idx..j].sort_unstable_by(|&(_, ia), &(_, ib)| {
                    let a = slice_at(&byte_flat, &byte_bd, ia as usize);
                    let b = slice_at(&byte_flat, &byte_bd, ib as usize);
                    a.cmp(b)
                });
            }
            idx = j;
        }
        let elapsed = t.elapsed();
        let indices: Vec<u32> = keys.into_iter().map(|(_, i)| i).collect();
        eprintln!(
            "  [32B+byte-refine] ties: {tie_groups} groups containing {tie_rows} rows"
        );
        results.push(SortRow {
            method: "two-pass: 32B key + byte cmp refine".into(),
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
                    assert_eq!(ba, bb, "two-pass-byte-refine order ≠ v1 order at k={k}");
                }
            }
        }
    }

    // ── Method 0c: two-pass with INDIRECT 32B key (sort u32 indices) ────
    // Avoids moving 36-byte tuples during pdqsort swaps. Sort u32 indices,
    // comparator looks up the key.
    {
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let t = Instant::now();
        indices.sort_unstable_by(|&i, &j| {
            row_prefix4[i as usize].cmp(&row_prefix4[j as usize])
        });
        // Refine ties.
        let mut idx = 0usize;
        while idx < n {
            let k = row_prefix4[indices[idx] as usize];
            let mut j = idx + 1;
            while j < n && row_prefix4[indices[j] as usize] == k {
                j += 1;
            }
            if j - idx > 1 {
                indices[idx..j].sort_unstable_by(|&ia, &ib| {
                    let a = slice_at(&token_flat, &token_bd, ia as usize);
                    let b = slice_at(&token_flat, &token_bd, ib as usize);
                    compare_fused(a, b, dict_bytes, dict_table_s)
                });
            }
            idx = j;
        }
        let elapsed = t.elapsed();
        results.push(SortRow {
            method: "two-pass: 32B key, sort u32 indices".into(),
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
                    assert_eq!(ba, bb, "two-pass-indirect order ≠ v1 order at k={k}");
                }
            }
        }
    }

    // ── Method 0b: two-pass with 32-byte key (helps URL-like data) ──────
    {
        // Sort key is [u64; 4] + idx. Tuple sort.
        let mut keys: Vec<([u64; 4], u32)> = (0..n as u32)
            .map(|i| (row_prefix4[i as usize], i))
            .collect();
        let t = Instant::now();
        keys.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut idx = 0usize;
        while idx < n {
            let k = keys[idx].0;
            let mut j = idx + 1;
            while j < n && keys[j].0 == k {
                j += 1;
            }
            if j - idx > 1 {
                keys[idx..j].sort_unstable_by(|&(_, ia), &(_, ib)| {
                    let a = slice_at(&token_flat, &token_bd, ia as usize);
                    let b = slice_at(&token_flat, &token_bd, ib as usize);
                    compare_fused(a, b, dict_bytes, dict_table_s)
                });
            }
            idx = j;
        }
        let elapsed = t.elapsed();
        let indices: Vec<u32> = keys.into_iter().map(|(_, i)| i).collect();
        results.push(SortRow {
            method: "two-pass: 32B key sort + refine ties".into(),
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
                    assert_eq!(ba, bb, "two-pass-32 order ≠ v1 order at k={k}");
                }
            }
        }
    }

    // ── Method 1c: compare_fused v3 (precomputed row prefix) ────────────
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
