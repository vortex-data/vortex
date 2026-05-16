// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OVC over compressed columnar encodings (Dict, RunEnd, FSST-like) vs.
//! OVC over decompressed primitives vs. materialize + memcmp.
//!
//! The hypothesis: for encodings whose compressed form preserves ordering
//! (sorted-dict codes, RunEnd values, order-preserving FSST), the OVC merge
//! can operate directly on the encoded form — fewer bytes touched, fewer
//! cache misses. Decompressing throws away that structural advantage.
//!
//! Exploratory; see `docs/developer-guide/internals/smj-ovc-design.md`.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::unwrap_used,
    clippy::panic
)]

// ───────────────────────────────────────────────────────────────────────────
// Minimal in-module encodings (cleaner than wiring into the full Vortex
// ArrayRef plumbing for an exploratory bench).
// ───────────────────────────────────────────────────────────────────────────

/// Plain primitive column.
pub(crate) struct PrimI64<'a> {
    pub data: &'a [i64],
}

/// Dict-encoded column. `dict` is **sorted ascending** so code order == value
/// order; comparing codes alone gives the same answer as comparing values.
/// Assumes dicts are rank-aligned across all merging sides (a per-merge
/// upfront pass; not measured here).
pub(crate) struct DictI64<'a> {
    pub codes: &'a [u32],
    pub dict: &'a [i64],
}

/// Run-end-encoded column. `run_ends[k]` is the first row index NOT in run k.
/// `values[k]` is the value of run k. Sorted runs (monotonically increasing
/// values) are the SMJ-friendly shape.
pub(crate) struct RunEndI64<'a> {
    pub run_ends: &'a [u32],
    pub values: &'a [i64],
    pub len: usize,
}

impl<'a> RunEndI64<'a> {
    /// Find the run index containing logical row `row` via binary search.
    #[inline]
    pub fn run_of(&self, row: usize) -> usize {
        self.run_ends.partition_point(|&e| (e as usize) <= row)
    }
    /// Find the run index containing `row`, starting the search at hint
    /// `start_run`. Cheaper than `run_of` for monotone iteration.
    #[inline]
    pub fn run_of_hint(&self, row: usize, start_run: usize) -> usize {
        let mut r = start_run;
        while r < self.run_ends.len() && (self.run_ends[r] as usize) <= row {
            r += 1;
        }
        r
    }
    #[inline]
    pub fn value_at(&self, row: usize) -> i64 {
        self.values[self.run_of(row)]
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Decompression: encoded → Vec<i64>
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn dict_decompress(d: &DictI64<'_>) -> Vec<i64> {
    d.codes.iter().map(|&c| d.dict[c as usize]).collect()
}

pub(crate) fn runend_decompress(r: &RunEndI64<'_>) -> Vec<i64> {
    let mut out = Vec::with_capacity(r.len);
    let mut prev_end = 0u32;
    for (i, &end) in r.run_ends.iter().enumerate() {
        for _ in prev_end..end {
            out.push(r.values[i]);
        }
        prev_end = end;
    }
    out
}

// ───────────────────────────────────────────────────────────────────────────
// Materialization to ord-bytes (8 bytes/row, sign-flipped BE i64).
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn materialize_prim(p: &PrimI64<'_>) -> Vec<u8> {
    let mut out = vec![0u8; p.data.len() * 8];
    for (i, &v) in p.data.iter().enumerate() {
        let u = (v as u64) ^ (1u64 << 63);
        out[i * 8..(i + 1) * 8].copy_from_slice(&u.to_be_bytes());
    }
    out
}

pub(crate) fn materialize_dict(d: &DictI64<'_>) -> Vec<u8> {
    let mut out = vec![0u8; d.codes.len() * 8];
    for (i, &c) in d.codes.iter().enumerate() {
        let v = d.dict[c as usize];
        let u = (v as u64) ^ (1u64 << 63);
        out[i * 8..(i + 1) * 8].copy_from_slice(&u.to_be_bytes());
    }
    out
}

pub(crate) fn materialize_runend(r: &RunEndI64<'_>) -> Vec<u8> {
    let mut out = vec![0u8; r.len * 8];
    let mut prev_end = 0u32;
    for (i, &end) in r.run_ends.iter().enumerate() {
        let v = r.values[i];
        let u = (v as u64) ^ (1u64 << 63);
        let bytes = u.to_be_bytes();
        for row in prev_end..end {
            out[(row as usize) * 8..(row as usize + 1) * 8].copy_from_slice(&bytes);
        }
        prev_end = end;
    }
    out
}

// ───────────────────────────────────────────────────────────────────────────
// Single-column OVC helpers (8-bit offset + 56-bit value).
// ───────────────────────────────────────────────────────────────────────────

#[inline]
fn pack(arity_minus_offset: u8, value_unsigned: u64) -> u64 {
    (u64::from(arity_minus_offset) << 56) | (value_unsigned >> 8)
}

#[inline]
fn i64_to_unsigned(v: i64) -> u64 {
    (v as u64) ^ (1u64 << 63)
}

// ───────────────────────────────────────────────────────────────────────────
// n-way OVC merge over PRIMITIVE columns.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_ovc_prim(sides: &[PrimI64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if !sides[i].data.is_empty() {
            ovcs[i] = pack(1, i64_to_unsigned(sides[i].data[0]));
        }
    }
    let mut count = 0usize;
    loop {
        let mut min_ovc = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if indices[i] < sides[i].data.len() && ovcs[i] < min_ovc {
                min_ovc = ovcs[i];
                min_side = i;
            }
        }
        if min_side == usize::MAX {
            break;
        }
        count += 1;
        let pred_row = indices[min_side];
        indices[min_side] += 1;
        if indices[min_side] < sides[min_side].data.len() {
            let cur = sides[min_side].data[indices[min_side]];
            let pred = sides[min_side].data[pred_row];
            ovcs[min_side] = if cur == pred {
                0
            } else {
                pack(1, i64_to_unsigned(cur))
            };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// n-way OVC merge over DICT columns — compares CODES (since dicts are sorted
// & rank-aligned, code order == value order). The OVC packs the code itself
// as the "value", so OVC compares are u32-cheap.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_ovc_dict(sides: &[DictI64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if !sides[i].codes.is_empty() {
            ovcs[i] = pack(1, u64::from(sides[i].codes[0]) << 32);
        }
    }
    let mut count = 0usize;
    loop {
        let mut min_ovc = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if indices[i] < sides[i].codes.len() && ovcs[i] < min_ovc {
                min_ovc = ovcs[i];
                min_side = i;
            }
        }
        if min_side == usize::MAX {
            break;
        }
        count += 1;
        let pred_row = indices[min_side];
        indices[min_side] += 1;
        if indices[min_side] < sides[min_side].codes.len() {
            let cur = sides[min_side].codes[indices[min_side]];
            let pred = sides[min_side].codes[pred_row];
            ovcs[min_side] = if cur == pred {
                0
            } else {
                pack(1, u64::from(cur) << 32)
            };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// n-way OVC merge over RUN-END columns — compares VALUES at the current run,
// with per-side cached run pointers. Within a run, all rows trivially equal
// → OVC == 0 (duplicate-of-predecessor), which the merge driver could
// special-case as "emit-without-priority-queue" (paper's optimization).
// Here we just emit normally; the gain is that value_at is O(1) amortised.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_ovc_runend(sides: &[RunEndI64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut run_idx = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if sides[i].len > 0 {
            run_idx[i] = sides[i].run_of_hint(0, 0);
            ovcs[i] = pack(1, i64_to_unsigned(sides[i].values[run_idx[i]]));
        }
    }
    let mut count = 0usize;
    loop {
        let mut min_ovc = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if indices[i] < sides[i].len && ovcs[i] < min_ovc {
                min_ovc = ovcs[i];
                min_side = i;
            }
        }
        if min_side == usize::MAX {
            break;
        }
        count += 1;
        let pred_row = indices[min_side];
        indices[min_side] += 1;
        if indices[min_side] < sides[min_side].len {
            // Advance the cached run pointer monotonically.
            let new_run = sides[min_side].run_of_hint(indices[min_side], run_idx[min_side]);
            run_idx[min_side] = new_run;
            let pred_run = sides[min_side].run_of_hint(pred_row, run_idx[min_side]);
            ovcs[min_side] = if new_run == pred_run {
                0 // same run → same value as predecessor → duplicate
            } else {
                pack(1, i64_to_unsigned(sides[min_side].values[new_run]))
            };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// n-way memcmp merge over pre-materialized 8-byte rows.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_memcmp(sides: &[&[u8]]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut count = 0usize;
    loop {
        let mut min_side = usize::MAX;
        let mut min_bytes: &[u8] = &[];
        for i in 0..n {
            let rows = sides[i].len() / 8;
            if indices[i] < rows {
                let row = &sides[i][indices[i] * 8..(indices[i] + 1) * 8];
                if min_side == usize::MAX || row < min_bytes {
                    min_side = i;
                    min_bytes = row;
                }
            }
        }
        if min_side == usize::MAX {
            break;
        }
        count += 1;
        indices[min_side] += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn primitive_sorted(n: usize, start: i64) -> Vec<i64> {
        (0..n as i64).map(|i| start + i).collect()
    }

    /// Build a sorted dict-encoded side: `distinct` ascending dict values,
    /// codes uniformly distributed (deterministic), `start` shifts the range.
    fn dict_sorted(n: usize, distinct: usize, start: i64) -> (Vec<u32>, Vec<i64>) {
        let dict: Vec<i64> = (0..distinct as i64).map(|i| start + i * 17).collect();
        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            // monotone codes for a sorted output
            codes.push((i * distinct / n) as u32);
        }
        (codes, dict)
    }

    /// Build a run-end column: `runs` runs, each with `run_len` rows of one
    /// value, monotonically increasing values.
    fn runend_sorted(runs: usize, run_len: usize, start: i64) -> (Vec<u32>, Vec<i64>, usize) {
        let n = runs * run_len;
        let mut ends = Vec::with_capacity(runs);
        let mut values = Vec::with_capacity(runs);
        for r in 0..runs {
            ends.push(((r + 1) * run_len) as u32);
            values.push(start + r as i64 * 13);
        }
        (ends, values, n)
    }

    #[test]
    fn agreement_primitive() {
        let s0 = primitive_sorted(20, 0);
        let s1 = primitive_sorted(20, 20);
        let sides = vec![PrimI64 { data: &s0 }, PrimI64 { data: &s1 }];
        assert_eq!(merge_n_way_ovc_prim(&sides), 40);
    }

    #[test]
    fn agreement_dict() {
        let (c0, d0) = dict_sorted(20, 10, 0);
        let (c1, d1) = dict_sorted(20, 10, 200);
        let sides = vec![
            DictI64 { codes: &c0, dict: &d0 },
            DictI64 { codes: &c1, dict: &d1 },
        ];
        assert_eq!(merge_n_way_ovc_dict(&sides), 40);
    }

    #[test]
    fn agreement_runend() {
        let (e0, v0, n0) = runend_sorted(10, 5, 0);
        let (e1, v1, n1) = runend_sorted(10, 5, 1000);
        let sides = vec![
            RunEndI64 { run_ends: &e0, values: &v0, len: n0 },
            RunEndI64 { run_ends: &e1, values: &v1, len: n1 },
        ];
        assert_eq!(merge_n_way_ovc_runend(&sides), n0 + n1);
    }

    /// 8-way merge bench across encodings.
    ///
    /// Per encoding, three paths are timed:
    ///   1. OVC over compressed (encoding-aware comparator)
    ///   2. OVC over decompressed (decode to Vec<i64>, then prim OVC)
    ///   3. Materialize ord-bytes + memcmp merge
    ///
    /// Run: cargo test --release -p vortex-array ovc_encoded::tests::bench \
    ///     -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "benchmark, run explicitly"]
    #[allow(clippy::cast_precision_loss)]
    fn bench_encoded_8way() {
        const N: usize = 50_000;
        const N_SIDES: usize = 8;
        const ITERS: u32 = 10;

        // --- Primitive baseline ---
        println!("\n== 8-way merge, single i64 column, {N} rows/side ==");

        let prim_data: Vec<Vec<i64>> = (0..N_SIDES)
            .map(|i| primitive_sorted(N, (i * N) as i64))
            .collect();
        let prim_sides: Vec<PrimI64> = prim_data.iter().map(|d| PrimI64 { data: d }).collect();
        bench_one("PRIMITIVE", N * N_SIDES, ITERS, || {
            (
                merge_n_way_ovc_prim(&prim_sides) as u64,
                {
                    let mats: Vec<Vec<u8>> = prim_sides.iter().map(materialize_prim).collect();
                    let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                    merge_n_way_memcmp(&refs) as u64
                },
                None,
            )
        });

        // --- Dict ---
        let dict_data: Vec<(Vec<u32>, Vec<i64>)> = (0..N_SIDES)
            .map(|i| dict_sorted(N, 256, (i * N) as i64 * 17))
            .collect();
        let dict_sides: Vec<DictI64> = dict_data
            .iter()
            .map(|(c, d)| DictI64 { codes: c, dict: d })
            .collect();
        bench_three(
            "DICT (256 distinct vals)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_dict(&dict_sides) as u64,
            || {
                let decoded: Vec<Vec<i64>> = dict_sides.iter().map(dict_decompress).collect();
                let prim_view: Vec<PrimI64> =
                    decoded.iter().map(|d| PrimI64 { data: d }).collect();
                merge_n_way_ovc_prim(&prim_view) as u64
            },
            || {
                let mats: Vec<Vec<u8>> = dict_sides.iter().map(materialize_dict).collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                merge_n_way_memcmp(&refs) as u64
            },
        );

        // --- RunEnd (long runs) ---
        let runend_data_long: Vec<(Vec<u32>, Vec<i64>, usize)> = (0..N_SIDES)
            .map(|i| runend_sorted(500, N / 500, (i * N) as i64 * 13))
            .collect();
        let runend_sides_long: Vec<RunEndI64> = runend_data_long
            .iter()
            .map(|(e, v, n)| RunEndI64 { run_ends: e, values: v, len: *n })
            .collect();
        bench_three(
            "RUN-END (avg 100 rows/run)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_runend(&runend_sides_long) as u64,
            || {
                let decoded: Vec<Vec<i64>> =
                    runend_sides_long.iter().map(runend_decompress).collect();
                let prim_view: Vec<PrimI64> =
                    decoded.iter().map(|d| PrimI64 { data: d }).collect();
                merge_n_way_ovc_prim(&prim_view) as u64
            },
            || {
                let mats: Vec<Vec<u8>> =
                    runend_sides_long.iter().map(materialize_runend).collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                merge_n_way_memcmp(&refs) as u64
            },
        );

        // --- RunEnd (short runs) ---
        let runend_data_short: Vec<(Vec<u32>, Vec<i64>, usize)> = (0..N_SIDES)
            .map(|i| runend_sorted(N / 5, 5, (i * N) as i64 * 13))
            .collect();
        let runend_sides_short: Vec<RunEndI64> = runend_data_short
            .iter()
            .map(|(e, v, n)| RunEndI64 { run_ends: e, values: v, len: *n })
            .collect();
        bench_three(
            "RUN-END (avg 5 rows/run)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_runend(&runend_sides_short) as u64,
            || {
                let decoded: Vec<Vec<i64>> =
                    runend_sides_short.iter().map(runend_decompress).collect();
                let prim_view: Vec<PrimI64> =
                    decoded.iter().map(|d| PrimI64 { data: d }).collect();
                merge_n_way_ovc_prim(&prim_view) as u64
            },
            || {
                let mats: Vec<Vec<u8>> =
                    runend_sides_short.iter().map(materialize_runend).collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                merge_n_way_memcmp(&refs) as u64
            },
        );
    }

    fn bench_one(
        label: &str,
        total_rows: usize,
        iters: u32,
        mut f: impl FnMut() -> (u64, u64, Option<u64>),
    ) {
        println!("\n  -- {label} --");
        let _ = f();
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            let (a, b, c) = f();
            acc = acc.wrapping_add(a).wrapping_add(b).wrapping_add(c.unwrap_or(0));
        }
        let d = t.elapsed();
        let total = d.as_nanos() as f64 / (u64::from(iters) * total_rows as u64) as f64;
        println!("    end-to-end (one iter pair): {total:>8.2} ns/row   acc={acc}");
    }

    fn bench_three(
        label: &str,
        total_rows: usize,
        iters: u32,
        mut ovc_compressed: impl FnMut() -> u64,
        mut ovc_decompressed: impl FnMut() -> u64,
        mut mat_memcmp: impl FnMut() -> u64,
    ) {
        println!("\n  -- {label} --");
        for (name, run) in [
            ("OVC over compressed", &mut ovc_compressed as &mut dyn FnMut() -> u64),
            ("OVC over decompressed", &mut ovc_decompressed),
            ("materialize + memcmp", &mut mat_memcmp),
        ] {
            let _ = run();
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..iters {
                acc = acc.wrapping_add(std::hint::black_box(run()));
            }
            let d = t.elapsed();
            let ns = d.as_nanos() as f64 / (u64::from(iters) * total_rows as u64) as f64;
            println!("    {:<28} {:>8.2} ns/row   acc={acc}", name, ns);
        }
    }
}
