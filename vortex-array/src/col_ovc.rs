// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recreate the experimental setup from Do & Graefe, "Robust and Efficient
//! Sorting with Offset-Value Coding" (arXiv:2209.08420), to validate the
//! paper's reported OVC speedup against its own baseline.
//!
//! Key choices from the paper:
//!  * Sort key = K columns of 8-byte integers with very few distinct values.
//!  * OVC is column-level: `offset` = first column where rows differ,
//!    `value` = column value at that offset. Packed into a u64.
//!  * Baseline `Full` comparator is interpreted: function-call-per-column,
//!    not SIMD memcmp.
//!
//! Exploratory; not wired into any operator. See
//! `docs/developer-guide/internals/smj-ovc-design.md`.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use std::cmp::Ordering;

/// Row-major i64 store: row `i` is `data[i*k .. (i+1)*k]`.
pub(crate) struct ColRows<'a> {
    data: &'a [i64],
    k: usize,
}

impl<'a> ColRows<'a> {
    pub(crate) fn new(data: &'a [i64], k: usize) -> Self {
        assert_eq!(data.len() % k, 0);
        Self { data, k }
    }
    pub(crate) fn len(&self) -> usize {
        self.data.len() / self.k
    }
    pub(crate) fn arity(&self) -> usize {
        self.k
    }
    #[inline]
    pub(crate) fn row(&self, i: usize) -> &[i64] {
        &self.data[i * self.k..(i + 1) * self.k]
    }
}

/// Paper's "Full" baseline: interpreted column-by-column comparison with
/// function-call dispatch per column. `black_box` defeats inlining so the
/// per-column cost is realistic for a dispatched comparator.
#[inline(never)]
fn cmp_full(a: &[i64], b: &[i64]) -> Ordering {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let cmp_fn: fn(&i64, &i64) -> Ordering = std::hint::black_box(i64::cmp);
        match cmp_fn(&a[i], &b[i]) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    Ordering::Equal
}

/// Column-level OVC packed into u64:
///   high 8 bits : (arity - offset)   -- larger offset = smaller OVC = sorts earlier
///   low 56 bits : sign-flipped value at offset, truncated to top 56 bits
/// A row equal to its predecessor (offset == arity) encodes to 0, which is
/// strictly less than any nonzero OVC under unsigned compare — duplicates
/// sort right after their predecessor.
#[inline]
fn encode_ovc(arity_minus_offset: u8, value_at_offset: i64) -> u64 {
    let v_unsigned = (value_at_offset as u64) ^ (1u64 << 63);
    (u64::from(arity_minus_offset) << 56) | (v_unsigned >> 8)
}

/// Compute OVC of `row` against `predecessor`. None predecessor is `-∞`:
/// every row diverges at column 0.
#[inline]
fn compute_ovc(row: &[i64], predecessor: Option<&[i64]>, arity: usize) -> u64 {
    match predecessor {
        None => encode_ovc(arity as u8, row[0]),
        Some(p) => {
            for i in 0..arity {
                if row[i] != p[i] {
                    return encode_ovc((arity - i) as u8, row[i]);
                }
            }
            0 // equal to predecessor
        }
    }
}

/// Compute OVC starting from a known minimum offset, exploiting the merge-
/// invariant that the new predecessor agrees with the old one for at least
/// `start_offset` columns.
#[inline]
fn compute_ovc_from(
    row: &[i64],
    predecessor: &[i64],
    arity: usize,
    start_offset: usize,
) -> u64 {
    for i in start_offset..arity {
        if row[i] != predecessor[i] {
            return encode_ovc((arity - i) as u8, row[i]);
        }
    }
    0
}

/// Decode the offset (in columns) from an OVC.
#[inline]
fn ovc_offset(ovc: u64, arity: usize) -> usize {
    arity - ((ovc >> 56) as usize)
}

/// Inner merge join using the "Full" interpreted baseline comparator.
pub(crate) fn merge_full(left: &ColRows<'_>, right: &ColRows<'_>) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let (mut l, mut r) = (0usize, 0usize);
    while l < left.len() && r < right.len() {
        match cmp_full(left.row(l), right.row(r)) {
            Ordering::Less => l += 1,
            Ordering::Greater => r += 1,
            Ordering::Equal => {
                let l_end = full_run_end(left, l);
                let r_end = full_run_end(right, r);
                for li in l..l_end {
                    for ri in r..r_end {
                        out.push((li as u32, ri as u32));
                    }
                }
                l = l_end;
                r = r_end;
            }
        }
    }
    out
}

#[inline]
fn full_run_end(side: &ColRows<'_>, start: usize) -> usize {
    let mut end = start + 1;
    while end < side.len() && cmp_full(side.row(start), side.row(end)) == Ordering::Equal {
        end += 1;
    }
    end
}

/// Inner merge join driven by column-level OVC, matching the paper's
/// algorithm:
///   * Each side carries an OVC against the most recently emitted row.
///   * On strict OVC inequality the winner advances and recomputes its OVC;
///     the loser's OVC is unchanged against the new predecessor (proven
///     invariant) so it carries forward as-is.
///   * On OVC tie the encoding (offset, value) was lossy — keys agree
///     through the divergence column but may differ deeper. Fall back to a
///     full row compare to break the tie, then recompute both OVCs against
///     the new predecessor.
pub(crate) fn merge_ovc(left: &ColRows<'_>, right: &ColRows<'_>) -> Vec<(u32, u32)> {
    let arity = left.arity();
    assert_eq!(arity, right.arity());
    let mut out = Vec::new();
    if left.len() == 0 || right.len() == 0 {
        return out;
    }

    let (mut l, mut r) = (0usize, 0usize);
    let mut ovc_l = compute_ovc(left.row(l), None, arity);
    let mut ovc_r = compute_ovc(right.row(r), None, arity);

    while l < left.len() && r < right.len() {
        let (cmp, tie) = match ovc_l.cmp(&ovc_r) {
            Ordering::Less => (Ordering::Less, false),
            Ordering::Greater => (Ordering::Greater, false),
            Ordering::Equal => (left.row(l).cmp(right.row(r)), true),
        };
        match cmp {
            Ordering::Less => {
                let pred_idx = l;
                l += 1;
                if l == left.len() {
                    break;
                }
                let pred = left.row(pred_idx);
                ovc_l = compute_ovc(left.row(l), Some(pred), arity);
                if tie {
                    ovc_r = compute_ovc(right.row(r), Some(pred), arity);
                }
            }
            Ordering::Greater => {
                let pred_idx = r;
                r += 1;
                if r == right.len() {
                    break;
                }
                let pred = right.row(pred_idx);
                ovc_r = compute_ovc(right.row(r), Some(pred), arity);
                if tie {
                    ovc_l = compute_ovc(left.row(l), Some(pred), arity);
                }
            }
            Ordering::Equal => {
                // True key equality. Emit N:M cross product over the runs.
                let l_end = run_end_eq(left, l);
                let r_end = run_end_eq(right, r);
                for li in l..l_end {
                    for ri in r..r_end {
                        out.push((li as u32, ri as u32));
                    }
                }
                let pred_idx = l_end - 1;
                l = l_end;
                r = r_end;
                if l < left.len() {
                    ovc_l = compute_ovc(left.row(l), Some(left.row(pred_idx)), arity);
                }
                if r < right.len() {
                    ovc_r = compute_ovc(right.row(r), Some(left.row(pred_idx)), arity);
                }
            }
        }
    }
    out
}

#[inline]
fn run_end_eq(side: &ColRows<'_>, start: usize) -> usize {
    let mut end = start + 1;
    let s = side.row(start);
    while end < side.len() && side.row(end) == s {
        end += 1;
    }
    end
}

/// Encode a row-major i64 stream into ord-bytes (sign-flipped big-endian
/// per column) so memcmp on the resulting buffer matches the logical sort.
pub(crate) fn to_ord_bytes(rows: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * 8);
    for &v in rows {
        let u = (v as u64) ^ (1u64 << 63);
        out.extend_from_slice(&u.to_be_bytes());
    }
    out
}

/// Byte-memcmp baseline operating on a pre-encoded ord-byte buffer. This is
/// the comparator that wins in our ord-byte SMJ pipeline; the paper does
/// not include it in its baselines.
pub(crate) fn merge_memcmp_bytes(
    left_bytes: &[u8],
    right_bytes: &[u8],
    row_bytes: usize,
) -> Vec<(u32, u32)> {
    let l_n = left_bytes.len() / row_bytes;
    let r_n = right_bytes.len() / row_bytes;
    let mut out = Vec::new();
    let (mut l, mut r) = (0usize, 0usize);
    while l < l_n && r < r_n {
        let lb = &left_bytes[l * row_bytes..(l + 1) * row_bytes];
        let rb = &right_bytes[r * row_bytes..(r + 1) * row_bytes];
        match lb.cmp(rb) {
            Ordering::Less => l += 1,
            Ordering::Greater => r += 1,
            Ordering::Equal => {
                let mut le = l + 1;
                while le < l_n && &left_bytes[le * row_bytes..(le + 1) * row_bytes] == lb {
                    le += 1;
                }
                let mut re = r + 1;
                while re < r_n && &right_bytes[re * row_bytes..(re + 1) * row_bytes] == rb {
                    re += 1;
                }
                for li in l..le {
                    for ri in r..re {
                        out.push((li as u32, ri as u32));
                    }
                }
                l = le;
                r = re;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// Build N sorted i64 rows with K columns. The first `shared_prefix`
    /// columns hold the same value across all rows; the remaining columns
    /// take values from a small distinct set, mapped from a row index so
    /// rows are in sorted order. This is the paper's workload shape.
    fn make_sorted(
        n: usize,
        k: usize,
        shared_prefix: usize,
        distinct_per_col: u32,
        offset: u64,
    ) -> Vec<i64> {
        assert!(shared_prefix <= k);
        let varying = k - shared_prefix;
        let mut data = Vec::with_capacity(n * k);
        // Encode each row as a base-`distinct_per_col` number where each
        // digit is a column value. Add `offset` so left/right can overlap
        // partially.
        for row_i in 0..(n as u64) {
            let mut idx = row_i + offset;
            // Prefix
            data.extend(std::iter::repeat_n(7, shared_prefix));
            // Varying columns: highest-order digit first
            let mut digits = Vec::with_capacity(varying);
            for _ in 0..varying {
                digits.push((idx % u64::from(distinct_per_col)) as i64);
                idx /= u64::from(distinct_per_col);
            }
            digits.reverse();
            data.extend(digits);
        }
        // Sort by the row (lex on i64s). The construction above is already
        // close to sorted but if `varying == 0` we'd be all-equal; sort to
        // be safe.
        let mut rows: Vec<Vec<i64>> = data.chunks(k).map(<[i64]>::to_vec).collect();
        rows.sort();
        rows.into_iter().flatten().collect()
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn agreement_small() {
        let l = make_sorted(50, 4, 1, 4, 0);
        let r = make_sorted(50, 4, 1, 4, 7);
        let lr = ColRows::new(&l, 4);
        let rr = ColRows::new(&r, 4);
        let lb = to_ord_bytes(&l);
        let rb = to_ord_bytes(&r);
        let a = merge_full(&lr, &rr);
        let b = merge_ovc(&lr, &rr);
        let c = merge_memcmp_bytes(&lb, &rb, 4 * 8);
        assert_eq!(a, b, "full and ovc disagree");
        assert_eq!(a, c, "full and memcmp disagree");
    }

    /// Recreate the paper's Table 12 setup as closely as possible: K i64
    /// columns with `shared_prefix` leading columns constant, vary the
    /// shared prefix length, measure CPU per row for each comparator.
    ///
    /// To isolate the comparator/merge cost from N:M cross-product emit
    /// cost, this uses **disjoint ranges**: left has indices [0, N), right
    /// has [N, 2N). The merge advances all left then all right (or
    /// interleaved when prefix=0), with zero matches emitted. This is what
    /// the paper effectively measures since their cost numbers are for the
    /// priority queue and comparator, not output assembly.
    ///
    /// Run: cargo test --release -p vortex-array col_ovc::tests::bench -- \
    ///     --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "benchmark, run explicitly"]
    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn bench_paper_recreation() {
        const N_PER_SIDE: usize = 50_000;
        const ITERS: u32 = 20;
        const K: usize = 8;
        const DISTINCT: u32 = 20; // enough distinct values for disjoint ranges up to p=4

        println!(
            "\n== OVC paper recreation: K={K} cols, {} rows/side, {DISTINCT} distinct/col, disjoint sides ==",
            N_PER_SIDE,
        );
        println!(
            "{:<10} {:>14} {:>14} {:>14}   {:>10} {:>10}",
            "prefix",
            "Full ns/row",
            "OVC ns/row",
            "memcmp ns/row",
            "OVC/Full",
            "memcmp/Full",
        );

        for &shared_prefix in &[0usize, 2, 4] {
            // For disjoint ranges we need distinct^(K - prefix) >= 2N.
            let varying = K - shared_prefix;
            let max_distinct: u64 = (DISTINCT as u64).saturating_pow(varying as u32);
            if max_distinct < (N_PER_SIDE as u64) * 2 {
                println!("{:<10}  (skipped — not enough distinct values for disjoint sides)", shared_prefix);
                continue;
            }
            let right_offset = N_PER_SIDE as u64;

            let left_data = make_sorted(N_PER_SIDE, K, shared_prefix, DISTINCT, 0);
            let right_data = make_sorted(N_PER_SIDE, K, shared_prefix, DISTINCT, right_offset);
            let l = ColRows::new(&left_data, K);
            let r = ColRows::new(&right_data, K);
            let lb = to_ord_bytes(&left_data);
            let rb = to_ord_bytes(&right_data);
            let row_bytes = K * 8;

            let measure = |mut f: Box<dyn FnMut() -> Vec<(u32, u32)>>| -> (f64, Vec<(u32, u32)>) {
                drop(f());
                let t = Instant::now();
                let mut last = Vec::new();
                for _ in 0..ITERS {
                    last = f();
                }
                let total_rows = u64::from(ITERS) * (N_PER_SIDE as u64) * 2;
                let ns = t.elapsed().as_nanos() as f64 / total_rows as f64;
                (ns, last)
            };

            let (full_ns, out_full) = measure(Box::new(|| merge_full(&l, &r)));
            let (ovc_ns, out_ovc) = measure(Box::new(|| merge_ovc(&l, &r)));
            let (mc_ns, out_mc) = measure(Box::new(|| merge_memcmp_bytes(&lb, &rb, row_bytes)));
            assert_eq!(out_full, out_ovc, "full and ovc disagree");
            assert_eq!(out_full, out_mc, "full and memcmp disagree");

            println!(
                "{:<10} {:>14.2} {:>14.2} {:>14.2}   {:>9.2}x {:>9.2}x   pairs={}",
                shared_prefix,
                full_ns,
                ovc_ns,
                mc_ns,
                ovc_ns / full_ns,
                mc_ns / full_ns,
                out_full.len(),
            );
        }
    }
}
