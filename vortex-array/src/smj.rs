// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sorted merge join sketch over normalized ord-byte rows.
//!
//! Exploratory. See `docs/developer-guide/internals/smj-ovc-design.md` for the
//! design discussion this corresponds to.
//!
//! Rows are presented as a slice of [`BinaryView`] entries plus the backing
//! data buffers. Each row is one variable-length byte sequence — the row's
//! normalized sort key, with per-column ord-byte contributions concatenated
//! in sort order. Comparison is byte-wise [`Ord`] and matches the logical
//! sort order by construction.
//!
//! This module implements only the merge driver. Producing the ord-bytes from
//! a Vortex array (the "row encoder") is intentionally out of scope here.

#![allow(dead_code)] // exploratory sketch; consumers not yet wired up

use std::cmp::Ordering;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

use crate::arrays::varbinview::BinaryView;

/// One side of a sorted merge join, viewed as ord-byte rows.
///
/// Short rows (≤ 12 bytes) live inline in the view header; longer rows
/// reference into `buffers`. Held by borrow for the duration of a merge.
pub(crate) struct OrdRows<'a> {
    views: &'a [BinaryView],
    buffers: &'a [ByteBuffer],
}

impl<'a> OrdRows<'a> {
    /// Wrap a slice of views over their backing data buffers.
    pub(crate) fn new(views: &'a [BinaryView], buffers: &'a [ByteBuffer]) -> Self {
        Self { views, buffers }
    }

    /// Number of rows.
    pub(crate) fn len(&self) -> usize {
        self.views.len()
    }

    /// Byte slice for row `i`.
    #[inline]
    pub(crate) fn row(&self, i: usize) -> &[u8] {
        let view = &self.views[i];
        if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let r = view.as_view();
            &self.buffers[r.buffer_index as usize].as_slice()[r.as_range()]
        }
    }
}

/// Inner sorted merge join over two pre-sorted ord-byte sides.
///
/// Returns `(left_idx, right_idx)` pairs for rows whose ord-bytes compare
/// equal. Both inputs must already be sorted ascending by their ord-bytes.
///
/// Runs of equal keys produce a full cross product, matching standard SMJ
/// semantics for N:M cardinality. Inputs are limited to `u32::MAX` rows per
/// side.
pub(crate) fn merge_inner_join(left: &OrdRows<'_>, right: &OrdRows<'_>) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let (mut l, mut r) = (0usize, 0usize);
    while l < left.len() && r < right.len() {
        match left.row(l).cmp(right.row(r)) {
            Ordering::Less => l += 1,
            Ordering::Greater => r += 1,
            Ordering::Equal => {
                let l_end = run_end(left, l);
                let r_end = run_end(right, r);
                for li in l..l_end {
                    for ri in r..r_end {
                        out.push((
                            u32::try_from(li).vortex_expect("row index fits in u32"),
                            u32::try_from(ri).vortex_expect("row index fits in u32"),
                        ));
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
fn run_end(side: &OrdRows<'_>, start: usize) -> usize {
    let key = side.row(start);
    let mut end = start + 1;
    while end < side.len() && side.row(end) == key {
        end += 1;
    }
    end
}

/// Byte-level Offset-Value Code against a predecessor row.
///
/// `offset` is the number of leading bytes the row shares with the
/// predecessor. `next` is the row's byte at `offset`, or `None` if the row
/// equals the predecessor up to its end.
///
/// Smaller `Ovc` = smaller key (under ascending sort): a row that agrees
/// with the predecessor longer is closer to it and therefore comes first;
/// at equal `offset` the smaller next byte wins, with `None` beating any
/// `Some(_)` (a row equal to the predecessor sorts right after it).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Ovc {
    offset: u32,
    next: Option<u8>,
}

impl Ord for Ovc {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .offset
            .cmp(&self.offset)
            .then(self.next.cmp(&other.next))
    }
}

impl PartialOrd for Ovc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute `row`'s OVC against `predecessor`. A `None` predecessor is `-∞`:
/// offset = 0, next = the row's first byte (or `None` if the row is empty).
fn ovc_of(row: &[u8], predecessor: Option<&[u8]>) -> Ovc {
    let p = predecessor.unwrap_or(&[]);
    let common = row.iter().zip(p.iter()).take_while(|(a, b)| a == b).count();
    Ovc {
        offset: u32::try_from(common).vortex_expect("prefix length fits in u32"),
        next: row.get(common).copied(),
    }
}

/// Inner sorted merge join driven by byte-level OVC.
///
/// Equivalent in output to [`merge_inner_join`]; the difference is that most
/// row comparisons reduce to one `Ovc` integer compare instead of a full
/// `memcmp` from byte 0. On `Ovc` tie a tail compare from `offset` onward
/// resolves the ordering and forces a loser OVC recompute.
pub(crate) fn merge_inner_join_ovc(left: &OrdRows<'_>, right: &OrdRows<'_>) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    if left.len() == 0 || right.len() == 0 {
        return out;
    }

    let (mut l, mut r) = (0usize, 0usize);
    let mut pred: Option<&[u8]> = None;
    let mut ovc_l = ovc_of(left.row(l), pred);
    let mut ovc_r = ovc_of(right.row(r), pred);

    while l < left.len() && r < right.len() {
        // Strict OVC compare resolves most rows in a single integer compare.
        // On tie, fall through to a tail compare past the matched byte; in
        // that case the loser's (offset, next) tuple is a lossy approximation
        // and must be recomputed against the new predecessor.
        let (cmp, loser_needs_recompute) = match ovc_l.cmp(&ovc_r) {
            Ordering::Less => (Ordering::Less, false),
            Ordering::Greater => (Ordering::Greater, false),
            Ordering::Equal => {
                let from = ovc_l.offset as usize + usize::from(ovc_l.next.is_some());
                let l_tail = left.row(l).get(from..).unwrap_or(&[]);
                let r_tail = right.row(r).get(from..).unwrap_or(&[]);
                (l_tail.cmp(r_tail), true)
            }
        };

        match cmp {
            Ordering::Less => {
                pred = Some(left.row(l));
                l += 1;
                if l == left.len() {
                    break;
                }
                ovc_l = ovc_of(left.row(l), pred);
                if loser_needs_recompute {
                    ovc_r = ovc_of(right.row(r), pred);
                }
            }
            Ordering::Greater => {
                pred = Some(right.row(r));
                r += 1;
                if r == right.len() {
                    break;
                }
                ovc_r = ovc_of(right.row(r), pred);
                if loser_needs_recompute {
                    ovc_l = ovc_of(left.row(l), pred);
                }
            }
            Ordering::Equal => {
                let l_end = run_end(left, l);
                let r_end = run_end(right, r);
                for li in l..l_end {
                    for ri in r..r_end {
                        out.push((
                            u32::try_from(li).vortex_expect("row index fits in u32"),
                            u32::try_from(ri).vortex_expect("row index fits in u32"),
                        ));
                    }
                }
                pred = Some(left.row(l_end - 1));
                l = l_end;
                r = r_end;
                if l < left.len() {
                    ovc_l = ovc_of(left.row(l), pred);
                }
                if r < right.len() {
                    ovc_r = ovc_of(right.row(r), pred);
                }
            }
        }
    }
    out
}

/// Inner sorted merge join with a shared per-side prefix.
///
/// Both sides carry a constant prefix factored out of every row; payload rows
/// in `left` and `right` hold only the suffix bytes. If the two prefixes
/// differ, no rows can possibly match and the result is empty. If they're
/// equal, the merge reduces to a memcmp merge over the suffix payload rows.
///
/// This is the framework's degenerate "all rows in this block share one
/// prefix" case; see `docs/developer-guide/internals/smj-ovc-design.md`.
pub(crate) fn merge_inner_join_with_prefix(
    left_prefix: &[u8],
    left: &OrdRows<'_>,
    right_prefix: &[u8],
    right: &OrdRows<'_>,
) -> Vec<(u32, u32)> {
    if left_prefix != right_prefix {
        return Vec::new();
    }
    merge_inner_join(left, right)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBuffer;

    use super::*;
    use crate::arrays::varbinview::BinaryView;

    fn views_from(rows: &[&[u8]]) -> (Vec<BinaryView>, Vec<ByteBuffer>) {
        let mut data: Vec<u8> = Vec::new();
        let mut views = Vec::with_capacity(rows.len());
        for row in rows {
            if row.len() <= BinaryView::MAX_INLINED_SIZE {
                views.push(BinaryView::new_inlined(row));
            } else {
                let offset = u32::try_from(data.len()).vortex_expect("offset fits in u32");
                data.extend_from_slice(row);
                views.push(BinaryView::make_view(row, 0, offset));
            }
        }
        (views, vec![ByteBuffer::copy_from(&data)])
    }

    /// Run both merge variants and assert they agree.
    fn join_both(left: &[&[u8]], right: &[&[u8]]) -> Vec<(u32, u32)> {
        let (lv, lb) = views_from(left);
        let (rv, rb) = views_from(right);
        let l = OrdRows::new(&lv, &lb);
        let r = OrdRows::new(&rv, &rb);
        let memcmp = merge_inner_join(&l, &r);
        let ovc = merge_inner_join_ovc(&l, &r);
        assert_eq!(memcmp, ovc, "memcmp and ovc variants must agree");
        memcmp
    }

    #[rstest]
    #[case::no_overlap(&[&b"a"[..], &b"b"[..]], &[&b"c"[..], &b"d"[..]], vec![])]
    #[case::all_match(&[&b"a"[..], &b"b"[..]], &[&b"a"[..], &b"b"[..]], vec![(0, 0), (1, 1)])]
    #[case::interleaved(&[&b"a"[..], &b"c"[..]], &[&b"b"[..], &b"c"[..]], vec![(1, 1)])]
    #[case::left_empty(&[], &[&b"a"[..]], vec![])]
    #[case::right_empty(&[&b"a"[..]], &[], vec![])]
    fn merge_basic(
        #[case] left: &[&[u8]],
        #[case] right: &[&[u8]],
        #[case] expected: Vec<(u32, u32)>,
    ) {
        assert_eq!(join_both(left, right), expected);
    }

    #[test]
    fn cross_product_on_run() {
        // Left has two "a"s, right has three "a"s — emit 2x3 = 6 pairs.
        let result = join_both(
            &[&b"a"[..], &b"a"[..], &b"b"[..]],
            &[&b"a"[..], &b"a"[..], &b"a"[..]],
        );
        assert_eq!(result, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],);
    }

    #[test]
    fn mixed_inline_and_ref_rows() {
        let short = b"short" as &[u8];
        let long = b"longer than twelve bytes" as &[u8];
        let result = join_both(&[short, long], &[short, long]);
        assert_eq!(result, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn lex_order_across_lengths() {
        // memcmp gives "ab" < "abc" < "ac" — prefix sort.
        let result = join_both(&[&b"ab"[..], &b"abc"[..], &b"ac"[..]], &[&b"abc"[..]]);
        assert_eq!(result, vec![(1, 0)]);
    }

    /// Long shared prefix between consecutive rows on each side — the case
    /// where OVC's "loser-invariant" optimization should fire.
    #[test]
    fn long_shared_prefix() {
        let l: Vec<Vec<u8>> = (0..16u8)
            .map(|i| {
                let mut v = vec![0u8; 32];
                v[31] = i;
                v
            })
            .collect();
        let r = l.clone();
        let l_refs: Vec<&[u8]> = l.iter().map(|v| v.as_slice()).collect();
        let r_refs: Vec<&[u8]> = r.iter().map(|v| v.as_slice()).collect();
        let result = join_both(&l_refs, &r_refs);
        let expected: Vec<(u32, u32)> = (0..16u32).map(|i| (i, i)).collect();
        assert_eq!(result, expected);
    }

    /// Microbenchmark for the three merge variants on wide, prefix-heavy keys.
    ///
    /// Run with:
    ///   cargo test --release -p vortex-array smj::tests::bench -- \
    ///     --ignored --nocapture
    #[test]
    #[ignore = "benchmark, run explicitly"]
    #[allow(clippy::use_debug, clippy::cast_precision_loss)]
    fn bench_wide_prefix_heavy() {
        use std::time::Instant;

        // Workload: every row's key is `k1 || k2 || k3` of 8 bytes each.
        // k1 and k2 are identical across all rows on each side (the shared
        // "block prefix"); k3 varies. This is the case the user described:
        // wide keys, sorted, leading columns repeat across many rows, with
        // the discriminating bytes in the tail.
        const K1: [u8; 8] = [0xAA; 8];
        const K2: [u8; 8] = [0xBB; 8];
        const N_PER_SIDE: u64 = 100_000;
        const OVERLAP_START: u64 = 50_000; // right starts here -> 50k matches
        const ITERS: u32 = 20;

        let prefix: Vec<u8> = [&K1[..], &K2[..]].concat(); // 16 bytes
        let mk_full = |k3: u64| {
            let mut v = Vec::with_capacity(24);
            v.extend_from_slice(&K1);
            v.extend_from_slice(&K2);
            v.extend_from_slice(&k3.to_be_bytes());
            v
        };
        let mk_suffix = |k3: u64| k3.to_be_bytes().to_vec();

        let left_full: Vec<Vec<u8>> = (0..N_PER_SIDE).map(mk_full).collect();
        let right_full: Vec<Vec<u8>> = (OVERLAP_START..OVERLAP_START + N_PER_SIDE)
            .map(mk_full)
            .collect();
        let left_suf: Vec<Vec<u8>> = (0..N_PER_SIDE).map(mk_suffix).collect();
        let right_suf: Vec<Vec<u8>> = (OVERLAP_START..OVERLAP_START + N_PER_SIDE)
            .map(mk_suffix)
            .collect();

        fn to_refs(v: &[Vec<u8>]) -> Vec<&[u8]> {
            v.iter().map(|r| r.as_slice()).collect()
        }
        let (lvf, lbf) = views_from(&to_refs(&left_full));
        let (rvf, rbf) = views_from(&to_refs(&right_full));
        let (lvs, lbs) = views_from(&to_refs(&left_suf));
        let (rvs, rbs) = views_from(&to_refs(&right_suf));

        let l_full = OrdRows::new(&lvf, &lbf);
        let r_full = OrdRows::new(&rvf, &rbf);
        let l_suf = OrdRows::new(&lvs, &lbs);
        let r_suf = OrdRows::new(&rvs, &rbs);

        let time = |label: &str, mut f: Box<dyn FnMut() -> Vec<(u32, u32)>>| {
            // Warm-up.
            let warm = f();
            let t = Instant::now();
            let mut last = warm;
            for _ in 0..ITERS {
                last = f();
            }
            let elapsed = t.elapsed();
            let per_iter = elapsed / ITERS;
            let total_rows = u64::from(ITERS) * (N_PER_SIDE * 2);
            let ns_per_row = elapsed.as_nanos() as f64 / total_rows as f64;
            println!(
                "{:<10} {:>10?}/iter   {:>6.2} ns/input-row   pairs={}",
                label,
                per_iter,
                ns_per_row,
                last.len(),
            );
            last
        };

        println!(
            "\n== smj microbench: {} rows/side, 24B keys (16B common prefix), {} expected matches ==",
            N_PER_SIDE,
            N_PER_SIDE - OVERLAP_START,
        );

        let o1 = time("memcmp", Box::new(|| merge_inner_join(&l_full, &r_full)));
        let o2 = time("ovc", Box::new(|| merge_inner_join_ovc(&l_full, &r_full)));
        let o3 = time(
            "prefix",
            Box::new(|| merge_inner_join_with_prefix(&prefix, &l_suf, &prefix, &r_suf)),
        );

        assert_eq!(o1, o2, "memcmp and ovc disagree");
        assert_eq!(o1, o3, "memcmp and prefix disagree");
    }

    /// Sweep the shared-prefix length to find OVC's crossover point.
    ///
    /// Run with the same `--ignored --nocapture --release` flags as above.
    #[test]
    #[ignore = "benchmark, run explicitly"]
    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn bench_prefix_length_sweep() {
        use std::time::Instant;

        const N_PER_SIDE: u64 = 50_000;
        const OVERLAP_START: u64 = 25_000;
        const ITERS: u32 = 20;
        const PREFIX_LENS: &[usize] = &[8, 32, 128, 512, 2048];

        println!(
            "\n== smj prefix-length sweep: {} rows/side, 50% overlap ==",
            N_PER_SIDE,
        );
        println!(
            "{:<10}  {:>14}  {:>14}  {:>10}",
            "prefix B", "memcmp ns/row", "ovc ns/row", "ovc/memcmp",
        );

        for &prefix_len in PREFIX_LENS {
            let prefix = vec![0xCC; prefix_len];
            let mk = |k3: u64| {
                let mut v = Vec::with_capacity(prefix_len + 8);
                v.extend_from_slice(&prefix);
                v.extend_from_slice(&k3.to_be_bytes());
                v
            };
            let left: Vec<Vec<u8>> = (0..N_PER_SIDE).map(mk).collect();
            let right: Vec<Vec<u8>> = (OVERLAP_START..OVERLAP_START + N_PER_SIDE).map(mk).collect();
            let l_refs: Vec<&[u8]> = left.iter().map(|v| v.as_slice()).collect();
            let r_refs: Vec<&[u8]> = right.iter().map(|v| v.as_slice()).collect();
            let (lv, lb) = views_from(&l_refs);
            let (rv, rb) = views_from(&r_refs);
            let l = OrdRows::new(&lv, &lb);
            let r = OrdRows::new(&rv, &rb);

            // Warm
            drop(merge_inner_join(&l, &r));
            drop(merge_inner_join_ovc(&l, &r));

            let t = Instant::now();
            let mut last = Vec::new();
            for _ in 0..ITERS {
                last = merge_inner_join(&l, &r);
            }
            let memcmp_ns =
                t.elapsed().as_nanos() as f64 / (f64::from(ITERS) * (N_PER_SIDE * 2) as f64);

            let t = Instant::now();
            for _ in 0..ITERS {
                last = merge_inner_join_ovc(&l, &r);
            }
            let ovc_ns =
                t.elapsed().as_nanos() as f64 / (f64::from(ITERS) * (N_PER_SIDE * 2) as f64);

            println!(
                "{:<10}  {:>14.2}  {:>14.2}  {:>9.2}x   pairs={}",
                prefix_len,
                memcmp_ns,
                ovc_ns,
                ovc_ns / memcmp_ns,
                last.len(),
            );
        }
    }

    #[test]
    fn ovc_invariant_three_columns() {
        // Multi-column keys (8 bytes wide) with varied divergence positions
        // exercise both strict-OVC and OVC-tie paths.
        let mk = |a: u8, b: u8, c: u32| {
            let mut v = vec![a, b];
            v.extend_from_slice(&c.to_be_bytes());
            v
        };
        let left = [mk(1, 1, 10), mk(1, 2, 10), mk(1, 2, 20), mk(2, 0, 0)];
        let right = [mk(1, 1, 10), mk(1, 2, 20), mk(3, 0, 0)];
        let l_refs: Vec<&[u8]> = left.iter().map(|v| v.as_slice()).collect();
        let r_refs: Vec<&[u8]> = right.iter().map(|v| v.as_slice()).collect();
        let result = join_both(&l_refs, &r_refs);
        assert_eq!(result, [(0, 0), (2, 1)]);
    }
}
