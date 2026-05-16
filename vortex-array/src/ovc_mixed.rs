// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Mixed-column n-way merge bench: OVC over columnar vs materialize +
//! memcmp. 5 sort-key columns: 2 i64 + 3 binary (sized to hit a target
//! total-key average like 200 B or 400 B). Two workloads: disjoint (no
//! matches) and dense (every row on every side is the same key, max
//! duplication).
//!
//! Per-phase timings are printed: int materialization, binary
//! materialization, full row materialization, memcmp merge, OVC merge.
//! Exploratory; see `docs/developer-guide/internals/smj-ovc-design.md`.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::unwrap_used,
    clippy::panic,
    clippy::if_same_then_else
)]

use std::cmp::Ordering;

/// Heterogeneous sort-key column. Mirrors Arrow value access (raw slices).
pub(crate) enum Col<'a> {
    Int(&'a [i64]),
    Binary { offsets: &'a [u32], data: &'a [u8] },
}

impl<'a> Col<'a> {
    fn len(&self) -> usize {
        match self {
            Col::Int(s) => s.len(),
            Col::Binary { offsets, .. } => offsets.len().saturating_sub(1),
        }
    }
}

/// One side of the merge: K columns, all the same row count.
pub(crate) struct MixedCols<'a> {
    pub(crate) cols: Vec<Col<'a>>,
}

impl<'a> MixedCols<'a> {
    pub(crate) fn new(cols: Vec<Col<'a>>) -> Self {
        let n = cols.first().map_or(0, Col::len);
        assert!(cols.iter().all(|c| c.len() == n));
        Self { cols }
    }
    pub(crate) fn arity(&self) -> usize {
        self.cols.len()
    }
    pub(crate) fn len(&self) -> usize {
        self.cols.first().map_or(0, Col::len)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Materialization: columnar → ord-byte varlen rows.
// Layout per row: concatenate per-column order-preserving bytes.
//   Int(i64): [0x01 null-tag] + 8 bytes sign-flipped big-endian   (9 B)
//   Binary:   [0x01 null-tag] + escape(0x00→0x00 0xFF) + [0x00 0x01] terminator
// ───────────────────────────────────────────────────────────────────────────

const INT_ORD_BYTES: usize = 9;

/// Escape-encode `src` into `dst`. Uses `slice::position` for run-finding
/// which the compiler vectorises on x86 to vpcmpeqb scans.
#[inline]
fn write_binary_ord(dst: &mut Vec<u8>, src: &[u8]) {
    dst.push(0x01);
    let mut i = 0;
    while i < src.len() {
        match src[i..].iter().position(|&b| b == 0x00) {
            Some(j) => {
                if j > 0 {
                    dst.extend_from_slice(&src[i..i + j]);
                }
                dst.extend_from_slice(&[0x00, 0xFF]);
                i += j + 1;
            }
            None => {
                dst.extend_from_slice(&src[i..]);
                break;
            }
        }
    }
    dst.extend_from_slice(&[0x00, 0x01]);
}

fn est_binary_per_row(offsets: &[u32]) -> usize {
    if offsets.len() > 1 {
        1 + (*offsets.last().unwrap() as usize / (offsets.len() - 1)) + 2
    } else {
        0
    }
}

/// Materialize all rows in `cols` to a single varlen ord-byte buffer.
pub(crate) fn materialize(cols: &MixedCols<'_>) -> (Vec<u32>, Vec<u8>) {
    let n = cols.len();
    let est: usize = cols
        .cols
        .iter()
        .map(|c| match c {
            Col::Int(_) => INT_ORD_BYTES,
            Col::Binary { offsets, .. } => est_binary_per_row(offsets),
        })
        .sum();
    let mut data = Vec::with_capacity(n * est);
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0u32);

    for row in 0..n {
        for col in &cols.cols {
            match col {
                Col::Int(s) => {
                    let v = s[row];
                    let u = (v as u64) ^ (1u64 << 63);
                    data.push(0x01);
                    data.extend_from_slice(&u.to_be_bytes());
                }
                Col::Binary { offsets: bo, data: bd } => {
                    let start = bo[row] as usize;
                    let end = bo[row + 1] as usize;
                    write_binary_ord(&mut data, &bd[start..end]);
                }
            }
        }
        offsets.push(data.len() as u32);
    }
    (offsets, data)
}

/// Materialize only the int columns. Used for phase-cost attribution.
pub(crate) fn materialize_int_only(cols: &MixedCols<'_>) -> Vec<u8> {
    let n = cols.len();
    let int_cols: Vec<&[i64]> = cols
        .cols
        .iter()
        .filter_map(|c| if let Col::Int(s) = c { Some(*s) } else { None })
        .collect();
    let k = int_cols.len();
    let mut out = vec![0u8; n * k * INT_ORD_BYTES];
    for row in 0..n {
        let row_off = row * k * INT_ORD_BYTES;
        for (ci, col) in int_cols.iter().enumerate() {
            let u = (col[row] as u64) ^ (1u64 << 63);
            let off = row_off + ci * INT_ORD_BYTES;
            out[off] = 0x01;
            out[off + 1..off + 9].copy_from_slice(&u.to_be_bytes());
        }
    }
    out
}

/// Materialize only the binary columns. Used for phase-cost attribution.
pub(crate) fn materialize_binary_only(cols: &MixedCols<'_>) -> (Vec<u32>, Vec<u8>) {
    let n = cols.len();
    let bin_cols: Vec<(&[u32], &[u8])> = cols
        .cols
        .iter()
        .filter_map(|c| match c {
            Col::Binary { offsets, data } => Some((*offsets, *data)),
            _ => None,
        })
        .collect();
    let est: usize = bin_cols.iter().map(|(o, _)| est_binary_per_row(o)).sum();
    let mut data = Vec::with_capacity(n * est);
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0u32);
    for row in 0..n {
        for (o, d) in &bin_cols {
            let s = o[row] as usize;
            let e = o[row + 1] as usize;
            write_binary_ord(&mut data, &d[s..e]);
        }
        offsets.push(data.len() as u32);
    }
    (offsets, data)
}

// ───────────────────────────────────────────────────────────────────────────
// n-way memcmp merge over pre-materialized varlen ord-byte rows.
// Linear-scan minimum across n sides per emit (n=8 is well within scan range).
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_memcmp(sides: &[(&[u32], &[u8])]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];

    let row_ptr = |side: usize, idx: usize| -> (*const u8, usize) {
        // Return raw pointer + length so we can hold one head reference across
        // loop iterations without lifetime conflicts.
        let (o, d) = sides[side];
        let s = o[idx] as usize;
        let e = o[idx + 1] as usize;
        (d[s..e].as_ptr(), e - s)
    };

    let mut count = 0usize;
    loop {
        let mut min_side = usize::MAX;
        let mut min_ptr: *const u8 = std::ptr::null();
        let mut min_len: usize = 0;
        for i in 0..n {
            let n_rows = sides[i].0.len().saturating_sub(1);
            if indices[i] < n_rows {
                let (p, l) = row_ptr(i, indices[i]);
                if min_side == usize::MAX {
                    min_side = i;
                    min_ptr = p;
                    min_len = l;
                } else {
                    // SAFETY: pointers/lengths are valid for `sides` borrow
                    let a = unsafe { std::slice::from_raw_parts(p, l) };
                    let b = unsafe { std::slice::from_raw_parts(min_ptr, min_len) };
                    if a < b {
                        min_side = i;
                        min_ptr = p;
                        min_len = l;
                    }
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

// ───────────────────────────────────────────────────────────────────────────
// OVC merge over mixed columnar inputs.
// ───────────────────────────────────────────────────────────────────────────

#[inline]
fn pack_ovc(arity_minus_offset: u8, value_u64: u64) -> u64 {
    (u64::from(arity_minus_offset) << 56) | (value_u64 >> 8)
}

#[inline]
fn ovc_value_at(col: &Col<'_>, row: usize) -> u64 {
    match col {
        Col::Int(s) => (s[row] as u64) ^ (1u64 << 63),
        Col::Binary { offsets, data } => {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let bytes = &data[start..end];
            let mut buf = [0u8; 8];
            let n = bytes.len().min(8);
            buf[..n].copy_from_slice(&bytes[..n]);
            u64::from_be_bytes(buf)
        }
    }
}

#[inline]
fn cmp_at(
    left: &MixedCols<'_>,
    lr: usize,
    right: &MixedCols<'_>,
    rr: usize,
    col: usize,
) -> Ordering {
    match (&left.cols[col], &right.cols[col]) {
        (Col::Int(a), Col::Int(b)) => a[lr].cmp(&b[rr]),
        (
            Col::Binary { offsets: ao, data: ad },
            Col::Binary { offsets: bo, data: bd },
        ) => {
            let a_s = ao[lr] as usize;
            let a_e = ao[lr + 1] as usize;
            let b_s = bo[rr] as usize;
            let b_e = bo[rr + 1] as usize;
            ad[a_s..a_e].cmp(&bd[b_s..b_e])
        }
        _ => panic!("column type mismatch"),
    }
}

#[inline]
fn cmp_row(left: &MixedCols<'_>, lr: usize, right: &MixedCols<'_>, rr: usize) -> Ordering {
    let arity = left.arity();
    for c in 0..arity {
        match cmp_at(left, lr, right, rr, c) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    Ordering::Equal
}

#[inline]
fn ovc_against(
    target: &MixedCols<'_>,
    tr: usize,
    pred: &MixedCols<'_>,
    pr: usize,
) -> u64 {
    let arity = target.arity();
    for c in 0..arity {
        if cmp_at(target, tr, pred, pr, c) != Ordering::Equal {
            return pack_ovc((arity - c) as u8, ovc_value_at(&target.cols[c], tr));
        }
    }
    0 // equal to predecessor
}

#[inline]
fn ovc_initial(target: &MixedCols<'_>, tr: usize) -> u64 {
    pack_ovc(target.arity() as u8, ovc_value_at(&target.cols[0], tr))
}

/// n-way OVC merge. Handles tie-breaking via full row compare AND recomputes
/// the OVCs of all sides that tied with the winner against the new
/// predecessor — the byte-OVC encoding is lossy on ties so those OVCs are
/// no longer valid as bare integers.
pub(crate) fn merge_n_way_ovc(sides: &[MixedCols<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let arity = sides[0].arity();
    assert!(sides.iter().all(|s| s.arity() == arity));

    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for (i, side) in sides.iter().enumerate() {
        if side.len() > 0 {
            ovcs[i] = ovc_initial(side, 0);
        }
    }

    let mut count = 0usize;
    loop {
        // Pass 1: find the smallest OVC integer.
        let mut min_ovc = u64::MAX;
        for i in 0..n {
            if indices[i] < sides[i].len() && ovcs[i] < min_ovc {
                min_ovc = ovcs[i];
            }
        }
        if min_ovc == u64::MAX {
            break;
        }

        // Pass 2: among sides with that OVC, pick the smallest by full row
        // compare. Record which sides tied — they all need OVC recompute.
        let mut min_side = usize::MAX;
        for i in 0..n {
            if indices[i] < sides[i].len() && ovcs[i] == min_ovc {
                if min_side == usize::MAX {
                    min_side = i;
                } else if cmp_row(&sides[i], indices[i], &sides[min_side], indices[min_side])
                    == Ordering::Less
                {
                    min_side = i;
                }
            }
        }

        count += 1;
        let pred_side = min_side;
        let pred_row = indices[min_side];
        indices[min_side] += 1;

        // Recompute the advancing side's OVC against its own previous row
        // (which is the new merge predecessor).
        if indices[min_side] < sides[min_side].len() {
            ovcs[min_side] = ovc_against(
                &sides[min_side],
                indices[min_side],
                &sides[pred_side],
                pred_row,
            );
        } else {
            ovcs[min_side] = u64::MAX;
        }

        // Recompute all OTHER sides that tied with the winner — their old
        // OVC was against the previous predecessor and is no longer correct.
        // Sides that did not tie keep their OVC (loser invariant).
        for i in 0..n {
            if i == min_side {
                continue;
            }
            if indices[i] < sides[i].len() && ovcs[i] == min_ovc {
                ovcs[i] = ovc_against(&sides[i], indices[i], &sides[pred_side], pred_row);
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    fn fill_random_bytes(rng: &mut StdRng, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = rng.random_range(0u8..=255u8);
        }
    }

    /// Build one side: 5 columns = (i64, i64, binary, binary, binary).
    ///
    /// `int_offset`: starting value of the leading int column (controls
    /// disjoint vs overlapping ranges across sides).
    /// `bin_seed`: rng seed for the binary content. Pass the same value to
    /// multiple sides to make their binary content identical (dense mode).
    /// `avg_binary`: bytes per binary value per row.
    #[allow(clippy::type_complexity)]
    fn build_side(
        n: usize,
        avg_binary: usize,
        int_offset: u64,
        bin_seed: u64,
    ) -> (Vec<i64>, Vec<i64>, [(Vec<u32>, Vec<u8>); 3]) {
        let mut rng = StdRng::seed_from_u64(bin_seed);
        let int0: Vec<i64> = (0..n as i64).map(|i| int_offset as i64 + i).collect();
        let int1: Vec<i64> = (0..n as i64).collect();
        let mut bins: [(Vec<u32>, Vec<u8>); 3] = std::array::from_fn(|_| {
            (Vec::with_capacity(n + 1), Vec::with_capacity(n * avg_binary))
        });
        for b in &mut bins {
            b.0.push(0u32);
        }
        for _row in 0..n {
            for bin in &mut bins {
                let mut buf = vec![0u8; avg_binary];
                fill_random_bytes(&mut rng, &mut buf);
                bin.1.extend_from_slice(&buf);
                bin.0.push(bin.1.len() as u32);
            }
        }
        (int0, int1, bins)
    }

    fn build_mixed<'a>(
        int0: &'a [i64],
        int1: &'a [i64],
        bins: &'a [(Vec<u32>, Vec<u8>); 3],
    ) -> MixedCols<'a> {
        MixedCols::new(vec![
            Col::Int(int0),
            Col::Int(int1),
            Col::Binary { offsets: &bins[0].0, data: &bins[0].1 },
            Col::Binary { offsets: &bins[1].0, data: &bins[1].1 },
            Col::Binary { offsets: &bins[2].0, data: &bins[2].1 },
        ])
    }

    #[test]
    fn agreement_disjoint() {
        let s1 = build_side(40, 60, 0, 1);
        let s2 = build_side(40, 60, 40, 2);
        let m1 = build_mixed(&s1.0, &s1.1, &s1.2);
        let m2 = build_mixed(&s2.0, &s2.1, &s2.2);
        let (o1, d1) = materialize(&m1);
        let (o2, d2) = materialize(&m2);
        let bytes = vec![(o1.as_slice(), d1.as_slice()), (o2.as_slice(), d2.as_slice())];
        let cols = vec![m1, m2];
        assert_eq!(merge_n_way_ovc(&cols), 80);
        assert_eq!(merge_n_way_memcmp(&bytes), 80);
    }

    #[test]
    fn agreement_dense() {
        // Same content across both sides — every row is a duplicate.
        let s1 = build_side(20, 60, 0, 42);
        let s2 = build_side(20, 60, 0, 42);
        let m1 = build_mixed(&s1.0, &s1.1, &s1.2);
        let m2 = build_mixed(&s2.0, &s2.1, &s2.2);
        let (o1, d1) = materialize(&m1);
        let (o2, d2) = materialize(&m2);
        let bytes = vec![(o1.as_slice(), d1.as_slice()), (o2.as_slice(), d2.as_slice())];
        let cols = vec![m1, m2];
        assert_eq!(merge_n_way_ovc(&cols), 40);
        assert_eq!(merge_n_way_memcmp(&bytes), 40);
    }

    /// 8-way merge bench across 4 configurations:
    ///   {200B key, 400B key} × {disjoint sides, dense sides}.
    ///
    /// Run: cargo test --release -p vortex-array ovc_mixed::tests::bench \
    ///     -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "benchmark, run explicitly"]
    #[allow(clippy::cast_precision_loss)]
    fn bench_8way_mixed() {
        const N: usize = 5_000;
        const N_SIDES: usize = 8;
        const ITERS: u32 = 5;

        // (label, avg_binary_per_col, expected_total_key_bytes_approx)
        // total ≈ 2 ints * 9 + 3 bins * (1 + avg + 2) = 18 + 3*(avg+3)
        // For total ≈ 200 → avg ≈ 60. For total ≈ 400 → avg ≈ 124.
        let key_sizes: &[(&str, usize)] = &[("200 B total", 60), ("400 B total", 124)];

        // Workload labels: (label, build_sides_fn).
        let workloads: &[&str] = &["disjoint", "dense"];

        for (key_label, avg_binary) in key_sizes {
            for &workload in workloads {
                let sides_raw: Vec<_> = (0..N_SIDES)
                    .map(|i| match workload {
                        "disjoint" => build_side(N, *avg_binary, (i * N) as u64, (i + 1) as u64),
                        "dense" => build_side(N, *avg_binary, 0, 42),
                        _ => unreachable!(),
                    })
                    .collect();

                let sides: Vec<MixedCols> = sides_raw
                    .iter()
                    .map(|r| build_mixed(&r.0, &r.1, &r.2))
                    .collect();

                let total_rows = (N * N_SIDES) as u64;
                let total_bytes: usize = sides_raw
                    .iter()
                    .map(|r| {
                        r.0.len() * 8
                            + r.1.len() * 8
                            + r.2.iter().map(|b| b.1.len()).sum::<usize>()
                    })
                    .sum();

                println!(
                    "\n== 8-way, K=5 (2 i64 + 3 binary), {key_label}, workload={workload}, \
                     {N} rows/side, ~{:.1} MB input ==",
                    total_bytes as f64 / 1e6,
                );

                let measure = |label: &str, mut f: Box<dyn FnMut() -> u64>| -> f64 {
                    let _ = f();
                    let t = Instant::now();
                    let mut acc = 0u64;
                    for _ in 0..ITERS {
                        acc = acc.wrapping_add(std::hint::black_box(f()));
                    }
                    let d = t.elapsed();
                    let ns = d.as_nanos() as f64 / (u64::from(ITERS) * total_rows) as f64;
                    println!("  {:<32} {:>10.2} ns/row   acc={acc}", label, ns);
                    ns
                };

                let ns_mat_int = measure(
                    "build: materialize int cols",
                    Box::new(|| sides.iter().map(|s| materialize_int_only(s).len() as u64).sum()),
                );
                let ns_mat_bin = measure(
                    "build: materialize binary cols",
                    Box::new(|| {
                        sides.iter().map(|s| materialize_binary_only(s).1.len() as u64).sum()
                    }),
                );
                let ns_mat_full = measure(
                    "build: materialize full rows",
                    Box::new(|| sides.iter().map(|s| materialize(s).1.len() as u64).sum()),
                );

                let mat: Vec<(Vec<u32>, Vec<u8>)> = sides.iter().map(materialize).collect();
                let mat_refs: Vec<(&[u32], &[u8])> =
                    mat.iter().map(|(o, d)| (o.as_slice(), d.as_slice())).collect();

                let ns_mc_merge = measure(
                    "merge: memcmp over ord rows",
                    Box::new(|| merge_n_way_memcmp(&mat_refs) as u64),
                );
                let ns_ovc_merge = measure(
                    "merge: OVC over columns",
                    Box::new(|| merge_n_way_ovc(&sides) as u64),
                );

                let mat_total = ns_mat_full + ns_mc_merge;
                let ovc_total = ns_ovc_merge;
                println!("  ──────────────────────────────────────────────────────────");
                println!(
                    "  ord-byte pipeline (mat+merge): {:>8.2} ns/row    mat-int %: {:>5.1}    \
                     mat-bin %: {:>5.1}    merge %: {:>5.1}",
                    mat_total,
                    100.0 * ns_mat_int / mat_total,
                    100.0 * ns_mat_bin / mat_total,
                    100.0 * ns_mc_merge / mat_total,
                );
                println!("  OVC pipeline      (merge):     {:>8.2} ns/row", ovc_total);
                println!(
                    "  OVC / Mat:                     {:>8.2}x    speedup: {:>5.2}x",
                    ovc_total / mat_total,
                    mat_total / ovc_total,
                );

                // Correctness check.
                assert_eq!(merge_n_way_ovc(&sides), merge_n_way_memcmp(&mat_refs));
            }
        }
    }
}
