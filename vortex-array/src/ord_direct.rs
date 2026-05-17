// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Design 1: Direct per-encoding OVC kernels.**
//!
//! Each encoding gets a bespoke n-way merge function that knows the
//! encoding's physical shape and uses it directly: dict compares codes,
//! run-end compares run headers, primitive compares values, varbin
//! compares first-8-bytes-then-tail. No trait, no chunked dispatch.
//!
//! This is the "baseline" against which the other two designs are
//! measured — hand-specialised code, manually optimised per encoding.
//! Fastest at small n; lacks open extensibility (each new encoding
//! requires a new top-level function).
//!
//! See `docs/developer-guide/internals/smj-ovc-design.md` for the
//! comparison with `ord_iter` (trait) and `ord_memcmp` (universal byte).

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

use crate::ord_common::{ConstantI64, DictI64, PrimI64, RunEndI64, VarBin};

/// Pack `(arity_minus_offset, value)` into a u64 for OVC compare.
#[inline]
fn pack_ovc(arity_minus_offset: u8, value: u64) -> u64 {
    (u64::from(arity_minus_offset) << 56) | (value >> 8)
}

#[inline]
fn i64_to_unsigned(v: i64) -> u64 {
    (v as u64) ^ (1u64 << 63)
}

// ───────────────────────────────────────────────────────────────────────────
// Primitive
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_prim(sides: &[PrimI64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for (i, side) in sides.iter().enumerate() {
        if !side.data.is_empty() {
            ovcs[i] = pack_ovc(1, i64_to_unsigned(side.data[0]));
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
            ovcs[min_side] = if cur == pred { 0 } else { pack_ovc(1, i64_to_unsigned(cur)) };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// Dict (sorted, rank-aligned dictionary)
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_dict(sides: &[DictI64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for (i, side) in sides.iter().enumerate() {
        if !side.codes.is_empty() {
            ovcs[i] = pack_ovc(1, u64::from(side.codes[0]) << 32);
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
            ovcs[min_side] = if cur == pred { 0 } else { pack_ovc(1, u64::from(cur) << 32) };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// RunEnd (sorted run values, with cached per-side run pointer)
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_runend(sides: &[RunEndI64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut run_idx = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for (i, side) in sides.iter().enumerate() {
        if side.len > 0 {
            run_idx[i] = side.run_of_hint(0, 0);
            ovcs[i] = pack_ovc(1, i64_to_unsigned(side.values[run_idx[i]]));
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
            let new_run = sides[min_side].run_of_hint(indices[min_side], run_idx[min_side]);
            run_idx[min_side] = new_run;
            let pred_run = sides[min_side].run_of_hint(pred_row, run_idx[min_side]);
            ovcs[min_side] = if new_run == pred_run {
                0
            } else {
                pack_ovc(1, i64_to_unsigned(sides[min_side].values[new_run]))
            };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// Constant
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn merge_n_way_constant(sides: &[ConstantI64]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for (i, side) in sides.iter().enumerate() {
        if side.len > 0 {
            ovcs[i] = pack_ovc(1, i64_to_unsigned(side.value));
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
        indices[min_side] += 1;
        if indices[min_side] < sides[min_side].len {
            ovcs[min_side] = 0; // same constant → duplicate of predecessor
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// VarBin (first-8-byte prefix, full byte compare on tie)
// ───────────────────────────────────────────────────────────────────────────

#[inline]
fn varbin_ovc_value(side: &VarBin<'_>, row: usize) -> u64 {
    let bytes = side.bytes_at(row);
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_be_bytes(buf)
}

pub(crate) fn merge_n_way_varbin(sides: &[VarBin<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for (i, side) in sides.iter().enumerate() {
        if side.len() > 0 {
            ovcs[i] = pack_ovc(1, varbin_ovc_value(side, 0));
        }
    }
    let mut count = 0usize;
    loop {
        // Pass 1: find smallest OVC integer.
        let mut min_ovc = u64::MAX;
        for i in 0..n {
            if indices[i] < sides[i].len() && ovcs[i] < min_ovc {
                min_ovc = ovcs[i];
            }
        }
        if min_ovc == u64::MAX {
            break;
        }
        // Pass 2: tie-break by full bytes compare.
        let mut min_side = usize::MAX;
        for i in 0..n {
            if indices[i] < sides[i].len()
                && ovcs[i] == min_ovc
                && (min_side == usize::MAX
                    || sides[i].bytes_at(indices[i]) < sides[min_side].bytes_at(indices[min_side]))
            {
                min_side = i;
            }
        }
        count += 1;
        let pred_row = indices[min_side];
        indices[min_side] += 1;
        if indices[min_side] < sides[min_side].len() {
            let cur = sides[min_side].bytes_at(indices[min_side]);
            let pred = sides[min_side].bytes_at(pred_row);
            ovcs[min_side] = if cur == pred {
                0
            } else {
                pack_ovc(1, varbin_ovc_value(&sides[min_side], indices[min_side]))
            };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// Per-row ord-value accessors used by the cross-design benchmark.
// These are what the merge functions inline; exposed separately so the
// "ord-generation cost" can be measured in isolation.
// ───────────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn prim_ord_at(p: &PrimI64<'_>, row: usize) -> u64 {
    (p.data[row] as u64) ^ (1u64 << 63)
}

#[inline]
pub(crate) fn dict_ord_at(d: &DictI64<'_>, row: usize) -> u64 {
    u64::from(d.codes[row]) << 32
}

#[inline]
pub(crate) fn runend_ord_at(r: &RunEndI64<'_>, row: usize) -> u64 {
    let run = r.run_of(row);
    (r.values[run] as u64) ^ (1u64 << 63)
}

#[inline]
pub(crate) fn constant_ord_at(c: &ConstantI64) -> u64 {
    (c.value as u64) ^ (1u64 << 63)
}

#[inline]
pub(crate) fn varbin_ord_at(v: &VarBin<'_>, row: usize) -> u64 {
    varbin_ovc_value(v, row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ord_common::{build_dict, build_prim, build_runend, build_varbin};

    #[test]
    fn prim_disjoint() {
        let s0 = build_prim(20, 0);
        let s1 = build_prim(20, 20);
        let sides = vec![PrimI64 { data: &s0 }, PrimI64 { data: &s1 }];
        assert_eq!(merge_n_way_prim(&sides), 40);
    }

    #[test]
    fn dict_disjoint() {
        let (c0, d0) = build_dict(20, 10, 0);
        let (c1, d1) = build_dict(20, 10, 200);
        let sides = vec![
            DictI64 { codes: &c0, dict: &d0 },
            DictI64 { codes: &c1, dict: &d1 },
        ];
        assert_eq!(merge_n_way_dict(&sides), 40);
    }

    #[test]
    fn runend_disjoint() {
        let (e0, v0, n0) = build_runend(10, 5, 0);
        let (e1, v1, n1) = build_runend(10, 5, 1000);
        let sides = vec![
            RunEndI64 { run_ends: &e0, values: &v0, len: n0 },
            RunEndI64 { run_ends: &e1, values: &v1, len: n1 },
        ];
        assert_eq!(merge_n_way_runend(&sides), n0 + n1);
    }

    #[test]
    fn constant_disjoint() {
        let sides = vec![
            ConstantI64 { value: 1, len: 10 },
            ConstantI64 { value: 5, len: 10 },
        ];
        assert_eq!(merge_n_way_constant(&sides), 20);
    }

    #[test]
    fn varbin_disjoint() {
        let (o0, d0) = build_varbin(20, 50, 0);
        let (o1, d1) = build_varbin(20, 50, 100);
        let sides = vec![
            VarBin { offsets: &o0, data: &d0 },
            VarBin { offsets: &o1, data: &d1 },
        ];
        assert_eq!(merge_n_way_varbin(&sides), 40);
    }
}

