// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Design 2: Materialize to ord-bytes, then memcmp merge.**
//!
//! Every encoding is normalised into a single contiguous u8 buffer
//! (sign-flipped big-endian per primitive value). The merge driver is
//! a single byte-slice comparison loop — uniform, encoding-agnostic.
//!
//! Trade-off: simple driver, slow build. The materialization pass is
//! O(N · row_bytes) writes, dominating the pipeline for wide keys.
//! Recovers only for narrow keys or when the byte buffer is amortised
//! across multiple downstream operators.
//!
//! See `docs/developer-guide/internals/smj-ovc-design.md`.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use crate::ord_common::{ConstantI64, DictI64, PrimI64, RunEndI64, VarBin};

/// Materialize a primitive column into a contiguous 8-bytes-per-row buffer.
pub(crate) fn materialize_prim(p: &PrimI64<'_>) -> Vec<u8> {
    let mut out = vec![0u8; p.data.len() * 8];
    for (i, &v) in p.data.iter().enumerate() {
        let u = (v as u64) ^ (1u64 << 63);
        out[i * 8..(i + 1) * 8].copy_from_slice(&u.to_be_bytes());
    }
    out
}

/// Materialize a dict-encoded column by gathering and re-encoding the
/// dict value per row. Output is 8 bytes per row.
pub(crate) fn materialize_dict(d: &DictI64<'_>) -> Vec<u8> {
    let mut out = vec![0u8; d.codes.len() * 8];
    for (i, &c) in d.codes.iter().enumerate() {
        let u = (d.dict[c as usize] as u64) ^ (1u64 << 63);
        out[i * 8..(i + 1) * 8].copy_from_slice(&u.to_be_bytes());
    }
    out
}

/// Materialize a run-end column by replicating run values into the row
/// buffer. The "cheap" encoding becomes the most expensive materialization
/// because each value gets written `run_len` times.
pub(crate) fn materialize_runend(r: &RunEndI64<'_>) -> Vec<u8> {
    let mut out = vec![0u8; r.len * 8];
    let mut prev_end = 0u32;
    for (i, &end) in r.run_ends.iter().enumerate() {
        let u = (r.values[i] as u64) ^ (1u64 << 63);
        let bytes = u.to_be_bytes();
        for row in prev_end..end {
            out[(row as usize) * 8..(row as usize + 1) * 8].copy_from_slice(&bytes);
        }
        prev_end = end;
    }
    out
}

/// Materialize a constant column by writing the same 8 bytes `len` times.
pub(crate) fn materialize_constant(c: &ConstantI64) -> Vec<u8> {
    let u = (c.value as u64) ^ (1u64 << 63);
    let bytes = u.to_be_bytes();
    let mut out = vec![0u8; c.len * 8];
    for row in 0..c.len {
        out[row * 8..(row + 1) * 8].copy_from_slice(&bytes);
    }
    out
}

/// Materialize a varbin column into a flat buffer at `stride` bytes per row.
/// Padded with zeros if a row's value is shorter than `stride`.
pub(crate) fn materialize_varbin(v: &VarBin<'_>, stride: usize) -> Vec<u8> {
    let n = v.len();
    let mut out = vec![0u8; n * stride];
    for row in 0..n {
        let bytes = v.bytes_at(row);
        let n_copy = bytes.len().min(stride);
        out[row * stride..row * stride + n_copy].copy_from_slice(&bytes[..n_copy]);
    }
    out
}

/// n-way merge over byte rows of fixed stride. Linear scan for min.
pub(crate) fn merge_n_way_memcmp(sides: &[&[u8]], stride: usize) -> usize {
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
            let rows = sides[i].len() / stride;
            if indices[i] < rows {
                let row = &sides[i][indices[i] * stride..(indices[i] + 1) * stride];
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
    use super::*;
    use crate::ord_common::{build_dict, build_prim, build_runend, build_varbin};

    #[test]
    fn prim_disjoint() {
        let s0 = build_prim(20, 0);
        let s1 = build_prim(20, 20);
        let b0 = materialize_prim(&PrimI64 { data: &s0 });
        let b1 = materialize_prim(&PrimI64 { data: &s1 });
        let refs: Vec<&[u8]> = vec![&b0, &b1];
        assert_eq!(merge_n_way_memcmp(&refs, 8), 40);
    }

    #[test]
    fn dict_disjoint() {
        let (c0, d0) = build_dict(20, 10, 0);
        let (c1, d1) = build_dict(20, 10, 200);
        let b0 = materialize_dict(&DictI64 { codes: &c0, dict: &d0 });
        let b1 = materialize_dict(&DictI64 { codes: &c1, dict: &d1 });
        let refs: Vec<&[u8]> = vec![&b0, &b1];
        assert_eq!(merge_n_way_memcmp(&refs, 8), 40);
    }

    #[test]
    fn runend_disjoint() {
        let (e0, v0, n0) = build_runend(10, 5, 0);
        let (e1, v1, n1) = build_runend(10, 5, 1000);
        let b0 = materialize_runend(&RunEndI64 { run_ends: &e0, values: &v0, len: n0 });
        let b1 = materialize_runend(&RunEndI64 { run_ends: &e1, values: &v1, len: n1 });
        let refs: Vec<&[u8]> = vec![&b0, &b1];
        assert_eq!(merge_n_way_memcmp(&refs, 8), n0 + n1);
    }

    #[test]
    fn varbin_disjoint() {
        let (o0, d0) = build_varbin(20, 50, 0);
        let (o1, d1) = build_varbin(20, 50, 100);
        let b0 = materialize_varbin(&VarBin { offsets: &o0, data: &d0 }, 50);
        let b1 = materialize_varbin(&VarBin { offsets: &o1, data: &d1 }, 50);
        let refs: Vec<&[u8]> = vec![&b0, &b1];
        assert_eq!(merge_n_way_memcmp(&refs, 50), 40);
    }
}
