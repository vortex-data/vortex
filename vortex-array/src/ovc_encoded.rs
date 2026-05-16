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
// CONSTANT: every row has the same value. OVC value is the constant; offset
// is always >= 1 vs same-side predecessor (within a side, every row equals
// the previous), so within-side OVC = 0 (duplicate). Cross-side compare is a
// single constant-vs-constant compare resolved at column open.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) struct ConstantI64 {
    pub value: i64,
    pub len: usize,
}

pub(crate) fn merge_n_way_ovc_constant(sides: &[ConstantI64]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if sides[i].len > 0 {
            ovcs[i] = pack(1, i64_to_unsigned(sides[i].value));
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
            // Same constant → equal to predecessor.
            ovcs[min_side] = 0;
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// FRAME-OF-REFERENCE (FoR): value = base + delta[row]. Same base across the
// column. OVC value is the reconstructed i64.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) struct ForI64<'a> {
    pub base: i64,
    pub deltas: &'a [i32],
}

impl<'a> ForI64<'a> {
    #[inline]
    fn value_at(&self, row: usize) -> i64 {
        self.base + i64::from(self.deltas[row])
    }
    #[inline]
    fn len(&self) -> usize {
        self.deltas.len()
    }
}

pub(crate) fn merge_n_way_ovc_for(sides: &[ForI64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if sides[i].len() > 0 {
            ovcs[i] = pack(1, i64_to_unsigned(sides[i].value_at(0)));
        }
    }
    let mut count = 0usize;
    loop {
        let mut min_ovc = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if indices[i] < sides[i].len() && ovcs[i] < min_ovc {
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
        if indices[min_side] < sides[min_side].len() {
            let cur = sides[min_side].value_at(indices[min_side]);
            let pred = sides[min_side].value_at(pred_row);
            ovcs[min_side] = if cur == pred { 0 } else { pack(1, i64_to_unsigned(cur)) };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// ZIGZAG: signed integer ⟷ unsigned via (n << 1) ^ (n >> 63). Per-row decode
// is one shift+xor. OVC value is the decoded i64.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) struct ZigzagI64<'a> {
    pub zigzagged: &'a [u64],
}

impl<'a> ZigzagI64<'a> {
    #[inline]
    fn value_at(&self, row: usize) -> i64 {
        let z = self.zigzagged[row];
        ((z >> 1) as i64) ^ -((z & 1) as i64)
    }
    #[inline]
    fn len(&self) -> usize {
        self.zigzagged.len()
    }
}

pub(crate) fn merge_n_way_ovc_zigzag(sides: &[ZigzagI64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if sides[i].len() > 0 {
            ovcs[i] = pack(1, i64_to_unsigned(sides[i].value_at(0)));
        }
    }
    let mut count = 0usize;
    loop {
        let mut min_ovc = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if indices[i] < sides[i].len() && ovcs[i] < min_ovc {
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
        if indices[min_side] < sides[min_side].len() {
            let cur = sides[min_side].value_at(indices[min_side]);
            let pred = sides[min_side].value_at(pred_row);
            ovcs[min_side] = if cur == pred { 0 } else { pack(1, i64_to_unsigned(cur)) };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// ALP-LITE: float value = mantissa * 10^-exp. Per-row reconstruction is a
// multiply + cast. OVC value is the reconstructed f64 bits, sign/bit-flipped
// so memcmp/u64-cmp matches numeric float order.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) struct AlpF64<'a> {
    pub mantissas: &'a [i64],
    pub exp: u8,
}

impl<'a> AlpF64<'a> {
    #[inline]
    fn value_at(&self, row: usize) -> f64 {
        const POW10: [f64; 16] = [
            1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13,
            1e-14, 1e-15,
        ];
        self.mantissas[row] as f64 * POW10[self.exp as usize]
    }
    #[inline]
    fn len(&self) -> usize {
        self.mantissas.len()
    }
}

#[inline]
fn f64_to_unsigned(v: f64) -> u64 {
    // Order-preserving f64 → u64: flip sign bit on positives, flip all bits on negatives.
    let bits = v.to_bits();
    if bits & (1u64 << 63) == 0 {
        bits | (1u64 << 63)
    } else {
        !bits
    }
}

pub(crate) fn merge_n_way_ovc_alp(sides: &[AlpF64<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if sides[i].len() > 0 {
            ovcs[i] = pack(1, f64_to_unsigned(sides[i].value_at(0)));
        }
    }
    let mut count = 0usize;
    loop {
        let mut min_ovc = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if indices[i] < sides[i].len() && ovcs[i] < min_ovc {
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
        if indices[min_side] < sides[min_side].len() {
            let cur = sides[min_side].value_at(indices[min_side]);
            let pred = sides[min_side].value_at(pred_row);
            ovcs[min_side] = if cur.to_bits() == pred.to_bits() {
                0
            } else {
                pack(1, f64_to_unsigned(cur))
            };
        } else {
            ovcs[min_side] = u64::MAX;
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// VARBIN: variable-length byte values. OVC value = first 8 bytes of the
// binary, taken as BE u64. On OVC tie, full byte compare resolves order.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) struct VarBin<'a> {
    pub offsets: &'a [u32],
    pub data: &'a [u8],
}

impl<'a> VarBin<'a> {
    #[inline]
    fn bytes_at(&self, row: usize) -> &'a [u8] {
        let s = self.offsets[row] as usize;
        let e = self.offsets[row + 1] as usize;
        &self.data[s..e]
    }
    #[inline]
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    #[inline]
    fn ovc_value_at(&self, row: usize) -> u64 {
        let bytes = self.bytes_at(row);
        let mut buf = [0u8; 8];
        let n = bytes.len().min(8);
        buf[..n].copy_from_slice(&bytes[..n]);
        u64::from_be_bytes(buf)
    }
}

pub(crate) fn merge_n_way_ovc_varbin(sides: &[VarBin<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if sides[i].len() > 0 {
            ovcs[i] = pack(1, sides[i].ovc_value_at(0));
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
        // Pass 2: tie-break by full bytes compare among sides with same OVC.
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
                pack(1, sides[min_side].ovc_value_at(indices[min_side]))
            };
        } else {
            ovcs[min_side] = u64::MAX;
        }
        // Recompute OVCs of other tied sides against new pred.
        // For varbin tie case, both rows had matching first-8 bytes; their
        // OVC vs new pred (= winner's previous row) is still vs first-8 of
        // their own row, which is unchanged. Loser invariant holds.
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// CHUNKED: a side composed of multiple sub-arrays of varying encodings.
// Per-row access dispatches to the chunk containing that row. For OVC, the
// "value" is whatever the inner encoding produces.
//
// We represent it as Vec<OvcCol<'a>> with cumulative offsets, but only
// implement a single-encoding wrap (PrimI64 chunks) here as proof of shape.
// The same skeleton extends to any inner encoding by polymorphism.
// ───────────────────────────────────────────────────────────────────────────

pub(crate) struct ChunkedPrim<'a> {
    pub chunks: Vec<&'a [i64]>,
    /// cumulative row counts: chunk_offsets[k] = total rows in chunks 0..k
    pub chunk_offsets: Vec<usize>,
    pub len: usize,
}

impl<'a> ChunkedPrim<'a> {
    pub(crate) fn new(chunks: Vec<&'a [i64]>) -> Self {
        let mut chunk_offsets = Vec::with_capacity(chunks.len() + 1);
        chunk_offsets.push(0);
        let mut total = 0usize;
        for c in &chunks {
            total += c.len();
            chunk_offsets.push(total);
        }
        Self { chunks, chunk_offsets, len: total }
    }
    #[inline]
    fn chunk_of(&self, row: usize, hint: usize) -> usize {
        let mut k = hint;
        while k < self.chunks.len() && self.chunk_offsets[k + 1] <= row {
            k += 1;
        }
        k
    }
    #[inline]
    fn value_at(&self, row: usize, hint: usize) -> (i64, usize) {
        let k = self.chunk_of(row, hint);
        (self.chunks[k][row - self.chunk_offsets[k]], k)
    }
}

pub(crate) fn merge_n_way_ovc_chunked(sides: &[ChunkedPrim<'_>]) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut indices = vec![0usize; n];
    let mut chunk_hint = vec![0usize; n];
    // Cache the last-seen value per side instead of re-fetching pred_row
    // (which may live in an earlier chunk than the current hint).
    let mut last_value = vec![0i64; n];
    let mut ovcs = vec![u64::MAX; n];
    for i in 0..n {
        if sides[i].len > 0 {
            let (v, k) = sides[i].value_at(0, 0);
            chunk_hint[i] = k;
            last_value[i] = v;
            ovcs[i] = pack(1, i64_to_unsigned(v));
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
        let pred = last_value[min_side];
        indices[min_side] += 1;
        if indices[min_side] < sides[min_side].len {
            let (cur, k) = sides[min_side].value_at(indices[min_side], chunk_hint[min_side]);
            chunk_hint[min_side] = k;
            last_value[min_side] = cur;
            ovcs[min_side] = if cur == pred { 0 } else { pack(1, i64_to_unsigned(cur)) };
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

    /// RunEnd OVC handles sides with totally different run boundaries.
    /// Side 0: 10 runs of 5 rows each; side 1: 5 runs of 13 rows each.
    /// Sort orders interleave at arbitrary run boundaries.
    #[test]
    fn agreement_constant() {
        let sides = vec![
            ConstantI64 { value: 1, len: 10 },
            ConstantI64 { value: 5, len: 10 },
            ConstantI64 { value: 3, len: 10 },
        ];
        assert_eq!(merge_n_way_ovc_constant(&sides), 30);
    }

    #[test]
    fn agreement_for() {
        let d0: Vec<i32> = (0..20i32).collect();
        let d1: Vec<i32> = (0..20i32).collect();
        let sides = vec![
            ForI64 { base: 0, deltas: &d0 },
            ForI64 { base: 100, deltas: &d1 },
        ];
        assert_eq!(merge_n_way_ovc_for(&sides), 40);
    }

    #[test]
    fn agreement_zigzag() {
        // Encode some sorted i64s as zigzag for both sides.
        let enc = |x: i64| -> u64 { ((x << 1) ^ (x >> 63)) as u64 };
        let d0: Vec<u64> = (0i64..20).map(enc).collect();
        let d1: Vec<u64> = (20i64..40).map(enc).collect();
        let sides = vec![ZigzagI64 { zigzagged: &d0 }, ZigzagI64 { zigzagged: &d1 }];
        assert_eq!(merge_n_way_ovc_zigzag(&sides), 40);
    }

    #[test]
    fn agreement_alp() {
        let m0: Vec<i64> = (0..20i64).collect();
        let m1: Vec<i64> = (20..40i64).collect();
        let sides = vec![
            AlpF64 { mantissas: &m0, exp: 2 },
            AlpF64 { mantissas: &m1, exp: 2 },
        ];
        assert_eq!(merge_n_way_ovc_alp(&sides), 40);
    }

    #[test]
    fn agreement_varbin() {
        let mut data0 = Vec::new();
        let mut offs0 = vec![0u32];
        let mut data1 = Vec::new();
        let mut offs1 = vec![0u32];
        for i in 0..20u8 {
            data0.extend_from_slice(&[i, 0xAA]);
            offs0.push(data0.len() as u32);
            data1.extend_from_slice(&[i + 20, 0xBB]);
            offs1.push(data1.len() as u32);
        }
        let sides = vec![
            VarBin { offsets: &offs0, data: &data0 },
            VarBin { offsets: &offs1, data: &data1 },
        ];
        assert_eq!(merge_n_way_ovc_varbin(&sides), 40);
    }

    #[test]
    fn agreement_chunked() {
        let c0a: Vec<i64> = (0..10).collect();
        let c0b: Vec<i64> = (10..20).collect();
        let c1a: Vec<i64> = (20..30).collect();
        let c1b: Vec<i64> = (30..40).collect();
        let sides = vec![
            ChunkedPrim::new(vec![&c0a, &c0b]),
            ChunkedPrim::new(vec![&c1a, &c1b]),
        ];
        assert_eq!(merge_n_way_ovc_chunked(&sides), 40);
    }

    #[test]
    fn agreement_runend_different_shapes() {
        // Side 0: ends [5,10,15,...,50], values [0,2,4,...,18]   (50 rows)
        // Side 1: ends [13,26,39,52,65], values [1,3,5,7,9]       (65 rows)
        let e0: Vec<u32> = (1u32..=10).map(|i| i * 5).collect();
        let v0: Vec<i64> = (0..10i64).map(|i| i * 2).collect();
        let e1: Vec<u32> = (1u32..=5).map(|i| i * 13).collect();
        let v1: Vec<i64> = (0..5i64).map(|i| i * 2 + 1).collect();
        let sides = vec![
            RunEndI64 { run_ends: &e0, values: &v0, len: 50 },
            RunEndI64 { run_ends: &e1, values: &v1, len: 65 },
        ];
        // Cross-check against the decompressed merge.
        let d0 = runend_decompress(&sides[0]);
        let d1 = runend_decompress(&sides[1]);
        let prim = vec![PrimI64 { data: &d0 }, PrimI64 { data: &d1 }];
        assert_eq!(merge_n_way_ovc_runend(&sides), merge_n_way_ovc_prim(&prim));
        assert_eq!(merge_n_way_ovc_runend(&sides), 50 + 65);
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

        // --- Constant ---
        let constants: Vec<ConstantI64> = (0..N_SIDES)
            .map(|i| ConstantI64 { value: (i * N) as i64, len: N })
            .collect();
        let const_to_prim: Vec<Vec<i64>> =
            constants.iter().map(|c| vec![c.value; c.len]).collect();
        let const_prim: Vec<PrimI64> = const_to_prim.iter().map(|d| PrimI64 { data: d }).collect();
        bench_three(
            "CONSTANT (1 value/side)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_constant(&constants) as u64,
            || merge_n_way_ovc_prim(&const_prim) as u64,
            || {
                let mats: Vec<Vec<u8>> = const_prim.iter().map(materialize_prim).collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                merge_n_way_memcmp(&refs) as u64
            },
        );

        // --- FoR ---
        let for_data: Vec<Vec<i32>> = (0..N_SIDES).map(|_| (0..N as i32).collect()).collect();
        let for_sides: Vec<ForI64> = for_data
            .iter()
            .enumerate()
            .map(|(i, d)| ForI64 { base: (i * N) as i64, deltas: d })
            .collect();
        bench_three(
            "FoR (base + i32 delta)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_for(&for_sides) as u64,
            || {
                let decoded: Vec<Vec<i64>> = for_sides
                    .iter()
                    .map(|f| f.deltas.iter().map(|&d| f.base + i64::from(d)).collect())
                    .collect();
                let prim_view: Vec<PrimI64> = decoded.iter().map(|d| PrimI64 { data: d }).collect();
                merge_n_way_ovc_prim(&prim_view) as u64
            },
            || {
                let mats: Vec<Vec<u8>> = for_sides
                    .iter()
                    .map(|f| {
                        let mut out = vec![0u8; f.deltas.len() * 8];
                        for (i, &d) in f.deltas.iter().enumerate() {
                            let v = f.base + i64::from(d);
                            let u = (v as u64) ^ (1u64 << 63);
                            out[i * 8..(i + 1) * 8].copy_from_slice(&u.to_be_bytes());
                        }
                        out
                    })
                    .collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                merge_n_way_memcmp(&refs) as u64
            },
        );

        // --- Zigzag ---
        let zz_data: Vec<Vec<u64>> = (0..N_SIDES)
            .map(|i| {
                (0..N as i64)
                    .map(|j| {
                        let v = (i * N) as i64 + j;
                        ((v << 1) ^ (v >> 63)) as u64
                    })
                    .collect()
            })
            .collect();
        let zz_sides: Vec<ZigzagI64> = zz_data.iter().map(|d| ZigzagI64 { zigzagged: d }).collect();
        bench_three(
            "ZIGZAG (signed varint base)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_zigzag(&zz_sides) as u64,
            || {
                let decoded: Vec<Vec<i64>> = zz_sides
                    .iter()
                    .map(|z| {
                        z.zigzagged
                            .iter()
                            .map(|&v| ((v >> 1) as i64) ^ -((v & 1) as i64))
                            .collect()
                    })
                    .collect();
                let prim_view: Vec<PrimI64> = decoded.iter().map(|d| PrimI64 { data: d }).collect();
                merge_n_way_ovc_prim(&prim_view) as u64
            },
            || {
                let mats: Vec<Vec<u8>> = zz_sides
                    .iter()
                    .map(|z| {
                        let mut out = vec![0u8; z.zigzagged.len() * 8];
                        for (i, &v) in z.zigzagged.iter().enumerate() {
                            let dec = ((v >> 1) as i64) ^ -((v & 1) as i64);
                            let u = (dec as u64) ^ (1u64 << 63);
                            out[i * 8..(i + 1) * 8].copy_from_slice(&u.to_be_bytes());
                        }
                        out
                    })
                    .collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                merge_n_way_memcmp(&refs) as u64
            },
        );

        // --- ALP-lite (floats) ---
        let alp_data: Vec<Vec<i64>> = (0..N_SIDES)
            .map(|i| (0..N as i64).map(|j| (i * N) as i64 + j).collect())
            .collect();
        let alp_sides: Vec<AlpF64> =
            alp_data.iter().map(|m| AlpF64 { mantissas: m, exp: 2 }).collect();
        bench_three(
            "ALP-lite (mantissa+exp f64)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_alp(&alp_sides) as u64,
            || {
                let decoded: Vec<Vec<i64>> = alp_sides
                    .iter()
                    .map(|a| {
                        a.mantissas
                            .iter()
                            .enumerate()
                            .map(|(i, _)| a.value_at(i).to_bits() as i64)
                            .collect()
                    })
                    .collect();
                let prim_view: Vec<PrimI64> = decoded.iter().map(|d| PrimI64 { data: d }).collect();
                merge_n_way_ovc_prim(&prim_view) as u64
            },
            || {
                let mats: Vec<Vec<u8>> = alp_sides
                    .iter()
                    .map(|a| {
                        let mut out = vec![0u8; a.mantissas.len() * 8];
                        for (i, _) in a.mantissas.iter().enumerate() {
                            let f = a.value_at(i);
                            let u = f64_to_unsigned(f);
                            out[i * 8..(i + 1) * 8].copy_from_slice(&u.to_be_bytes());
                        }
                        out
                    })
                    .collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                merge_n_way_memcmp(&refs) as u64
            },
        );

        // --- VarBin (50-byte values) ---
        const VB_LEN: usize = 50;
        let vb_offsets: Vec<Vec<u32>> = (0..N_SIDES)
            .map(|_| (0..=N).map(|i| (i * VB_LEN) as u32).collect())
            .collect();
        let vb_data: Vec<Vec<u8>> = (0..N_SIDES)
            .map(|side| {
                let mut buf = vec![0u8; N * VB_LEN];
                for row in 0..N {
                    let key = ((side * N + row) as u64).to_be_bytes();
                    buf[row * VB_LEN..row * VB_LEN + 8].copy_from_slice(&key);
                }
                buf
            })
            .collect();
        let vb_sides: Vec<VarBin> = vb_offsets
            .iter()
            .zip(vb_data.iter())
            .map(|(o, d)| VarBin { offsets: o, data: d })
            .collect();
        bench_three(
            "VARBIN (50B values, leading key)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_varbin(&vb_sides) as u64,
            || {
                // Decompressed-equivalent: copy bytes into a flat Vec<u8> with
                // fixed stride 50; merge with memcmp (no escape-encoding).
                let mats: Vec<Vec<u8>> = vb_sides
                    .iter()
                    .map(|s| {
                        let n = s.len();
                        let mut out = vec![0u8; n * VB_LEN];
                        for i in 0..n {
                            let b = s.bytes_at(i);
                            out[i * VB_LEN..i * VB_LEN + b.len()].copy_from_slice(b);
                        }
                        out
                    })
                    .collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                // re-use memcmp merge but stride is VB_LEN, not 8
                let mut indices = vec![0usize; refs.len()];
                let mut count = 0usize;
                loop {
                    let mut min_side = usize::MAX;
                    let mut min_b: &[u8] = &[];
                    for i in 0..refs.len() {
                        let rows = refs[i].len() / VB_LEN;
                        if indices[i] < rows {
                            let row = &refs[i][indices[i] * VB_LEN..(indices[i] + 1) * VB_LEN];
                            if min_side == usize::MAX || row < min_b {
                                min_side = i;
                                min_b = row;
                            }
                        }
                    }
                    if min_side == usize::MAX {
                        break;
                    }
                    count += 1;
                    indices[min_side] += 1;
                }
                count as u64
            },
            || {
                let mats: Vec<Vec<u8>> = vb_sides
                    .iter()
                    .map(|s| {
                        let n = s.len();
                        let mut out = vec![0u8; n * VB_LEN];
                        for i in 0..n {
                            let b = s.bytes_at(i);
                            out[i * VB_LEN..i * VB_LEN + b.len()].copy_from_slice(b);
                        }
                        out
                    })
                    .collect();
                let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
                let mut indices = vec![0usize; refs.len()];
                let mut count = 0usize;
                loop {
                    let mut min_side = usize::MAX;
                    let mut min_b: &[u8] = &[];
                    for i in 0..refs.len() {
                        let rows = refs[i].len() / VB_LEN;
                        if indices[i] < rows {
                            let row = &refs[i][indices[i] * VB_LEN..(indices[i] + 1) * VB_LEN];
                            if min_side == usize::MAX || row < min_b {
                                min_side = i;
                                min_b = row;
                            }
                        }
                    }
                    if min_side == usize::MAX {
                        break;
                    }
                    count += 1;
                    indices[min_side] += 1;
                }
                count as u64
            },
        );

        // --- Chunked primitive (4 chunks/side) ---
        let chunked_raw: Vec<Vec<Vec<i64>>> = (0..N_SIDES)
            .map(|i| {
                let base = (i * N) as i64;
                let q = N / 4;
                (0..4)
                    .map(|c| (0..q as i64).map(|j| base + (c as i64) * q as i64 + j).collect())
                    .collect()
            })
            .collect();
        let chunked_sides: Vec<ChunkedPrim> = chunked_raw
            .iter()
            .map(|cs| ChunkedPrim::new(cs.iter().map(Vec::as_slice).collect()))
            .collect();
        bench_three(
            "CHUNKED (4 prim chunks/side)",
            N * N_SIDES,
            ITERS,
            || merge_n_way_ovc_chunked(&chunked_sides) as u64,
            || {
                let decoded: Vec<Vec<i64>> = chunked_sides
                    .iter()
                    .map(|c| {
                        let mut out = Vec::with_capacity(c.len);
                        for ch in &c.chunks {
                            out.extend_from_slice(ch);
                        }
                        out
                    })
                    .collect();
                let prim_view: Vec<PrimI64> = decoded.iter().map(|d| PrimI64 { data: d }).collect();
                merge_n_way_ovc_prim(&prim_view) as u64
            },
            || {
                let mats: Vec<Vec<u8>> = chunked_sides
                    .iter()
                    .map(|c| {
                        let mut out = Vec::with_capacity(c.len * 8);
                        for ch in &c.chunks {
                            for &v in ch.iter() {
                                let u = (v as u64) ^ (1u64 << 63);
                                out.extend_from_slice(&u.to_be_bytes());
                            }
                        }
                        out
                    })
                    .collect();
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
