// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Design 3 — the converged design: `OrdIter` transformed iterator.**
//!
//! A single trait that any encoding implements. Each encoding emits its
//! rows as a stream of structurally-shaped `OrdChunk`s (Constant /
//! RunEnd / Dense). The merge driver is one function generic over
//! `&mut [Box<dyn OrdIter>]` — encoding-agnostic, open to extension.
//!
//! Two key optimisations make this beat hand-specialised direct kernels
//! (see `ord_direct`):
//!
//! 1. **Heads array** — current head u64 per side stored in a flat
//!    `Vec<u64>`. Min-search is a tight typed loop, no enum match per
//!    iteration.
//! 2. **Bulk-emit** — when the winner's value stays constant for a known
//!    span (Constant chunk: entire `len`; RunEnd: rest of the current
//!    run), the driver emits the whole span in one operation. Safe
//!    because the other sides' heads are unchanged while only the winner
//!    advances.
//!
//! The cost: one dyn call per chunk refill (~5 ns amortised over
//! chunk_size rows). The benefit: open extensibility, single driver, and
//! at chunk_size=1024 the dyn overhead is ~0.005 ns/row.
//!
//! See `docs/developer-guide/internals/smj-ovc-design.md` for the
//! full design rationale and benchmark comparison against `ord_direct`
//! (specialised) and `ord_memcmp` (universal byte form).

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

use crate::ord_common::{ConstantI64, DictI64, PrimI64, RunEndI64, VarBin};

// ───────────────────────────────────────────────────────────────────────────
// Public types
// ───────────────────────────────────────────────────────────────────────────

/// Caller-owned scratch backing one iter's chunk output.
pub struct Scratch {
    dense: Vec<u64>,
    run_ends: Vec<u32>,
    run_values: Vec<u64>,
}

impl Scratch {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            dense: vec![0u64; chunk_size],
            run_ends: vec![0u32; chunk_size],
            run_values: vec![0u64; chunk_size],
        }
    }
}

/// Structural shape of a chunk. Borrows from caller-supplied `Scratch`.
///
/// The merge driver fast-paths each variant: Constant + RunEnd emit
/// multiple rows per iteration when they win; Dense is the universal
/// fallback (one row per iteration).
pub enum OrdChunk<'a> {
    /// `len` rows all equal to `value`.
    Constant { value: u64, len: usize },
    /// `n_runs` runs. `run_ends[k]` is the first chunk-local row NOT in
    /// run k; `values[k]` is run k's ord value. `len` is the total rows.
    RunEnd { run_ends: &'a [u32], values: &'a [u64], len: usize },
    /// One u64 per row.
    Dense(&'a [u64]),
}

impl OrdChunk<'_> {
    pub fn len(&self) -> usize {
        match self {
            OrdChunk::Constant { len, .. } => *len,
            OrdChunk::RunEnd { len, .. } => *len,
            OrdChunk::Dense(v) => v.len(),
        }
    }
}

/// A transformed lending iterator yielding order-preserving u64 chunks.
///
/// Each impl owns its cursor; new encodings plug in by implementing the
/// trait (default `next_chunk` could canonicalize-and-recurse for an
/// extension story not yet implemented here).
pub trait OrdIter {
    fn ord_len(&self) -> usize;

    /// Pull the next chunk into `scratch`. Returns `None` when exhausted.
    fn next_chunk<'s>(
        &mut self,
        max_rows: usize,
        scratch: &'s mut Scratch,
    ) -> Option<OrdChunk<'s>>;

    /// Advance `n` rows without producing values (duplicate-bypass shortcut).
    fn skip(&mut self, n: usize);
}

// ───────────────────────────────────────────────────────────────────────────
// Per-encoding iter impls
// ───────────────────────────────────────────────────────────────────────────

pub struct PrimIter<'a> {
    data: &'a [i64],
    pos: usize,
}

impl<'a> PrimIter<'a> {
    pub fn new(data: &'a [i64]) -> Self {
        Self { data, pos: 0 }
    }
}

impl OrdIter for PrimIter<'_> {
    fn ord_len(&self) -> usize {
        self.data.len()
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        if self.pos >= self.data.len() {
            return None;
        }
        let take = max_rows.min(self.data.len() - self.pos);
        for (i, &v) in self.data[self.pos..self.pos + take].iter().enumerate() {
            scratch.dense[i] = (v as u64) ^ (1u64 << 63);
        }
        self.pos += take;
        Some(OrdChunk::Dense(&scratch.dense[..take]))
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.data.len());
    }
}

pub struct DictIter<'a> {
    codes: &'a [u32],
    pos: usize,
}

impl<'a> DictIter<'a> {
    pub fn new(codes: &'a [u32]) -> Self {
        Self { codes, pos: 0 }
    }
}

impl OrdIter for DictIter<'_> {
    fn ord_len(&self) -> usize {
        self.codes.len()
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        if self.pos >= self.codes.len() {
            return None;
        }
        let take = max_rows.min(self.codes.len() - self.pos);
        for (i, &c) in self.codes[self.pos..self.pos + take].iter().enumerate() {
            scratch.dense[i] = u64::from(c) << 32;
        }
        self.pos += take;
        Some(OrdChunk::Dense(&scratch.dense[..take]))
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.codes.len());
    }
}

pub struct ConstantIter {
    value: u64,
    total: usize,
    pos: usize,
}

impl ConstantIter {
    pub fn new(value: i64, len: usize) -> Self {
        Self {
            value: (value as u64) ^ (1u64 << 63),
            total: len,
            pos: 0,
        }
    }
}

impl OrdIter for ConstantIter {
    fn ord_len(&self) -> usize {
        self.total
    }
    fn next_chunk<'s>(
        &mut self,
        max_rows: usize,
        _scratch: &'s mut Scratch,
    ) -> Option<OrdChunk<'s>> {
        if self.pos >= self.total {
            return None;
        }
        let take = max_rows.min(self.total - self.pos);
        self.pos += take;
        Some(OrdChunk::Constant { value: self.value, len: take })
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.total);
    }
}

pub struct RunEndIter<'a> {
    run_ends: &'a [u32],
    values: &'a [i64],
    total: usize,
    pos: usize,
    run_idx: usize,
}

impl<'a> RunEndIter<'a> {
    pub fn new(run_ends: &'a [u32], values: &'a [i64], total: usize) -> Self {
        Self { run_ends, values, total, pos: 0, run_idx: 0 }
    }
}

impl OrdIter for RunEndIter<'_> {
    fn ord_len(&self) -> usize {
        self.total
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        if self.pos >= self.total {
            return None;
        }
        let end = (self.pos + max_rows).min(self.total);
        let take = end - self.pos;
        while self.run_idx < self.run_ends.len()
            && (self.run_ends[self.run_idx] as usize) <= self.pos
        {
            self.run_idx += 1;
        }
        let mut local_run = 0;
        let mut row = self.pos;
        let mut run = self.run_idx;
        while row < end {
            let run_end_global = (self.run_ends[run] as usize).min(end);
            scratch.run_ends[local_run] = (run_end_global - self.pos) as u32;
            scratch.run_values[local_run] = (self.values[run] as u64) ^ (1u64 << 63);
            local_run += 1;
            row = run_end_global;
            if run_end_global == self.run_ends[run] as usize {
                run += 1;
            } else {
                break;
            }
        }
        self.pos = end;
        Some(OrdChunk::RunEnd {
            run_ends: &scratch.run_ends[..local_run],
            values: &scratch.run_values[..local_run],
            len: take,
        })
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.total);
    }
}

pub struct VarBinIter<'a> {
    offsets: &'a [u32],
    data: &'a [u8],
    pos: usize,
}

impl<'a> VarBinIter<'a> {
    pub fn new(offsets: &'a [u32], data: &'a [u8]) -> Self {
        Self { offsets, data, pos: 0 }
    }
}

impl OrdIter for VarBinIter<'_> {
    fn ord_len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        let total = self.offsets.len().saturating_sub(1);
        if self.pos >= total {
            return None;
        }
        let take = max_rows.min(total - self.pos);
        for i in 0..take {
            let row = self.pos + i;
            let s = self.offsets[row] as usize;
            let e = self.offsets[row + 1] as usize;
            let bytes = &self.data[s..e];
            let mut buf = [0u8; 8];
            let n = bytes.len().min(8);
            buf[..n].copy_from_slice(&bytes[..n]);
            scratch.dense[i] = u64::from_be_bytes(buf);
        }
        self.pos += take;
        Some(OrdChunk::Dense(&scratch.dense[..take]))
    }
    fn skip(&mut self, n: usize) {
        let total = self.offsets.len().saturating_sub(1);
        self.pos = (self.pos + n).min(total);
    }
}

// Convenience constructors from the shared source structs.
impl<'a> From<&PrimI64<'a>> for PrimIter<'a> {
    fn from(p: &PrimI64<'a>) -> Self {
        Self::new(p.data)
    }
}
impl<'a> From<&DictI64<'a>> for DictIter<'a> {
    fn from(d: &DictI64<'a>) -> Self {
        Self::new(d.codes)
    }
}
impl<'a> From<&RunEndI64<'a>> for RunEndIter<'a> {
    fn from(r: &RunEndI64<'a>) -> Self {
        Self::new(r.run_ends, r.values, r.len)
    }
}
impl From<&ConstantI64> for ConstantIter {
    fn from(c: &ConstantI64) -> Self {
        Self::new(c.value, c.len)
    }
}
impl<'a> From<&VarBin<'a>> for VarBinIter<'a> {
    fn from(v: &VarBin<'a>) -> Self {
        Self::new(v.offsets, v.data)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Merge driver
// ───────────────────────────────────────────────────────────────────────────

enum SideState {
    Empty,
    Constant { remaining: usize },
    RunEnd { run_idx: usize, n_runs: usize, within: usize, run_len: usize },
    Dense { pos: usize, fill: usize },
}

/// n-way merge over heterogeneous `OrdIter` sides. Linear-scan O(n) min
/// per emit-batch; bulk-emits whole Constant chunks and whole RunEnd runs.
pub fn merge_n_way(iters: &mut [Box<dyn OrdIter + '_>], chunk_size: usize) -> usize {
    let n = iters.len();
    if n == 0 {
        return 0;
    }
    let mut scratches: Vec<Scratch> = (0..n).map(|_| Scratch::new(chunk_size)).collect();
    let mut state: Vec<SideState> = (0..n).map(|_| SideState::Empty).collect();
    let mut heads: Vec<u64> = vec![u64::MAX; n];

    fn refill(
        i: usize,
        iters: &mut [Box<dyn OrdIter + '_>],
        scratches: &mut [Scratch],
        state: &mut [SideState],
        heads: &mut [u64],
        chunk_size: usize,
    ) {
        match iters[i].next_chunk(chunk_size, &mut scratches[i]) {
            None => {
                state[i] = SideState::Empty;
                heads[i] = u64::MAX;
            }
            Some(OrdChunk::Constant { value, len }) => {
                state[i] = SideState::Constant { remaining: len };
                heads[i] = value;
            }
            Some(OrdChunk::RunEnd { run_ends, values: _, len: _ }) => {
                let n_runs = run_ends.len();
                let first_run_len = if n_runs > 0 { run_ends[0] as usize } else { 0 };
                state[i] = SideState::RunEnd { run_idx: 0, n_runs, within: 0, run_len: first_run_len };
                heads[i] = scratches[i].run_values[0];
            }
            Some(OrdChunk::Dense(buf)) => {
                let fill = buf.len();
                state[i] = SideState::Dense { pos: 0, fill };
                heads[i] = scratches[i].dense[0];
            }
        }
    }

    for i in 0..n {
        refill(i, iters, &mut scratches, &mut state, &mut heads, chunk_size);
    }

    let mut count = 0usize;
    loop {
        let mut min_v = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if heads[i] < min_v {
                min_v = heads[i];
                min_side = i;
            }
        }
        if min_side == usize::MAX {
            break;
        }

        // Bulk-emit: how many rows the winner can produce while head == min_v.
        let runway = match &state[min_side] {
            SideState::Constant { remaining } => *remaining,
            SideState::RunEnd { within, run_len, .. } => *run_len - *within,
            SideState::Dense { .. } => 1,
            SideState::Empty => unreachable!(),
        };
        count += runway;

        let mut needs_refill = false;
        match &mut state[min_side] {
            SideState::Constant { remaining } => {
                *remaining = 0;
                needs_refill = true;
            }
            SideState::RunEnd { run_idx, n_runs, within, run_len } => {
                *within = *run_len;
                *run_idx += 1;
                if *run_idx >= *n_runs {
                    needs_refill = true;
                } else {
                    let prev = scratches[min_side].run_ends[*run_idx - 1] as usize;
                    let cur = scratches[min_side].run_ends[*run_idx] as usize;
                    *within = 0;
                    *run_len = cur - prev;
                    heads[min_side] = scratches[min_side].run_values[*run_idx];
                }
            }
            SideState::Dense { pos, fill } => {
                *pos += 1;
                if *pos >= *fill {
                    needs_refill = true;
                } else {
                    heads[min_side] = scratches[min_side].dense[*pos];
                }
            }
            SideState::Empty => unreachable!(),
        }
        if needs_refill {
            refill(min_side, iters, &mut scratches, &mut state, &mut heads, chunk_size);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ord_common::{build_dict, build_prim, build_runend, build_varbin};

    #[test]
    fn prim_disjoint() {
        let s0 = build_prim(50, 0);
        let s1 = build_prim(50, 50);
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(PrimIter::new(&s0)),
            Box::new(PrimIter::new(&s1)),
        ];
        assert_eq!(merge_n_way(&mut iters, 16), 100);
    }

    #[test]
    fn dict_disjoint() {
        let (c0, _) = build_dict(20, 10, 0);
        let (c1, _) = build_dict(20, 10, 200);
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(DictIter::new(&c0)),
            Box::new(DictIter::new(&c1)),
        ];
        assert_eq!(merge_n_way(&mut iters, 16), 40);
    }

    #[test]
    fn runend_disjoint() {
        let (e0, v0, n0) = build_runend(10, 5, 0);
        let (e1, v1, n1) = build_runend(10, 5, 1000);
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(RunEndIter::new(&e0, &v0, n0)),
            Box::new(RunEndIter::new(&e1, &v1, n1)),
        ];
        assert_eq!(merge_n_way(&mut iters, 16), n0 + n1);
    }

    #[test]
    fn runend_different_shapes() {
        let e0: Vec<u32> = (1u32..=10).map(|i| i * 5).collect();
        let v0: Vec<i64> = (0..10).map(|i| i * 2).collect();
        let e1: Vec<u32> = (1u32..=5).map(|i| i * 13).collect();
        let v1: Vec<i64> = (0..5).map(|i| i * 2 + 1).collect();
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(RunEndIter::new(&e0, &v0, 50)),
            Box::new(RunEndIter::new(&e1, &v1, 65)),
        ];
        assert_eq!(merge_n_way(&mut iters, 16), 115);
    }

    #[test]
    fn constant_disjoint() {
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(ConstantIter::new(1, 10)),
            Box::new(ConstantIter::new(5, 10)),
        ];
        assert_eq!(merge_n_way(&mut iters, 4), 20);
    }

    #[test]
    fn varbin_disjoint() {
        let (o0, d0) = build_varbin(20, 50, 0);
        let (o1, d1) = build_varbin(20, 50, 100);
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(VarBinIter::new(&o0, &d0)),
            Box::new(VarBinIter::new(&o1, &d1)),
        ];
        assert_eq!(merge_n_way(&mut iters, 16), 40);
    }
}
