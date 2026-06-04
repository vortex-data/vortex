// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `RowEncode` variadic scalar function: encode N input columns into a single `ListView<u8>`.
//!
//! The output's `(elements, offsets, sizes)` triple is built up in a single left-to-right
//! pass over the input columns. The `sizes` array doubles as the per-row write cursor, so
//! when the last column finishes encoding, the accumulator is the final array - no separate
//! conversion step is needed.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::codec;
use crate::options::RowEncodingOptions;
use crate::options::deserialize_row_encoding_options;
use crate::options::serialize_row_encoding_options;
use crate::size::compute_sizes;

/// Variadic scalar function that encodes N input columns into a single `List<u8>`
/// [`ListViewArray`] where row `i` contains the row-encoded bytes for column values
/// `cols[0][i], cols[1][i], ...` concatenated left-to-right.
///
/// This scalar function is public for session registration and encoding extension work.
/// Most callers should use [`RowEncoder`](crate::RowEncoder) rather than invoking the scalar
/// function directly.
#[derive(Clone, Debug)]
pub struct RowEncode;

impl ScalarFnVTable for RowEncode {
    type Options = RowEncodingOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.row_encode")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(serialize_row_encoding_options(options)))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        deserialize_row_encoding_options(metadata)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Variadic { min: 1, max: None }
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::from(Arc::from(format!("col_{}", child_idx)))
    }

    fn return_dtype(&self, _options: &Self::Options, _args: &[DType]) -> VortexResult<DType> {
        Ok(DType::List(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            Nullability::NonNullable,
        ))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_row_encode(options, args, ctx)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

fn execute_row_encode(
    options: &RowEncodingOptions,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nrows = args.row_count();
    if u32::try_from(nrows).is_err() {
        vortex_bail!("row-encoded input has {} rows, exceeds u32::MAX", nrows);
    }

    // ===== Phase 1: classify + size pass =====
    let crate::size::SizePassResult {
        fixed_per_row,
        var_lengths,
        columns,
    } = compute_sizes(options, args, ctx)?;

    // ===== Phase 2: totals + buffer =====
    let var_total: u64 = var_lengths
        .as_ref()
        .map_or(0, |v| v.iter().map(|&x| u64::from(x)).sum());
    let total: u64 = (nrows as u64)
        .checked_mul(u64::from(fixed_per_row))
        .and_then(|t| t.checked_add(var_total))
        .ok_or_else(|| {
            vortex_error::vortex_err!("row-encoded total bytes overflow u64 (nrows * fixed + var)")
        })?;
    if total > u32::MAX as u64 {
        vortex_bail!("row-encoded output size {} bytes exceeds u32::MAX", total);
    }
    let total_len =
        usize::try_from(total).vortex_expect("validated row-encoded output size must fit usize");

    let mut out_buf: BufferMut<u8> = BufferMut::with_capacity(total_len);
    // Every encoder writes every byte in its row range: fixed-width values write
    // sentinel + value (null rows write sentinel + explicit zero-fill); varlen blocks
    // zero-pad their final partial block; struct/FSL fixed children are written for all
    // rows then null parent rows are overwritten with the canonical null body. So the
    // size-pass + encoder contract guarantees `[0, total_len)` is fully written before
    // the buffer is read out, making the pre-zero-init redundant. Skipping it saves a
    // `total_len`-byte memset per call (significant for varlen-heavy inputs, where
    // `total_len` reaches multiple MB).
    //
    // SAFETY: `total_len` bytes of capacity were just reserved, and by the contract above
    // every byte in that range is written before `out_buf` is frozen and read.
    unsafe { out_buf.set_len(total_len) };

    // ===== Phase 3: per-row offsets =====
    // listview_offsets[i] is the absolute byte offset where row `i` begins.
    // For pure-fixed: i * fixed_per_row.
    // For mixed: i * fixed_per_row + exclusive prefix sum of var_lengths.
    // Build directly into a BufferMut to avoid a Vec→Buffer copy at the end.
    let mut listview_offsets: BufferMut<u32> = BufferMut::with_capacity(nrows);
    // SAFETY: `nrows` of capacity reserved above; every index in `[0, nrows)` is written
    // before the buffer is read out. `nrows` was validated to fit `u32` at function entry,
    // so `i as u32` below is exact and the multiplications can't overflow.
    unsafe { listview_offsets.set_len(nrows) };
    let off = listview_offsets.as_mut_slice();
    match var_lengths.as_ref() {
        None => {
            // Pure-fixed: offsets[i] = i * fixed_per_row. `iter_mut().enumerate()` elides
            // per-element bounds checks, so LLVM auto-vectorizes this multiply.
            for (i, slot) in off.iter_mut().enumerate() {
                *slot = (i as u32) * fixed_per_row;
            }
        }
        Some(v) => {
            // Mixed: offsets[i] = i * fixed_per_row + var_prefix[i], where var_prefix is the
            // exclusive cumsum of varlen lengths. `iter_mut().zip` elides per-element bounds
            // checks; the total was validated to fit u32 upstream so the wrapping arithmetic
            // is exact (it never actually wraps).
            let mut acc: u32 = 0;
            for (i, (slot, &l)) in off.iter_mut().zip(v.iter()).enumerate() {
                *slot = (i as u32).wrapping_mul(fixed_per_row).wrapping_add(acc);
                acc = acc.wrapping_add(l);
            }
        }
    }
    let listview_offsets_slice: &[u32] = listview_offsets.as_slice();

    // Per-row write cursor (also doubles as the ListView `sizes` slot when done). We build
    // it as a BufferMut so we can hand it directly to the output PrimitiveArray.
    let mut row_cursors: BufferMut<u32> = BufferMut::with_capacity(nrows);
    row_cursors.push_n(0u32, nrows);

    // ===== Phase 4: encode columns via the cursor path =====
    // Each column was canonicalized once during the size pass; reuse that canonical form.
    for (i, canonical) in columns.iter().enumerate() {
        codec::field_encode(
            canonical,
            options.fields[i],
            listview_offsets_slice,
            row_cursors.as_mut_slice(),
            &mut out_buf,
            ctx,
        )?;
    }

    // ===== Phase 5: build ListView output =====
    let elements = PrimitiveArray::new(out_buf.freeze(), Validity::NonNullable).into_array();
    let offsets_arr =
        PrimitiveArray::new(listview_offsets.freeze(), Validity::NonNullable).into_array();
    let sizes_arr = PrimitiveArray::new(row_cursors.freeze(), Validity::NonNullable).into_array();
    // SAFETY: this encoder constructs `elements`, `offsets_arr`, and `sizes_arr` itself:
    // - `elements` is a `PrimitiveArray<u8>` of length `total_len`.
    // - `offsets_arr[i]` is `i * fixed_per_row + var_prefix[i]`, monotonically increasing and
    //   in `0..=total_len`.
    // - `offsets_arr[i] + sizes_arr[i] <= total_len` by construction, and each row's slice is
    //   disjoint from every other row's.
    // `try_new`'s validation re-walks every row to check exactly these invariants, which we
    // already guarantee by construction, so we skip it.
    Ok(unsafe {
        ListViewArray::new_unchecked(elements, offsets_arr, sizes_arr, Validity::NonNullable)
    }
    .into_array())
}
