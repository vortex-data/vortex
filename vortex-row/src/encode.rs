// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    reason = "row encoding indexes into u32-sized buffers; lengths are validated to fit in u32"
)]

//! `RowEncode` variadic scalar function: encode N input columns into a single `ListView<u8>`.
//!
//! The output's `(elements, offsets, sizes)` triple is built up in a single left-to-right
//! pass over the input columns. The `sizes` array doubles as the per-row write cursor, so
//! when the last column finishes encoding, the accumulator is the final array - no separate
//! conversion step is needed.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
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
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::codec;
use crate::options::RowEncodeOptions;
use crate::options::SortField;
use crate::options::deserialize_row_encode_options;
use crate::options::serialize_row_encode_options;
use crate::size::compute_sizes;

/// Variadic scalar function that encodes N input columns into a single `List<u8>`
/// [`ListViewArray`] where row `i` contains the row-encoded bytes for column values
/// `cols[0][i], cols[1][i], ...` concatenated left-to-right.
#[derive(Clone, Debug)]
pub struct RowEncode;

impl ScalarFnVTable for RowEncode {
    type Options = RowEncodeOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.row_encode")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(serialize_row_encode_options(options)))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        deserialize_row_encode_options(metadata)
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
    options: &RowEncodeOptions,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nrows = args.row_count();

    // ===== Phase 1: classify + size pass =====
    let crate::size::SizePassResult {
        fixed_per_row,
        var_lengths,
        col_kinds: _,
        first_varlen_idx: _,
        columns,
    } = compute_sizes(options, args, ctx, "RowEncode")?;

    // ===== Phase 2: totals + buffer =====
    let var_total: u64 = var_lengths
        .as_ref()
        .map_or(0, |v| v.iter().map(|&x| u64::from(x)).sum());
    let total: u64 = (nrows as u64)
        .checked_mul(u64::from(fixed_per_row))
        .and_then(|t| t.checked_add(var_total))
        .vortex_expect("row-encoded total bytes overflow");
    if total > u32::MAX as u64 {
        vortex_bail!("row-encoded output size {} bytes exceeds u32::MAX", total);
    }
    let total_len = total as usize;

    let mut out_buf: BufferMut<u8> = BufferMut::with_capacity(total_len);
    // Every encoder writes every byte in its row range: non-null values are written
    // directly; null fixed-width slots are sentinel + explicit zero-fill; varlen partial
    // blocks zero-pad via the encoder's own loop; null struct/FSL bodies are zero-filled
    // after the child encoders run. So the pre-zero-init of the buffer is redundant;
    // skipping it saves a memset of `total_len` bytes per call (significant for
    // varlen-heavy inputs where total_len reaches multiple MB).
    //
    // SAFETY: we just allocated `total_len` capacity. By the size-pass + encoder
    // contract every byte in [0, total_len) is written before the buffer is read out.
    unsafe { out_buf.set_len(total_len) };

    // ===== Phase 3: per-row offsets =====
    // listview_offsets[i] is the absolute byte offset where row `i` begins.
    // For pure-fixed: i * fixed_per_row.
    // For mixed: i * fixed_per_row + exclusive prefix sum of var_lengths.
    let mut listview_offsets: Vec<u32> = Vec::with_capacity(nrows);
    match var_lengths.as_ref() {
        None => {
            // Pure-fixed: offsets[i] = i * fixed_per_row. Materialize via a tight
            // pointer-write loop that LLVM auto-vectorizes; we already validated total
            // fits in u32 above so the multiplications can't overflow.
            // SAFETY: reserved nrows; pointers within [0, nrows) are valid.
            unsafe {
                let ptr = listview_offsets.as_mut_ptr();
                for i in 0..nrows {
                    ptr.add(i).write((i as u32) * fixed_per_row);
                }
                listview_offsets.set_len(nrows);
            }
        }
        Some(v) => {
            let mut acc: u32 = 0;
            for (i, &l) in v.iter().enumerate() {
                let off = (i as u32)
                    .checked_mul(fixed_per_row)
                    .and_then(|t| t.checked_add(acc))
                    .vortex_expect("row offset overflow");
                listview_offsets.push(off);
                acc = acc.checked_add(l).vortex_expect("varlen prefix overflow");
            }
        }
    }

    // Per-row write cursor (also doubles as the ListView `sizes` slot when done).
    let mut row_cursors = vec![0u32; nrows];

    // ===== Phase 4: encode columns via the cursor path =====
    for (i, col) in columns.iter().enumerate() {
        dispatch_encode(
            col,
            options.fields[i],
            &listview_offsets,
            &mut row_cursors,
            &mut out_buf,
            ctx,
        )?;
    }

    // ===== Phase 5: build ListView output =====
    let elements = PrimitiveArray::new(out_buf.freeze(), Validity::NonNullable).into_array();
    let offsets_arr = PrimitiveArray::new(
        Buffer::<u32>::copy_from(&listview_offsets),
        Validity::NonNullable,
    )
    .into_array();
    let sizes_arr = PrimitiveArray::new(
        Buffer::<u32>::copy_from(&row_cursors),
        Validity::NonNullable,
    )
    .into_array();
    // SAFETY: The encoder constructs `elements`, `offsets_arr`, and `sizes_arr` itself.
    // - `elements` is a `PrimitiveArray<u8>` of length `total_bytes`.
    // - `offsets[i]` is `i * fixed_per_row + var_prefix[i]`, monotonically increasing,
    //   each value in `0..total_bytes`.
    // - `sizes[i]` is the per-row size; `offsets[i] + sizes[i] <= total_bytes` by
    //   construction of the buffer.
    // - Each row's slice is disjoint from every other row's slice.
    // The constructor's `validate` re-walks every row to verify these invariants; we know
    // they hold by construction, so we skip that walk.
    Ok(unsafe {
        ListViewArray::new_unchecked(elements, offsets_arr, sizes_arr, Validity::NonNullable)
    }
    .into_array())
}

/// Dispatch a single column's encoding into the shared `out` buffer.
///
/// For PR 1 this is just the canonicalize-then-`codec::field_encode` fallback path.
/// In-crate fast paths for `Constant`/`Dict`/`Patched` and the inventory-based registry
/// for downstream encodings are added in PR 3.
pub fn dispatch_encode(
    col: &ArrayRef,
    field: SortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let canonical = col.clone().execute::<Canonical>(ctx)?;
    codec::field_encode(&canonical, field, offsets, cursors, out, ctx)
}

/// Mutate-buffer kernel: write this column's per-row bytes into `out` at
/// `offsets[i] + cursors[i]`, advancing `cursors[i]` by the bytes written.
///
/// Return `Ok(None)` to decline and fall back to the canonical path.
///
/// Trait is defined now; per-encoding impls and dispatch wiring land in PR 3.
pub trait RowEncodeKernel: VTable {
    /// Write this column's per-row bytes into `out` at `offsets[i] + cursors[i]`, advancing
    /// `cursors[i]` by the bytes written.
    fn row_encode_into(
        column: ArrayView<'_, Self>,
        field: SortField,
        offsets: &[u32],
        cursors: &mut [u32],
        out: &mut [u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>>;
}
