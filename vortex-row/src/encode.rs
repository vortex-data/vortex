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
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::Dict;
use vortex_array::arrays::patched::Patched;
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
use crate::options::RowSortField;
use crate::options::deserialize_row_encoding_options;
use crate::options::serialize_row_encoding_options;
use crate::registry;
use crate::size::ColKind;
use crate::size::ColumnEncodeInput;
use crate::size::compute_sizes;

/// Variadic scalar function that encodes N input columns into a single `List<u8>`
/// [`ListViewArray`] where row `i` contains the row-encoded bytes for column values
/// `cols[0][i], cols[1][i], ...` concatenated left-to-right.
///
/// This scalar function is public for session registration and row-encoding work.
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
        col_kinds,
        first_varlen_idx,
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
    //
    // When fixed-before-varlen columns coexist with a varlen column, we additionally build
    // `var_prefix_for_arith[i] = exclusive cumsum of var_lengths[..i]` and hand it to the
    // arithmetic encoders so they can compute per-row write positions without a cursor.
    let need_arith_prefix = first_varlen_idx.is_some()
        && col_kinds.iter().any(|k| {
            matches!(
                k,
                ColKind::Fixed {
                    before_varlen: true,
                    ..
                }
            )
        });

    // Build directly into a BufferMut to avoid a Vec→Buffer copy at the end.
    let mut listview_offsets: BufferMut<u32> = BufferMut::with_capacity(nrows);
    // SAFETY: `nrows` of capacity reserved above; every index in `[0, nrows)` is written
    // before the buffer is read out. `nrows` was validated to fit `u32` at function entry,
    // so the `0u32..` counters below are exact and the multiplications can't overflow.
    unsafe { listview_offsets.set_len(nrows) };
    let off = listview_offsets.as_mut_slice();
    let mut var_prefix_for_arith: Option<Vec<u32>> = None;
    match var_lengths.as_ref() {
        None => {
            // Pure-fixed: offsets[i] = i * fixed_per_row. Zipping against a `u32` counter
            // elides per-element bounds checks, so LLVM auto-vectorizes this multiply.
            for (slot, i) in off.iter_mut().zip(0u32..) {
                *slot = i * fixed_per_row;
            }
        }
        Some(v) => {
            // Mixed: offsets[i] = i * fixed_per_row + var_prefix[i], where var_prefix is the
            // exclusive cumsum of varlen lengths. The total was validated to fit u32 upstream
            // so the wrapping arithmetic is exact (it never actually wraps).
            let mut vp: Option<Vec<u32>> = need_arith_prefix.then(|| Vec::with_capacity(nrows));
            let mut acc: u32 = 0;
            for ((slot, &l), i) in off.iter_mut().zip(v.iter()).zip(0u32..) {
                if let Some(p) = vp.as_mut() {
                    p.push(acc);
                }
                *slot = i.wrapping_mul(fixed_per_row).wrapping_add(acc);
                acc = acc.wrapping_add(l);
            }
            var_prefix_for_arith = vp;
        }
    }
    let listview_offsets_slice: &[u32] = listview_offsets.as_slice();

    // Per-row write cursor (also doubles as the ListView `sizes` slot when done). We build
    // it as a BufferMut so we can hand it directly to the output PrimitiveArray.
    //
    // The cursor path begins at the first cursor-path column. Fixed-before-varlen columns
    // are written by the arithmetic path and do not touch the cursor, so the cursor is
    // pre-seeded with the within-row offset of the first varlen column (its `fixed_prefix`).
    // When there are no varlen columns at all, every column takes the arithmetic path and
    // the cursor loop runs zero iterations; seeding with `fixed_per_row` then leaves the
    // cursors already correct as per-row sizes.
    let initial_cursor: u32 = match first_varlen_idx {
        Some(idx) => match col_kinds[idx] {
            ColKind::Variable { fixed_prefix } => fixed_prefix,
            ColKind::Fixed { .. } => unreachable!("first_varlen_idx points at a varlen column"),
        },
        None => fixed_per_row,
    };
    let mut row_cursors: BufferMut<u32> = BufferMut::with_capacity(nrows);
    row_cursors.push_n(initial_cursor, nrows);

    // ===== Phase 4: encode columns =====
    // Fixed-before-varlen columns take the arithmetic-write path (constant within-row
    // offset, no cursor mutation), where a fixed-width kernel can fuse decompression with the
    // write. Fixed-after-varlen and varlen columns take the cursor path. Variable-width
    // fallback columns reuse the canonical form materialized during the size pass.
    for (i, col_input) in columns.iter().enumerate() {
        let field = options.fields[i];
        match (col_kinds[i], col_input) {
            (
                ColKind::Fixed {
                    prefix,
                    before_varlen: true,
                    ..
                },
                ColumnEncodeInput::Raw(raw),
            ) => {
                dispatch_encode_fixed_arith(
                    raw,
                    field,
                    prefix,
                    fixed_per_row,
                    var_prefix_for_arith.as_deref(),
                    nrows,
                    &mut out_buf,
                    ctx,
                )?;
            }
            (_, ColumnEncodeInput::Raw(raw)) => {
                dispatch_encode(
                    raw,
                    field,
                    listview_offsets_slice,
                    row_cursors.as_mut_slice(),
                    &mut out_buf,
                    ctx,
                )?;
            }
            (_, ColumnEncodeInput::Canonical(canonical)) => {
                codec::field_encode(
                    canonical,
                    field,
                    listview_offsets_slice,
                    row_cursors.as_mut_slice(),
                    &mut out_buf,
                    ctx,
                )?;
            }
        }
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

/// Dispatch a single column's encoding into the shared `out` buffer at `offsets[i] +
/// cursors[i]`, advancing each cursor by the bytes written.
///
/// Tries the in-crate per-encoding fast paths ([`Constant`], [`Dict`], [`Patched`]) and then
/// the downstream-encoding [`registry`], falling back to canonicalization. This is the encode
/// counterpart to [`dispatch_size`](crate::dispatch_size) and is public so downstream encoding
/// kernels can recurse into a child array's encoding.
pub fn dispatch_encode(
    col: &ArrayRef,
    field: RowSortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if let Some(view) = col.as_opt::<Constant>()
        && Constant::row_encode_into(view, field, offsets, cursors, out, ctx)?.is_some()
    {
        return Ok(());
    }
    if let Some(view) = col.as_opt::<Dict>()
        && Dict::row_encode_into(view, field, offsets, cursors, out, ctx)?.is_some()
    {
        return Ok(());
    }
    if let Some(view) = col.as_opt::<Patched>()
        && Patched::row_encode_into(view, field, offsets, cursors, out, ctx)?.is_some()
    {
        return Ok(());
    }
    if let Some((_, encode_fn, _)) = registry::lookup(&col.encoding_id())
        && encode_fn(col, field, offsets, cursors, out, ctx)?.is_some()
    {
        return Ok(());
    }
    let canonical = col.clone().execute::<Canonical>(ctx)?;
    codec::field_encode(&canonical, field, offsets, cursors, out, ctx)
}

/// Dispatch a fixed-width column through the arithmetic-write fast path, for columns that
/// appear before any variable-length column. Row `i` is written at the constant within-row
/// position `i * row_stride + col_prefix (+ var_prefix[i])`, with no per-row cursor.
///
/// A fixed-width kernel (e.g. FastLanes BitPacked) can fuse decompression with the write here,
/// skipping both the intermediate canonical array and the cursor/offset array traffic. Columns
/// with no kernel fall back to canonicalization plus
/// [`codec::field_encode_fixed_arithmetic`].
#[allow(clippy::too_many_arguments)]
pub fn dispatch_encode_fixed_arith(
    col: &ArrayRef,
    field: RowSortField,
    col_prefix: u32,
    row_stride: u32,
    var_prefix: Option<&[u32]>,
    nrows: usize,
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // Already-canonical primitive: hand straight to the codec's arithmetic primitive path
    // without re-running the canonicalization machinery.
    if col.as_opt::<Primitive>().is_some()
        && let Ok(parr) = col.clone().try_downcast::<Primitive>()
    {
        let canonical = Canonical::Primitive(parr);
        return codec::field_encode_fixed_arithmetic(
            &canonical, field, col_prefix, row_stride, var_prefix, nrows, out, ctx,
        );
    }
    // Constant: write the same encoded bytes at every per-row position.
    if let Some(view) = col.as_opt::<Constant>()
        && Constant::row_encode_fixed_arith(
            view, field, col_prefix, row_stride, var_prefix, out, ctx,
        )?
        .is_some()
    {
        return Ok(());
    }
    // Downstream fixed-width kernels (e.g. FastLanes BitPacked / FoR / Delta).
    if let Some((_, _, Some(arith_fn))) = registry::lookup(&col.encoding_id())
        && arith_fn(
            col, field, col_prefix, row_stride, var_prefix, nrows, out, ctx,
        )?
        .is_some()
    {
        return Ok(());
    }
    let canonical = col.clone().execute::<Canonical>(ctx)?;
    codec::field_encode_fixed_arithmetic(
        &canonical, field, col_prefix, row_stride, var_prefix, nrows, out, ctx,
    )
}

/// Per-encoding fast path that writes a column's per-row bytes into `out` at `offsets[i] +
/// cursors[i]`, advancing `cursors[i]` by the bytes written.
///
/// Return `Ok(Some(()))` when the kernel handled the column, or `Ok(None)` to decline and let
/// the dispatcher fall back to the canonical path.
pub trait RowEncodeKernel: VTable {
    /// Write this column's per-row bytes into `out` at `offsets[i] + cursors[i]`, advancing
    /// `cursors[i]` by the bytes written.
    fn row_encode_into(
        column: ArrayView<'_, Self>,
        field: RowSortField,
        offsets: &[u32],
        cursors: &mut [u32],
        out: &mut [u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>>;

    /// Fixed-width arithmetic write: write row `i`'s bytes at the constant within-row position
    /// `i * row_stride + col_prefix (+ var_prefix[i])`, without a per-row cursor.
    ///
    /// Only called for fixed-width columns that precede every variable-length column. The
    /// default declines (`Ok(None)`), so the dispatcher falls back to canonicalization; an
    /// encoding overrides it to fuse decompression with the arithmetic write.
    fn row_encode_fixed_arith(
        _column: ArrayView<'_, Self>,
        _field: RowSortField,
        _col_prefix: u32,
        _row_stride: u32,
        _var_prefix: Option<&[u32]>,
        _out: &mut [u8],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        Ok(None)
    }
}
