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

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::Dict;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::row::codec;
use crate::row::codec::RowWidth;
use crate::row::options::RowEncodeOptions;
use crate::row::options::SortField;
use crate::row::options::deserialize_row_encode_options;
use crate::row::options::serialize_row_encode_options;
use crate::row::registry;
use crate::row::size::dispatch_size;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::validity::Validity;

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

/// Classification of a single input column at row-encode time.
#[derive(Clone, Copy, Debug)]
enum ColKind {
    /// Column has fixed width `width`. `prefix` is the within-row byte offset of this
    /// column's first byte, summed over the widths of preceding fixed columns. If
    /// `before_varlen` is true, no varlen column precedes this one, so the within-row
    /// offset is the same constant for every row and we use the arithmetic fast path.
    Fixed {
        width: u32,
        prefix: u32,
        before_varlen: bool,
    },
    /// Column has variable width per row. `fixed_prefix` is the sum of widths of all
    /// preceding fixed columns (constant). The varlen prefix sum is added per row.
    Variable { fixed_prefix: u32 },
}

fn execute_row_encode(
    options: &RowEncodeOptions,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let n_inputs = args.num_inputs();
    if n_inputs == 0 {
        vortex_bail!("RowEncode requires at least one input column");
    }
    if options.fields.len() != n_inputs {
        vortex_bail!(
            "RowEncode options.fields.len()={} does not match num_inputs={}",
            options.fields.len(),
            n_inputs
        );
    }
    let nrows = args.row_count();

    // ===== Phase 1: classify + size pass (stack accumulators) =====
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(n_inputs);
    let mut col_kinds: Vec<ColKind> = Vec::with_capacity(n_inputs);
    let mut fixed_per_row: u32 = 0;
    let mut var_lengths: Option<Vec<u32>> = None;
    let mut first_varlen_idx: Option<usize> = None;
    let mut running_fixed_prefix: u32 = 0;

    for i in 0..n_inputs {
        let col = args.get(i)?;
        if col.len() != nrows {
            vortex_bail!(
                "RowEncode: column {} has length {} but expected {}",
                i,
                col.len(),
                nrows
            );
        }
        let width = codec::row_width_for_dtype(col.dtype())?;
        match width {
            RowWidth::Fixed(w) => {
                col_kinds.push(ColKind::Fixed {
                    width: w,
                    prefix: running_fixed_prefix,
                    before_varlen: first_varlen_idx.is_none(),
                });
                fixed_per_row = fixed_per_row
                    .checked_add(w)
                    .vortex_expect("row width overflow");
                running_fixed_prefix = running_fixed_prefix
                    .checked_add(w)
                    .vortex_expect("row width overflow");
            }
            RowWidth::Variable => {
                if first_varlen_idx.is_none() {
                    first_varlen_idx = Some(i);
                }
                let v = var_lengths.get_or_insert_with(|| vec![0u32; nrows]);
                dispatch_size(&col, options.fields[i], v, ctx)?;
                col_kinds.push(ColKind::Variable {
                    fixed_prefix: running_fixed_prefix,
                });
            }
        }
        columns.push(col);
    }

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
    // Pure-fixed inputs are written byte-for-byte (sentinel + value or sentinel +
    // zero-fill for nulls) by the encoders, so we can skip the zero-init memset. Mixed
    // inputs rely on a zeroed buffer for padding inside varlen partial blocks and for
    // null-struct/null-FSL bodies in the cursor-based encode path.
    if first_varlen_idx.is_some() {
        out_buf.push_n(0u8, total_len);
    } else {
        // Reserve space without zero-filling; encoders write every byte.
        // SAFETY: we just allocated `total_len` capacity above; setting len is safe as
        // long as every byte is written before the buffer is read, which is enforced by
        // the encoders below (they cover every row's [i*stride, (i+1)*stride) range).
        unsafe { out_buf.set_len(total_len) };
    }

    // ===== Phase 3: per-row write context =====
    // We build a single per-row `listview_offsets` array: the absolute byte offset where
    // row `i` begins. It equals `i * fixed_per_row + var_prefix[i]` when varlen columns are
    // present, or `i * fixed_per_row` for pure-fixed. The cursor-based encode path is fed
    // these listview offsets as its `row_offsets` and a zeroed cursor that absorbs
    // `prefix_at_first_varlen` initially (so the encoded position is
    // `listview_offsets[i] + cursors[i]`). The fixed-arithmetic encode path receives
    // `var_prefix` directly when fixed-before-varlen columns exist.
    let mut listview_offsets: Vec<u32> = Vec::with_capacity(nrows);
    let mut var_prefix_for_arith: Option<Vec<u32>> = None;

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

    match var_lengths.as_ref() {
        None => {
            // Pure-fixed.
            let mut acc: u32 = 0;
            for _ in 0..nrows {
                listview_offsets.push(acc);
                acc = acc
                    .checked_add(fixed_per_row)
                    .vortex_expect("offset overflow");
            }
        }
        Some(v) => {
            let mut vp: Option<Vec<u32>> = need_arith_prefix.then(|| Vec::with_capacity(nrows));
            let mut acc: u32 = 0;
            for (i, &l) in v.iter().enumerate() {
                if let Some(p) = vp.as_mut() {
                    p.push(acc);
                }
                listview_offsets.push((i as u32) * fixed_per_row + acc);
                acc = acc.checked_add(l).vortex_expect("var prefix overflow");
            }
            var_prefix_for_arith = vp;
        }
    }

    // For the cursor path: each row's initial cursor starts at `prefix_at_first_varlen`
    // (the within-row offset of the first cursor-based column) so that
    // `listview_offsets[i] + cursors[i]` lands at the right position.
    let mut row_cursors: Option<Vec<u32>> = first_varlen_idx.map(|idx| {
        let prefix_at_first_varlen = match col_kinds[idx] {
            ColKind::Variable { fixed_prefix } => fixed_prefix,
            ColKind::Fixed { .. } => unreachable!("first_varlen_idx points to a varlen column"),
        };
        vec![prefix_at_first_varlen; nrows]
    });

    // ===== Phase 4: encode columns =====
    for (i, col) in columns.iter().enumerate() {
        match col_kinds[i] {
            ColKind::Fixed {
                width,
                prefix,
                before_varlen: true,
            } => {
                dispatch_encode_fixed_arith(
                    col,
                    options.fields[i],
                    prefix,
                    fixed_per_row,
                    var_prefix_for_arith.as_deref(),
                    width,
                    &mut out_buf,
                    ctx,
                )?;
            }
            ColKind::Fixed { .. } | ColKind::Variable { .. } => {
                let curs = row_cursors
                    .as_mut()
                    .vortex_expect("row_cursors initialized when a varlen column exists");
                dispatch_encode(
                    col,
                    options.fields[i],
                    &listview_offsets,
                    curs,
                    &mut out_buf,
                    ctx,
                )?;
            }
        }
    }

    // ===== Phase 5: build ListView output =====
    let elements = PrimitiveArray::new(out_buf.freeze(), Validity::NonNullable).into_array();

    let offsets_arr = PrimitiveArray::new(
        vortex_buffer::Buffer::<u32>::copy_from(&listview_offsets),
        Validity::NonNullable,
    )
    .into_array();

    let sizes_arr = match var_lengths {
        None => ConstantArray::new(Scalar::from(fixed_per_row), nrows).into_array(),
        Some(mut v) => {
            // In-place: rewrite var_lengths to total per-row size to avoid a second
            // allocation and the per-element push that loses memcpy-style vectorization.
            if fixed_per_row != 0 {
                for x in v.iter_mut() {
                    *x += fixed_per_row;
                }
            }
            PrimitiveArray::new(
                vortex_buffer::Buffer::<u32>::copy_from(&v),
                Validity::NonNullable,
            )
            .into_array()
        }
    };

    // SAFETY: The encoder constructs `elements`, `offsets_arr`, and `sizes_arr` itself.
    // - `elements` is a `PrimitiveArray<u8>` of length `total_bytes`.
    // - `offsets[i]` is `i * fixed_per_row + var_prefix[i]`, monotonically increasing,
    //   each value in `0..total_bytes`.
    // - `sizes[i]` is the per-row size; `offsets[i] + sizes[i] <= total_bytes` by
    //   construction of the buffer.
    // - Each row's slice is disjoint from every other row's slice.
    // The constructor's `validate` re-walks every row to verify these invariants; we know
    // they hold so we skip it.
    Ok(unsafe {
        ListViewArray::new_unchecked(elements, offsets_arr, sizes_arr, Validity::NonNullable)
    }
    .into_array())
}

/// Dispatch a single column's encoding through the arithmetic fast path. This is used for
/// fixed-width columns that appear before any variable-length column in the row layout.
#[allow(clippy::too_many_arguments)]
fn dispatch_encode_fixed_arith(
    col: &ArrayRef,
    field: SortField,
    col_prefix: u32,
    row_stride: u32,
    var_prefix: Option<&[u32]>,
    width: u32,
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // Already-canonical PrimitiveArray: bypass the canonicalization machinery entirely so
    // the hot loop can be reached without going through `execute_until::<AnyCanonical>`.
    if col.as_opt::<Primitive>().is_some()
        && let Ok(parr) = col.clone().try_downcast::<Primitive>()
    {
        let canonical = Canonical::Primitive(parr);
        return codec::field_encode_fixed_arithmetic(
            &canonical, field, col_prefix, row_stride, var_prefix, width, out, ctx,
        );
    }
    // Constant fast path: write the same scalar bytes at each per-row position.
    if let Some(view) = col.as_opt::<Constant>() {
        return encode_constant_arith(view, field, col_prefix, row_stride, var_prefix, width, out);
    }
    // For other fixed columns route through canonicalization and the codec helpers. This
    // is sufficient because Dict / RunEnd values are varlen-only in the dominant benchmark
    // scenarios; if a fixed Dict/RunEnd appears, we accept canonicalization on this path.
    let canonical = col.clone().execute::<Canonical>(ctx)?;
    codec::field_encode_fixed_arithmetic(
        &canonical, field, col_prefix, row_stride, var_prefix, width, out, ctx,
    )
}

fn encode_constant_arith(
    view: ArrayView<'_, Constant>,
    field: SortField,
    col_prefix: u32,
    row_stride: u32,
    var_prefix: Option<&[u32]>,
    _width: u32,
    out: &mut [u8],
) -> VortexResult<()> {
    let bytes = codec::encode_scalar(view.scalar(), field)?;
    let len = bytes.len();
    if len == 0 {
        return Ok(());
    }
    let n = view.len();
    // SAFETY: encoded scalar length matches the per-row width contributed to the size pass,
    // so `pos + len <= out.len()` by buffer construction. For small fixed lengths (the
    // common case: bool=2, i32=5, i64=9, i128=17) we hoist the encoded bytes into
    // register-sized loads before the loop and emit direct word stores per row. This is
    // faster than copy_nonoverlapping for small `len` because the compiler emits a real
    // memcpy call rather than inlining the 1- or 2-word store sequence.
    unsafe {
        let src = bytes.as_ptr();
        let stride = row_stride as usize;
        match (var_prefix, len) {
            // i64-typical: 1 sentinel + 8 value bytes = 9 bytes, no varlen prefix.
            (None, 9) => {
                let v_lo = std::ptr::read_unaligned(src as *const u64);
                let v_hi = *src.add(8);
                let mut dst = out.as_mut_ptr().add(col_prefix as usize);
                for _ in 0..n {
                    std::ptr::write_unaligned(dst as *mut u64, v_lo);
                    *dst.add(8) = v_hi;
                    dst = dst.add(stride);
                }
            }
            // i32-typical: 1 sentinel + 4 value bytes = 5 bytes, no varlen prefix.
            (None, 5) => {
                let v_lo = std::ptr::read_unaligned(src as *const u32);
                let v_hi = *src.add(4);
                let mut dst = out.as_mut_ptr().add(col_prefix as usize);
                for _ in 0..n {
                    std::ptr::write_unaligned(dst as *mut u32, v_lo);
                    *dst.add(4) = v_hi;
                    dst = dst.add(stride);
                }
            }
            // bool / i8: 1 sentinel + 1 value byte = 2 bytes, no varlen prefix.
            (None, 2) => {
                let v = std::ptr::read_unaligned(src as *const u16);
                let mut dst = out.as_mut_ptr().add(col_prefix as usize);
                for _ in 0..n {
                    std::ptr::write_unaligned(dst as *mut u16, v);
                    dst = dst.add(stride);
                }
            }
            // i128: 1 sentinel + 16 value bytes = 17 bytes, no varlen prefix.
            (None, 17) => {
                let v_lo = std::ptr::read_unaligned(src as *const u128);
                let v_hi = *src.add(16);
                let mut dst = out.as_mut_ptr().add(col_prefix as usize);
                for _ in 0..n {
                    std::ptr::write_unaligned(dst as *mut u128, v_lo);
                    *dst.add(16) = v_hi;
                    dst = dst.add(stride);
                }
            }
            // General fallback for other lengths or for the var_prefix case.
            (None, _) => {
                let mut dst = out.as_mut_ptr().add(col_prefix as usize);
                for _ in 0..n {
                    std::ptr::copy_nonoverlapping(src, dst, len);
                    dst = dst.add(stride);
                }
            }
            (Some(vp), _) => {
                let base = out.as_mut_ptr();
                for i in 0..n {
                    let pos = (i as u32) * row_stride + col_prefix + vp[i];
                    std::ptr::copy_nonoverlapping(src, base.add(pos as usize), len);
                }
            }
        }
    }
    Ok(())
}

/// Dispatch a single column's encoding into the shared `out` buffer.
pub fn dispatch_encode(
    col: &ArrayRef,
    field: SortField,
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
    if let Some((_, encode_fn)) = registry::lookup(&col.encoding_id())
        && encode_fn(col, field, offsets, cursors, out, ctx)?.is_some()
    {
        return Ok(());
    }
    let canonical = col.clone().execute::<Canonical>(ctx)?;
    codec::field_encode(&canonical, field, offsets, cursors, out, ctx)
}

/// Mutate-buffer kernel: write this column's per-row bytes into `out` at
/// `offsets[i] + cursors[i]`, advancing `cursors[i]` by the bytes written.
///
/// Return `Ok(None)` to decline and fall back to the canonical path.
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
