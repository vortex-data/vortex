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
use vortex_array::arrays::Constant;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::Primitive;
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
use crate::size::ColKind;
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
        col_kinds,
        first_varlen_idx,
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
    //
    // When fixed-before-varlen columns exist alongside a varlen column, we also build
    // `var_prefix_for_arith[i] = exclusive cumsum of var_lengths[..i]` and pass it to
    // the arithmetic encoders so they can compute per-row positions without a cursor.
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

    let mut listview_offsets: Vec<u32> = Vec::with_capacity(nrows);
    let mut var_prefix_for_arith: Option<Vec<u32>> = None;
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
            // Mixed path: offsets[i] = i * fixed_per_row + var_prefix[i] where
            // var_prefix is the exclusive cumsum of varlen lengths. Same raw-pointer
            // write loop as the pure-fixed branch (auto-vectorized); the total was
            // validated to fit in u32 upstream so `wrapping_add` is sound here.
            let mut vp: Option<Vec<u32>> = need_arith_prefix.then(|| Vec::with_capacity(nrows));
            // SAFETY: we just reserved nrows; writes at indices [0, nrows) are valid.
            // Likewise `vp` (if Some) has reserved nrows.
            unsafe {
                let off_ptr = listview_offsets.as_mut_ptr();
                let vp_ptr = vp.as_mut().map(|p| p.as_mut_ptr());
                let mut acc: u32 = 0;
                for (i, &l) in v.iter().enumerate() {
                    if let Some(p) = vp_ptr {
                        p.add(i).write(acc);
                    }
                    off_ptr
                        .add(i)
                        .write((i as u32).wrapping_mul(fixed_per_row).wrapping_add(acc));
                    acc = acc.wrapping_add(l);
                }
                listview_offsets.set_len(nrows);
                if let Some(p) = vp.as_mut() {
                    p.set_len(nrows);
                }
            }
            var_prefix_for_arith = vp;
        }
    }

    // Per-row write cursor (also doubles as the ListView `sizes` slot when done).
    //
    // The cursor path starts at `prefix_at_first_varlen` so that `listview_offsets[i] +
    // cursors[i]` lands at the position of the first cursor-path column (i.e. after the
    // bytes already written by the arithmetic path for fixed-before-varlen columns).
    //
    // When there are no varlen columns at all, every column went through the arith path,
    // so the cursor path runs zero iterations. Pre-seeding the cursors with
    // `fixed_per_row` makes them already correct as per-row sizes in that case.
    let initial_cursor: u32 = match first_varlen_idx {
        Some(idx) => match col_kinds[idx] {
            ColKind::Variable { fixed_prefix } => fixed_prefix,
            ColKind::Fixed { .. } => unreachable!("first_varlen_idx points to a varlen column"),
        },
        None => fixed_per_row,
    };
    let mut row_cursors = vec![initial_cursor; nrows];

    // ===== Phase 4: encode columns =====
    // Fixed-before-varlen columns take the arithmetic write path (no cursor mutation).
    // Fixed-after-varlen and varlen columns take the cursor path, which already runs
    // through `dispatch_encode`.
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
                dispatch_encode(
                    col,
                    options.fields[i],
                    &listview_offsets,
                    &mut row_cursors,
                    &mut out_buf,
                    ctx,
                )?;
            }
        }
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

/// Dispatch a single column's encoding through the arithmetic fast path. This is used for
/// fixed-width columns that appear before any variable-length column in the row layout: the
/// within-row write offset is a constant `col_prefix + var_prefix[i]` (or just `col_prefix`
/// for the pure-fixed case), so we can skip the per-row cursor read/write entirely.
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
    // the hot loop is reached without going through `execute_until::<AnyCanonical>`.
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
    // For other fixed columns route through canonicalization and the codec helpers.
    let canonical = col.clone().execute::<Canonical>(ctx)?;
    codec::field_encode_fixed_arithmetic(
        &canonical, field, col_prefix, row_stride, var_prefix, width, out, ctx,
    )
}

/// Constant-specific arithmetic writer. Encodes the scalar bytes once, then writes the same
/// bytes into each per-row slot via direct register-sized stores for the common small
/// lengths (2/5/9/17), or `copy_nonoverlapping` as a fallback.
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
    // register-sized loads before the loop and emit direct write_unaligned stores per row.
    // This is faster than copy_nonoverlapping for small `len` because the compiler emits a
    // real memcpy call rather than inlining the 1- or 2-word store sequence.
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
            // General fallback for other lengths.
            (None, _) => {
                let mut dst = out.as_mut_ptr().add(col_prefix as usize);
                for _ in 0..n {
                    std::ptr::copy_nonoverlapping(src, dst, len);
                    dst = dst.add(stride);
                }
            }
            (Some(vp), 9) => {
                let v_lo = std::ptr::read_unaligned(src as *const u64);
                let v_hi = *src.add(8);
                let base = out.as_mut_ptr();
                for i in 0..n {
                    let pos = (i as u32) * row_stride + col_prefix + vp[i];
                    let dst = base.add(pos as usize);
                    std::ptr::write_unaligned(dst as *mut u64, v_lo);
                    *dst.add(8) = v_hi;
                }
            }
            (Some(vp), 5) => {
                let v_lo = std::ptr::read_unaligned(src as *const u32);
                let v_hi = *src.add(4);
                let base = out.as_mut_ptr();
                for i in 0..n {
                    let pos = (i as u32) * row_stride + col_prefix + vp[i];
                    let dst = base.add(pos as usize);
                    std::ptr::write_unaligned(dst as *mut u32, v_lo);
                    *dst.add(4) = v_hi;
                }
            }
            (Some(vp), 2) => {
                let v = std::ptr::read_unaligned(src as *const u16);
                let base = out.as_mut_ptr();
                for i in 0..n {
                    let pos = (i as u32) * row_stride + col_prefix + vp[i];
                    std::ptr::write_unaligned(base.add(pos as usize) as *mut u16, v);
                }
            }
            (Some(vp), 17) => {
                let v_lo = std::ptr::read_unaligned(src as *const u128);
                let v_hi = *src.add(16);
                let base = out.as_mut_ptr();
                for i in 0..n {
                    let pos = (i as u32) * row_stride + col_prefix + vp[i];
                    let dst = base.add(pos as usize);
                    std::ptr::write_unaligned(dst as *mut u128, v_lo);
                    *dst.add(16) = v_hi;
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
