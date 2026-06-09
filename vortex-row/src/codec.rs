// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pure byte-encoding kernels for row-oriented output, operating on `Canonical` variants.
//!
//! The encoded byte format produces a lexicographically byte-comparable representation:
//! comparing the byte slices of two encoded rows yields the same ordering as the
//! original logical (tuple) comparison of their values, modulo nulls placement and
//! descending-ness as configured by [`RowSortField`].
//!
//! Conventions:
//! - Every fixed-width value is preceded by a 1-byte sentinel that orders nulls relative to
//!   non-nulls. For `descending`, only the **value** bytes are bit-inverted (XOR with 0xFF),
//!   not the sentinel.
//! - Variable-length (Utf8, Binary) values use **three** distinct leading sentinels — one each
//!   for null, empty, and non-empty — so byte comparison at position 0 fully categorizes the
//!   value and column-byte boundaries stay aligned across rows. See
//!   [`varlen_null_sentinel`], [`varlen_empty_sentinel`], [`varlen_non_empty_sentinel`].
//! - Fixed-width integers are big-endian, with the sign bit flipped for signed types.
//! - Floats are bit-pattern big-endian with sign-aware mask: non-negative flips the top
//!   bit; negative flips all bits.
//! - Nullable structs and fixed-size lists encode null parent rows with a **canonical null
//!   body** so two null parent rows produce byte-equal encodings: fixed-width children
//!   contribute their fixed null encoding, and variable-width children collapse to a single
//!   null sentinel byte.

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half::f16;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::options::RowSortField;

/// Size in bytes of the encoded form of a single bool value (sentinel + 1 content byte).
pub(crate) const BOOL_ENCODED_SIZE: u32 = 2;

/// Block size used in the variable-length encoding.
pub(crate) const VARLEN_BLOCK_SIZE: usize = 32;
/// Total bytes per varlen block including the trailing continuation marker.
pub(crate) const VARLEN_BLOCK_TOTAL: usize = VARLEN_BLOCK_SIZE + 1;
const VARLEN_BLOCK_TOTAL_U32: u32 = 33;

/// Size in bytes of an encoded null varlen value (just the sentinel byte).
pub(crate) const VARLEN_NULL_SIZE: u32 = 1;
/// Size in bytes of an encoded empty varlen value (just the sentinel byte).
pub(crate) const VARLEN_EMPTY_SIZE: u32 = 1;

/// Returns the size in bytes of the encoded form of a non-empty variable-length value.
///
/// Includes the leading sentinel byte plus `ceil(len/32) * 33` block bytes (32 content + 1
/// continuation/length byte). Callers must use [`VARLEN_NULL_SIZE`] for null values and
/// [`VARLEN_EMPTY_SIZE`] for empty values. A `u32` always suffices because a `BinaryView`
/// length is itself a `u32`, so `blocks <= ceil(u32::MAX / 32) < 2^27`.
#[inline]
fn encoded_size_for_non_empty_varlen(len: usize) -> u32 {
    debug_assert!(len > 0);
    let blocks = u32::try_from(len.div_ceil(VARLEN_BLOCK_SIZE))
        .vortex_expect("varlen block count must fit in u32");
    1 + blocks * VARLEN_BLOCK_TOTAL_U32
}

/// Constant per-row size in bytes for fixed-width encodings (including 1-byte sentinel).
#[inline]
const fn encoded_size_for_fixed(value_bytes: u32) -> u32 {
    1 + value_bytes
}

fn byte_width_u32(width: usize) -> u32 {
    u32::try_from(width).vortex_expect("native byte width must fit in u32")
}

/// Pre-resolved per-row validity for the row encoders.
///
/// Encoders pattern-match on this once before their inner loop so the no-nulls fast path
/// avoids per-row `mask.value(i)` branches entirely, and the nullable path materializes the
/// mask exactly once.
pub(crate) enum ValidityKind {
    /// Column statically has no nulls (`Validity::NonNullable` or `AllValid`); no mask needed.
    AllValid,
    /// Column may have nulls; carries the materialized per-row mask.
    Mask(vortex_mask::Mask),
}

/// Resolve a [`Validity`] into a [`ValidityKind`], materializing the mask only when the column
/// may actually have nulls.
#[inline]
pub(crate) fn resolve_validity(
    validity: Validity,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ValidityKind> {
    Ok(match validity {
        Validity::NonNullable | Validity::AllValid => ValidityKind::AllValid,
        other => ValidityKind::Mask(other.execute_mask(len, ctx)?),
    })
}

/// Returns the sentinel byte for a null varlen value.
///
/// The choice is positional (0x00 when nulls sort first, 0xFF when nulls sort last) and
/// independent of `descending`, matching the convention used by `arrow-row`.
#[inline]
fn varlen_null_sentinel(field: RowSortField) -> u8 {
    if field.nulls_first { 0x00 } else { 0xFF }
}

/// Returns the sentinel byte for an empty varlen value.
///
/// Equal to `0x01` in ascending mode and `!0x01 = 0xFE` in descending mode.
#[inline]
fn varlen_empty_sentinel(field: RowSortField) -> u8 {
    if field.descending { !0x01u8 } else { 0x01u8 }
}

/// Returns the sentinel byte for a non-empty varlen value.
///
/// Equal to `0x02` in ascending mode and `!0x02 = 0xFD` in descending mode.
#[inline]
fn varlen_non_empty_sentinel(field: RowSortField) -> u8 {
    if field.descending { !0x02u8 } else { 0x02u8 }
}

/// Returns the single-byte null sentinel used when a child contributes its canonical null
/// encoding inside a null parent struct/FSL row.
///
/// For varlen children that is the varlen null sentinel; for everything else (including
/// nested struct/FSL when used as a variable-width child) it is the fixed-width null sentinel.
fn child_canonical_null_byte(child_dtype: &DType, field: RowSortField) -> u8 {
    match child_dtype {
        DType::Utf8(_) | DType::Binary(_) => varlen_null_sentinel(field),
        _ => field.null_sentinel(),
    }
}

/// Per-row width classification for a column.
///
/// `Fixed(w)` means every row encodes to exactly `w` bytes (sentinel + value), regardless
/// of null-ness or value. `Variable` means per-row sizes depend on the data (Utf8/Binary,
/// List, or any composite that recurses through a variable-width field).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowWidth {
    /// Per-row width is the same constant for every row in the column.
    Fixed(u32),
    /// Per-row width is data-dependent.
    Variable,
}

/// Classify a column's per-row encoded width by inspecting only its [`DType`].
///
/// Returns `Fixed(w)` when every row encodes to exactly `w` bytes (sentinel + value),
/// regardless of null-ness or value. Returns `Variable` when per-row sizes depend on the
/// data.
///
/// Classification does not depend on the [`RowSortField`]: null-vs-non-null encoding width is
/// the same for fixed-width types (the sentinel byte plus zero-fill for nulls).
///
/// # Errors
///
/// Returns an error for dtypes that the row encoder does not support. Width arithmetic that
/// would overflow `u32` is also reported as an error rather than silently saturating.
pub(crate) fn row_width_for_dtype(dtype: &DType) -> VortexResult<RowWidth> {
    match dtype {
        DType::Null => Ok(RowWidth::Fixed(1)),
        DType::Bool(_) => Ok(RowWidth::Fixed(BOOL_ENCODED_SIZE)),
        DType::Primitive(ptype, _) => Ok(RowWidth::Fixed(encoded_size_for_fixed(byte_width_u32(
            ptype.byte_width(),
        )))),
        DType::Decimal(dt, _) => {
            let vt = DecimalType::smallest_decimal_value_type(dt);
            if matches!(vt, DecimalType::I256) {
                vortex_bail!("row encoding for Decimal256 is not yet implemented");
            }
            Ok(RowWidth::Fixed(encoded_size_for_fixed(byte_width_u32(
                vt.byte_width(),
            ))))
        }
        DType::Utf8(_) | DType::Binary(_) => Ok(RowWidth::Variable),
        DType::FixedSizeList(elem, n, _) => match row_width_for_dtype(elem)? {
            // FSL is fixed iff its element type is fixed. Add a sentinel byte for the FSL
            // itself, then `n` copies of the element width.
            RowWidth::Fixed(w) => {
                let body = w
                    .checked_mul(*n)
                    .ok_or_else(|| vortex_error::vortex_err!("FSL row width overflows u32"))?;
                let total = body
                    .checked_add(1)
                    .ok_or_else(|| vortex_error::vortex_err!("FSL row width overflows u32"))?;
                Ok(RowWidth::Fixed(total))
            }
            RowWidth::Variable => Ok(RowWidth::Variable),
        },
        DType::Struct(fields, _) => {
            // Struct is fixed iff all its fields are fixed; sum their widths plus a sentinel.
            let mut total: u32 = 1; // outer sentinel
            for field_dtype in fields.fields() {
                match row_width_for_dtype(&field_dtype)? {
                    RowWidth::Fixed(w) => {
                        total = total.checked_add(w).ok_or_else(|| {
                            vortex_error::vortex_err!("Struct row width overflows u32")
                        })?;
                    }
                    RowWidth::Variable => return Ok(RowWidth::Variable),
                }
            }
            Ok(RowWidth::Fixed(total))
        }
        DType::List(..) => {
            vortex_bail!(
                "row encoding does not support variable-size List arrays (no well-defined ordering)"
            )
        }
        DType::Variant(_) => {
            vortex_bail!("row encoding does not support Variant arrays (no well-defined ordering)")
        }
        DType::Union(_) => vortex_bail!("row encoding does not support Union arrays"),
        dtype => vortex_bail!("row encoding does not support dtype: {dtype:?}"),
    }
}

/// Compute the per-row size in bytes for the given canonical view, adding into `sizes`.
///
/// `sizes` is expected to be initialized (typically zeroed). This function *adds* the
/// per-row size to each entry so multiple columns can accumulate into the same buffer.
///
/// # Errors
///
/// Returns an error for unsupported canonical variants.
pub(crate) fn field_size(
    canonical: &Canonical,
    field: RowSortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match canonical {
        Canonical::Null(arr) => add_size_null(arr, sizes),
        Canonical::Bool(_) => add_size_const(sizes, encoded_size_for_fixed(1)),
        Canonical::Primitive(arr) => add_size_primitive(arr, sizes),
        Canonical::Decimal(arr) => add_size_decimal(arr, sizes),
        Canonical::VarBinView(arr) => add_size_varbinview(arr, sizes, ctx)?,
        Canonical::Struct(arr) => add_size_struct(arr, field, sizes, ctx)?,
        Canonical::FixedSizeList(arr) => add_size_fsl(arr, field, sizes, ctx)?,
        Canonical::List(_) => vortex_bail!(
            "row encoding does not support canonical List arrays: {:?}",
            canonical.dtype()
        ),
        Canonical::Variant(_) => {
            vortex_bail!("row encoding does not support Variant arrays (no well-defined ordering)")
        }
        unsupported => {
            vortex_bail!(
                "row encoding does not support canonical array: {:?}",
                unsupported.dtype()
            )
        }
    }
    Ok(())
}

/// Encode a fixed-width column at arithmetic offsets, without reading or writing any per-row
/// cursor.
///
/// For row `i`, the column's bytes are written starting at `i * row_stride + col_prefix
/// (+ var_prefix[i])`, where `var_prefix` is the exclusive prefix sum of the varlen
/// contributions (`None` when the row layout has no variable-length columns). This is the
/// fast path for fixed-width columns that appear before any varlen column, so their
/// within-row position is a constant offset rather than a running cursor.
///
/// For primitive columns in the pure-fixed case it uses a `chunks_exact_mut` hot loop that
/// removes the per-row offset/cursor indirection (matching `arrow-row`'s `encode_not_null`).
/// All other types reuse [`field_encode`] at the materialized offsets, so the bytes written
/// are byte-identical to the cursor path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn field_encode_fixed_arithmetic(
    canonical: &Canonical,
    field: RowSortField,
    col_prefix: u32,
    row_stride: u32,
    var_prefix: Option<&[u32]>,
    nrows: usize,
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if var_prefix.is_none()
        && let Canonical::Primitive(arr) = canonical
    {
        return encode_primitive_arith(arr, field, col_prefix, row_stride, out, ctx);
    }

    // General path: materialize this column's per-row start offsets and reuse the cursor
    // encoder with zero-initialized cursors, so every row is written at its arithmetic
    // offset with the exact same bytes the cursor path would produce.
    let mut offsets: Vec<u32> = Vec::with_capacity(nrows);
    let mut base = col_prefix;
    match var_prefix {
        None => {
            for _ in 0..nrows {
                offsets.push(base);
                base = base.wrapping_add(row_stride);
            }
        }
        Some(vp) => {
            for &p in vp.iter().take(nrows) {
                offsets.push(base.wrapping_add(p));
                base = base.wrapping_add(row_stride);
            }
        }
    }
    let mut cursors = vec![0u32; nrows];
    field_encode(canonical, field, &offsets, &mut cursors, out, ctx)
}

/// Encode each row's bytes for the given canonical view into `out`, writing starting at
/// `offsets[i] + cursors[i]` for row `i` and advancing `cursors[i]` by the number of
/// bytes written.
///
/// After this call returns successfully, `cursors[i]` will have advanced by exactly the
/// per-row contribution previously computed by [`field_size`] for the same column.
pub(crate) fn field_encode(
    canonical: &Canonical,
    field: RowSortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match canonical {
        Canonical::Null(arr) => encode_null(arr, field, offsets, cursors, out),
        Canonical::Bool(arr) => encode_bool(arr, field, offsets, cursors, out, ctx)?,
        Canonical::Primitive(arr) => encode_primitive(arr, field, offsets, cursors, out, ctx)?,
        Canonical::Decimal(arr) => encode_decimal(arr, field, offsets, cursors, out, ctx)?,
        Canonical::VarBinView(arr) => encode_varbinview(arr, field, offsets, cursors, out, ctx)?,
        Canonical::Struct(arr) => encode_struct(arr, field, offsets, cursors, out, ctx)?,
        Canonical::FixedSizeList(arr) => encode_fsl(arr, field, offsets, cursors, out, ctx)?,
        Canonical::List(_) => vortex_bail!(
            "row encoding does not support canonical List arrays: {:?}",
            canonical.dtype()
        ),
        Canonical::Variant(_) => {
            vortex_bail!("row encoding does not support Variant arrays (no well-defined ordering)")
        }
        unsupported => {
            vortex_bail!(
                "row encoding does not support canonical array: {:?}",
                unsupported.dtype()
            )
        }
    }
    Ok(())
}

fn add_size_const(sizes: &mut [u32], add: u32) {
    for s in sizes.iter_mut() {
        *s += add;
    }
}

fn add_size_null(arr: &NullArray, sizes: &mut [u32]) {
    debug_assert_eq!(arr.len(), sizes.len());
    // Just a sentinel byte per row.
    for s in sizes.iter_mut() {
        *s += 1;
    }
}

fn add_size_primitive(arr: &PrimitiveArray, sizes: &mut [u32]) {
    let width = byte_width_u32(arr.ptype().byte_width());
    add_size_const(sizes, encoded_size_for_fixed(width));
}

fn add_size_decimal(arr: &DecimalArray, sizes: &mut [u32]) {
    let width = byte_width_u32(arr.values_type().byte_width());
    add_size_const(sizes, encoded_size_for_fixed(width));
}

fn add_size_varbinview(
    arr: &VarBinViewArray,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let views = arr.views();
    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            for (i, view) in views.iter().enumerate() {
                let contribution = if view.is_empty() {
                    VARLEN_EMPTY_SIZE
                } else {
                    encoded_size_for_non_empty_varlen(view.len() as usize)
                };
                sizes[i] = sizes[i]
                    .checked_add(contribution)
                    .vortex_expect("per-row size overflow");
            }
        }
        ValidityKind::Mask(mask) => {
            for (i, view) in views.iter().enumerate() {
                let contribution = if !mask.value(i) {
                    VARLEN_NULL_SIZE
                } else if view.is_empty() {
                    VARLEN_EMPTY_SIZE
                } else {
                    encoded_size_for_non_empty_varlen(view.len() as usize)
                };
                sizes[i] = sizes[i]
                    .checked_add(contribution)
                    .vortex_expect("per-row size overflow");
            }
        }
    }
    Ok(())
}

fn add_size_struct(
    arr: &StructArray,
    field: RowSortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    let mask = arr.as_ref().validity()?.execute_mask(n, ctx)?;
    // Outer sentinel: 1 byte per row.
    for s in sizes.iter_mut() {
        *s = s.checked_add(1).vortex_expect("per-row size overflow");
    }
    // Each child contributes its per-row size when the parent is non-null, and a canonical
    // null contribution when the parent is null. For fixed-width children both are equal,
    // so we can simply add the fixed width to every row. For variable-width children the
    // null contribution collapses to 1 byte, ensuring null parent rows have a constant body.
    for child in arr.iter_unmasked_fields() {
        match row_width_for_dtype(child.dtype())? {
            RowWidth::Fixed(w) => add_size_const(sizes, w),
            RowWidth::Variable => {
                let canonical = child.clone().execute::<Canonical>(ctx)?;
                let mut child_sizes = vec![0u32; n];
                field_size(&canonical, field, &mut child_sizes, ctx)?;
                for i in 0..n {
                    let contribution = if mask.value(i) { child_sizes[i] } else { 1u32 };
                    sizes[i] = sizes[i]
                        .checked_add(contribution)
                        .vortex_expect("per-row size overflow");
                }
            }
        }
    }
    Ok(())
}

fn add_size_fsl(
    arr: &FixedSizeListArray,
    field: RowSortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    debug_assert_eq!(n, sizes.len());
    let list_size = arr.list_size() as usize;
    let mask = arr.as_ref().validity()?.execute_mask(n, ctx)?;
    let elem_dtype = arr.elements().dtype();
    // Outer sentinel: 1 byte per row.
    for s in sizes.iter_mut() {
        *s = s.checked_add(1).vortex_expect("per-row size overflow");
    }
    match row_width_for_dtype(elem_dtype)? {
        RowWidth::Fixed(w) => {
            // Each row has `list_size` fixed-width elements regardless of null parent mask.
            let body = w
                .checked_mul(u32::try_from(list_size).vortex_expect("list_size fits u32"))
                .vortex_expect("FSL body width overflow");
            add_size_const(sizes, body);
        }
        RowWidth::Variable => {
            let elements = arr.elements().clone().execute::<Canonical>(ctx)?;
            debug_assert_eq!(elements.len(), n * list_size);
            let mut elem_sizes = vec![0u32; n * list_size];
            field_size(&elements, field, &mut elem_sizes, ctx)?;
            for i in 0..n {
                let body: u32 = if mask.value(i) {
                    let base = i * list_size;
                    let mut sum: u32 = 0;
                    for j in 0..list_size {
                        sum = sum
                            .checked_add(elem_sizes[base + j])
                            .vortex_expect("FSL row body overflow");
                    }
                    sum
                } else {
                    // Canonical null body for FSL with variable element: one null sentinel
                    // per element. (Each element contributes `child_null_width = 1`.)
                    u32::try_from(list_size).vortex_expect("list_size fits u32")
                };
                sizes[i] = sizes[i]
                    .checked_add(body)
                    .vortex_expect("FSL per-row size overflow");
            }
        }
    }
    Ok(())
}

fn encode_null(
    arr: &NullArray,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
) {
    let sentinel = field.null_sentinel();
    for i in 0..arr.len() {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        out[pos] = sentinel;
        col_offset[i] += 1;
    }
}

fn encode_bool(
    arr: &BoolArray,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let bits = arr.clone().into_bit_buffer();
    let non_null = field.non_null_sentinel();
    let xor = if field.descending { 0xFF } else { 0x00 };
    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            for i in 0..bits.len() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                out[pos] = non_null;
                // false=0x01, true=0x02 so false < true; XOR for descending
                let raw = if bits.value(i) { 0x02u8 } else { 0x01u8 };
                out[pos + 1] = raw ^ xor;
                col_offset[i] += BOOL_ENCODED_SIZE;
            }
        }
        ValidityKind::Mask(mask) => {
            let null = field.null_sentinel();
            for i in 0..bits.len() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                if mask.value(i) {
                    out[pos] = non_null;
                    let raw = if bits.value(i) { 0x02u8 } else { 0x01u8 };
                    out[pos + 1] = raw ^ xor;
                } else {
                    out[pos] = null;
                    out[pos + 1] = 0;
                }
                col_offset[i] += BOOL_ENCODED_SIZE;
            }
        }
    }
    Ok(())
}

fn encode_primitive(
    arr: &PrimitiveArray,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_native_ptype!(arr.ptype(), |T| {
        encode_primitive_typed::<T>(arr, field, row_offsets, col_offset, out, ctx)?;
    });
    Ok(())
}

fn encode_primitive_typed<T: NativePType + RowEncode>(
    arr: &PrimitiveArray,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let slice: &[T] = arr.as_slice();
    let non_null = field.non_null_sentinel();
    let value_bytes = size_of::<T>();
    let stride = encoded_size_for_fixed(byte_width_u32(value_bytes));
    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            for (i, &v) in slice.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                out[pos] = non_null;
                v.encode_to(&mut out[pos + 1..pos + 1 + value_bytes], field.descending);
                col_offset[i] += stride;
            }
        }
        ValidityKind::Mask(mask) => {
            let null = field.null_sentinel();
            for (i, &v) in slice.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                if mask.value(i) {
                    out[pos] = non_null;
                    v.encode_to(&mut out[pos + 1..pos + 1 + value_bytes], field.descending);
                } else {
                    out[pos] = null;
                    // Zero-fill the value bytes.
                    for b in &mut out[pos + 1..pos + 1 + value_bytes] {
                        *b = 0;
                    }
                }
                col_offset[i] += stride;
            }
        }
    }
    Ok(())
}

fn encode_decimal(
    arr: &DecimalArray,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mask = arr.as_ref().validity()?.execute_mask(arr.len(), ctx)?;
    match arr.values_type() {
        DecimalType::I8 => {
            encode_decimal_typed::<i8>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I16 => {
            encode_decimal_typed::<i16>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I32 => {
            encode_decimal_typed::<i32>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I64 => {
            encode_decimal_typed::<i64>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I128 => {
            encode_decimal_typed::<i128>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I256 => {
            vortex_bail!("row encoding for Decimal256 is not yet implemented")
        }
    }
    Ok(())
}

fn encode_decimal_typed<T>(
    arr: &DecimalArray,
    mask: &vortex_mask::Mask,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
) where
    T: vortex_array::dtype::NativeDecimalType + RowEncode,
{
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let value_bytes = size_of::<T>();
    let total = encoded_size_for_fixed(byte_width_u32(value_bytes));
    let slice = arr.buffer::<T>();
    for i in 0..slice.len() {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        if mask.value(i) {
            out[pos] = non_null;
            slice[i].encode_to(&mut out[pos + 1..pos + 1 + value_bytes], field.descending);
        } else {
            out[pos] = null;
            for b in &mut out[pos + 1..pos + 1 + value_bytes] {
                *b = 0;
            }
        }
        col_offset[i] += total;
    }
}

fn encode_varbinview(
    arr: &VarBinViewArray,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let null_byte = varlen_null_sentinel(field);
    let empty_byte = varlen_empty_sentinel(field);
    let non_empty_byte = varlen_non_empty_sentinel(field);
    let descending = field.descending;

    let views = arr.views();
    // Cache the data-buffer slices once. Inlined views (len <= 12) carry their bytes inline,
    // so they never touch `buffers`; referenced views index into the pre-validated buffer at
    // `offset..offset + len`. Walking views directly avoids the per-row bounds and branch work
    // of `with_iterator`.
    let buffers: smallvec::SmallVec<[&[u8]; 4]> = (0..arr.data_buffers().len())
        .map(|i| arr.buffer(i).as_slice())
        .collect();

    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            for (i, view) in views.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                let len = view.len() as usize;
                if len == 0 {
                    out[pos] = empty_byte;
                    col_offset[i] += VARLEN_EMPTY_SIZE;
                    continue;
                }
                let bytes: &[u8] = if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let r = view.as_view();
                    let off = r.offset as usize;
                    &buffers[r.buffer_index as usize][off..off + len]
                };
                out[pos] = non_empty_byte;
                let written = encode_non_empty_varlen_body(bytes, &mut out[pos + 1..], descending);
                col_offset[i] += 1 + written;
            }
        }
        ValidityKind::Mask(mask) => {
            for (i, view) in views.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                if !mask.value(i) {
                    out[pos] = null_byte;
                    col_offset[i] += VARLEN_NULL_SIZE;
                    continue;
                }
                let len = view.len() as usize;
                if len == 0 {
                    out[pos] = empty_byte;
                    col_offset[i] += VARLEN_EMPTY_SIZE;
                    continue;
                }
                let bytes: &[u8] = if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let r = view.as_view();
                    let off = r.offset as usize;
                    &buffers[r.buffer_index as usize][off..off + len]
                };
                out[pos] = non_empty_byte;
                let written = encode_non_empty_varlen_body(bytes, &mut out[pos + 1..], descending);
                col_offset[i] += 1 + written;
            }
        }
    }
    Ok(())
}

fn encode_struct(
    arr: &StructArray,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    let mask = arr.as_ref().validity()?.execute_mask(n, ctx)?;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();

    // Write the outer sentinel for each row.
    for i in 0..n {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        out[pos] = if mask.value(i) { non_null } else { null };
        col_offset[i] += 1;
    }

    // Encode each child. For non-null parent rows the child contributes its actual encoding;
    // for null parent rows the child contributes its canonical null encoding so that two null
    // parent rows produce byte-equal output regardless of underlying child values.
    for child in arr.iter_unmasked_fields() {
        match row_width_for_dtype(child.dtype())? {
            RowWidth::Fixed(w) => {
                let canonical = child.clone().execute::<Canonical>(ctx)?;
                field_encode(&canonical, field, row_offsets, col_offset, out, ctx)?;
                // Replace null parent rows with the canonical null encoding (the same as a
                // child-level null: null sentinel followed by zero-padded value bytes).
                let null_byte = child_canonical_null_byte(child.dtype(), field);
                for i in 0..n {
                    if !mask.value(i) {
                        let end = (row_offsets[i] + col_offset[i]) as usize;
                        let start = end - w as usize;
                        out[start] = null_byte;
                        for b in &mut out[start + 1..end] {
                            *b = 0;
                        }
                    }
                }
            }
            RowWidth::Variable => {
                encode_variable_child(child, field, &mask, row_offsets, col_offset, out, ctx)?;
            }
        }
    }

    Ok(())
}

fn encode_fsl(
    arr: &FixedSizeListArray,
    field: RowSortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let nrows = arr.len();
    let list_size = arr.list_size() as usize;
    let mask = arr.as_ref().validity()?.execute_mask(nrows, ctx)?;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let elem_dtype = arr.elements().dtype().clone();

    // Outer sentinel.
    for i in 0..nrows {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        out[pos] = if mask.value(i) { non_null } else { null };
        col_offset[i] += 1;
    }

    match row_width_for_dtype(&elem_dtype)? {
        RowWidth::Fixed(w) => {
            // Fixed-width elements: encode the elements array directly (its length is
            // nrows * list_size) using a derived (offsets, cursors) pair. Then overwrite
            // the body of null parent rows with the canonical null encoding per element.
            let elements = arr.elements().clone().execute::<Canonical>(ctx)?;
            debug_assert_eq!(elements.len(), nrows * list_size);
            let list_size_u32 = arr.list_size();
            let row_body_bytes = w
                .checked_mul(list_size_u32)
                .vortex_expect("FSL body width overflow");
            let mut elem_offsets = vec![0u32; nrows * list_size];
            for i in 0..nrows {
                let base = row_offsets[i] + col_offset[i];
                for j in 0u32..list_size_u32 {
                    elem_offsets[i * list_size + j as usize] = base + j * w;
                }
            }
            let mut elem_cursors = vec![0u32; nrows * list_size];
            field_encode(&elements, field, &elem_offsets, &mut elem_cursors, out, ctx)?;
            for i in 0..nrows {
                col_offset[i] = col_offset[i]
                    .checked_add(row_body_bytes)
                    .vortex_expect("FSL row body overflow");
            }
            // Canonical null body for null parent rows: one null encoding per element.
            let null_byte = child_canonical_null_byte(&elem_dtype, field);
            let elem_width = w as usize;
            for i in 0..nrows {
                if !mask.value(i) {
                    let end = (row_offsets[i] + col_offset[i]) as usize;
                    let start = end - row_body_bytes as usize;
                    let mut pos = start;
                    for _ in 0..list_size {
                        out[pos] = null_byte;
                        for b in &mut out[pos + 1..pos + elem_width] {
                            *b = 0;
                        }
                        pos += elem_width;
                    }
                }
            }
        }
        RowWidth::Variable => {
            // Variable-width elements: for null parent rows the canonical body is exactly
            // `list_size` null sentinel bytes (one per element). For non-null parent rows,
            // encode each element via a scratch buffer and copy into out.
            let elements = arr.elements().clone().execute::<Canonical>(ctx)?;
            debug_assert_eq!(elements.len(), nrows * list_size);
            let mut elem_sizes = vec![0u32; nrows * list_size];
            field_size(&elements, field, &mut elem_sizes, ctx)?;
            let total: u64 = elem_sizes.iter().map(|&s| u64::from(s)).sum();
            let total_usize =
                usize::try_from(total).vortex_expect("FSL scratch buffer size fits usize");
            let mut scratch = vec![0u8; total_usize];
            let mut scratch_offsets = Vec::with_capacity(nrows * list_size);
            let mut acc: u32 = 0;
            for &s in &elem_sizes {
                scratch_offsets.push(acc);
                acc = acc
                    .checked_add(s)
                    .vortex_expect("FSL scratch offset overflow");
            }
            let mut scratch_cursors = vec![0u32; nrows * list_size];
            field_encode(
                &elements,
                field,
                &scratch_offsets,
                &mut scratch_cursors,
                &mut scratch,
                ctx,
            )?;
            let null_byte = child_canonical_null_byte(&elem_dtype, field);
            for i in 0..nrows {
                let dst = (row_offsets[i] + col_offset[i]) as usize;
                if mask.value(i) {
                    let mut body_bytes: u32 = 0;
                    for j in 0..list_size {
                        let k = i * list_size + j;
                        let src = scratch_offsets[k] as usize;
                        let sz = elem_sizes[k] as usize;
                        out[dst + body_bytes as usize..dst + body_bytes as usize + sz]
                            .copy_from_slice(&scratch[src..src + sz]);
                        body_bytes = body_bytes
                            .checked_add(elem_sizes[k])
                            .vortex_expect("FSL body bytes overflow");
                    }
                    col_offset[i] = col_offset[i]
                        .checked_add(body_bytes)
                        .vortex_expect("FSL row offset overflow");
                } else {
                    for offset in 0..list_size {
                        out[dst + offset] = null_byte;
                    }
                    col_offset[i] = col_offset[i]
                        .checked_add(u32::try_from(list_size).vortex_expect("list_size fits u32"))
                        .vortex_expect("FSL row offset overflow");
                }
            }
        }
    }

    Ok(())
}

/// Encode one variable-width child of a struct: for non-null parent rows, copy the child's
/// natural encoding from a scratch buffer; for null parent rows, write a single
/// `child_canonical_null_byte`.
fn encode_variable_child(
    child: &vortex_array::ArrayRef,
    field: RowSortField,
    parent_mask: &vortex_mask::Mask,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = child.len();
    let canonical = child.clone().execute::<Canonical>(ctx)?;

    // Size and encode the child into a sequential scratch buffer.
    let mut child_sizes = vec![0u32; n];
    field_size(&canonical, field, &mut child_sizes, ctx)?;
    let total: u64 = child_sizes.iter().map(|&s| u64::from(s)).sum();
    let total_usize = usize::try_from(total).vortex_expect("child scratch buffer size fits usize");
    let mut scratch = vec![0u8; total_usize];
    let mut scratch_offsets = Vec::with_capacity(n);
    let mut acc: u32 = 0;
    for &s in &child_sizes {
        scratch_offsets.push(acc);
        acc = acc
            .checked_add(s)
            .vortex_expect("child scratch offset overflow");
    }
    let mut scratch_cursors = vec![0u32; n];
    field_encode(
        &canonical,
        field,
        &scratch_offsets,
        &mut scratch_cursors,
        &mut scratch,
        ctx,
    )?;

    let null_byte = child_canonical_null_byte(child.dtype(), field);
    for i in 0..n {
        let dst = (row_offsets[i] + col_offset[i]) as usize;
        if parent_mask.value(i) {
            let src = scratch_offsets[i] as usize;
            let sz = child_sizes[i] as usize;
            out[dst..dst + sz].copy_from_slice(&scratch[src..src + sz]);
            col_offset[i] = col_offset[i]
                .checked_add(child_sizes[i])
                .vortex_expect("col_offset overflow");
        } else {
            out[dst] = null_byte;
            col_offset[i] = col_offset[i]
                .checked_add(1)
                .vortex_expect("col_offset overflow");
        }
    }
    Ok(())
}

/// Arithmetic-write primitive encoder: writes each row's `sentinel + value` slot at a
/// constant within-row offset, iterating the output in `row_stride`-sized chunks so the
/// compiler can drop the per-row offset/cursor indirection.
fn encode_primitive_arith(
    arr: &PrimitiveArray,
    field: RowSortField,
    col_prefix: u32,
    row_stride: u32,
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_native_ptype!(arr.ptype(), |T| {
        encode_primitive_arith_typed::<T>(arr, field, col_prefix, row_stride, out, ctx)?;
    });
    Ok(())
}

fn encode_primitive_arith_typed<T: NativePType + RowEncode>(
    arr: &PrimitiveArray,
    field: RowSortField,
    col_prefix: u32,
    row_stride: u32,
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let slice: &[T] = arr.as_slice();
    let non_null = field.non_null_sentinel();
    let value_bytes = size_of::<T>();
    let slot_size = 1 + value_bytes;
    let stride = row_stride as usize;
    let prefix = col_prefix as usize;
    let descending = field.descending;

    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            // Hot path: each row's slot is a fixed window inside its `stride`-sized chunk,
            // so the inner write vectorizes the same way as `arrow-row`'s not-null path.
            for (chunk, &v) in out.chunks_exact_mut(stride).zip(slice.iter()) {
                let slot = &mut chunk[prefix..prefix + slot_size];
                slot[0] = non_null;
                v.encode_to(&mut slot[1..], descending);
            }
        }
        ValidityKind::Mask(mask) => {
            let null = field.null_sentinel();
            for (i, (chunk, &v)) in out.chunks_exact_mut(stride).zip(slice.iter()).enumerate() {
                let slot = &mut chunk[prefix..prefix + slot_size];
                if mask.value(i) {
                    slot[0] = non_null;
                    v.encode_to(&mut slot[1..], descending);
                } else {
                    slot[0] = null;
                    for b in &mut slot[1..] {
                        *b = 0;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Encode a non-empty variable-length byte slice into `out` in 32-byte blocks with
/// continuation/length markers. Returns the number of bytes written. Empty values are
/// encoded by the caller as a single sentinel byte and never reach this function.
///
/// For the ascending path the hot loop is a `copy_nonoverlapping` of 32 bytes per block
/// plus one stamped continuation byte. For the descending path it reads a u64 at a time and
/// XORs with `0xFF`, giving LLVM a vectorizable inner loop.
fn encode_non_empty_varlen_body(bytes: &[u8], out: &mut [u8], descending: bool) -> u32 {
    debug_assert!(!bytes.is_empty());
    let len = bytes.len();
    let full_blocks = len / VARLEN_BLOCK_SIZE;
    let partial = len % VARLEN_BLOCK_SIZE;
    let (full_to_write, partial_block_len) = if partial == 0 {
        // Length is an exact multiple of 32: emit (full_blocks - 1) full blocks with the
        // 0xFF continuation marker, then a final block whose continuation byte is 32.
        (full_blocks - 1, VARLEN_BLOCK_SIZE)
    } else {
        (full_blocks, partial)
    };
    let total = (full_to_write + 1) * VARLEN_BLOCK_TOTAL;
    debug_assert!(out.len() >= total);
    // The final block's continuation byte encodes its content length (1..=32).
    let len_byte =
        u8::try_from(partial_block_len).vortex_expect("varlen final block length (1..=32) fits u8");

    // SAFETY: `out` has at least `total` bytes — the caller sizes every varlen slot via
    // `encoded_size_for_non_empty_varlen` (which equals `1 + total`, the extra byte being the
    // leading sentinel that the caller wrote and that is not part of `out`). `bytes` is valid
    // for `len` reads, and every pointer advance below stays within `[0, total)` for `dst`
    // and `[0, len)` for `src`.
    unsafe {
        let mut src = bytes.as_ptr();
        let mut dst = out.as_mut_ptr();

        if !descending {
            // Ascending fast path: each full block is a 32-byte memcpy + a single 0xFF stamp.
            for _ in 0..full_to_write {
                std::ptr::copy_nonoverlapping(src, dst, VARLEN_BLOCK_SIZE);
                *dst.add(VARLEN_BLOCK_SIZE) = 0xFF;
                src = src.add(VARLEN_BLOCK_SIZE);
                dst = dst.add(VARLEN_BLOCK_TOTAL);
            }
            // Final block: copy the partial data, zero-pad the tail, write the length byte.
            std::ptr::copy_nonoverlapping(src, dst, partial_block_len);
            std::ptr::write_bytes(
                dst.add(partial_block_len),
                0,
                VARLEN_BLOCK_SIZE - partial_block_len,
            );
            *dst.add(VARLEN_BLOCK_SIZE) = len_byte;
        } else {
            // Descending: invert every value byte. A u64-stride XOR gives LLVM a vectorizable
            // inner loop; the tail handles the partial block byte-wise.
            for _ in 0..full_to_write {
                xor_copy_block(src, dst);
                *dst.add(VARLEN_BLOCK_SIZE) = 0x00; // descending counterpart of 0xFF
                src = src.add(VARLEN_BLOCK_SIZE);
                dst = dst.add(VARLEN_BLOCK_TOTAL);
            }
            for i in 0..partial_block_len {
                *dst.add(i) = *src.add(i) ^ 0xFF;
            }
            std::ptr::write_bytes(
                dst.add(partial_block_len),
                0xFF, // 0x00 XOR 0xFF
                VARLEN_BLOCK_SIZE - partial_block_len,
            );
            *dst.add(VARLEN_BLOCK_SIZE) = len_byte ^ 0xFF;
        }
    }
    u32::try_from(total).vortex_expect("encoded varlen byte length fits u32")
}

/// Copy 32 bytes from `src` to `dst`, XORing each with `0xFF`. LLVM auto-vectorizes the
/// four u64-wide iterations into SIMD on x86.
///
/// # Safety
/// `src` must be valid for 32 reads, `dst` valid for 32 writes, and the regions must not
/// overlap.
#[inline(always)]
unsafe fn xor_copy_block(src: *const u8, dst: *mut u8) {
    // Four u64 lanes of 8 bytes each = 32 bytes total.
    for i in 0..4 {
        let off = i * 8;
        // SAFETY: the caller guarantees src/dst are valid for the full 32-byte block.
        let v = unsafe { std::ptr::read_unaligned(src.add(off) as *const u64) };
        unsafe { std::ptr::write_unaligned(dst.add(off) as *mut u64, v ^ u64::MAX) };
    }
}

/// Internal trait for encoding a fixed-width native value into byte slots.
///
/// Implementations must produce a sequence of `size_of::<Self>()` bytes that is
/// lexicographically byte-comparable according to the natural ordering of the type.
pub(crate) trait RowEncode: Copy {
    /// Encode this value into `out`, inverting the bytes for descending order.
    fn encode_to(self, out: &mut [u8], descending: bool);
}

macro_rules! impl_row_encode_unsigned {
    ($t:ty) => {
        impl RowEncode for $t {
            #[inline]
            fn encode_to(self, out: &mut [u8], descending: bool) {
                let bytes = self.to_be_bytes();
                if descending {
                    for (i, b) in bytes.iter().enumerate() {
                        out[i] = b ^ 0xFF;
                    }
                } else {
                    out.copy_from_slice(&bytes);
                }
            }
        }
    };
}

macro_rules! impl_row_encode_signed {
    ($t:ty) => {
        impl RowEncode for $t {
            #[inline]
            fn encode_to(self, out: &mut [u8], descending: bool) {
                let mut bytes = self.to_be_bytes();
                // Flip sign bit so negatives < non-negatives lexicographically.
                bytes[0] ^= 0x80;
                if descending {
                    for (i, b) in bytes.iter().enumerate() {
                        out[i] = b ^ 0xFF;
                    }
                } else {
                    out.copy_from_slice(&bytes);
                }
            }
        }
    };
}

impl_row_encode_unsigned!(u8);
impl_row_encode_unsigned!(u16);
impl_row_encode_unsigned!(u32);
impl_row_encode_unsigned!(u64);
impl_row_encode_signed!(i8);
impl_row_encode_signed!(i16);
impl_row_encode_signed!(i32);
impl_row_encode_signed!(i64);
impl_row_encode_signed!(i128);

impl RowEncode for f32 {
    fn encode_to(self, out: &mut [u8], descending: bool) {
        let bits = self.to_bits();
        let mask: u32 = if (bits >> 31) == 0 {
            0x8000_0000
        } else {
            0xFFFF_FFFF
        };
        let mut bytes = (bits ^ mask).to_be_bytes();
        if descending {
            for b in bytes.iter_mut() {
                *b ^= 0xFF;
            }
        }
        out.copy_from_slice(&bytes);
    }
}

impl RowEncode for f64 {
    fn encode_to(self, out: &mut [u8], descending: bool) {
        let bits = self.to_bits();
        let mask: u64 = if (bits >> 63) == 0 {
            0x8000_0000_0000_0000
        } else {
            0xFFFF_FFFF_FFFF_FFFF
        };
        let mut bytes = (bits ^ mask).to_be_bytes();
        if descending {
            for b in bytes.iter_mut() {
                *b ^= 0xFF;
            }
        }
        out.copy_from_slice(&bytes);
    }
}

impl RowEncode for f16 {
    fn encode_to(self, out: &mut [u8], descending: bool) {
        let bits = self.to_bits();
        let mask: u16 = if (bits >> 15) == 0 { 0x8000 } else { 0xFFFF };
        let mut bytes = (bits ^ mask).to_be_bytes();
        if descending {
            for b in bytes.iter_mut() {
                *b ^= 0xFF;
            }
        }
        out.copy_from_slice(&bytes);
    }
}
