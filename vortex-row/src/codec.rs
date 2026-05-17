// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    reason = "row encoding indexes into u32-sized buffers; lengths are validated to fit in u32 elsewhere"
)]

//! Pure byte-encoding kernels for row-oriented output, operating on `Canonical` variants.
//!
//! The encoded byte format produces a lexicographically byte-comparable representation:
//! comparing the byte slices of two encoded rows yields the same ordering as the
//! original logical (tuple) comparison of their values, modulo nulls placement and
//! descending-ness as configured by [`SortField`].
//!
//! Conventions:
//! - Every value is preceded by a 1-byte sentinel that orders nulls relative to non-nulls.
//! - For `descending`, only the **value** bytes are bit-inverted (XOR with 0xFF), not the
//!   sentinel.
//! - Fixed-width integers are big-endian, with the sign bit flipped for signed types.
//! - Floats are bit-pattern big-endian with sign-aware mask: non-negative flips the top
//!   bit; negative flips all bits.
//!
//! This commit covers only the fixed-width canonical variants (Null, Bool, Primitive,
//! Decimal); variable-length and nested canonical variants land in later commits.

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::options::SortField;

/// Size in bytes of the encoded form of a single bool value (sentinel + 1 content byte).
pub const BOOL_ENCODED_SIZE: u32 = 2;

/// Block size used in the variable-length encoding.
pub const VARLEN_BLOCK_SIZE: usize = 32;
/// Total bytes per varlen block including the trailing continuation marker.
pub const VARLEN_BLOCK_TOTAL: usize = VARLEN_BLOCK_SIZE + 1;

/// Returns the size in bytes of the encoded form of a variable-length value of the given length.
#[inline]
fn encoded_size_for_varlen(len: usize) -> u32 {
    // 1 sentinel + ceil(len/32)*33 content bytes (or 1 zero terminator if empty)
    if len == 0 {
        1 + 1
    } else {
        let blocks = len.div_ceil(VARLEN_BLOCK_SIZE);
        1 + (blocks as u32) * (VARLEN_BLOCK_TOTAL as u32)
    }
}

/// Constant per-row size in bytes for fixed-width encodings (including 1-byte sentinel).
#[inline]
const fn encoded_size_for_fixed(value_bytes: u32) -> u32 {
    1 + value_bytes
}

/// Pre-resolved per-row validity for the row encoders.
///
/// Encoders pattern-match on this once before their inner loop so the
/// no-nulls fast path avoids per-row `mask.value(i)` branches entirely,
/// and the nullable path holds the materialized mask exactly once.
pub(crate) enum ValidityKind {
    /// Column statically has no nulls (`Validity::NonNullable` or `AllValid`); no mask
    /// allocation needed.
    AllValid,
    /// Column may have nulls; the materialized per-row mask is included.
    Mask(vortex_mask::Mask),
}

/// Resolve a [`Validity`] into a [`ValidityKind`], materializing the mask only when
/// the column may actually have nulls.
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

/// Per-row width classification for a column.
///
/// `Fixed(w)` means every row encodes to exactly `w` bytes (sentinel + value), regardless
/// of null-ness or value. `Variable` means per-row sizes depend on the data (Utf8/Binary,
/// List, or any composite that recurses through a variable-width field).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowWidth {
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
/// Classification does not depend on the [`SortField`]: null-vs-non-null encoding width is
/// the same for fixed-width types (the sentinel byte plus zero-fill for nulls).
///
/// # Errors
///
/// Returns an error for dtypes that the row encoder does not yet support. Variable-length
/// dtypes (Utf8/Binary), nested dtypes (Struct/FixedSizeList/Extension), and
/// Variant/Union/List arrive in later commits.
pub fn row_width_for_dtype(dtype: &DType) -> VortexResult<RowWidth> {
    match dtype {
        DType::Null => Ok(RowWidth::Fixed(1)),
        DType::Bool(_) => Ok(RowWidth::Fixed(BOOL_ENCODED_SIZE)),
        DType::Primitive(ptype, _) => Ok(RowWidth::Fixed(encoded_size_for_fixed(
            ptype.byte_width() as u32,
        ))),
        DType::Decimal(dt, _) => {
            let vt = DecimalType::smallest_decimal_value_type(dt);
            Ok(RowWidth::Fixed(encoded_size_for_fixed(
                vt.byte_width() as u32
            )))
        }
        DType::Utf8(_) | DType::Binary(_) => Ok(RowWidth::Variable),
        DType::FixedSizeList(elem, n, _) => match row_width_for_dtype(elem)? {
            // FSL is fixed iff its element type is fixed. Add a sentinel byte for the FSL
            // itself, then `n` copies of the element width.
            RowWidth::Fixed(w) => {
                let body = w.saturating_mul(*n);
                Ok(RowWidth::Fixed(body.saturating_add(1)))
            }
            RowWidth::Variable => Ok(RowWidth::Variable),
        },
        DType::Struct(fields, _) => {
            // Struct is fixed iff all its fields are fixed; sum their widths plus a sentinel.
            let mut total: u32 = 1; // outer sentinel
            for field_dtype in fields.fields() {
                match row_width_for_dtype(&field_dtype)? {
                    RowWidth::Fixed(w) => total = total.saturating_add(w),
                    RowWidth::Variable => return Ok(RowWidth::Variable),
                }
            }
            Ok(RowWidth::Fixed(total))
        }
        DType::List(..) => Ok(RowWidth::Variable),
        DType::Extension(ext) => row_width_for_dtype(ext.storage_dtype()),
        DType::Variant(_) => {
            vortex_bail!("row encoding does not support Variant arrays (no well-defined ordering)")
        }
        DType::Union(_) => vortex_bail!("row encoding does not support Union arrays"),
    }
}

/// Compute the per-row size in bytes for the given canonical view, adding into `sizes`.
///
/// `sizes` is expected to be initialized (typically zeroed). This function *adds* the
/// per-row size to each entry so multiple columns can accumulate into the same buffer.
///
/// # Errors
///
/// Returns an error for unsupported canonical variants. Variable-length and nested
/// variants land in later commits.
pub fn field_size(
    canonical: &Canonical,
    field: SortField,
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
        Canonical::Extension(arr) => add_size_extension(arr, field, sizes, ctx)?,
        Canonical::List(_) => vortex_bail!(
            "row encoding does not yet support canonical type {:?}",
            canonical.dtype()
        ),
        Canonical::Variant(_) => {
            vortex_bail!("row encoding does not support Variant arrays (no well-defined ordering)")
        }
    }
    Ok(())
}

/// Encode each row's bytes for the given canonical view into `out`, writing starting at
/// `offsets[i] + cursors[i]` for row `i` and advancing `cursors[i]` by the number of
/// bytes written.
///
/// After this call returns successfully, `cursors[i]` will have advanced by exactly the
/// per-row contribution previously computed by [`field_size`] for the same column.
pub fn field_encode(
    canonical: &Canonical,
    field: SortField,
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
        Canonical::Extension(arr) => encode_extension(arr, field, offsets, cursors, out, ctx)?,
        Canonical::List(_) => vortex_bail!(
            "row encoding does not yet support canonical type {:?}",
            canonical.dtype()
        ),
        Canonical::Variant(_) => {
            vortex_bail!("row encoding does not support Variant arrays (no well-defined ordering)")
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
    let width = arr.ptype().byte_width() as u32;
    add_size_const(sizes, encoded_size_for_fixed(width));
}

fn add_size_decimal(arr: &DecimalArray, sizes: &mut [u32]) {
    let width = arr.values_type().byte_width() as u32;
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
                sizes[i] += encoded_size_for_varlen(view.len() as usize);
            }
        }
        ValidityKind::Mask(mask) => {
            for (i, view) in views.iter().enumerate() {
                if mask.value(i) {
                    sizes[i] += encoded_size_for_varlen(view.len() as usize);
                } else {
                    sizes[i] += 1; // sentinel only
                }
            }
        }
    }
    Ok(())
}

fn add_size_struct(
    arr: &StructArray,
    field: SortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // null sentinel: 1 byte per row.
    for s in sizes.iter_mut() {
        *s += 1;
    }
    // Each field adds its own per-row size.
    for child in arr.iter_unmasked_fields() {
        let canonical = child.clone().execute::<Canonical>(ctx)?;
        field_size(&canonical, field, sizes, ctx)?;
    }
    Ok(())
}

fn add_size_fsl(
    arr: &FixedSizeListArray,
    field: SortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    debug_assert_eq!(n, sizes.len());
    let list_size = arr.list_size() as usize;
    let elements = arr.elements().clone().execute::<Canonical>(ctx)?;
    debug_assert_eq!(elements.len(), n * list_size);
    // Sizing: 1 sentinel + sum of element sizes (`list_size` per row).
    // We compute element-wise sizes into a contiguous scratch buffer then reduce by row.
    let mut elem_sizes = vec![0u32; n * list_size];
    field_size(&elements, field, &mut elem_sizes, ctx)?;
    for i in 0..n {
        let mut sum: u32 = 1; // sentinel
        let base = i * list_size;
        for j in 0..list_size {
            sum = sum.saturating_add(elem_sizes[base + j]);
        }
        sizes[i] += sum;
    }
    Ok(())
}

fn add_size_extension(
    arr: &ExtensionArray,
    field: SortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let storage = arr.storage_array().clone().execute::<Canonical>(ctx)?;
    field_size(&storage, field, sizes, ctx)
}

fn encode_null(
    arr: &NullArray,
    field: SortField,
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
    field: SortField,
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
                    // false=0x01, true=0x02 so false < true; XOR for descending
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
    field: SortField,
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
    field: SortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let slice: &[T] = arr.as_slice();
    let non_null = field.non_null_sentinel();
    let value_bytes = size_of::<T>();
    let stride = encoded_size_for_fixed(value_bytes as u32);
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
    field: SortField,
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
    field: SortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
) where
    T: vortex_array::dtype::NativeDecimalType + RowEncode,
{
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let value_bytes = size_of::<T>();
    let total = encoded_size_for_fixed(value_bytes as u32);
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
    field: SortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let non_null = field.non_null_sentinel();
    let descending = field.descending;
    let views = arr.views();
    let n_buffers = arr.data_buffers().len();
    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            // Cache data-buffer slices once. For inlined views (len <= 12), bytes live
            // inside the view itself.
            let buffers: smallvec::SmallVec<[&[u8]; 4]> =
                (0..n_buffers).map(|i| arr.buffer(i).as_slice()).collect();
            for (i, view) in views.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                out[pos] = non_null;
                let len = view.len() as usize;
                // SAFETY: BinaryView's inlined-vs-ref discriminant is its `size` field
                // (read by `view.len()`); for len <= 12 the bytes are inline in the view
                // (we read from `as_inlined().value()`); for larger we index into the
                // pre-validated buffer at `view_ref.offset..offset+size`. Both reads
                // produce a slice of exactly `len` valid bytes.
                let bytes: &[u8] = if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let r = view.as_view();
                    let off = r.offset as usize;
                    &buffers[r.buffer_index as usize][off..off + len]
                };
                let written = encode_varlen_value(bytes, &mut out[pos + 1..], descending);
                col_offset[i] += 1 + written;
            }
        }
        ValidityKind::Mask(mask) => {
            let null = field.null_sentinel();
            arr.with_iterator(|iter| {
                for (i, maybe) in iter.enumerate() {
                    let pos = (row_offsets[i] + col_offset[i]) as usize;
                    if !mask.value(i) {
                        out[pos] = null;
                        col_offset[i] += 1;
                        continue;
                    }
                    let bytes: &[u8] = maybe.unwrap_or(&[]);
                    out[pos] = non_null;
                    let written = encode_varlen_value(bytes, &mut out[pos + 1..], descending);
                    col_offset[i] += 1 + written;
                }
            });
        }
    }
    Ok(())
}

fn encode_struct(
    arr: &StructArray,
    field: SortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    let mask = arr.as_ref().validity()?.execute_mask(n, ctx)?;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();

    // First, write the sentinel for each row. We track the post-sentinel cursor offsets
    // for the body in `body_cursors` (which start exactly at +1 of the input cursor).
    // For null rows we additionally need to zero-fill the (uniform-width) field bytes,
    // but because struct widths are variable in general, we record null indexes first
    // and zero-fill after we know each row's contribution.
    //
    // To keep the implementation simple we:
    //   1) advance the cursor past the sentinel,
    //   2) recursively encode each field's bytes (the field encoders ignore nullness of
    //      the struct, but use their own per-field nullness),
    //   3) for null struct rows, overwrite the body bytes with zeros so the encoded form
    //      depends only on the sentinel.
    let body_start: Vec<u32> = (0..n).map(|i| col_offset[i] + 1).collect();
    for i in 0..n {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        out[pos] = if mask.value(i) { non_null } else { null };
        col_offset[i] += 1;
    }

    for child in arr.iter_unmasked_fields() {
        let canonical = child.clone().execute::<Canonical>(ctx)?;
        field_encode(&canonical, field, row_offsets, col_offset, out, ctx)?;
    }

    // Zero-fill body bytes of null rows (the field encoders may have written values).
    for i in 0..n {
        if !mask.value(i) {
            let start = (row_offsets[i] + body_start[i]) as usize;
            let end = (row_offsets[i] + col_offset[i]) as usize;
            for b in &mut out[start..end] {
                *b = 0;
            }
        }
    }

    Ok(())
}

fn encode_fsl(
    arr: &FixedSizeListArray,
    field: SortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    let list_size = arr.list_size() as usize;
    let mask = arr.as_ref().validity()?.execute_mask(n, ctx)?;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let elements = arr.elements().clone().execute::<Canonical>(ctx)?;
    debug_assert_eq!(elements.len(), n * list_size);

    // Write sentinels and remember body start for null zero-fill.
    let body_start: Vec<u32> = (0..n).map(|i| col_offset[i] + 1).collect();
    for i in 0..n {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        out[pos] = if mask.value(i) { non_null } else { null };
        col_offset[i] += 1;
    }

    // Encode all `n * list_size` elements into the body. Build a fresh
    // (offsets, cursors) pair where each element gets one slot. Then sum bytes back
    // into the parent col_offset.
    let mut elem_sizes = vec![0u32; n * list_size];
    field_size(&elements, field, &mut elem_sizes, ctx)?;
    // Element offsets are sequential starting at each parent's current cursor position.
    let mut elem_offsets = vec![0u32; n * list_size];
    for i in 0..n {
        let mut acc = row_offsets[i] + col_offset[i];
        for j in 0..list_size {
            elem_offsets[i * list_size + j] = acc;
            acc = acc.saturating_add(elem_sizes[i * list_size + j]);
        }
    }
    let mut elem_cursors = vec![0u32; n * list_size];
    field_encode(&elements, field, &elem_offsets, &mut elem_cursors, out, ctx)?;
    // Advance the parent cursors by the total per-row element bytes.
    for i in 0..n {
        let mut sum: u32 = 0;
        for j in 0..list_size {
            sum = sum.saturating_add(elem_sizes[i * list_size + j]);
        }
        col_offset[i] = col_offset[i].saturating_add(sum);
    }

    // Zero-fill null bodies.
    for i in 0..n {
        if !mask.value(i) {
            let start = (row_offsets[i] + body_start[i]) as usize;
            let end = (row_offsets[i] + col_offset[i]) as usize;
            for b in &mut out[start..end] {
                *b = 0;
            }
        }
    }

    Ok(())
}

fn encode_extension(
    arr: &ExtensionArray,
    field: SortField,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let storage = arr.storage_array().clone().execute::<Canonical>(ctx)?;
    field_encode(&storage, field, row_offsets, col_offset, out, ctx)
}

/// Encode a variable-length byte slice into `out` in 32-byte blocks with
/// continuation markers. Returns the number of bytes written.
///
/// For the ascending path (descending == false), the hot loop is a `copy_nonoverlapping`
/// of 32 bytes per block plus one stamped continuation byte — no per-byte work. For the
/// descending path, the hot loop reads u64-at-a-time and XORs with 0xFF to give LLVM
/// a vectorizable inner loop.
fn encode_varlen_value(bytes: &[u8], out: &mut [u8], descending: bool) -> u32 {
    if bytes.is_empty() {
        // Single zero terminator (descending flips it to 0xFF).
        out[0] = if descending { 0xFF } else { 0 };
        return 1;
    }
    let len = bytes.len();
    let full_blocks = len / VARLEN_BLOCK_SIZE;
    let partial = len % VARLEN_BLOCK_SIZE;
    let (full_to_write, partial_block_len) = if partial == 0 {
        // Length is an exact multiple of 32. The spec emits (full_blocks-1) full blocks
        // with 0xFF continuation, plus a final block whose continuation byte is 32.
        (full_blocks - 1, VARLEN_BLOCK_SIZE)
    } else {
        (full_blocks, partial)
    };
    let total = (full_to_write + 1) * VARLEN_BLOCK_TOTAL;
    debug_assert!(out.len() >= total);

    // SAFETY: bounds checked above. The encoder always invokes us with `out.len()`
    // >= encoded_size_for_varlen(bytes.len()) - 1 (the leading sentinel is written by the
    // caller and not counted here).
    unsafe {
        let mut src = bytes.as_ptr();
        let mut dst = out.as_mut_ptr();

        if !descending {
            // Ascending fast path: full blocks are memcpy + a single 0xFF stamp.
            for _ in 0..full_to_write {
                std::ptr::copy_nonoverlapping(src, dst, VARLEN_BLOCK_SIZE);
                *dst.add(VARLEN_BLOCK_SIZE) = 0xFF;
                src = src.add(VARLEN_BLOCK_SIZE);
                dst = dst.add(VARLEN_BLOCK_TOTAL);
            }
            // Final block: copy the partial data, zero-pad the tail, write the
            // length byte as the continuation marker.
            std::ptr::copy_nonoverlapping(src, dst, partial_block_len);
            std::ptr::write_bytes(
                dst.add(partial_block_len),
                0,
                VARLEN_BLOCK_SIZE - partial_block_len,
            );
            *dst.add(VARLEN_BLOCK_SIZE) = partial_block_len as u8;
        } else {
            // Descending: invert all value bytes. u64-stride XOR gives LLVM a
            // vectorizable inner loop; the tail handles the partial block.
            for _ in 0..full_to_write {
                xor_copy_block(src, dst);
                *dst.add(VARLEN_BLOCK_SIZE) = 0x00; // descending counterpart of 0xFF
                src = src.add(VARLEN_BLOCK_SIZE);
                dst = dst.add(VARLEN_BLOCK_TOTAL);
            }
            // Final block: XOR-copy the partial data, fill the tail with 0xFF
            // (which is 0x00 XOR 0xFF), then write the inverted length byte.
            for i in 0..partial_block_len {
                *dst.add(i) = *src.add(i) ^ 0xFF;
            }
            std::ptr::write_bytes(
                dst.add(partial_block_len),
                0xFF,
                VARLEN_BLOCK_SIZE - partial_block_len,
            );
            *dst.add(VARLEN_BLOCK_SIZE) = (partial_block_len as u8) ^ 0xFF;
        }
    }
    total as u32
}

/// Copy 32 bytes from `src` to `dst`, XORing each with 0xFF. Auto-vectorized by LLVM
/// into SIMD on x86 (verified via cargo asm in earlier iterations).
///
/// # Safety
/// `src` must be valid for 32 reads; `dst` must be valid for 32 writes; the regions
/// may not overlap.
#[inline(always)]
unsafe fn xor_copy_block(src: *const u8, dst: *mut u8) {
    // Use u64 chunks (4 lanes of 8 bytes = 32 bytes total).
    for i in 0..4 {
        let off = i * 8;
        // SAFETY: caller upholds the contract that src/dst are valid for 32 bytes.
        let v = unsafe { std::ptr::read_unaligned(src.add(off) as *const u64) };
        unsafe { std::ptr::write_unaligned(dst.add(off) as *mut u64, v ^ u64::MAX) };
    }
}

/// Internal trait for encoding a fixed-width native value into byte slots.
///
/// Implementations must produce a sequence of `size_of::<Self>()` bytes that is
/// lexicographically byte-comparable according to the natural ordering of the type.
pub trait RowEncode: Copy {
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

/// Encode a single scalar primitive value of a known PType into a buffer slot.
pub fn encode_scalar_primitive(
    ptype: PType,
    value: vortex_array::scalar::PValue,
    field: SortField,
    is_null: bool,
    out: &mut ByteBufferMut,
) -> VortexResult<()> {
    if is_null {
        out.push(field.null_sentinel());
        return Ok(());
    }
    out.push(field.non_null_sentinel());
    let width = ptype.byte_width();
    let mut tmp = [0u8; 16];
    let buf = &mut tmp[..width];
    match_each_native_ptype!(
        ptype,
        integral: |T| {
            let v: T = T::try_from(value)?;
            v.encode_to(buf, field.descending);
        },
        floating: |T| {
            let v: T = T::try_from(value)?;
            v.encode_to(buf, field.descending);
        }
    );
    out.extend_from_slice(buf);
    Ok(())
}

/// Encode a single varlen value into a buffer.
pub fn encode_scalar_varlen(value: Option<&[u8]>, field: SortField, out: &mut ByteBufferMut) {
    match value {
        None => out.push(field.null_sentinel()),
        Some(bytes) => {
            out.push(field.non_null_sentinel());
            let needed = if bytes.is_empty() {
                1
            } else {
                bytes.len().div_ceil(VARLEN_BLOCK_SIZE) * VARLEN_BLOCK_TOTAL
            };
            let start = out.len();
            for _ in 0..needed {
                out.push(0);
            }
            let written = encode_varlen_value(bytes, &mut out[start..], field.descending);
            debug_assert_eq!(written as usize, needed);
        }
    }
}

/// Encode a single boolean value.
pub fn encode_scalar_bool(value: Option<bool>, field: SortField, out: &mut ByteBufferMut) {
    match value {
        None => {
            out.push(field.null_sentinel());
            out.push(0);
        }
        Some(b) => {
            out.push(field.non_null_sentinel());
            let raw = if b { 0x02u8 } else { 0x01u8 };
            let xor = if field.descending { 0xFFu8 } else { 0 };
            out.push(raw ^ xor);
        }
    }
}

/// Encode a single null-type value (only the sentinel).
pub fn encode_scalar_null(field: SortField, is_null: bool, out: &mut ByteBufferMut) {
    if is_null {
        out.push(field.null_sentinel());
    } else {
        out.push(field.non_null_sentinel());
    }
}

/// Returns the per-row encoded size for a scalar value (used for the Constant fast path).
pub fn encoded_size_for_scalar(
    scalar: &vortex_array::scalar::Scalar,
    _field: SortField,
) -> VortexResult<u32> {
    if scalar.is_null() {
        match scalar.dtype() {
            DType::Null => Ok(1),
            DType::Bool(_) => Ok(BOOL_ENCODED_SIZE),
            DType::Primitive(ptype, _) => Ok(encoded_size_for_fixed(ptype.byte_width() as u32)),
            DType::Decimal(dt, _) => {
                let vt = DecimalType::smallest_decimal_value_type(dt);
                Ok(encoded_size_for_fixed(vt.byte_width() as u32))
            }
            DType::Utf8(_) | DType::Binary(_) => Ok(1),
            _ => vortex_bail!(
                "unsupported scalar dtype for row encoding: {}",
                scalar.dtype()
            ),
        }
    } else {
        match scalar.dtype() {
            DType::Null => Ok(1),
            DType::Bool(_) => Ok(BOOL_ENCODED_SIZE),
            DType::Primitive(ptype, _) => Ok(encoded_size_for_fixed(ptype.byte_width() as u32)),
            DType::Decimal(..) => {
                let dec = scalar.as_decimal();
                let vt = dec
                    .decimal_value()
                    .map(|v| v.decimal_type())
                    .unwrap_or(DecimalType::I128);
                Ok(encoded_size_for_fixed(vt.byte_width() as u32))
            }
            DType::Utf8(_) => {
                let bs = scalar
                    .as_utf8()
                    .value()
                    .map(|s| s.as_str().len())
                    .unwrap_or(0);
                Ok(encoded_size_for_varlen(bs))
            }
            DType::Binary(_) => {
                let bs = scalar.as_binary().value().map(|b| b.len()).unwrap_or(0);
                Ok(encoded_size_for_varlen(bs))
            }
            _ => vortex_bail!(
                "unsupported scalar dtype for row encoding: {}",
                scalar.dtype()
            ),
        }
    }
}

/// Encode a single scalar value into a fresh `Bytes` buffer.
pub fn encode_scalar(
    scalar: &vortex_array::scalar::Scalar,
    field: SortField,
) -> VortexResult<bytes::Bytes> {
    use vortex_array::scalar::PValue;
    let size = encoded_size_for_scalar(scalar, field)? as usize;
    let mut out = ByteBufferMut::with_capacity(size);
    if scalar.is_null() {
        match scalar.dtype() {
            DType::Null => out.push(field.null_sentinel()),
            DType::Bool(_) => {
                out.push(field.null_sentinel());
                out.push(0);
            }
            DType::Primitive(ptype, _) => {
                out.push(field.null_sentinel());
                let width = ptype.byte_width();
                for _ in 0..width {
                    out.push(0);
                }
            }
            DType::Decimal(dt, _) => {
                out.push(field.null_sentinel());
                let vt = DecimalType::smallest_decimal_value_type(dt);
                for _ in 0..vt.byte_width() {
                    out.push(0);
                }
            }
            DType::Utf8(_) | DType::Binary(_) => out.push(field.null_sentinel()),
            _ => vortex_bail!(
                "unsupported scalar dtype for row encoding: {}",
                scalar.dtype()
            ),
        }
    } else {
        match scalar.dtype() {
            DType::Null => out.push(field.non_null_sentinel()),
            DType::Bool(_) => {
                let v = scalar.as_bool().value().unwrap_or(false);
                encode_scalar_bool(Some(v), field, &mut out);
            }
            DType::Primitive(ptype, _) => {
                let v: PValue = scalar
                    .as_primitive()
                    .pvalue()
                    .ok_or_else(|| vortex_error::vortex_err!("missing primitive value"))?;
                encode_scalar_primitive(*ptype, v, field, false, &mut out)?;
            }
            DType::Decimal(..) => {
                let dec = scalar.as_decimal();
                out.push(field.non_null_sentinel());
                let value = dec
                    .decimal_value()
                    .ok_or_else(|| vortex_error::vortex_err!("missing decimal value"))?;
                match value {
                    vortex_array::scalar::DecimalValue::I8(v) => {
                        let mut tmp = [0u8; 1];
                        v.encode_to(&mut tmp, field.descending);
                        out.extend_from_slice(&tmp);
                    }
                    vortex_array::scalar::DecimalValue::I16(v) => {
                        let mut tmp = [0u8; 2];
                        v.encode_to(&mut tmp, field.descending);
                        out.extend_from_slice(&tmp);
                    }
                    vortex_array::scalar::DecimalValue::I32(v) => {
                        let mut tmp = [0u8; 4];
                        v.encode_to(&mut tmp, field.descending);
                        out.extend_from_slice(&tmp);
                    }
                    vortex_array::scalar::DecimalValue::I64(v) => {
                        let mut tmp = [0u8; 8];
                        v.encode_to(&mut tmp, field.descending);
                        out.extend_from_slice(&tmp);
                    }
                    vortex_array::scalar::DecimalValue::I128(v) => {
                        let mut tmp = [0u8; 16];
                        v.encode_to(&mut tmp, field.descending);
                        out.extend_from_slice(&tmp);
                    }
                    vortex_array::scalar::DecimalValue::I256(_) => {
                        vortex_bail!("row encoding for Decimal256 is not yet implemented")
                    }
                }
            }
            DType::Utf8(_) => {
                let v = scalar.as_utf8();
                let bytes = v.value().map(|s| s.as_str().as_bytes()).unwrap_or(&[]);
                encode_scalar_varlen(Some(bytes), field, &mut out);
            }
            DType::Binary(_) => {
                let v = scalar.as_binary();
                let bytes = v.value().map(|b| b.as_slice()).unwrap_or(&[]);
                encode_scalar_varlen(Some(bytes), field, &mut out);
            }
            _ => vortex_bail!(
                "unsupported scalar dtype for row encoding: {}",
                scalar.dtype()
            ),
        }
    }
    Ok(out.freeze().into_inner())
}
