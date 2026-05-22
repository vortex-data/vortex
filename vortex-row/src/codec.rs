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
//! - Every value is preceded by a 1-byte sentinel that orders nulls relative to non-nulls.
//! - For `descending`, only the **value** bytes are bit-inverted (XOR with 0xFF), not the
//!   sentinel.
//! - Fixed-width integers are big-endian, with the sign bit flipped for signed types.
//! - Floats are bit-pattern big-endian with sign-aware mask: non-negative flips the top
//!   bit; negative flips all bits.

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
use vortex_array::dtype::half::f16;
use vortex_array::match_each_native_ptype;
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

/// Returns the size in bytes of the encoded form of a variable-length value of the given length.
#[inline]
fn encoded_size_for_varlen(len: usize) -> u32 {
    // 1 sentinel + ceil(len/32)*33 content bytes (or 1 zero terminator if empty)
    if len == 0 {
        1 + 1
    } else {
        let blocks = u32::try_from(len.div_ceil(VARLEN_BLOCK_SIZE))
            .vortex_expect("varlen block count must fit in u32");
        1 + blocks * VARLEN_BLOCK_TOTAL_U32
    }
}

/// Constant per-row size in bytes for fixed-width encodings (including 1-byte sentinel).
#[inline]
const fn encoded_size_for_fixed(value_bytes: u32) -> u32 {
    1 + value_bytes
}

fn byte_width_u32(width: usize) -> u32 {
    u32::try_from(width).vortex_expect("native byte width must fit in u32")
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
/// Returns an error for dtypes that the row encoder does not support.
pub(crate) fn row_width_for_dtype(dtype: &DType) -> VortexResult<RowWidth> {
    match dtype {
        DType::Null => Ok(RowWidth::Fixed(1)),
        DType::Bool(_) => Ok(RowWidth::Fixed(BOOL_ENCODED_SIZE)),
        DType::Primitive(ptype, _) => Ok(RowWidth::Fixed(encoded_size_for_fixed(byte_width_u32(
            ptype.byte_width(),
        )))),
        DType::Decimal(dt, _) => {
            let vt = DecimalType::smallest_decimal_value_type(dt);
            Ok(RowWidth::Fixed(encoded_size_for_fixed(byte_width_u32(
                vt.byte_width(),
            ))))
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
        Canonical::Extension(arr) => add_size_extension(arr, field, sizes, ctx)?,
        Canonical::List(_) => vortex_bail!(
            "row encoding does not support canonical List arrays: {:?}",
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
        Canonical::Extension(arr) => encode_extension(arr, field, offsets, cursors, out, ctx)?,
        Canonical::List(_) => vortex_bail!(
            "row encoding does not support canonical List arrays: {:?}",
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
    let mask = arr.as_ref().validity()?.execute_mask(arr.len(), ctx)?;
    let views = arr.views();
    for (i, view) in views.iter().enumerate() {
        let valid = mask.value(i);
        if !valid {
            sizes[i] += 1; // sentinel only
        } else {
            let len = view.len() as usize;
            sizes[i] += encoded_size_for_varlen(len);
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
    field: RowSortField,
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
    field: RowSortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let storage = arr.storage_array().clone().execute::<Canonical>(ctx)?;
    field_size(&storage, field, sizes, ctx)
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
    let mask = arr.as_ref().validity()?.execute_mask(arr.len(), ctx)?;
    let bits = arr.clone().into_bit_buffer();
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let xor = if field.descending { 0xFF } else { 0x00 };
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
    let mask = arr.as_ref().validity()?.execute_mask(arr.len(), ctx)?;
    let slice: &[T] = arr.as_slice();
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let value_bytes = size_of::<T>();
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
        col_offset[i] += encoded_size_for_fixed(byte_width_u32(value_bytes));
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
    let mask = arr.as_ref().validity()?.execute_mask(arr.len(), ctx)?;
    let non_null = field.non_null_sentinel();
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
            let written = encode_varlen_value(bytes, &mut out[pos + 1..], field.descending);
            col_offset[i] += 1 + written;
        }
    });
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
    field: RowSortField,
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
    field: RowSortField,
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
fn encode_varlen_value(bytes: &[u8], out: &mut [u8], descending: bool) -> u32 {
    let xor = if descending { 0xFFu8 } else { 0x00 };
    if bytes.is_empty() {
        // Single zero terminator.
        out[0] = xor;
        return 1;
    }
    let mut written = 0usize;
    let mut remaining = bytes;
    while remaining.len() > VARLEN_BLOCK_SIZE {
        // Full block, continuation marker 0xFF (then XORed if descending).
        let block = &remaining[..VARLEN_BLOCK_SIZE];
        for (i, &b) in block.iter().enumerate() {
            out[written + i] = b ^ xor;
        }
        out[written + VARLEN_BLOCK_SIZE] = 0xFF ^ xor;
        written += VARLEN_BLOCK_TOTAL;
        remaining = &remaining[VARLEN_BLOCK_SIZE..];
    }
    // Final partial block: pad with zeros, last byte = remaining.len() (1..=32).
    let n = remaining.len();
    for (i, &b) in remaining.iter().enumerate() {
        out[written + i] = b ^ xor;
    }
    for j in n..VARLEN_BLOCK_SIZE {
        out[written + j] = xor;
    }
    out[written + VARLEN_BLOCK_SIZE] =
        u8::try_from(n).vortex_expect("final varlen block length must fit in u8") ^ xor;
    written += VARLEN_BLOCK_TOTAL;
    u32::try_from(written).vortex_expect("encoded varlen byte length must fit in u32")
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
