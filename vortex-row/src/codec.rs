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
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::match_each_native_ptype;
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
#[allow(
    dead_code,
    reason = "used once varlen support lands in a follow-up commit"
)]
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
        DType::Utf8(_) | DType::Binary(_) => {
            vortex_bail!("row encoding for {} is not yet supported", dtype)
        }
        DType::Struct(..) | DType::FixedSizeList(..) | DType::List(..) | DType::Extension(..) => {
            vortex_bail!("row encoding for {} is not yet supported", dtype)
        }
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
    _field: SortField,
    sizes: &mut [u32],
    _ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match canonical {
        Canonical::Null(arr) => add_size_null(arr, sizes),
        Canonical::Bool(_) => add_size_const(sizes, encoded_size_for_fixed(1)),
        Canonical::Primitive(arr) => add_size_primitive(arr, sizes),
        Canonical::Decimal(arr) => add_size_decimal(arr, sizes),
        Canonical::VarBinView(_)
        | Canonical::Struct(_)
        | Canonical::FixedSizeList(_)
        | Canonical::Extension(_)
        | Canonical::List(_) => vortex_bail!(
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
        Canonical::VarBinView(_)
        | Canonical::Struct(_)
        | Canonical::FixedSizeList(_)
        | Canonical::Extension(_)
        | Canonical::List(_) => vortex_bail!(
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
        col_offset[i] += encoded_size_for_fixed(value_bytes as u32);
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
            _ => vortex_bail!(
                "unsupported scalar dtype for row encoding: {}",
                scalar.dtype()
            ),
        }
    }
    Ok(out.freeze().into_inner())
}
