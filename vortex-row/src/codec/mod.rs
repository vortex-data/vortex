// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pure byte-encoding kernels for row-oriented output, operating on `Canonical` variants.
//!
//! The encoded byte format produces a lexicographically byte-comparable representation:
//! comparing the byte slices of two encoded rows yields the same ordering as the
//! original logical (tuple) comparison of their values, modulo nulls placement and
//! descending-ness as configured by [`RowSortFieldOptions`].
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
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half::f16;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::options::RowSortFieldOptions;

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
/// [`VARLEN_EMPTY_SIZE`] for empty values.
///
/// # Errors
///
/// Returns an error if the encoded length overflows `u32`. The block count itself always fits
/// (a `BinaryView` length is a `u32`, so `blocks <= ceil(u32::MAX / 32) < 2^27`), but the
/// `blocks * 33 + 1` byte total can exceed `u32::MAX` for multi-gigabyte values.
#[inline]
fn encoded_size_for_non_empty_varlen(len: usize) -> VortexResult<u32> {
    debug_assert!(len > 0);
    let blocks = u32::try_from(len.div_ceil(VARLEN_BLOCK_SIZE))
        .vortex_expect("varlen block count must fit in u32");
    blocks
        .checked_mul(VARLEN_BLOCK_TOTAL_U32)
        .and_then(|b| b.checked_add(1))
        .ok_or_else(|| vortex_err!("varlen encoded size overflows u32"))
}

/// Constant per-row size in bytes for fixed-width encodings (including 1-byte sentinel).
#[inline]
const fn encoded_size_for_fixed(value_bytes: u32) -> u32 {
    1 + value_bytes
}

/// A native byte width (at most 32 for `i256`) always fits in a `u32`.
#[inline]
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
fn varlen_null_sentinel(field: RowSortFieldOptions) -> u8 {
    if field.nulls_first { 0x00 } else { 0xFF }
}

/// Returns the sentinel byte for an empty varlen value.
///
/// Equal to `0x01` in ascending mode and `!0x01 = 0xFE` in descending mode.
#[inline]
fn varlen_empty_sentinel(field: RowSortFieldOptions) -> u8 {
    if field.descending { !0x01u8 } else { 0x01u8 }
}

/// Returns the sentinel byte for a non-empty varlen value.
///
/// Equal to `0x02` in ascending mode and `!0x02 = 0xFD` in descending mode.
#[inline]
fn varlen_non_empty_sentinel(field: RowSortFieldOptions) -> u8 {
    if field.descending { !0x02u8 } else { 0x02u8 }
}

/// The sentinel byte that precedes a non-null fixed-width value.
///
/// Fixed-width values always lead with `0x01`. Null values use a sentinel that sorts either
/// below (`0x00`) or above (`0x02`) it (see [`fixed_null_sentinel`]), so a single leading-byte
/// comparison orders nulls relative to non-nulls. Unlike the value bytes, this sentinel is never
/// inverted for `descending`: null placement is positional and independent of sort direction.
const FIXED_NON_NULL_SENTINEL: u8 = 0x01;

/// Returns the sentinel byte that precedes a null fixed-width value.
///
/// `nulls_first` writes `0x00` (sorts before the [`FIXED_NON_NULL_SENTINEL`] `0x01`); otherwise
/// `0x02` (sorts after). Like the non-null sentinel, the choice is positional and independent of
/// `descending`, matching the convention used by `arrow-row`.
#[inline]
fn fixed_null_sentinel(field: RowSortFieldOptions) -> u8 {
    if field.nulls_first { 0x00 } else { 0x02 }
}

/// Returns the single-byte null sentinel used when a child contributes its canonical null
/// encoding inside a null parent struct/FSL row.
///
/// For varlen children that is the varlen null sentinel; for everything else (including
/// nested struct/FSL when used as a variable-width child) it is the fixed-width null sentinel.
fn child_canonical_null_byte(child_dtype: &DType, field: RowSortFieldOptions) -> u8 {
    match child_dtype {
        DType::Utf8(_) | DType::Binary(_) => varlen_null_sentinel(field),
        _ => fixed_null_sentinel(field),
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
/// Classification does not depend on the [`RowSortFieldOptions`]: null-vs-non-null encoding width is
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
                    .ok_or_else(|| vortex_err!("FSL row width overflows u32"))?;
                let total = body
                    .checked_add(1)
                    .ok_or_else(|| vortex_err!("FSL row width overflows u32"))?;
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
                        total = total
                            .checked_add(w)
                            .ok_or_else(|| vortex_err!("Struct row width overflows u32"))?;
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
    field: RowSortFieldOptions,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match canonical {
        Canonical::Null(arr) => add_size_null(arr, sizes)?,
        Canonical::Bool(_) => add_size_const(sizes, encoded_size_for_fixed(1))?,
        Canonical::Primitive(arr) => add_size_primitive(arr, sizes)?,
        Canonical::Decimal(arr) => add_size_decimal(arr, sizes)?,
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
    field: RowSortFieldOptions,
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
    field: RowSortFieldOptions,
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

mod encoding;
mod native;
mod sizing;
mod varlen;

use encoding::*;
use native::*;
use sizing::*;
use varlen::*;
