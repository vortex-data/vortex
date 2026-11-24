// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-exact floating-point narrowing for [`WidenArray`].
//!
//! A floating-point value fits in a narrower IEEE-754 type iff narrowing and widening back
//! reproduces the original bit pattern and does not leave a signaling NaN in the narrow child.
//! Comparing bit patterns preserves `-0.0` vs `+0.0` and canonical quiet NaNs while rejecting
//! signaling NaNs and non-representable quiet NaN payloads.

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::WidenArray;
use crate::arrays::widen::compute::wrap_values;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::scalar::PValue;

/// Number of lanes checked per branchless chunk so LLVM autovectorizes the scan while still
/// exiting early on non-narrowable columns.
const FIT_CHECK_CHUNK_LEN: usize = 64;

/// Returns whether an `f16` value is a signaling NaN (exponent all ones, nonzero payload, quiet
/// bit `0x0200` unset).
#[inline]
fn is_f16_snan(value: f16) -> bool {
    let abs_bits = value.to_bits() & 0x7fff;
    abs_bits > 0x7c00 && (abs_bits & 0x0200) == 0
}

/// Returns whether an `f32` value is a signaling NaN (exponent all ones, nonzero payload, quiet
/// bit `1 << 22` unset).
#[inline]
fn is_f32_snan(value: f32) -> bool {
    let abs_bits = value.to_bits() & 0x7fff_ffff;
    abs_bits > 0x7f80_0000 && (abs_bits & 0x0040_0000) == 0
}

/// Returns whether `value` round-trips bit-for-bit through `f32` without producing a signaling NaN.
///
/// Using `.to_bits()` preserves the sign of zero (`-0.0` vs `+0.0`) and canonical quiet NaNs while
/// rejecting signaling NaNs and quiet NaNs whose payload bits cannot be represented in `f32`.
#[inline]
pub fn f64_fits_in_f32(value: f64) -> bool {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "round-trip bit comparison immediately verifies that narrowing to f32 was lossless"
    )]
    let narrowed = value as f32;
    f64::from(narrowed).to_bits() == value.to_bits() && !is_f32_snan(narrowed)
}

/// Returns whether `value` round-trips bit-for-bit through `f16` without producing a signaling NaN.
#[inline]
pub fn f32_fits_in_f16(value: f32) -> bool {
    let narrowed = f16::from_f32(value);
    narrowed.to_f32().to_bits() == value.to_bits() && !is_f16_snan(narrowed)
}

/// Returns whether `value` round-trips bit-for-bit through `f16` without producing a signaling NaN.
#[inline]
pub fn f64_fits_in_f16(value: f64) -> bool {
    let narrowed = f16::from_f64(value);
    narrowed.to_f64().to_bits() == value.to_bits() && !is_f16_snan(narrowed)
}

/// Returns whether every element in `values` fits bit-exactly in `f32`.
pub fn f64_slice_fits_in_f32(values: &[f64]) -> bool {
    slice_fits_chunked(values, f64_fits_in_f32)
}

/// Returns whether every element in `values` fits bit-exactly in `f16`.
pub fn f32_slice_fits_in_f16(values: &[f32]) -> bool {
    slice_fits_chunked(values, f32_fits_in_f16)
}

/// Returns whether every element in `values` fits bit-exactly in `f16`.
pub fn f64_slice_fits_in_f16(values: &[f64]) -> bool {
    slice_fits_chunked(values, f64_fits_in_f16)
}

#[inline]
fn slice_fits_chunked<T: Copy>(values: &[T], fits: impl Fn(T) -> bool) -> bool {
    for chunk in values.chunks(FIT_CHECK_CHUNK_LEN) {
        let mut all_fit = true;
        for &value in chunk {
            all_fit &= fits(value);
        }
        if !all_fit {
            return false;
        }
    }
    true
}

fn all_valid_slices_match<T>(slice: &[T], mask: &Mask, check: impl Fn(&[T]) -> bool) -> bool {
    match mask {
        Mask::AllTrue(_) => check(slice),
        Mask::AllFalse(_) => true,
        Mask::Values(values) => values
            .slices()
            .iter()
            .all(|&(start, end)| check(&slice[start..end])),
    }
}

/// Returns whether every valid element of floating-point `array` fits bit-exactly in the strictly
/// narrower floating-point type `target`.
///
/// `mask` **must** be the validity mask of `array`.
pub fn float_array_fits_in_ptype(
    array: &PrimitiveArray,
    target: PType,
    mask: &Mask,
) -> VortexResult<bool> {
    match (array.ptype(), target) {
        (PType::F64, PType::F32) => Ok(all_valid_slices_match(
            array.as_slice::<f64>(),
            mask,
            f64_slice_fits_in_f32,
        )),
        (PType::F64, PType::F16) => Ok(all_valid_slices_match(
            array.as_slice::<f64>(),
            mask,
            f64_slice_fits_in_f16,
        )),
        (PType::F32, PType::F16) => Ok(all_valid_slices_match(
            array.as_slice::<f32>(),
            mask,
            f32_slice_fits_in_f16,
        )),
        (src, dst) => {
            vortex_bail!("expected a strictly narrower float target than {src}, got {dst}")
        }
    }
}

/// Narrows floating-point `array` to the strictly narrower floating-point type `target`,
/// preserving its validity and writing zero into null slots.
///
/// Null slots are zeroed so arbitrary null payloads cannot introduce signaling NaNs into the
/// narrowed child. `mask` **must** be the validity mask of `array`.
pub fn narrow_float_values(
    array: &PrimitiveArray,
    target: PType,
    mask: &Mask,
) -> VortexResult<PrimitiveArray> {
    let validity = array.validity()?;
    match (array.ptype(), target) {
        (PType::F64, PType::F32) => {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "caller verified bit-exact f32 representation on valid lanes; null lanes are zeroed"
            )]
            let buffer = narrow_slice_with_mask(array.as_slice::<f64>(), mask, |v| v as f32);
            Ok(PrimitiveArray::new(buffer, validity))
        }
        (PType::F64, PType::F16) => {
            let buffer = narrow_slice_with_mask(array.as_slice::<f64>(), mask, f16::from_f64);
            Ok(PrimitiveArray::new(buffer, validity))
        }
        (PType::F32, PType::F16) => {
            let buffer = narrow_slice_with_mask(array.as_slice::<f32>(), mask, f16::from_f32);
            Ok(PrimitiveArray::new(buffer, validity))
        }
        (src, dst) => {
            vortex_bail!("expected a strictly narrower float target than {src}, got {dst}")
        }
    }
}

fn narrow_slice_with_mask<W: Copy, N: Copy>(
    slice: &[W],
    mask: &Mask,
    narrow: impl Fn(W) -> N,
) -> Buffer<N> {
    let len = slice.len();
    match mask {
        Mask::AllTrue(_) => {
            let mut out = BufferMut::<N>::with_capacity(len);
            for &value in slice {
                out.push(narrow(value));
            }
            out.freeze()
        }
        Mask::AllFalse(_) => Buffer::<N>::zeroed(len),
        Mask::Values(values) => {
            let mut out = BufferMut::<N>::zeroed(len);
            for &(start, end) in values.slices() {
                for (dst, &src) in out[start..end].iter_mut().zip(&slice[start..end]) {
                    *dst = narrow(src);
                }
            }
            out.freeze()
        }
    }
}

/// Returns whether the non-null floating-point `value` fits bit-exactly in `target`, without
/// leaving a signaling NaN.
pub(in crate::arrays::widen) fn float_value_fits_in_ptype(
    value: PValue,
    target: PType,
) -> Option<bool> {
    match (value, target) {
        (PValue::F64(v), PType::F32) => Some(f64_fits_in_f32(v)),
        (PValue::F64(v), PType::F16) => Some(f64_fits_in_f16(v)),
        (PValue::F32(v), PType::F16) => Some(f32_fits_in_f16(v)),
        (PValue::F16(v), PType::F16 | PType::F32 | PType::F64) => Some(!is_f16_snan(v)),
        (PValue::F32(v), PType::F32 | PType::F64) => Some(!is_f32_snan(v)),
        (PValue::F64(_), PType::F64) => Some(true),
        _ => None,
    }
}

/// Returns the smallest floating-point type that holds `value` bit-exactly without leaving a
/// signaling NaN in a narrow child.
pub(in crate::arrays::widen) fn smallest_float_ptype_for_value(value: PValue) -> Option<PType> {
    match value {
        PValue::F16(_) => Some(PType::F16),
        PValue::F32(v) => {
            if f32_fits_in_f16(v) {
                Some(PType::F16)
            } else {
                Some(PType::F32)
            }
        }
        PValue::F64(v) => {
            if f64_fits_in_f16(v) {
                Some(PType::F16)
            } else if f64_fits_in_f32(v) {
                Some(PType::F32)
            } else {
                Some(PType::F64)
            }
        }
        _ => None,
    }
}

impl WidenArray {
    /// Stores `array` as a `Widen` over the narrowest floating-point type (`f16` or `f32`) that
    /// reproduces every valid value bit-for-bit on widening back, or returns `array` unchanged
    /// when no narrower floating-point type holds them.
    ///
    /// Bit-exact checking preserves `-0.0` vs `+0.0` and canonical quiet NaNs while rejecting
    /// signaling NaNs and non-representable NaN payloads. Null rows do not count, so their
    /// payloads may hold arbitrary values and are zeroed in the narrowed child. An empty or
    /// all-null `f32` or `f64` array narrows to `f16`. Fails when `array` is not a floating-point
    /// array.
    pub fn encode_float(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ptype = array.ptype();
        vortex_ensure!(
            ptype.is_float(),
            "Widen float encoding requires floating-point values, got {ptype}"
        );
        if ptype == PType::F16 {
            return Ok(array.into_array());
        }

        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        let target = match ptype {
            PType::F64 => {
                // Every `f16` value is exact in `f32`, so a fast `f32` scan rejects general `f64`
                // columns before attempting the narrower `f16` check.
                if !float_array_fits_in_ptype(&array, PType::F32, &mask)? {
                    return Ok(array.into_array());
                }
                if float_array_fits_in_ptype(&array, PType::F16, &mask)? {
                    PType::F16
                } else {
                    PType::F32
                }
            }
            PType::F32 => {
                if float_array_fits_in_ptype(&array, PType::F16, &mask)? {
                    PType::F16
                } else {
                    return Ok(array.into_array());
                }
            }
            _ => unreachable!("guarded by ptype.is_float() and ptype != PType::F16"),
        };

        let dtype = array.dtype().clone();
        let narrowed = narrow_float_values(&array, target, &mask)?;
        wrap_values(narrowed.into_array(), dtype)
    }
}
