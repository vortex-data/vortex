// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-exact floating-point narrowing for [`WidenArray`].
//!
//! A floating-point value fits in a narrower IEEE-754 type iff narrowing and widening back
//! reproduces the original bit pattern, and any NaN is the positive canonical quiet NaN.
//! Comparing bit patterns preserves `-0.0` vs `+0.0`, while restricting NaNs to the canonical
//! quiet NaN keeps narrowing portable across architectures whose float conversions do not
//! preserve NaN sign or payload bits.

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

/// Returns whether `value` round-trips bit-for-bit through `f32`, accepting only the positive
/// canonical quiet NaN (`f64::NAN`) among NaN bit patterns.
#[inline]
pub(in crate::arrays::widen) fn f64_fits_in_f32(value: f64) -> bool {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "round-trip bit comparison immediately verifies that narrowing to f32 was lossless"
    )]
    let narrowed = value as f32;
    f64::from(narrowed).to_bits() == value.to_bits()
        && (!value.is_nan() || value.to_bits() == f64::NAN.to_bits())
}

/// Returns whether `value` round-trips bit-for-bit through `f16`, accepting only the positive
/// canonical quiet NaN (`f32::NAN`) among NaN bit patterns.
#[inline]
pub(in crate::arrays::widen) fn f32_fits_in_f16(value: f32) -> bool {
    let narrowed = f16::from_f32(value);
    narrowed.to_f32().to_bits() == value.to_bits()
        && (!value.is_nan() || value.to_bits() == f32::NAN.to_bits())
}

/// Returns whether `value` round-trips bit-for-bit through `f16`, accepting only the positive
/// canonical quiet NaN (`f64::NAN`) among NaN bit patterns.
#[inline]
pub(in crate::arrays::widen) fn f64_fits_in_f16(value: f64) -> bool {
    let narrowed = f16::from_f64(value);
    narrowed.to_f64().to_bits() == value.to_bits()
        && (!value.is_nan() || value.to_bits() == f64::NAN.to_bits())
}

fn f64_slice_fits_in_f32(values: &[f64]) -> bool {
    slice_fits_chunked(values, f64_fits_in_f32)
}

fn f32_slice_fits_in_f16(values: &[f32]) -> bool {
    slice_fits_chunked(values, f32_fits_in_f16)
}

fn f64_slice_fits_in_f16(values: &[f64]) -> bool {
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

/// Returns the narrowest floating-point [`PType`] strictly narrower than `array`'s type that
/// reproduces every valid element bit-for-bit on widening back, or `None` when no narrower float
/// type fits (including when `array` is already `f16`, not a float, or not host-resident).
///
/// `mask` **must** be the validity mask of `array`.
pub fn narrowest_float_ptype(array: &PrimitiveArray, mask: &Mask) -> Option<PType> {
    if !array.buffer_handle().is_on_host() {
        return None;
    }
    match array.ptype() {
        PType::F64 => {
            let slice = array.as_slice::<f64>();
            // Every `f16` value is exact in `f32`, so a fast `f32` scan rejects general `f64`
            // columns before attempting the narrower `f16` check.
            if !all_valid_slices_match(slice, mask, f64_slice_fits_in_f32) {
                return None;
            }
            if all_valid_slices_match(slice, mask, f64_slice_fits_in_f16) {
                Some(PType::F16)
            } else {
                Some(PType::F32)
            }
        }
        PType::F32 => all_valid_slices_match(array.as_slice::<f32>(), mask, f32_slice_fits_in_f16)
            .then_some(PType::F16),
        _ => None,
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
    vortex_ensure!(
        array.buffer_handle().is_on_host(),
        "Widen float narrowing requires host-resident values"
    );
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

/// Returns the smallest floating-point type that holds `value` bit-exactly, accepting only the
/// positive canonical quiet NaN among NaN bit patterns.
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
    /// Bit-exact checking preserves `-0.0` vs `+0.0` and the positive canonical quiet NaN while
    /// rejecting signaling NaNs, negative NaNs, and non-canonical NaN payloads. Null rows do not
    /// count, so their payloads may hold arbitrary values and are zeroed in the narrowed child.
    /// An empty or all-null `f32` or `f64` array narrows to `f16`. Fails when `array` is not a
    /// floating-point array.
    pub fn encode_float(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ptype = array.ptype();
        vortex_ensure!(
            ptype.is_float(),
            "Widen float encoding requires floating-point values, got {ptype}"
        );

        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        let Some(target) = narrowest_float_ptype(&array, &mask) else {
            return Ok(array.into_array());
        };

        let dtype = array.dtype().clone();
        let narrowed = narrow_float_values(&array, target, &mask)?;
        wrap_values(narrowed.into_array(), dtype)
    }
}
