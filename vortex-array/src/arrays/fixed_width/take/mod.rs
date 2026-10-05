// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2;
mod records;
mod scalar;
mod slices;
#[cfg(test)]
mod tests;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use std::sync::LazyLock;

use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use self::records::take_records;
use self::scalar::take_values_scalar;
use self::slices::take_slices;
use self::slices::take_slices_constant_length;
use super::FixedWidthArray;
use super::with_values;
use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::piecewise_sequence::constant_unsigned_usize;
use crate::arrays::piecewise_sequence::maybe_contiguous_slices;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::UnsignedPType;
use crate::dtype::half::f16;
use crate::dtype::i256;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
static HAS_AVX2: LazyLock<bool> = LazyLock::new(|| is_x86_feature_detected!("avx2"));

impl<V: FixedWidthArray> TakeExecute for V {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take(array, indices, ctx)
    }
}

/// A fixed-width value whose initialized bytes may be moved through integer SIMD lanes.
///
/// # Safety
///
/// Implementors must have no uninitialized bytes. The shared AVX2 gather reads the complete
/// representation through a same-width integer lane before writing those bytes back unchanged.
pub(crate) unsafe trait FixedWidthTakeValue: Copy {}

macro_rules! impl_fixed_width_take_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            // SAFETY: These scalar representations contain no padding or uninitialized bytes.
            unsafe impl FixedWidthTakeValue for $ty {}
        )+
    };
}

impl_fixed_width_take_value!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i256, f16, f32, f64,
);

// SAFETY: Byte arrays have no padding and every byte is initialized.
unsafe impl<const N: usize> FixedWidthTakeValue for [u8; N] {}

pub(crate) fn take_values<T: FixedWidthTakeValue, I: UnsignedPType>(
    values: &[T],
    indices: &[I],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if *HAS_AVX2 {
        // SAFETY: AVX2 was detected above and `FixedWidthTakeValue` guarantees an initialized byte
        // representation. The AVX2 dispatcher retains Primitive's existing scalar fallbacks and
        // out-of-bounds behavior for every value width.
        return unsafe { avx2::take_avx2(values, indices, allocator) };
    }

    take_values_scalar(values, indices, allocator)
}

pub(crate) fn take<V: FixedWidthArray>(
    array: ArrayView<'_, V>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>() {
        let taken = match V::byte_width(array) {
            1 => take_contiguous_ranges::<V, u8>(array, piecewise_indices, indices, ctx)?,
            2 => take_contiguous_ranges::<V, u16>(array, piecewise_indices, indices, ctx)?,
            4 => take_contiguous_ranges::<V, u32>(array, piecewise_indices, indices, ctx)?,
            8 => take_contiguous_ranges::<V, u64>(array, piecewise_indices, indices, ctx)?,
            16 => take_contiguous_ranges::<V, u128>(array, piecewise_indices, indices, ctx)?,
            32 => take_contiguous_ranges::<V, i256>(array, piecewise_indices, indices, ctx)?,
            _ => None,
        };
        if taken.is_some() {
            return Ok(taken);
        }
    }

    let DType::Primitive(ptype, nullability) = indices.dtype() else {
        vortex_bail!("Invalid indices dtype: {}", indices.dtype())
    };
    if !ptype.is_int() {
        vortex_bail!("Invalid indices dtype: {}", indices.dtype())
    }

    let indices_validity = indices.validity()?;
    let indices_nulls_zeroed = match indices_validity.execute_mask(indices.len(), ctx)? {
        Mask::AllTrue(_) => indices.clone(),
        Mask::AllFalse(_) => {
            return Ok(Some(
                ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                    .into_array(),
            ));
        }
        Mask::Values(_) => indices
            .clone()
            .fill_null(Scalar::from(0).cast(indices.dtype())?)?,
    };

    let indices = if ptype.is_unsigned_int() {
        indices_nulls_zeroed.execute::<PrimitiveArray>(ctx)?
    } else {
        indices_nulls_zeroed
            .cast(DType::Primitive(ptype.to_unsigned(), *nullability))?
            .execute::<PrimitiveArray>(ctx)?
    };
    let validity = array
        .validity()?
        .take(&indices.clone().into_array())?
        .and(indices_validity)?;

    let values = match V::byte_width(array) {
        1 => take_records::<V, u8>(array, &indices, ctx.allocator())?,
        2 => take_records::<V, u16>(array, &indices, ctx.allocator())?,
        4 => take_records::<V, u32>(array, &indices, ctx.allocator())?,
        8 => take_records::<V, u64>(array, &indices, ctx.allocator())?,
        16 => take_records::<V, u128>(array, &indices, ctx.allocator())?,
        32 => take_records::<V, i256>(array, &indices, ctx.allocator())?,
        _ => return Ok(None),
    };
    Ok(Some(
        with_values(array, values, indices.len(), validity)?.into_array(),
    ))
}

// Avoid duplicating the starts/lengths dispatch in every record-width arm of `take`.
#[inline(never)]
fn take_contiguous_ranges<V: FixedWidthArray, T: Copy>(
    array: ArrayView<'_, V>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = maybe_contiguous_slices(indices, ctx)? else {
        return Ok(None);
    };

    let values = V::values::<T>(array);
    vortex_ensure!(
        values.len() == array.len(),
        "Fixed-width values buffer length does not match record count"
    );
    let output_len = indices_ref.len();
    let taken = match lengths {
        Columnar::Constant(lengths) => {
            let length = constant_unsigned_usize(&lengths);
            match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
                take_slices_constant_length(
                    &values,
                    starts.as_slice::<S>(),
                    length,
                    output_len,
                    ctx.allocator(),
                )
            })
        }
        Columnar::Canonical(lengths) => {
            let lengths = lengths.into_primitive();
            match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
                match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                    take_slices(
                        &values,
                        starts.as_slice::<S>(),
                        lengths.as_slice::<L>(),
                        output_len,
                        ctx.allocator(),
                    )
                })
            })
        }
    }?;
    let validity = array.validity()?.take(indices_ref)?;
    Ok(Some(
        with_values(array, taken.into_byte_buffer(), output_len, validity)?.into_array(),
    ))
}
