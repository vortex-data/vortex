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
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use self::records::take_byte_records;
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

impl_fixed_width_take_value!(u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64,);

// SAFETY: Byte arrays have no padding and every byte is initialized.
unsafe impl<const N: usize> FixedWidthTakeValue for [u8; N] {}

pub(crate) fn take_values<T: FixedWidthTakeValue, I: UnsignedPType>(
    values: &[T],
    indices: &[I],
) -> Buffer<T> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if *HAS_AVX2 {
        // SAFETY: AVX2 was detected above and `FixedWidthTakeValue` guarantees an initialized byte
        // representation. The AVX2 dispatcher retains Primitive's existing scalar fallbacks and
        // out-of-bounds behavior for every value width.
        return unsafe { avx2::take_avx2(values, indices) };
    }

    take_values_scalar(values, indices)
}

pub(crate) fn take<V: FixedWidthArray>(
    array: ArrayView<'_, V>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
        && let Some(taken) = take_contiguous_ranges(array, piecewise_indices, indices, ctx)?
    {
        return Ok(Some(taken));
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

    let source = V::values(array);
    let values = match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
        take_byte_records(
            &source,
            V::byte_width(array),
            array.len(),
            indices.as_slice::<I>(),
        )
    })?;
    Ok(Some(
        with_values(array, values, indices.len(), validity)?.into_array(),
    ))
}

fn take_contiguous_ranges<V: FixedWidthArray>(
    array: ArrayView<'_, V>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = maybe_contiguous_slices(indices, ctx)? else {
        return Ok(None);
    };

    let values = V::values(array);
    let byte_width = V::byte_width(array);
    let output_len = indices_ref.len();
    let taken = match lengths {
        Columnar::Constant(lengths) => {
            let length = constant_unsigned_usize(&lengths);
            match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
                take_slices_constant_length(
                    &values,
                    byte_width,
                    array.len(),
                    starts.as_slice::<S>(),
                    length,
                    output_len,
                )
            })
        }
        Columnar::Canonical(lengths) => {
            let lengths = lengths.into_primitive();
            match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
                match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                    take_slices(
                        &values,
                        byte_width,
                        array.len(),
                        starts.as_slice::<S>(),
                        lengths.as_slice::<L>(),
                        output_len,
                    )
                })
            })
        }
    }?;
    let validity = array.validity()?.take(indices_ref)?;
    Ok(Some(
        with_values(array, taken, output_len, validity)?.into_array(),
    ))
}
