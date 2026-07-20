// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2;

use std::sync::LazyLock;

use itertools::Itertools as _;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Columnar;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::piecewise_sequence::SpareBufferWriter;
use crate::arrays::piecewise_sequence::constant_unsigned_usize;
use crate::arrays::piecewise_sequence::maybe_contiguous_slices;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::UnsignedPType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

// Kernel selection happens on the first call to `take` and uses a combination of compile-time
// and runtime feature detection to infer the best kernel for the platform.
static PRIMITIVE_TAKE_KERNEL: LazyLock<&'static dyn TakeImpl> = LazyLock::new(|| {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            &avx2::TakeKernelAVX2
        } else {
            &TakeKernelScalar
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        &TakeKernelScalar
    }
});

trait TakeImpl: Send + Sync {
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef>;
}

struct TakeKernelScalar;

impl TakeImpl for TakeKernelScalar {
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        match_each_native_ptype!(array.ptype(), |T| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let values = take_primitive_scalar(array.as_slice::<T>(), indices.as_slice::<I>());
                Ok(PrimitiveArray::new(values, validity).into_array())
            })
        })
    }
}

impl TakeExecute for Primitive {
    fn take(
        array: ArrayView<'_, Primitive>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
            && let Some(taken) = take_contiguous_ranges(array, piecewise_indices, indices, ctx)?
        {
            return Ok(Some(taken));
        }

        let DType::Primitive(ptype, null) = indices.dtype() else {
            vortex_bail!("Invalid indices dtype: {}", indices.dtype())
        };

        let indices_validity = indices.validity()?;
        // Null index lanes are semantically ignored, but their physical values may be out of
        // bounds. Redirect those lanes to zero for the cast/gather, then restore the original index
        // validity below.
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

        let unsigned_indices = if ptype.is_unsigned_int() {
            indices_nulls_zeroed.execute::<PrimitiveArray>(ctx)?
        } else {
            // This will fail if all values cannot be converted to unsigned
            indices_nulls_zeroed
                .cast(DType::Primitive(ptype.to_unsigned(), *null))?
                .execute::<PrimitiveArray>(ctx)?
        };

        let validity = array
            .validity()?
            .take(&unsigned_indices.clone().into_array())?
            .and(indices_validity)?;
        // Delegate to the best kernel based on the target CPU
        {
            let unsigned_indices = unsigned_indices.as_view();
            PRIMITIVE_TAKE_KERNEL
                .take(array, unsigned_indices, validity)
                .map(Some)
        }
    }
}

fn take_contiguous_ranges(
    array: ArrayView<'_, Primitive>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = maybe_contiguous_slices(indices, ctx)? else {
        return Ok(None);
    };
    let validity = array.validity()?.take(indices_ref)?;
    let output_len = indices_ref.len();
    let taken = match lengths {
        Columnar::Constant(lengths) => {
            let length = constant_unsigned_usize(&lengths);
            take_slices_constant_length(array, &starts, length, validity, output_len)?
        }
        Columnar::Canonical(lengths) => {
            let lengths = lengths.into_primitive();
            take_slices(array, &starts, &lengths, validity, output_len)?
        }
    };
    Ok(Some(taken))
}

// Compiler may see this as unused based on enabled features
#[inline(always)]
fn take_primitive_scalar<T: Copy, I: IntegerPType>(buffer: &[T], indices: &[I]) -> Buffer<T> {
    // NB: The simpler `indices.iter().map(|idx| buffer[idx.as_()]).collect()` generates suboptimal
    // assembly where the buffer length is repeatedly loaded from the stack on each iteration.

    let mut result = BufferMut::with_capacity(indices.len());
    let ptr = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    // This explicit loop with pointer writes keeps the length in a register and avoids per-element
    // capacity checks from `push()`.
    for (i, idx) in indices.iter().enumerate() {
        // SAFETY: We reserved `indices.len()` capacity, so `ptr.add(i)` is valid.
        unsafe { ptr.add(i).write(buffer[idx.as_()]) };
    }

    // SAFETY: We just wrote exactly `indices.len()` elements.
    unsafe { result.set_len(indices.len()) };
    result.freeze()
}

fn take_slices(
    array: ArrayView<'_, Primitive>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef> {
    match_each_native_ptype!(array.ptype(), |T| {
        take_slices_typed::<T>(array, starts, lengths, validity, output_len)
    })
}

fn take_slices_typed<T>(
    array: ArrayView<'_, Primitive>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
{
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        take_slices_start_typed::<T, S>(array, starts, lengths, validity, output_len)
    })
}

fn take_slices_start_typed<T, S>(
    array: ArrayView<'_, Primitive>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    S: UnsignedPType,
{
    match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
        let values = take_slices_to_buffer::<T, S, L>(
            array.as_slice::<T>(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
        )?;
        Ok(PrimitiveArray::new(values, validity).into_array())
    })
}

fn take_slices_to_buffer<T, S, L>(
    source: &[T],
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: Copy,
    S: UnsignedPType,
    L: UnsignedPType,
{
    let mut values = BufferMut::<T>::with_capacity(output_len);
    let mut writer = SpareBufferWriter::new(&mut values, output_len)?;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = start.as_();
        let length = length.as_();
        writer.copy_slice(&source[start..][..length])?;
    }
    writer.finish()?;
    Ok(values.freeze())
}

fn take_slices_constant_length(
    array: ArrayView<'_, Primitive>,
    starts: &PrimitiveArray,
    length: usize,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef> {
    match_each_native_ptype!(array.ptype(), |T| {
        take_slices_constant_length_typed::<T>(array, starts, length, validity, output_len)
    })
}

fn take_slices_constant_length_typed<T>(
    array: ArrayView<'_, Primitive>,
    starts: &PrimitiveArray,
    length: usize,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
{
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        let values = take_slices_constant_length_to_buffer::<T, S>(
            array.as_slice::<T>(),
            starts.as_slice::<S>(),
            length,
            output_len,
        )?;
        Ok(PrimitiveArray::new(values, validity).into_array())
    })
}

fn take_slices_constant_length_to_buffer<T, S>(
    source: &[T],
    starts: &[S],
    length: usize,
    output_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: Copy,
    S: UnsignedPType,
{
    let computed_len = starts
        .len()
        .checked_mul(length)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );

    let mut values = BufferMut::<T>::with_capacity(output_len);
    let mut writer = SpareBufferWriter::new(&mut values, output_len)?;
    for &start in starts {
        let start = start.as_();
        writer.copy_slice(&source[start..][..length])?;
    }
    writer.finish()?;
    Ok(values.freeze())
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::primitive::compute::take::take_primitive_scalar;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn test_take() {
        let a = vec![1i32, 2, 3, 4, 5];
        let result = take_primitive_scalar(&a, &[0, 0, 4, 2]);
        assert_eq!(result.as_slice(), &[1i32, 1, 5, 3]);
    }

    #[test]
    fn test_take_with_null_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5],
            Validity::Array(BoolArray::from_iter([true, true, false, false, true]).into_array()),
        );
        let indices = PrimitiveArray::new(
            buffer![0, 3, 4],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        assert_eq!(
            actual.execute_scalar(0, &mut ctx).vortex_expect("no fail"),
            Scalar::from(Some(1))
        );
        // position 3 is null
        assert_eq!(
            actual.execute_scalar(1, &mut ctx).vortex_expect("no fail"),
            Scalar::null_native::<i32>()
        );
        // the third index is null
        assert_eq!(
            actual.execute_scalar(2, &mut ctx).vortex_expect("no fail"),
            Scalar::null_native::<i32>()
        );
    }

    #[rstest]
    #[case(PrimitiveArray::new(buffer![42i32], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4, 5, 6, 7], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid))]
    #[case(PrimitiveArray::new(
        buffer![0, 1, 2, 3, 4, 5],
        Validity::Array(BoolArray::from_iter([true, false, true, false, true, true]).into_array()),
    ))]
    #[case(PrimitiveArray::from_option_iter([Some(1), None, Some(3), Some(4), None]))]
    fn test_take_primitive_conformance(#[case] array: PrimitiveArray) {
        test_take_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::validity::Validity;

    #[test]
    fn take_null_index_skips_out_of_bounds_value() {
        let mut ctx = array_session().create_execution_ctx();
        let values = PrimitiveArray::from_iter([10i32, 20, 30]);
        let indices = PrimitiveArray::new(
            buffer![1u64, 3],
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        );

        let taken = values.take(indices.into_array()).unwrap();

        assert_arrays_eq!(
            taken,
            PrimitiveArray::from_option_iter([Some(20i32), None]).into_array(),
            &mut ctx
        );
    }
}
