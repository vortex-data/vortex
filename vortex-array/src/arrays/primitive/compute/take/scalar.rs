// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::IntegerPType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_integer_ptype;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

#[allow(unused)]
pub(super) struct TakeKernelScalar;

impl TakeImpl for TakeKernelScalar {
    #[allow(clippy::cognitive_complexity)]
    fn take(
        &self,
        array: &PrimitiveArray,
        indices: &PrimitiveArray,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        match_each_native_ptype!(array.ptype(), |T| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let indices_slice = indices.as_slice::<I>();
                let indices_validity = indices.validity();
                let values = if indices_validity.all_valid(indices_slice.len()) {
                    // Fast path: indices have no nulls, safe to index directly
                    take_primitive_scalar(array.as_slice::<T>(), indices_slice)
                } else {
                    // Slow path: indices may have nulls with garbage values
                    take_primitive_scalar_with_nulls(
                        array.as_slice::<T>(),
                        indices_slice,
                        indices_validity,
                    )
                };
                Ok(PrimitiveArray::new(values, validity).into_array())
            })
        })
    }
}

// Compiler may see this as unused based on enabled features
#[allow(unused)]
#[inline(always)]
pub(super) fn take_primitive_scalar<T: NativePType, I: IntegerPType>(
    array: &[T],
    indices: &[I],
) -> Buffer<T> {
    indices.iter().map(|idx| array[idx.as_()]).collect()
}

/// Slow path for take when indices may contain nulls with garbage values.
/// Uses 0 as a safe index for null positions (the value will be masked out by validity).
#[allow(unused)]
#[inline(always)]
fn take_primitive_scalar_with_nulls<T: NativePType, I: IntegerPType>(
    array: &[T],
    indices: &[I],
    validity: &Validity,
) -> Buffer<T> {
    indices
        .iter()
        .enumerate()
        .map(|(i, idx)| {
            if validity.is_valid(i) {
                array[idx.as_()]
            } else {
                T::zero()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::primitive::compute::take::TakeImpl;
    use crate::arrays::primitive::compute::take::scalar::TakeKernelScalar;
    use crate::validity::Validity;

    #[test]
    fn test_scalar_basic() {
        let values = buffer![1, 2, 3, 4, 5].into_array().to_primitive();
        let indices = buffer![0, 1, 1, 2, 2, 3, 4].into_array().to_primitive();

        let result = TakeKernelScalar
            .take(&values, &indices, Validity::NonNullable)
            .unwrap()
            .to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[1, 2, 2, 3, 3, 4, 5]);
    }

    #[test]
    fn test_scalar_with_nulls() {
        let values = buffer![1, 2, 3, 4, 5].into_array().to_primitive();
        let validity = Validity::from_iter([true, false, true, true, true]);
        let indices = PrimitiveArray::new(buffer![0, 100, 2, 3, 4], validity.clone());

        let result = TakeKernelScalar
            .take(&values, &indices, validity.clone())
            .unwrap()
            .to_primitive();

        assert_eq!(result.as_slice::<i32>(), &[1, 0, 3, 4, 5]);
        assert_eq!(result.validity, validity);
    }
}
