// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_scalar::{NativeDecimalType, match_each_decimal_value_type};

use crate::arrays::{DecimalArray, DecimalVTable};
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, ToCanonical, register_kernel};

impl TakeKernel for DecimalVTable {
    fn take(&self, array: &DecimalArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;
        let validity = array.validity().take(indices.as_ref())?;

        // TODO(joe): if the true count of take indices validity is low, only take array values with
        // valid indices.
        let decimal = match_each_decimal_value_type!(array.values_type(), |D| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let buffer =
                    take_to_buffer::<I, D>(indices.as_slice::<I>(), array.buffer::<D>().as_slice());
                DecimalArray::new(buffer, array.decimal_dtype(), validity)
            })
        });

        Ok(decimal.to_array())
    }
}

register_kernel!(TakeKernelAdapter(DecimalVTable).lift());

#[inline]
fn take_to_buffer<I: NativePType + AsPrimitive<usize>, T: NativeDecimalType>(
    indices: &[I],
    values: &[T],
) -> Buffer<T> {
    indices.iter().map(|idx| values[idx.as_()]).collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::{DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::IntoArray;
    use crate::arrays::{DecimalArray, DecimalVTable, PrimitiveArray};
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;
    use crate::validity::Validity;

    #[test]
    fn test_take() {
        let array = DecimalArray::new(
            buffer![10i128, 11i128, 12i128, 13i128],
            DecimalDType::new(19, 1),
            Validity::NonNullable,
        );

        let indices = buffer![0, 2, 3].into_array();
        let taken = take(array.as_ref(), indices.as_ref()).unwrap();
        let taken_decimals = taken.as_::<DecimalVTable>();
        assert_eq!(
            taken_decimals.buffer::<i128>(),
            buffer![10i128, 12i128, 13i128]
        );
        assert_eq!(taken_decimals.decimal_dtype(), DecimalDType::new(19, 1));
    }

    #[test]
    fn test_take_null_indices() {
        let array = DecimalArray::new(
            buffer![i128::MAX, 11i128, 12i128, 13i128],
            DecimalDType::new(19, 1),
            Validity::NonNullable,
        );

        let indices = PrimitiveArray::from_option_iter([None, Some(2), Some(3)]).into_array();
        let taken = take(array.as_ref(), indices.as_ref()).unwrap();

        assert!(taken.scalar_at(0).is_null());
        assert_eq!(
            taken.scalar_at(1),
            Scalar::decimal(
                DecimalValue::I128(12i128),
                array.decimal_dtype(),
                Nullability::Nullable
            )
        );

        assert_eq!(
            taken.scalar_at(2),
            Scalar::decimal(
                DecimalValue::I128(13i128),
                array.decimal_dtype(),
                Nullability::Nullable
            )
        );
    }

    #[rstest]
    #[case(DecimalArray::new(
        buffer![100i128, 200i128, 300i128, 400i128, 500i128],
        DecimalDType::new(19, 2),
        Validity::NonNullable,
    ))]
    #[case(DecimalArray::new(
        buffer![10i64, 20i64, 30i64, 40i64, 50i64],
        DecimalDType::new(10, 1),
        Validity::NonNullable,
    ))]
    #[case(DecimalArray::new(
        buffer![1i32, 2i32, 3i32, 4i32, 5i32],
        DecimalDType::new(5, 0),
        Validity::NonNullable,
    ))]
    #[case(DecimalArray::new(
        buffer![1000i128, 2000i128, 3000i128, 4000i128, 5000i128],
        DecimalDType::new(19, 3),
        Validity::from_iter([true, false, true, true, false]),
    ))]
    #[case(DecimalArray::new(
        buffer![42i128],
        DecimalDType::new(19, 0),
        Validity::NonNullable,
    ))]
    #[case({
        let values: Vec<i128> = (0..100).map(|i| i * 1000).collect();
        DecimalArray::new(
            Buffer::from_iter(values),
            DecimalDType::new(19, 4),
            Validity::NonNullable,
        )
    })]
    fn test_take_decimal_conformance(#[case] array: DecimalArray) {
        test_take_conformance(array.as_ref());
    }
}
