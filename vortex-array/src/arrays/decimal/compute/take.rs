// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::executor::ExecutionCtx;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;

impl TakeExecute for Decimal {
    fn take(
        array: ArrayView<'_, Decimal>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let validity = array.validity().take(&indices.clone().into_array())?;

        // TODO(joe): if the true count of take indices validity is low, only take array values with
        // valid indices.
        let decimal = match_each_decimal_value_type!(array.values_type(), |D| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let buffer =
                    take_to_buffer::<I, D>(indices.as_slice::<I>(), array.buffer::<D>().as_slice());
                // SAFETY: Take operation preserves decimal dtype and creates valid buffer.
                // Validity is computed correctly from the parent array and indices.
                unsafe { DecimalArray::new_unchecked(buffer, array.decimal_dtype(), validity) }
            })
        });

        Ok(Some(decimal.into_array()))
    }
}

fn take_to_buffer<I: IntegerPType, T: NativeDecimalType>(indices: &[I], values: &[T]) -> Buffer<T> {
    indices.iter().map(|idx| values[idx.as_()]).collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DecimalDType;
    use crate::validity::Validity;

    #[test]
    fn test_take() {
        let ddtype = DecimalDType::new(19, 1);
        let array = DecimalArray::new(
            buffer![10i128, 11i128, 12i128, 13i128],
            ddtype,
            Validity::NonNullable,
        );

        let indices = buffer![0, 2, 3].into_array();
        let taken = array.take(indices).unwrap();

        let expected = DecimalArray::from_iter([10i128, 12, 13], ddtype);
        assert_arrays_eq!(expected, taken);
    }

    #[test]
    fn test_take_null_indices() {
        let ddtype = DecimalDType::new(19, 1);
        let array = DecimalArray::new(
            buffer![i128::MAX, 11i128, 12i128, 13i128],
            ddtype,
            Validity::NonNullable,
        );

        let indices = PrimitiveArray::from_option_iter([None, Some(2), Some(3)]).into_array();
        let taken = array.take(indices).unwrap();

        let expected = DecimalArray::from_option_iter([None, Some(12i128), Some(13)], ddtype);
        assert_arrays_eq!(expected, taken);
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
        test_take_conformance(&array.into_array());
    }
}
