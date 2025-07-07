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
    use vortex_buffer::buffer;
    use vortex_dtype::{DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::IntoArray;
    use crate::arrays::{DecimalArray, DecimalVTable, PrimitiveArray};
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

        assert!(taken.scalar_at(0).unwrap().is_null());
        assert_eq!(
            taken.scalar_at(1).unwrap(),
            Scalar::decimal(
                DecimalValue::I128(12i128),
                array.decimal_dtype(),
                Nullability::Nullable
            )
        );

        assert_eq!(
            taken.scalar_at(2).unwrap(),
            Scalar::decimal(
                DecimalValue::I128(13i128),
                array.decimal_dtype(),
                Nullability::Nullable
            )
        );
    }
}
