use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexResult, vortex_err};

use crate::arrays::{DecimalArray, DecimalEncoding, NativeDecimalType, PrimitiveArray};
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, match_each_decimal_value_type, register_kernel};

impl TakeKernel for DecimalEncoding {
    fn take(&self, array: &DecimalArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices
            .as_any()
            .downcast_ref::<PrimitiveArray>()
            .ok_or_else(|| vortex_err!("indices must be a PrimitiveArray"))?;

        let decimal = match_each_decimal_value_type!(array.values_type(), |$D| {
                match_each_integer_ptype!(indices.ptype(), |$I| {
                    let buffer = take_to_buffer::<$I, $D>(indices.as_slice::<$I>(), array.buffer::<$D>().as_slice());
                    DecimalArray::new(buffer, array.decimal_dtype(), array.validity().clone())
                })
        });

        Ok(decimal.to_array())
    }
}

register_kernel!(TakeKernelAdapter(DecimalEncoding).lift());

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
    use vortex_dtype::DecimalDType;

    use crate::arrays::DecimalArray;
    use crate::compute::take;
    use crate::validity::Validity;
    use crate::{Array, IntoArray};

    #[test]
    fn test_take() {
        let array = DecimalArray::new(
            buffer![10i128, 11i128, 12i128, 13i128],
            DecimalDType::new(19, 1),
            Validity::NonNullable,
        );

        let indices = buffer![0, 2, 3].into_array();
        let taken = take(&array, indices.as_ref()).unwrap();
        let taken_decimals = taken.as_any().downcast_ref::<DecimalArray>().unwrap();
        assert_eq!(
            taken_decimals.buffer::<i128>(),
            buffer![10i128, 12i128, 13i128]
        );
        assert_eq!(taken_decimals.decimal_dtype(), DecimalDType::new(19, 1));
    }
}
