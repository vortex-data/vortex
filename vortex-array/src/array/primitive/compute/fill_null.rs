use std::ops::Not;

use vortex_buffer::BufferMut;
use vortex_dtype::{match_each_native_ptype, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::{ConstantArray, PrimitiveEncoding};
use crate::compute::FillNullFn;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant as _};

impl FillNullFn<PrimitiveArray> for PrimitiveEncoding {
    fn fill_null(&self, array: &PrimitiveArray, fill_value: Scalar) -> VortexResult<ArrayData> {
        let result_validity = match fill_value.dtype().nullability() {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };

        Ok(match array.validity() {
            Validity::NonNullable | Validity::AllValid => {
                match_each_native_ptype!(array.ptype(), |$T| {
                    PrimitiveArray::new::<$T>(array.buffer().clone(), result_validity).into_array()
                })
            }
            Validity::AllInvalid => ConstantArray::new(fill_value, array.len()).into_array(),
            Validity::Array(is_valid) => {
                // TODO(danking): when we take PrimitiveArray by value, we should mutate in-place
                let is_invalid = is_valid.into_bool()?.boolean_buffer().not();
                match_each_native_ptype!(array.ptype(), |$T| {
                    let mut buffer = BufferMut::copy_from(array.as_slice::<$T>());
                    let fill_value = fill_value
                        .as_primitive()
                        .typed_value::<$T>()
                        .vortex_expect("top-level fill_null ensure non-null fill value");
                    for invalid_index in is_invalid.set_indices() {
                        buffer[invalid_index] = fill_value;
                    }
                    PrimitiveArray::new(buffer, result_validity).into_array()
                })
            }
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::array::primitive::PrimitiveArray;
    use crate::array::BoolArray;
    use crate::compute::fill_null;
    use crate::validity::{ArrayValidity, Validity};
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn fill_null_leading_none() {
        let arr =
            PrimitiveArray::from_option_iter([None, Some(8u8), None, Some(10), None]).into_array();
        let p = fill_null(&arr, Scalar::from(42u8))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![42, 8, 42, 10, 42]);
        assert!(p.logical_validity().all_valid());
    }

    #[test]
    fn fill_null_all_none() {
        let arr = PrimitiveArray::from_option_iter([Option::<u8>::None, None, None, None, None])
            .into_array();

        let p = fill_null(&arr, Scalar::from(255u8))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![255, 255, 255, 255, 255]);
        assert!(p.logical_validity().all_valid());
    }

    #[test]
    fn fill_null_nullable_non_null() {
        let arr = PrimitiveArray::new(
            buffer![8u8, 10, 12, 14, 16],
            Validity::Array(BoolArray::from_iter([true, true, true, true, true]).into_array()),
        )
        .into_array();
        let p = fill_null(&arr, Scalar::from(255u8))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![8, 10, 12, 14, 16]);
        assert!(p.logical_validity().all_valid());
    }

    #[test]
    fn fill_null_non_nullable() {
        let arr = buffer![8u8, 10, 12, 14, 16].into_array();
        let p = fill_null(&arr, Scalar::from(255u8))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![8u8, 10, 12, 14, 16]);
        assert!(p.logical_validity().all_valid());
    }
}
