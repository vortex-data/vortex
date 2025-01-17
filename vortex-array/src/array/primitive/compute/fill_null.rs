use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, Nullability};
use vortex_error::{vortex_err, VortexExpect, VortexResult};

use crate::array::primitive::PrimitiveArray;
use crate::array::{ConstantArray, PrimitiveEncoding};
use crate::compute::FillNullFn;
use crate::validity::{ArrayValidity, Validity};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, ToArrayData};

impl FillNullFn<PrimitiveArray> for PrimitiveEncoding {
    fn fill_null(
        &self,
        array: &PrimitiveArray,
        fill_value: vortex_scalar::Scalar,
    ) -> VortexResult<ArrayData> {
        let result_validity = match fill_value.dtype().nullability() {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };

        if array.dtype().nullability() == Nullability::NonNullable
            && fill_value.dtype().nullability() == Nullability::NonNullable
        {
            return Ok(array.to_array());
        }

        let array_validity = array.logical_validity();
        if array_validity.all_valid() {
            return Ok(PrimitiveArray::from_byte_buffer(
                array.byte_buffer().clone(),
                array.ptype(),
                result_validity,
            )
            .into_array());
        }

        if array_validity.all_invalid() {
            return Ok(ConstantArray::new(fill_value, array.len()).into_array());
        }

        let nulls = array_validity
            .to_null_buffer()?
            .ok_or_else(|| vortex_err!("Failed to convert array validity to null buffer"))?;

        // TODO(danking): when we take PrimitiveArray by value, we should mutate in-place
        match_each_native_ptype!(array.ptype(), |$T| {
            let as_slice = array.as_slice::<$T>();
            let fill_value = fill_value
                .as_primitive()
                .typed_value::<$T>()
                .vortex_expect("top-level fill_null ensure non-null fill value");
            let filled = Buffer::from_iter(
                as_slice
                    .iter()
                    .zip(nulls.into_iter())
                    .map(|(value, valid)| {
                        if valid {
                            *value
                        } else {
                            fill_value
                        }
                    })
            );
            Ok(PrimitiveArray::new(filled, result_validity).into_array())
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
