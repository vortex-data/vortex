use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, Nullability};
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::{ConstantArray, PrimitiveEncoding};
use crate::compute::FillForwardFn;
use crate::validity::{ArrayValidity, Validity};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, ToArrayData};

impl FillForwardFn<PrimitiveArray> for PrimitiveEncoding {
    fn fill_forward(&self, array: &PrimitiveArray) -> VortexResult<ArrayData> {
        if array.dtype().nullability() == Nullability::NonNullable {
            return Ok(array.to_array());
        }

        let validity = array.logical_validity();
        if validity.all_valid() {
            return Ok(PrimitiveArray::from_byte_buffer(
                array.byte_buffer().clone(),
                array.ptype(),
                Validity::AllValid,
            )
            .into_array());
        }

        if validity.all_invalid() {
            match_each_native_ptype!(array.ptype(), |$T| {
                let fill_value = Scalar::from($T::default()).cast(array.dtype())?;
                return Ok(ConstantArray::new(fill_value, array.len()).into_array())
            })
        }

        let nulls = validity
            .to_null_buffer()?
            .ok_or_else(|| vortex_err!("Failed to convert array validity to null buffer"))?;

        // TODO(ngates): when we take PrimitiveArray by value, we should mutate in-place
        match_each_native_ptype!(array.ptype(), |$T| {
            let as_slice = array.as_slice::<$T>();
            let mut last_value = $T::default();
            let filled = Buffer::from_iter(
                as_slice
                    .iter()
                    .zip(nulls.into_iter())
                    .map(|(v, valid)| {
                        if valid {
                            last_value = *v;
                        }
                        last_value
                    })
            );
            Ok(PrimitiveArray::new(filled, Validity::AllValid).into_array())
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;

    use crate::array::primitive::PrimitiveArray;
    use crate::array::BoolArray;
    use crate::compute::fill_forward;
    use crate::validity::{ArrayValidity, Validity};
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn leading_none() {
        let arr =
            PrimitiveArray::from_option_iter([None, Some(8u8), None, Some(10), None]).into_array();
        let p = fill_forward(&arr).unwrap().into_primitive().unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![0, 8, 8, 10, 10]);
        assert!(p.logical_validity().all_valid());
    }

    #[test]
    fn all_none() {
        let arr = PrimitiveArray::from_option_iter([Option::<u8>::None, None, None, None, None])
            .into_array();

        let p = fill_forward(&arr).unwrap().into_primitive().unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![0, 0, 0, 0, 0]);
        assert!(p.logical_validity().all_valid());
    }

    #[test]
    fn nullable_non_null() {
        let arr = PrimitiveArray::new(
            buffer![8u8, 10, 12, 14, 16],
            Validity::Array(BoolArray::from_iter([true, true, true, true, true]).into_array()),
        )
        .into_array();
        let p = fill_forward(&arr).unwrap().into_primitive().unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![8u8, 10, 12, 14, 16]);
        assert!(p.logical_validity().all_valid());
    }
}
