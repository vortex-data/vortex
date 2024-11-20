use vortex_dtype::{match_each_native_ptype, Nullability};
use vortex_error::{vortex_err, VortexResult};

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::unary::FillForwardFn;
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
            return Ok(PrimitiveArray::new(
                array.buffer().clone(),
                array.ptype(),
                Validity::AllValid,
            )
            .into_array());
        }

        match_each_native_ptype!(array.ptype(), |$T| {
            if validity.all_invalid() {
                return Ok(PrimitiveArray::from_vec(vec![$T::default(); array.len()], Validity::AllValid).into_array());
            }

            let nulls = validity.to_null_buffer()?.ok_or_else(|| vortex_err!("Failed to convert array validity to null buffer"))?;
            let maybe_null_slice = array.maybe_null_slice::<$T>();
            let mut last_value = $T::default();
            let filled = maybe_null_slice
                .iter()
                .zip(nulls.into_iter())
                .map(|(v, valid)| {
                    if valid {
                        last_value = *v;
                    }
                    last_value
                })
                .collect::<Vec<_>>();
            Ok(PrimitiveArray::from_vec(filled, Validity::AllValid).into_array())
        })
    }
}

#[cfg(test)]
mod test {
    use crate::array::primitive::PrimitiveArray;
    use crate::array::BoolArray;
    use crate::compute::unary::fill_forward;
    use crate::validity::{ArrayValidity, Validity};
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn leading_none() {
        let arr = PrimitiveArray::from_nullable_vec(vec![None, Some(8u8), None, Some(10), None])
            .into_array();
        let p = fill_forward(&arr).unwrap().into_primitive().unwrap();
        assert_eq!(p.maybe_null_slice::<u8>(), vec![0, 8, 8, 10, 10]);
        assert!(p.logical_validity().all_valid());
    }

    #[test]
    fn all_none() {
        let arr =
            PrimitiveArray::from_nullable_vec(vec![Option::<u8>::None, None, None, None, None])
                .into_array();

        let p = fill_forward(&arr).unwrap().into_primitive().unwrap();
        assert_eq!(p.maybe_null_slice::<u8>(), vec![0, 0, 0, 0, 0]);
        assert!(p.logical_validity().all_valid());
    }

    #[test]
    fn nullable_non_null() {
        let arr = PrimitiveArray::from_vec(
            vec![8u8, 10u8, 12u8, 14u8, 16u8],
            Validity::Array(BoolArray::from_iter([true, true, true, true, true]).into_array()),
        )
        .into_array();
        let p = fill_forward(&arr).unwrap().into_primitive().unwrap();
        assert_eq!(p.maybe_null_slice::<u8>(), vec![8, 10, 12, 14, 16]);
        assert!(p.logical_validity().all_valid());
    }
}
