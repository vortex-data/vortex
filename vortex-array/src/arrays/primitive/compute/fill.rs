use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, Nullability};
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_scalar::Scalar;

use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::{ConstantArray, PrimitiveEncoding};
use crate::compute::FillForwardFn;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef};

impl FillForwardFn<&PrimitiveArray> for PrimitiveEncoding {
    fn fill_forward(&self, array: &PrimitiveArray) -> VortexResult<ArrayRef> {
        if array.dtype().nullability() == Nullability::NonNullable {
            return Ok(array.to_array().into_array());
        }

        match array.validity_mask()?.boolean_buffer() {
            AllOr::All => Ok(PrimitiveArray::from_byte_buffer(
                array.byte_buffer().clone(),
                array.ptype(),
                Validity::AllValid,
            )
            .into_array()),
            AllOr::None => {
                match_each_native_ptype!(array.ptype(), |$T| {
                    let fill_value = Scalar::from($T::default()).cast(array.dtype())?;
                    return Ok(ConstantArray::new(fill_value, array.len()).into_array())
                })
            }
            AllOr::Some(validity) => {
                // TODO(ngates): when we take PrimitiveArray by value, we should mutate in-place
                match_each_native_ptype!(array.ptype(), |$T| {
                    let as_slice = array.as_slice::<$T>();
                    let mut last_value = $T::default();
                    let filled = Buffer::from_iter(
                        as_slice
                            .iter()
                            .zip(validity.into_iter())
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
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;

    use crate::array::Array;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::arrays::BoolArray;
    use crate::canonical::ToCanonical;
    use crate::compute::fill_forward;
    use crate::validity::Validity;

    #[test]
    fn leading_none() {
        let arr = PrimitiveArray::from_option_iter([None, Some(8u8), None, Some(10), None]);
        let p = fill_forward(&arr).unwrap().to_primitive().unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![0, 8, 8, 10, 10]);
        assert!(p.validity_mask().unwrap().all_true());
    }

    #[test]
    fn all_none() {
        let arr = PrimitiveArray::from_option_iter([Option::<u8>::None, None, None, None, None]);

        let p = fill_forward(&arr).unwrap().to_primitive().unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![0, 0, 0, 0, 0]);
        assert!(p.validity_mask().unwrap().all_true());
    }

    #[test]
    fn nullable_non_null() {
        let arr = PrimitiveArray::new(
            buffer![8u8, 10, 12, 14, 16],
            Validity::Array(BoolArray::from_iter([true, true, true, true, true]).into_array()),
        );
        let p = fill_forward(&arr).unwrap().to_primitive().unwrap();
        assert_eq!(p.as_slice::<u8>(), vec![8u8, 10, 12, 14, 16]);
        assert!(p.validity_mask().unwrap().all_true());
    }
}
