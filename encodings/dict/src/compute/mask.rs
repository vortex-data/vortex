use vortex_array::array::PrimitiveArray;
use vortex_array::compute::{add_scalar, FilterIter, FilterMask, MaskFn};
use vortex_array::variants::PrimitiveArrayTrait as _;
use vortex_array::{ArrayDType as _, ArrayData, IntoArrayData, IntoArrayVariant as _};
use vortex_buffer::BufferMut;
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{dict_values_validity, DictArray, DictEncoding};

impl MaskFn<DictArray> for DictEncoding {
    fn mask(&self, array: &DictArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let (codes, new_values) = if array.dtype().is_nullable() {
            (array.codes().into_primitive()?, array.values().clone())
        } else {
            let values = array.values().into_primitive()?;
            let values_with_null =
                match_each_integer_ptype!(values.ptype(), |$T| add_null_value::<$T>(values))?;
            let codes_with_null = add_scalar(
                array.codes(),
                Scalar::from(1u8).cast(array.codes().dtype())?,
            )?
            .into_primitive()?;
            (codes_with_null, values_with_null)
        };

        let new_codes = match_each_integer_ptype!(codes.ptype(), |$T| {
            let mut codes = codes.into_buffer_mut();
            typed_mask::<$T>(&mut codes, mask)?;
            codes.into_array()
        });
        DictArray::try_new(new_codes, new_values).map(IntoArrayData::into_array)
    }
}

fn add_null_value<T: NativePType>(values: PrimitiveArray) -> VortexResult<ArrayData> {
    let buf: BufferMut<T> = values.into_buffer_mut::<T>();
    let mut new_buf: BufferMut<T> = BufferMut::<T>::with_capacity(buf.len() + 1);
    new_buf.push(T::zero());
    new_buf.extend(buf);
    let len = new_buf.len();
    Ok(PrimitiveArray::new(new_buf, dict_values_validity(true, len)).into_array())
}

fn typed_mask<T: NativePType>(codes: &mut BufferMut<T>, mask: FilterMask) -> VortexResult<()> {
    match mask.iter() {
        FilterIter::Indices(indices) => {
            for index in indices {
                codes[*index] = T::zero();
            }
        }
        FilterIter::Slices(slices) => {
            for slice in slices {
                codes[slice.0..slice.1].fill(T::zero());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::test_harness::test_mask;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayData as _;
    use vortex_buffer::buffer;

    use crate::{dict_encode_primitive, DictArray};

    #[test]
    fn test_mask_nullable_dict_array() {
        let reference =
            PrimitiveArray::from_option_iter([None, Some(42), Some(-9), Some(42), Some(5)]);
        let (codes, values) = dict_encode_primitive(&reference);
        test_mask(
            DictArray::try_new(codes.into_array(), values.into_array())
                .unwrap()
                .into_array(),
        )
    }

    #[test]
    fn test_mask_non_nullable_dict_array() {
        let reference = PrimitiveArray::new(buffer![5, 42, -9, 42, 5], Validity::NonNullable);
        let (codes, values) = dict_encode_primitive(&reference);
        test_mask(
            DictArray::try_new(codes.into_array(), values.into_array())
                .unwrap()
                .into_array(),
        )
    }
}
