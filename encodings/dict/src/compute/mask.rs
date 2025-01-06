use vortex_array::compute::{FilterIter, FilterMask, MaskFn};
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant as _};
use vortex_buffer::BufferMut;
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl MaskFn<DictArray> for DictEncoding {
    fn mask(&self, array: &DictArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let new_codes = match_each_integer_ptype!(array.codes_ptype(), |$T| {
            let mut codes = array.codes().into_primitive()?.into_buffer_mut();
            typed_mask::<$T>(&mut codes, mask)?;
            codes.into_array()
        });
        DictArray::try_new(new_codes, array.values()).map(IntoArrayData::into_array)
    }
}

fn typed_mask<T: NativePType>(codes: &mut BufferMut<T>, mask: FilterMask) -> VortexResult<()> {
    match mask.iter()? {
        FilterIter::Indices(indices) => {
            for index in indices {
                codes[*index] = T::zero();
            }
        }
        FilterIter::IndicesIter(bit_index_iterator) => {
            for index in bit_index_iterator {
                codes[index] = T::zero();
            }
        }
        FilterIter::Slices(slices) => {
            for slice in slices {
                for index in slice.0..slice.1 {
                    codes[index] = T::zero();
                }
            }
        }
        FilterIter::SlicesIter(bit_slice_iterator) => {
            for slice in bit_slice_iterator {
                for index in slice.0..slice.1 {
                    codes[index] = T::zero();
                }
            }
        }
    }
    Ok(())
}
