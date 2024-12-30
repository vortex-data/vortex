use vortex_error::VortexResult;

use crate::array::sparse::SparseArray;
use crate::array::{ConstantArray, SparseEncoding};
use crate::compute::SliceFn;
use crate::{ArrayData, IntoArrayData};

impl SliceFn<SparseArray> for SparseEncoding {
    fn slice(&self, array: &SparseArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let new_patches = array.patches().slice(
            array.indices_offset() + start,
            array.indices_offset() + stop,
        )?;

        let Some(new_patches) = new_patches else {
            return Ok(ConstantArray::new(array.fill_scalar(), stop - start).into_array());
        };

        SparseArray::try_new_from_patches(
            new_patches,
            stop - start,
            // NB: Patches::slice adjusts the indices
            0,
            array.fill_scalar(),
        )
        .map(IntoArrayData::into_array)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::compute::slice;
    use crate::IntoArrayVariant;

    #[test]
    fn test_slice() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        let sparse = SparseArray::try_new(indices, values, 101, 0_u32.into())
            .unwrap()
            .into_array();

        let sliced = slice(&sparse, 15, 100).unwrap();
        assert_eq!(sliced.len(), 100 - 15);
        let primitive = SparseArray::try_from(sliced)
            .unwrap()
            .patches()
            .into_values()
            .into_primitive()
            .unwrap();

        assert_eq!(primitive.as_slice::<u32>(), &[13531]);
    }

    #[test]
    fn doubly_sliced() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        let sparse = SparseArray::try_new(indices, values, 101, 0_u32.into())
            .unwrap()
            .into_array();

        let sliced = slice(&sparse, 15, 100).unwrap();
        assert_eq!(sliced.len(), 100 - 15);
        let primitive = SparseArray::try_from(sliced.clone())
            .unwrap()
            .patches()
            .into_values()
            .into_primitive()
            .unwrap();

        assert_eq!(primitive.as_slice::<u32>(), &[13531]);

        let doubly_sliced = slice(&sliced, 35, 36).unwrap();
        let primitive_doubly_sliced = SparseArray::try_from(doubly_sliced)
            .unwrap()
            .patches()
            .into_values()
            .into_primitive()
            .unwrap();

        assert_eq!(primitive_doubly_sliced.as_slice::<u32>(), &[13531]);
    }

    #[test]
    fn slice_partially_invalid() {
        let values = buffer![0u64].into_array();
        let indices = buffer![0u8].into_array();

        let sparse = SparseArray::try_new(indices, values, 1000, 999u64.into()).unwrap();
        let sliced = slice(&sparse, 0, 1000).unwrap();
        let mut expected = vec![999u64; 1000];
        expected[0] = 0;

        let actual = sliced.into_primitive().unwrap().as_slice::<u64>().to_vec();
        assert_eq!(expected, actual);
    }
}
