use vortex_array::array::ConstantArray;
use vortex_array::compute::TakeFn;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{SparseArray, SparseEncoding};

impl TakeFn<SparseArray> for SparseEncoding {
    fn take(&self, array: &SparseArray, take_indices: &ArrayData) -> VortexResult<ArrayData> {
        // FIXME(DK): add_scalar to the take_indices if they are shorter
        let resolved_patches = array.resolved_patches()?;

        let Some(new_patches) = resolved_patches.take(take_indices)? else {
            return Ok(ConstantArray::new(array.fill_scalar(), take_indices.len()).into_array());
        };

        SparseArray::try_new_from_patches(new_patches, take_indices.len(), 0, array.fill_scalar())
            .map(IntoArrayData::into_array)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{scalar_at, slice, take};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    fn test_array_fill_value() -> Scalar {
        // making this const is annoying
        Scalar::null_typed::<f64>()
    }

    fn sparse_array() -> ArrayData {
        SparseArray::try_new(
            buffer![0u64, 37, 47, 99].into_array(),
            PrimitiveArray::new(buffer![1.23f64, 0.47, 9.99, 3.5], Validity::AllValid).into_array(),
            100,
            test_array_fill_value(),
        )
        .unwrap()
        .into_array()
    }

    #[test]
    fn take_with_non_zero_offset() {
        let sparse = sparse_array();
        let sparse = slice(sparse, 30, 40).unwrap();
        let sparse = take(sparse, buffer![6, 7, 8].into_array()).unwrap();
        assert_eq!(scalar_at(&sparse, 0).unwrap(), test_array_fill_value());
        assert_eq!(scalar_at(&sparse, 1).unwrap(), Scalar::from(Some(0.47)));
        assert_eq!(scalar_at(&sparse, 2).unwrap(), test_array_fill_value());
    }

    #[test]
    fn sparse_take() {
        let sparse = sparse_array();
        let taken =
            SparseArray::try_from(take(sparse, buffer![0, 47, 47, 0, 99].into_array()).unwrap())
                .unwrap();
        assert_eq!(
            taken
                .patches()
                .into_indices()
                .into_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(
            taken
                .patches()
                .into_values()
                .into_primitive()
                .unwrap()
                .as_slice::<f64>(),
            [1.23f64, 9.99, 9.99, 1.23, 3.5]
        );
    }

    #[test]
    fn nonexistent_take() {
        let sparse = sparse_array();
        let taken = take(sparse, buffer![69].into_array()).unwrap();
        assert!(taken.len() == 1);
        assert_eq!(scalar_at(taken, 0).unwrap(), test_array_fill_value());
    }

    #[test]
    fn ordered_take() {
        let sparse = sparse_array();
        let taken =
            SparseArray::try_from(take(&sparse, buffer![69, 37].into_array()).unwrap()).unwrap();
        assert_eq!(
            taken
                .patches()
                .into_indices()
                .into_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [1]
        );
        assert_eq!(
            taken
                .patches()
                .into_values()
                .into_primitive()
                .unwrap()
                .as_slice::<f64>(),
            [0.47f64]
        );
        assert_eq!(taken.len(), 2);
    }
}
