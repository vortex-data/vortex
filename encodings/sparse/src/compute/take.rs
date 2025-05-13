use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{TakeKernel, TakeKernelAdapter};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;

use crate::{SparseArray, SparseVTable};

impl TakeKernel for SparseVTable {
    fn take(&self, array: &SparseArray, take_indices: &dyn Array) -> VortexResult<ArrayRef> {
        let Some(new_patches) = array.patches().take(take_indices)? else {
            let result_fill_scalar = array.fill_scalar().cast(
                &array
                    .dtype()
                    .union_nullability(take_indices.dtype().nullability()),
            )?;
            return Ok(ConstantArray::new(result_fill_scalar, take_indices.len()).into_array());
        };

        // See `SparseEncoding::slice`.
        if new_patches.array_len() == new_patches.values().len() {
            return Ok(new_patches.into_values());
        }

        Ok(
            SparseArray::try_new_from_patches(new_patches, array.fill_scalar().clone())?
                .into_array(),
        )
    }
}

register_kernel!(TakeKernelAdapter(SparseVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::take;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayExt, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::{SparseArray, SparseVTable};

    fn test_array_fill_value() -> Scalar {
        // making this const is annoying
        Scalar::null_typed::<f64>()
    }

    fn sparse_array() -> ArrayRef {
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
        let sparse = sparse.slice(30, 40).unwrap();
        let sparse = take(&sparse, &buffer![6, 7, 8].into_array()).unwrap();
        assert_eq!(sparse.scalar_at(0).unwrap(), test_array_fill_value());
        assert_eq!(sparse.scalar_at(1).unwrap(), Scalar::from(Some(0.47)));
        assert_eq!(sparse.scalar_at(2).unwrap(), test_array_fill_value());
    }

    #[test]
    fn sparse_take() {
        let sparse = sparse_array();
        let prim = take(&sparse, &buffer![0, 47, 47, 0, 99].into_array())
            .unwrap()
            .to_primitive()
            .unwrap();
        assert_eq!(prim.as_slice::<f64>(), [1.23f64, 9.99, 9.99, 1.23, 3.5]);
    }

    #[test]
    fn nonexistent_take() {
        let sparse = sparse_array();
        let taken = take(&sparse, &buffer![69].into_array()).unwrap();
        assert_eq!(taken.len(), 1);
        assert_eq!(taken.scalar_at(0).unwrap(), test_array_fill_value());
    }

    #[test]
    fn ordered_take() {
        let sparse = sparse_array();
        let taken_arr = take(&sparse, &buffer![69, 37].into_array()).unwrap();
        let taken = taken_arr.as_::<SparseVTable>();

        assert_eq!(
            taken
                .patches()
                .indices()
                .to_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [1]
        );
        assert_eq!(
            taken
                .patches()
                .values()
                .to_primitive()
                .unwrap()
                .as_slice::<f64>(),
            [0.47f64]
        );
        assert_eq!(taken.len(), 2);
    }
}
