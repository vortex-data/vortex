use vortex_array::compute::ScalarAtFn;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{SparseArray, SparseEncoding};

impl ScalarAtFn<&SparseArray> for SparseEncoding {
    fn scalar_at(&self, array: &SparseArray, index: usize) -> VortexResult<Scalar> {
        Ok(array
            .patches()
            .get_patched(index)?
            .filter(|s| s.is_valid())
            .unwrap_or_else(|| array.fill_scalar().clone()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{ConstantArray, PrimitiveArray};
    use vortex_array::compute::{scalar_at, slice, try_cast};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexError;
    use vortex_scalar::{PrimitiveScalar, Scalar};

    use crate::SparseArray;

    fn sparse_array() -> ArrayRef {
        let fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        // merged array: [null, null, 100, null, null, 200, null, null, 300, null]
        let mut values = buffer![100i32, 200, 300].into_array();
        values = try_cast(&values, fill_value.dtype()).unwrap();

        SparseArray::try_new(buffer![2u64, 5, 8].into_array(), values, 10, fill_value)
            .unwrap()
            .into_array()
    }

    #[test]
    fn invalid_patches() {
        let fill_value = Scalar::primitive(0i32, Nullability::Nullable);
        let array = SparseArray::try_new(
            buffer![1u32, 4].into_array(),
            PrimitiveArray::new(buffer![0, 1], Validity::from_iter(vec![false, true])).into_array(),
            5,
            fill_value.clone(),
        )
        .unwrap();

        assert_eq!(scalar_at(&array, 1).unwrap(), fill_value);
    }

    #[test]
    pub fn test_scalar_at() {
        let array = sparse_array();

        assert_eq!(
            scalar_at(&array, 0).unwrap(),
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
        );
        assert_eq!(scalar_at(&array, 2).unwrap(), Scalar::from(Some(100_i32)));
        assert_eq!(scalar_at(&array, 5).unwrap(), Scalar::from(Some(200_i32)));

        let error = scalar_at(&array, 10).err().unwrap();
        let VortexError::OutOfBounds(i, start, stop, _) = error else {
            unreachable!()
        };
        assert_eq!(i, 10);
        assert_eq!(start, 0);
        assert_eq!(stop, 10);
    }

    #[test]
    pub fn test_scalar_at_again() {
        let arr = SparseArray::try_new(
            ConstantArray::new(10u32, 1).into_array(),
            ConstantArray::new(Scalar::primitive(1234u32, Nullability::Nullable), 1).into_array(),
            100,
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable)),
        )
        .unwrap();

        assert_eq!(
            PrimitiveScalar::try_from(&scalar_at(&arr, 10).unwrap())
                .unwrap()
                .typed_value::<u32>(),
            Some(1234)
        );
        assert!(scalar_at(&arr, 0).unwrap().is_null());
        assert!(scalar_at(&arr, 99).unwrap().is_null());
    }

    #[test]
    pub fn scalar_at_sliced() {
        let sliced = slice(&sparse_array(), 2, 7).unwrap();
        assert_eq!(
            usize::try_from(&scalar_at(&sliced, 0).unwrap()).unwrap(),
            100
        );
        let error = scalar_at(&sliced, 5).err().unwrap();
        let VortexError::OutOfBounds(i, start, stop, _) = error else {
            unreachable!()
        };
        assert_eq!(i, 5);
        assert_eq!(start, 0);
        assert_eq!(stop, 5);
    }

    #[test]
    pub fn scalar_at_sliced_twice() {
        let sliced_once = slice(&sparse_array(), 1, 8).unwrap();
        assert_eq!(
            usize::try_from(&scalar_at(&sliced_once, 1).unwrap()).unwrap(),
            100
        );
        let error = scalar_at(&sliced_once, 7).err().unwrap();
        let VortexError::OutOfBounds(i, start, stop, _) = error else {
            unreachable!()
        };
        assert_eq!(i, 7);
        assert_eq!(start, 0);
        assert_eq!(stop, 7);

        let sliced_twice = slice(&sliced_once, 1, 6).unwrap();
        assert_eq!(
            usize::try_from(&scalar_at(&sliced_twice, 3).unwrap()).unwrap(),
            200
        );
        let error2 = scalar_at(&sliced_twice, 5).err().unwrap();
        let VortexError::OutOfBounds(i, start, stop, _) = error2 else {
            unreachable!()
        };
        assert_eq!(i, 5);
        assert_eq!(start, 0);
        assert_eq!(stop, 5);
    }
}
