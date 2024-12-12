use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::chunked::ChunkedArray;
use crate::array::ChunkedEncoding;
use crate::compute::{
    try_cast, BinaryBooleanFn, CastFn, CompareFn, ComputeVTable, FillNullFn, FilterFn, InvertFn,
    ScalarAtFn, SliceFn, SubtractScalarFn, TakeFn,
};
use crate::{ArrayData, IntoArrayData};

mod boolean;
mod compare;
mod fill_null;
mod filter;
mod invert;
mod scalar_at;
mod slice;
mod take;

impl ComputeVTable for ChunkedEncoding {
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<ArrayData>> {
        Some(self)
    }

    fn cast_fn(&self) -> Option<&dyn CastFn<ArrayData>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn subtract_scalar_fn(&self) -> Option<&dyn SubtractScalarFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl CastFn<ChunkedArray> for ChunkedEncoding {
    fn cast(&self, array: &ChunkedArray, dtype: &DType) -> VortexResult<ArrayData> {
        let mut cast_chunks = Vec::new();
        for chunk in array.chunks() {
            cast_chunks.push(try_cast(&chunk, dtype)?);
        }

        Ok(ChunkedArray::try_new(cast_chunks, dtype.clone())?.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability, PType};

    use crate::array::chunked::ChunkedArray;
    use crate::array::primitive::PrimitiveArray;
    use crate::compute::try_cast;
    use crate::validity::Validity;
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn test_cast_chunked() {
        let arr0 = PrimitiveArray::from_vec(vec![0u32, 1], Validity::NonNullable).into_array();
        let arr1 = PrimitiveArray::from_vec(vec![2u32, 3], Validity::NonNullable).into_array();

        let chunked = ChunkedArray::try_new(
            vec![arr0, arr1],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        // Two levels of chunking, just to be fancy.
        let root = ChunkedArray::try_new(
            vec![chunked],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        assert_eq!(
            try_cast(
                &root,
                &DType::Primitive(PType::U64, Nullability::NonNullable)
            )
            .unwrap()
            .into_primitive()
            .unwrap()
            .into_maybe_null_slice::<u64>(),
            vec![0u64, 1, 2, 3],
        );
    }
}
