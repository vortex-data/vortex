use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;

use crate::array::chunked::ChunkedArray;
use crate::compute::unary::{try_cast, CastFn, ScalarAtFn, SubtractScalarFn};
use crate::compute::{
    compare, slice, ArrayCompute, CompareFn, FilterFn, Operator, SliceFn, TakeFn,
};
use crate::{ArrayData, IntoArrayData};

mod filter;
mod scalar_at;
mod slice;
mod take;

impl ArrayCompute for ChunkedArray {
    fn cast(&self) -> Option<&dyn CastFn> {
        Some(self)
    }

    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        Some(CompareFn::compare(self, other, operator))
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn subtract_scalar(&self) -> Option<&dyn SubtractScalarFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }

    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }
}

impl CastFn for ChunkedArray {
    fn cast(&self, dtype: &DType) -> VortexResult<ArrayData> {
        let mut cast_chunks = Vec::new();
        for chunk in self.chunks() {
            cast_chunks.push(try_cast(&chunk, dtype)?);
        }

        Ok(ChunkedArray::try_new(cast_chunks, dtype.clone())?.into_array())
    }
}

impl CompareFn for ChunkedArray {
    fn compare(&self, array: &ArrayData, operator: Operator) -> VortexResult<ArrayData> {
        let mut idx = 0;
        let mut compare_chunks = Vec::with_capacity(self.nchunks());

        for chunk in self.chunks() {
            let sliced = slice(array, idx, idx + chunk.len())?;
            let cmp_result = compare(&chunk, &sliced, operator)?;
            compare_chunks.push(cmp_result);

            idx += chunk.len();
        }

        Ok(ChunkedArray::try_new(compare_chunks, DType::Bool(Nullability::Nullable))?.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability, PType};

    use crate::array::chunked::ChunkedArray;
    use crate::array::primitive::PrimitiveArray;
    use crate::compute::unary::try_cast;
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
