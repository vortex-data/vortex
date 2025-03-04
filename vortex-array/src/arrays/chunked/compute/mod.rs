use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::ChunkedEncoding;
use crate::arrays::chunked::ChunkedArray;
use crate::compute::{
    BinaryBooleanFn, BinaryNumericFn, CastFn, CompareFn, FillNullFn, FilterFn, InvertFn,
    IsConstantFn, IsSortedFn, MaskFn, MinMaxFn, ScalarAtFn, SliceFn, TakeFn, UncompressedSizeFn,
    try_cast,
};
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayRef};

mod binary_numeric;
mod boolean;
mod compare;
mod fill_null;
mod filter;
mod invert;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod scalar_at;
mod slice;
mod sum;
mod take;
mod uncompressed_size;

impl ComputeVTable for ChunkedEncoding {
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<&dyn Array>> {
        Some(self)
    }

    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<&dyn Array>> {
        Some(self)
    }

    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<&dyn Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<&dyn Array>> {
        Some(self)
    }

    fn is_constant_fn(&self) -> Option<&dyn IsConstantFn<&dyn Array>> {
        Some(self)
    }

    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}

impl CastFn<&ChunkedArray> for ChunkedEncoding {
    fn cast(&self, array: &ChunkedArray, dtype: &DType) -> VortexResult<ArrayRef> {
        let mut cast_chunks = Vec::new();
        for chunk in array.chunks() {
            cast_chunks.push(try_cast(chunk, dtype)?);
        }

        Ok(ChunkedArray::new_unchecked(cast_chunks, dtype.clone()).into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::IntoArray;
    use crate::array::Array;
    use crate::arrays::chunked::ChunkedArray;
    use crate::canonical::ToCanonical;
    use crate::compute::try_cast;

    #[test]
    fn test_cast_chunked() {
        let arr0 = buffer![0u32, 1].into_array();
        let arr1 = buffer![2u32, 3].into_array();

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
            .to_primitive()
            .unwrap()
            .as_slice::<u64>(),
            &[0u64, 1, 2, 3],
        );
    }
}
