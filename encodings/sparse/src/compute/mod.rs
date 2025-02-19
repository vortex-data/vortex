use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{
    BinaryNumericFn, FilterFn, InvertFn, ScalarAtFn, SearchResult, SearchSortedFn,
    SearchSortedSide, SearchSortedUsizeFn, SliceFn, TakeFn,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::{SparseArray, SparseEncoding};

mod binary_numeric;
mod invert;
mod slice;
mod take;

impl ComputeVTable for SparseEncoding {
    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<Array>> {
        Some(self)
    }

    fn search_sorted_usize_fn(&self) -> Option<&dyn SearchSortedUsizeFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }
}

impl ScalarAtFn<SparseArray> for SparseEncoding {
    fn scalar_at(&self, array: &SparseArray, index: usize) -> VortexResult<Scalar> {
        Ok(array
            .patches()
            .get_patched(index)?
            .unwrap_or_else(|| array.fill_scalar()))
    }
}

// FIXME(ngates): these are broken in a way that works for array patches, this will be fixed soon.
impl SearchSortedFn<SparseArray> for SparseEncoding {
    fn search_sorted(
        &self,
        array: &SparseArray,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        array.patches().search_sorted(value.clone(), side)
    }
}

// FIXME(ngates): these are broken in a way that works for array patches, this will be fixed soon.
impl SearchSortedUsizeFn<SparseArray> for SparseEncoding {
    fn search_sorted_usize(
        &self,
        array: &SparseArray,
        value: usize,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        let Ok(target) = Scalar::from(value).cast(array.dtype()) else {
            // If the downcast fails, then the target is too large for the dtype.
            return Ok(SearchResult::NotFound(array.len()));
        };
        SearchSortedFn::search_sorted(self, array, &target, side)
    }
}

impl FilterFn<SparseArray> for SparseEncoding {
    fn filter(&self, array: &SparseArray, mask: &Mask) -> VortexResult<Array> {
        let new_length = mask.true_count();

        let Some(new_patches) = array.patches().filter(mask)? else {
            return Ok(ConstantArray::new(array.fill_scalar(), new_length).into_array());
        };

        SparseArray::try_new_from_patches(new_patches, new_length, array.fill_scalar())
            .map(IntoArray::into_array)
    }
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::test_harness::{test_binary_numeric, test_mask};
    use vortex_array::compute::{
        filter, search_sorted, slice, try_cast, SearchResult, SearchSortedSide,
    };
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    #[fixture]
    fn array() -> Array {
        SparseArray::try_new(
            buffer![2u64, 9, 15].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            20,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    fn search_larger_than(array: Array) {
        let res = search_sorted(&array, 66, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_less_than(array: Array) {
        let res = search_sorted(&array, 22, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(2));
    }

    #[rstest]
    fn search_found(array: Array) {
        let res = search_sorted(&array, 44, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::Found(9));
    }

    #[rstest]
    fn search_not_found_right(array: Array) {
        let res = search_sorted(&array, 56, SearchSortedSide::Right).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_sliced(array: Array) {
        let array = slice(&array, 7, 20).unwrap();
        assert_eq!(
            search_sorted(&array, 22, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
    }

    #[test]
    fn search_right() {
        let array = SparseArray::try_new(
            buffer![0u64].into_array(),
            PrimitiveArray::new(buffer![0u8], Validity::AllValid).into_array(),
            2,
            Scalar::null_typed::<u8>(),
        )
        .unwrap()
        .into_array();

        assert_eq!(
            search_sorted(&array, 0, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(1)
        );
        assert_eq!(
            search_sorted(&array, 1, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(1)
        );
    }

    #[rstest]
    fn test_filter(array: Array) {
        let mut predicate = vec![false, false, true];
        predicate.extend_from_slice(&[false; 17]);
        let mask = Mask::from_iter(predicate);

        let filtered_array = filter(&array, &mask).unwrap();
        let filtered_array = SparseArray::try_from(filtered_array).unwrap();

        assert_eq!(filtered_array.len(), 1);
        assert_eq!(filtered_array.patches().values().len(), 1);
        assert_eq!(filtered_array.patches().indices().len(), 1);
    }

    #[test]
    fn true_fill_value() {
        let mask = Mask::from_iter([false, true, false, true, false, true, true]);
        let array = SparseArray::try_new(
            buffer![0_u64, 3, 6].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            7,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array();

        let filtered_array = filter(&array, &mask).unwrap();
        let filtered_array = SparseArray::try_from(filtered_array).unwrap();

        assert_eq!(filtered_array.len(), 4);
        let primitive = filtered_array
            .patches()
            .into_indices()
            .into_primitive()
            .unwrap();

        assert_eq!(primitive.as_slice::<u64>(), &[1, 3]);
    }

    #[rstest]
    fn test_sparse_binary_numeric(array: Array) {
        test_binary_numeric::<i32>(array)
    }

    #[test]
    fn test_mask_sparse_array() {
        let null_fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        test_mask(
            SparseArray::try_new(
                buffer![1u64, 2, 4].into_array(),
                try_cast(
                    buffer![100i32, 200, 300].into_array(),
                    null_fill_value.dtype(),
                )
                .unwrap(),
                5,
                null_fill_value,
            )
            .unwrap()
            .into_array(),
        );

        let ten_fill_value = Scalar::from(10i32);
        test_mask(
            SparseArray::try_new(
                buffer![1u64, 2, 4].into_array(),
                buffer![100i32, 200, 300].into_array(),
                5,
                ten_fill_value,
            )
            .unwrap()
            .into_array(),
        )
    }
}
