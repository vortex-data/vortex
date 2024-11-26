use vortex_dtype::match_each_integer_ptype;
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::array::sparse::SparseArray;
use crate::array::{PrimitiveArray, SparseEncoding};
use crate::compute::unary::{scalar_at, ScalarAtFn};
use crate::compute::{
    search_sorted, take, ComputeVTable, FilterFn, FilterMask, SearchResult, SearchSortedFn,
    SearchSortedSide, SliceFn, TakeFn, TakeOptions,
};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

mod slice;
mod take;

impl ComputeVTable for SparseEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<SparseArray> for SparseEncoding {
    fn scalar_at(&self, array: &SparseArray, index: usize) -> VortexResult<Scalar> {
        Ok(match array.search_index(index)?.to_found() {
            None => array.fill_scalar(),
            Some(idx) => scalar_at(array.values(), idx)?,
        })
    }
}

impl SearchSortedFn<SparseArray> for SparseEncoding {
    fn search_sorted(
        &self,
        array: &SparseArray,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        search_sorted(&array.values(), value.clone(), side).and_then(|sr| {
            let sidx = sr.to_offsets_index(array.metadata().indices_len);
            let index: usize = scalar_at(array.indices(), sidx)?.as_ref().try_into()?;
            Ok(match sr {
                SearchResult::Found(i) => SearchResult::Found(
                    if i == array.metadata().indices_len {
                        index + 1
                    } else {
                        index
                    } - array.indices_offset(),
                ),
                SearchResult::NotFound(i) => SearchResult::NotFound(
                    if i == 0 { index } else { index + 1 } - array.indices_offset(),
                ),
            })
        })
    }
}

impl FilterFn<SparseArray> for SparseEncoding {
    fn filter(&self, array: &SparseArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let buffer = mask.to_boolean_buffer()?;
        let mut coordinate_indices: Vec<u64> = Vec::new();
        let mut value_indices = Vec::new();
        let mut last_inserted_index = 0;

        let flat_indices = array
            .indices()
            .into_primitive()
            .vortex_expect("Failed to convert SparseArray indices to primitive array");
        match_each_integer_ptype!(flat_indices.ptype(), |$P| {
            let indices = flat_indices
                .maybe_null_slice::<$P>()
                .iter()
                .map(|v| (*v as usize) - array.indices_offset());
            for (value_idx, coordinate) in indices.enumerate() {
                if buffer.value(coordinate) {
                    // We count the number of truthy values between this coordinate and the previous truthy one
                    let adjusted_coordinate = buffer.slice(last_inserted_index, coordinate - last_inserted_index).count_set_bits() as u64;
                    coordinate_indices.push(adjusted_coordinate + coordinate_indices.last().copied().unwrap_or_default());
                    last_inserted_index = coordinate;
                    value_indices.push(value_idx as u64);
                }
            }
        });

        Ok(SparseArray::try_new(
            PrimitiveArray::from(coordinate_indices).into_array(),
            take(
                array.values(),
                PrimitiveArray::from(value_indices),
                TakeOptions::default(),
            )?,
            buffer.count_set_bits(),
            array.fill_scalar(),
        )?
        .into_array())
    }
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};
    use vortex_scalar::Scalar;

    use crate::array::primitive::PrimitiveArray;
    use crate::array::sparse::SparseArray;
    use crate::compute::{
        filter, search_sorted, slice, FilterMask, SearchResult, SearchSortedSide,
    };
    use crate::validity::Validity;
    use crate::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};

    #[fixture]
    fn array() -> ArrayData {
        SparseArray::try_new(
            PrimitiveArray::from(vec![2u64, 9, 15]).into_array(),
            PrimitiveArray::from_vec(vec![33_i32, 44, 55], Validity::AllValid).into_array(),
            20,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    fn search_larger_than(array: ArrayData) {
        let res = search_sorted(&array, 66, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_less_than(array: ArrayData) {
        let res = search_sorted(&array, 22, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(2));
    }

    #[rstest]
    fn search_found(array: ArrayData) {
        let res = search_sorted(&array, 44, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::Found(9));
    }

    #[rstest]
    fn search_not_found_right(array: ArrayData) {
        let res = search_sorted(&array, 56, SearchSortedSide::Right).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_sliced(array: ArrayData) {
        let array = slice(&array, 7, 20).unwrap();
        assert_eq!(
            search_sorted(&array, 22, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
    }

    #[test]
    fn search_right() {
        let array = SparseArray::try_new(
            PrimitiveArray::from(vec![0u64]).into_array(),
            PrimitiveArray::from_vec(vec![0u8], Validity::AllValid).into_array(),
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
    fn test_filter(array: ArrayData) {
        let mut predicate = vec![false, false, true];
        predicate.extend_from_slice(&[false; 17]);
        let mask = FilterMask::from_iter(predicate);

        let filtered_array = filter(&array, mask).unwrap();
        let filtered_array = SparseArray::try_from(filtered_array).unwrap();

        assert_eq!(filtered_array.len(), 1);
        assert_eq!(filtered_array.values().len(), 1);
        assert_eq!(filtered_array.indices().len(), 1);
    }

    #[test]
    fn true_fill_value() {
        let mask = FilterMask::from_iter([false, true, false, true, false, true, true]);
        let array = SparseArray::try_new(
            PrimitiveArray::from(vec![0_u64, 3, 6]).into_array(),
            PrimitiveArray::from_vec(vec![33_i32, 44, 55], Validity::AllValid).into_array(),
            7,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array();

        let filtered_array = filter(&array, mask).unwrap();
        let filtered_array = SparseArray::try_from(filtered_array).unwrap();

        assert_eq!(filtered_array.len(), 4);
        let primitive = filtered_array.indices().into_primitive().unwrap();

        assert_eq!(primitive.maybe_null_slice::<u64>(), &[1, 3]);
    }
}
