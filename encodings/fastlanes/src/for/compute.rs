use std::ops::AddAssign;

use num_traits::{CheckedShl, CheckedShr, WrappingAdd, WrappingSub};
use vortex_array::compute::unary::{scalar_at, ScalarAtFn};
use vortex_array::compute::{
    filter, search_sorted, slice, take, ComputeVTable, FilterFn, FilterMask, SearchResult,
    SearchSortedFn, SearchSortedSide, SliceFn, TakeFn, TakeOptions,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{VortexError, VortexExpect as _, VortexResult};
use vortex_scalar::{PValue, Scalar};

use crate::{FoRArray, FoREncoding};

impl ComputeVTable for FoREncoding {
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

impl TakeFn<FoRArray> for FoREncoding {
    fn take(
        &self,
        array: &FoRArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        FoRArray::try_new(
            take(array.encoded(), indices, options)?,
            array.reference_scalar(),
            array.shift(),
        )
        .map(|a| a.into_array())
    }
}

impl FilterFn<FoRArray> for FoREncoding {
    fn filter(&self, array: &FoRArray, mask: FilterMask) -> VortexResult<ArrayData> {
        FoRArray::try_new(
            filter(&array.encoded(), mask)?,
            array.reference_scalar(),
            array.shift(),
        )
        .map(|a| a.into_array())
    }
}

impl ScalarAtFn<FoRArray> for FoREncoding {
    fn scalar_at(&self, array: &FoRArray, index: usize) -> VortexResult<Scalar> {
        let encoded_pvalue = scalar_at(array.encoded(), index)?.reinterpret_cast(array.ptype());
        let encoded_pvalue = encoded_pvalue.as_primitive();
        let reference = array.reference_scalar();
        let reference = reference.as_primitive();

        Ok(match_each_integer_ptype!(array.ptype(), |$P| {
            encoded_pvalue
                .typed_value::<$P>()
                .map(|v|
                     v.checked_shl(array.shift() as u32)
                     .unwrap_or_default()
                     .wrapping_add(
                         reference
                             .typed_value::<$P>()
                             .vortex_expect("FoRArray Reference value cannot be null")))
                .map(|v| Scalar::primitive::<$P>(v, array.dtype().nullability()))
                .unwrap_or_else(|| Scalar::null(array.dtype().clone()))
        }))
    }
}

impl SliceFn<FoRArray> for FoREncoding {
    fn slice(&self, array: &FoRArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        FoRArray::try_new(
            slice(array.encoded(), start, stop)?,
            array.reference_scalar(),
            array.shift(),
        )
        .map(|a| a.into_array())
    }
}

impl SearchSortedFn<FoRArray> for FoREncoding {
    fn search_sorted(
        &self,
        array: &FoRArray,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        match_each_integer_ptype!(array.ptype(), |$P| {
            search_sorted_typed::<$P>(array, value, side)
        })
    }
}

fn search_sorted_typed<T>(
    array: &FoRArray,
    value: &Scalar,
    side: SearchSortedSide,
) -> VortexResult<SearchResult>
where
    T: NativePType
        + for<'a> TryFrom<&'a Scalar, Error = VortexError>
        + TryFrom<PValue, Error = VortexError>
        + CheckedShr
        + CheckedShl
        + WrappingSub
        + WrappingAdd
        + AddAssign
        + Into<PValue>,
{
    let min: T = array
        .reference_scalar()
        .as_primitive()
        .typed_value::<T>()
        .vortex_expect("Reference value cannot be null");
    let primitive_value: T = value.cast(array.dtype())?.as_ref().try_into()?;
    // Make sure that smaller values are still smaller and not larger than (which they would be after wrapping_sub)
    if primitive_value < min {
        return Ok(SearchResult::NotFound(0));
    }

    // When the values in the array are shifted, not all values in the domain are representable in the compressed
    // space. Multiple different search values can translate to same value in the compressed space.
    let encoded_value = primitive_value
        .wrapping_sub(&min)
        .checked_shr(array.shift() as u32)
        .unwrap_or_default();
    let decoded_value = encoded_value
        .checked_shl(array.shift() as u32)
        .unwrap_or_default()
        .wrapping_add(&min);

    // We first determine whether the value can be represented in the compressed array. For any value that is not
    // representable, it is by definition NotFound. For NotFound values, the correct insertion index is by definition
    // the same regardless of which side we search on.
    // However, to correctly handle repeated values in the array, we need to search left on the next *representable*
    // value (i.e., increment the translated value by 1).
    let representable = decoded_value == primitive_value;
    let (side, target) = if representable {
        (side, encoded_value)
    } else {
        (
            SearchSortedSide::Left,
            encoded_value.wrapping_add(&T::one()),
        )
    };

    let target_scalar = Scalar::primitive(target, value.dtype().nullability())
        .reinterpret_cast(array.ptype().to_unsigned());
    let search_result = search_sorted(&array.encoded(), target_scalar, side)?;
    Ok(
        if representable && matches!(search_result, SearchResult::Found(_)) {
            search_result
        } else {
            SearchResult::NotFound(search_result.to_index())
        },
    )
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::compute::{search_sorted, SearchResult, SearchSortedSide};
    use vortex_array::IntoArrayData;

    use crate::{for_compress, FoRArray};

    #[test]
    fn for_scalar_at() {
        let for_arr = for_compress(&PrimitiveArray::from(vec![-100, 1100, 1500, 1900])).unwrap();
        assert_eq!(scalar_at(&for_arr, 0).unwrap(), (-100).into());
        assert_eq!(scalar_at(&for_arr, 1).unwrap(), 1100.into());
        assert_eq!(scalar_at(&for_arr, 2).unwrap(), 1500.into());
        assert_eq!(scalar_at(&for_arr, 3).unwrap(), 1900.into());
    }

    #[test]
    fn for_search() {
        let for_arr = for_compress(&PrimitiveArray::from(vec![1100, 1500, 1900]))
            .unwrap()
            .into_array();
        assert_eq!(
            search_sorted(&for_arr, 1500, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(1)
        );
        assert_eq!(
            search_sorted(&for_arr, 2000, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(3)
        );
        assert_eq!(
            search_sorted(&for_arr, 1000, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
    }

    #[test]
    fn search_with_shift_notfound() {
        let for_arr = for_compress(&PrimitiveArray::from(vec![62, 114]))
            .unwrap()
            .into_array();
        assert_eq!(
            search_sorted(&for_arr, 63, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(1)
        );
        assert_eq!(
            search_sorted(&for_arr, 61, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            search_sorted(&for_arr, 113, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(1)
        );
        assert_eq!(
            search_sorted(&for_arr, 115, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
    }

    #[test]
    fn search_with_shift_repeated() {
        let arr = for_compress(&PrimitiveArray::from(vec![62, 62, 114, 114]))
            .unwrap()
            .into_array();
        let for_array = FoRArray::try_from(arr.clone()).unwrap();

        let min: i32 = for_array
            .reference_scalar()
            .as_primitive()
            .typed_value::<i32>()
            .unwrap();
        assert_eq!(min, 62);
        assert_eq!(for_array.shift(), 1);

        assert_eq!(
            search_sorted(&arr, 61, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            search_sorted(&arr, 61, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            search_sorted(&arr, 62, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(0)
        );
        assert_eq!(
            search_sorted(&arr, 62, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(2)
        );
        assert_eq!(
            search_sorted(&arr, 63, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
        assert_eq!(
            search_sorted(&arr, 63, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(2)
        );
        assert_eq!(
            search_sorted(&arr, 113, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
        assert_eq!(
            search_sorted(&arr, 113, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(2)
        );
        assert_eq!(
            search_sorted(&arr, 114, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(2)
        );
        assert_eq!(
            search_sorted(&arr, 114, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(4)
        );
        assert_eq!(
            search_sorted(&arr, 115, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(4)
        );
        assert_eq!(
            search_sorted(&arr, 115, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(4)
        );
    }
}
