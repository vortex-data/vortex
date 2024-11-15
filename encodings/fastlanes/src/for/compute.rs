use std::ops::{AddAssign, Shl, Shr};

use num_traits::{WrappingAdd, WrappingSub};
use vortex_array::compute::unary::{scalar_at_unchecked, ScalarAtFn};
use vortex_array::compute::{
    filter, search_sorted, slice, take, ArrayCompute, FilterFn, SearchResult, SearchSortedFn,
    SearchSortedSide, SliceFn, TakeFn,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, VortexUnwrap as _};
use vortex_scalar::{PValue, Scalar};

use crate::FoRArray;

impl ArrayCompute for FoRArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn search_sorted(&self) -> Option<&dyn SearchSortedFn> {
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

impl TakeFn for FoRArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        Self::try_new(
            take(self.encoded(), indices)?,
            self.owned_reference_scalar(),
            self.shift(),
        )
        .map(|a| a.into_array())
    }
}

impl FilterFn for FoRArray {
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData> {
        Self::try_new(
            filter(self.encoded(), predicate)?,
            self.owned_reference_scalar(),
            self.shift(),
        )
        .map(|a| a.into_array())
    }
}

impl ScalarAtFn for FoRArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(self.scalar_at_unchecked(index))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        let encoded_pvalue = scalar_at_unchecked(self.encoded(), index)
            .into_value()
            .as_pvalue()
            .vortex_expect("Encoded scalar must be primitive")
            .map(|p| p.reinterpret_cast(self.ptype()));
        let reference = self
            .reference()
            .as_pvalue()
            .vortex_expect("Reference scalar must be primitive")
            .vortex_expect("Reference scalar cannot be null");

        match_each_integer_ptype!(self.ptype(), |$P| {
            encoded_pvalue
                .map(|v| v.as_primitive::<$P>().vortex_unwrap())
                .map(|v| (v << self.shift()).wrapping_add(reference.as_primitive::<$P>().vortex_unwrap()))
                .map(|v| Scalar::primitive::<$P>(v, self.dtype().nullability()))
                .unwrap_or_else(|| Scalar::null(self.dtype().clone()))
        })
    }
}

impl SliceFn for FoRArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Self::try_new(
            slice(self.encoded(), start, stop)?,
            self.owned_reference_scalar(),
            self.shift(),
        )
        .map(|a| a.into_array())
    }
}

impl SearchSortedFn for FoRArray {
    fn search_sorted(&self, value: &Scalar, side: SearchSortedSide) -> VortexResult<SearchResult> {
        match_each_integer_ptype!(self.ptype(), |$P| {
            search_sorted_typed::<$P>(self, value, side)
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
        + Shr<u8, Output = T>
        + Shl<u8, Output = T>
        + WrappingSub
        + WrappingAdd
        + AddAssign
        + Into<PValue>,
{
    let min: T = array
        .reference()
        .as_pvalue()?
        .vortex_expect("Reference value cannot be null")
        .as_primitive::<T>()?;
    let primitive_value: T = value.cast(array.dtype())?.as_ref().try_into()?;
    // Make sure that smaller values are still smaller and not larger than (which they would be after wrapping_sub)
    if primitive_value < min {
        return Ok(SearchResult::NotFound(0));
    }

    // When the values in the array are shifted, not all values in the domain are representable in the compressed
    // space. Multiple different search values can translate to same value in the compressed space.
    let encoded_value = primitive_value.wrapping_sub(&min) >> array.shift();
    let decoded_value = (encoded_value << array.shift()).wrapping_add(&min);

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
        let for_arr = for_compress(&PrimitiveArray::from(vec![1100, 1500, 1900])).unwrap();
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
        let for_arr = for_compress(&PrimitiveArray::from(vec![62, 114])).unwrap();
        assert_eq!(
            search_sorted(&for_arr, 63, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(1)
        );
        let for_arr = for_compress(&PrimitiveArray::from(vec![62, 114])).unwrap();
        assert_eq!(
            search_sorted(&for_arr, 61, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
        let for_arr = for_compress(&PrimitiveArray::from(vec![62, 114])).unwrap();
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
        let arr = for_compress(&PrimitiveArray::from(vec![62, 62, 114, 114])).unwrap();
        let for_array = FoRArray::try_from(arr.clone()).unwrap();

        let min: i32 = for_array.reference().try_into().unwrap();
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
