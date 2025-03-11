mod compare;

use std::ops::AddAssign;

use num_traits::{CheckedShl, CheckedShr, WrappingAdd, WrappingSub};
use vortex_array::compute::{
    CompareFn, FilterKernel, FilterKernelAdapter, KernelRef, ScalarAtFn, SearchResult,
    SearchSortedFn, SearchSortedSide, SliceFn, TakeFn, filter, scalar_at, search_sorted, slice,
    take,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayComputeImpl, ArrayRef};
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexError, VortexExpect as _, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::{PValue, Scalar};

use crate::{FoRArray, FoREncoding};

impl ArrayComputeImpl for FoRArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(FoREncoding).some();
}

impl ComputeVTable for FoREncoding {
    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}

impl TakeFn<&FoRArray> for FoREncoding {
    fn take(&self, array: &FoRArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        FoRArray::try_new(
            take(array.encoded(), indices)?,
            array.reference_scalar().clone(),
        )
        .map(|a| a.into_array())
    }
}

impl FilterKernel for FoREncoding {
    fn filter(&self, array: &FoRArray, mask: &Mask) -> VortexResult<ArrayRef> {
        FoRArray::try_new(
            filter(array.encoded(), mask)?,
            array.reference_scalar().clone(),
        )
        .map(|a| a.into_array())
    }
}

impl ScalarAtFn<&FoRArray> for FoREncoding {
    fn scalar_at(&self, array: &FoRArray, index: usize) -> VortexResult<Scalar> {
        let encoded_pvalue = scalar_at(array.encoded(), index)?.reinterpret_cast(array.ptype());
        let encoded_pvalue = encoded_pvalue.as_primitive();
        let reference = array.reference_scalar();
        let reference = reference.as_primitive();

        Ok(match_each_integer_ptype!(array.ptype(), |$P| {
            encoded_pvalue
                .typed_value::<$P>()
                .map(|v|
                     v.wrapping_add(
                         reference
                             .typed_value::<$P>()
                             .vortex_expect("FoRArray Reference value cannot be null")))
                .map(|v| Scalar::primitive::<$P>(v, array.dtype().nullability()))
                .unwrap_or_else(|| Scalar::null(array.dtype().clone()))
        }))
    }
}

impl SliceFn<&FoRArray> for FoREncoding {
    fn slice(&self, array: &FoRArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        FoRArray::try_new(
            slice(array.encoded(), start, stop)?,
            array.reference_scalar().clone(),
        )
        .map(|a| a.into_array())
    }
}

impl SearchSortedFn<&FoRArray> for FoREncoding {
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
    if primitive_value.is_lt(min) {
        return Ok(SearchResult::NotFound(array.invalid_count()?));
    }

    // When the values in the array are shifted, not all values in the domain are representable in the compressed
    // space. Multiple different search values can translate to same value in the compressed space.
    let target = primitive_value.wrapping_sub(&min);
    let target_scalar = Scalar::primitive(target, value.dtype().nullability())
        .reinterpret_cast(array.ptype().to_unsigned());

    search_sorted(array.encoded(), target_scalar, side)
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::{SearchResult, SearchSortedSide, scalar_at, search_sorted};

    use crate::for_compress;

    #[test]
    fn for_scalar_at() {
        let for_arr = for_compress(PrimitiveArray::from_iter([-100, 1100, 1500, 1900])).unwrap();
        assert_eq!(scalar_at(&for_arr, 0).unwrap(), (-100).into());
        assert_eq!(scalar_at(&for_arr, 1).unwrap(), 1100.into());
        assert_eq!(scalar_at(&for_arr, 2).unwrap(), 1500.into());
        assert_eq!(scalar_at(&for_arr, 3).unwrap(), 1900.into());
    }

    #[test]
    fn for_search() {
        let for_arr = for_compress(PrimitiveArray::from_iter([1100, 1500, 1900])).unwrap();
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
        let for_arr = for_compress(PrimitiveArray::from_iter([62, 114])).unwrap();
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
    fn search_with_nulls() {
        let for_arr = for_compress(PrimitiveArray::from_option_iter([
            None,
            None,
            Some(-8739),
            Some(-29),
        ]))
        .unwrap();
        assert_eq!(
            search_sorted(&for_arr, -22360, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
    }
}
