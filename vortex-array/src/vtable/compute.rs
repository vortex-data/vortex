use crate::compute::{
    BinaryBooleanFn, BinaryNumericFn, CastFn, CompareFn, FillForwardFn, FillNullFn, FilterFn,
    InvertFn, LikeFn, ScalarAtFn, SearchSortedFn, SearchSortedUsizeFn, SliceFn, TakeFn, ToArrowFn,
};
use crate::ArrayData;

/// VTable for dispatching compute functions to Vortex encodings.
pub trait ComputeVTable {
    /// Implementation of binary boolean logic operations.
    ///
    /// See: [BinaryBooleanFn].
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<ArrayData>> {
        None
    }

    /// Implementation of binary numeric operations.
    ///
    /// See: [BinaryNumericFn].
    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<ArrayData>> {
        None
    }

    /// Implemented for arrays that can be casted to different types.
    ///
    /// See: [CastFn].
    fn cast_fn(&self) -> Option<&dyn CastFn<ArrayData>> {
        None
    }

    /// Binary operator implementation for arrays against other arrays.
    ///
    ///See: [CompareFn].
    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        None
    }

    /// Array function that returns new arrays a non-null value is repeated across runs of nulls.
    ///
    /// See: [FillForwardFn].
    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<ArrayData>> {
        None
    }

    /// Fill null values with given desired value. Resulting array is NonNullable
    ///
    /// See: [FillNullFn]
    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<ArrayData>> {
        None
    }

    /// Filter an array with a given mask.
    ///
    /// See: [FilterFn].
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        None
    }

    /// Invert a boolean array. Converts true -> false, false -> true, null -> null.
    ///
    /// See [InvertFn]
    fn invert_fn(&self) -> Option<&dyn InvertFn<ArrayData>> {
        None
    }

    /// Perform a SQL LIKE operation on two arrays.
    ///
    /// See: [LikeFn].
    fn like_fn(&self) -> Option<&dyn LikeFn<ArrayData>> {
        None
    }

    /// Single item indexing on Vortex arrays.
    ///
    /// See: [ScalarAtFn].
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        None
    }

    /// Perform a search over an ordered array.
    ///
    /// See: [SearchSortedFn].
    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<ArrayData>> {
        None
    }

    /// Perform a search over an ordered array.
    ///
    /// See: [SearchSortedUsizeFn].
    fn search_sorted_usize_fn(&self) -> Option<&dyn SearchSortedUsizeFn<ArrayData>> {
        None
    }

    /// Perform zero-copy slicing of an array.
    ///
    /// See: [SliceFn].
    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        None
    }

    /// Take a set of indices from an array. This often forces allocations and decoding of
    /// the receiver.
    ///
    /// See: [TakeFn].
    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        None
    }

    /// Convert the array to an Arrow array of the given type.
    ///
    /// See: [ToArrowFn].
    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<ArrayData>> {
        None
    }
}
