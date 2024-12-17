//! Compute kernels on top of Vortex Arrays.
//!
//! We aim to provide a basic set of compute kernels that can be used to efficiently index, slice,
//! and filter Vortex Arrays in their encoded forms.
//!
//! Every [array variant][crate::ArrayTrait] has the ability to implement their own efficient
//! implementations of these operators, else we will decode, and perform the equivalent operator
//! from Arrow.

pub use boolean::{
    and, and_kleene, binary_boolean, or, or_kleene, BinaryBooleanFn, BinaryOperator,
};
pub use cast::{try_cast, CastFn};
pub use compare::{compare, scalar_cmp, CompareFn, Operator};
pub use fill_forward::{fill_forward, FillForwardFn};
pub use fill_null::{fill_null, FillNullFn};
pub use filter::{filter, FilterFn, FilterIter, FilterMask};
pub use invert::{invert, InvertFn};
pub use like::{like, LikeFn, LikeOptions};
pub use scalar_at::{scalar_at, ScalarAtFn};
pub use scalar_subtract::{subtract_scalar, SubtractScalarFn};
pub use search_sorted::*;
pub use slice::{slice, SliceFn};
pub use take::{take, TakeFn};

use crate::ArrayData;

mod boolean;
mod cast;
mod compare;
mod fill_forward;
mod fill_null;
mod filter;
mod invert;
mod like;
mod scalar_at;
mod scalar_subtract;
mod search_sorted;
mod slice;
mod take;

/// VTable for dispatching compute functions to Vortex encodings.
pub trait ComputeVTable {
    /// Implementation of binary boolean logic operations.
    ///
    /// See: [BinaryBooleanFn].
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<ArrayData>> {
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

    /// Broadcast subtraction of scalar from Vortex array.
    ///
    /// See: [SubtractScalarFn].
    fn subtract_scalar_fn(&self) -> Option<&dyn SubtractScalarFn<ArrayData>> {
        None
    }

    /// Take a set of indices from an array. This often forces allocations and decoding of
    /// the receiver.
    ///
    /// See: [TakeFn].
    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        None
    }
}
