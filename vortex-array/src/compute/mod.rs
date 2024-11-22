//! Compute kernels on top of Vortex Arrays.
//!
//! We aim to provide a basic set of compute kernels that can be used to efficiently index, slice,
//! and filter Vortex Arrays in their encoded forms.
//!
//! Every [array variant][crate::ArrayTrait] has the ability to implement their own efficient
//! implementations of these operators, else we will decode, and perform the equivalent operator
//! from Arrow.

pub use boolean::{and, and_kleene, or, or_kleene, BinaryBooleanFn, BinaryOperator};
pub use compare::{compare, scalar_cmp, CompareFn, Operator};
pub use filter::*;
pub use search_sorted::*;
pub use slice::{slice, SliceFn};
pub use take::*;
use unary::{CastFn, FillForwardFn, ScalarAtFn, SubtractScalarFn};

use crate::ArrayData;

mod boolean;
mod compare;
mod filter;
mod search_sorted;
mod slice;
mod take;

pub mod unary;

/// VTable for dispatching compute functions to Vortex encodings.
pub trait ComputeVTable {
    /// Implementation of binary boolean logic operations.
    ///
    /// See: [BinaryBooleanFn].
    fn binary_boolean_fn(
        &self,
        _lhs: &ArrayData,
        _rhs: &ArrayData,
    ) -> Option<&dyn BinaryBooleanFn<ArrayData>> {
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

    /// Filter an array with a given mask.
    ///
    /// See: [FilterFn].
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
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
