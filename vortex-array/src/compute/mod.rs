//! Compute kernels on top of Vortex Arrays.
//!
//! We aim to provide a basic set of compute kernels that can be used to efficiently index, slice,
//! and filter Vortex Arrays in their encoded forms.
//!
//! Every array encoding has the ability to implement their own efficient implementations of these
//! operators, else we will decode, and perform the equivalent operator from Arrow.

pub use between::{BetweenFn, BetweenOptions, StrictComparison, between};
pub use binary_numeric::{
    BinaryNumericFn, add, add_scalar, binary_numeric, div, div_scalar, mul, mul_scalar, sub,
    sub_scalar,
};
pub use boolean::{
    BinaryBooleanFn, BinaryOperator, and, and_kleene, binary_boolean, or, or_kleene,
};
pub use cast::{CastFn, try_cast};
pub use compare::{CompareFn, Operator, compare, compare_lengths_to_empty, scalar_cmp};
pub use fill_forward::{FillForwardFn, fill_forward};
pub use fill_null::{FillNullFn, fill_null};
pub use filter::{FilterFn, filter};
pub use invert::{InvertFn, invert};
pub use is_constant::*;
pub use like::{LikeFn, LikeOptions, like};
pub use mask::{MaskFn, mask};
pub use min_max::{MinMaxFn, MinMaxResult, min_max};
pub use scalar_at::{ScalarAtFn, scalar_at};
pub use search_sorted::*;
pub use slice::{SliceFn, slice};
pub use sum::*;
pub use take::{TakeFn, take, take_into};
pub use take_from::TakeFromFn;
pub use to_arrow::*;
pub use uncompressed_size::*;

mod between;
mod binary_numeric;
mod boolean;
mod cast;
mod compare;
mod fill_forward;
mod fill_null;
mod filter;
mod invert;
mod is_constant;
mod like;
mod mask;
mod min_max;
mod scalar_at;
mod search_sorted;
mod slice;
mod sum;
mod take;
mod take_from;
mod to_arrow;
mod uncompressed_size;

#[cfg(feature = "test-harness")]
pub mod test_harness {
    pub use crate::compute::binary_numeric::test_harness::test_binary_numeric;
    pub use crate::compute::mask::test_harness::test_mask;
}
