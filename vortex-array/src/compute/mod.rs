//! Compute kernels on top of Vortex Arrays.
//!
//! We aim to provide a basic set of compute kernels that can be used to efficiently index, slice,
//! and filter Vortex Arrays in their encoded forms.
//!
//! Every array encoding has the ability to implement their own efficient implementations of these
//! operators, else we will decode, and perform the equivalent operator from Arrow.

pub use between::{between, BetweenFn, BetweenOptions, StrictComparison};
pub use binary_numeric::{
    add, add_scalar, binary_numeric, div, div_scalar, mul, mul_scalar, sub, sub_scalar,
    BinaryNumericFn,
};
pub use boolean::{
    and, and_kleene, binary_boolean, or, or_kleene, BinaryBooleanFn, BinaryOperator,
};
pub use cast::{try_cast, CastFn};
pub use compare::{compare, compare_lengths_to_empty, scalar_cmp, CompareFn, Operator};
pub use fill_forward::{fill_forward, FillForwardFn};
pub use fill_null::{fill_null, FillNullFn};
pub use filter::{filter, FilterFn};
pub use invert::{invert, InvertFn};
pub use like::{like, LikeFn, LikeOptions};
pub use mask::{mask, MaskFn};
pub use min_max::{min_max, MinMaxFn, MinMaxResult};
pub use scalar_at::{scalar_at, ScalarAtFn};
pub use search_sorted::*;
pub use slice::{slice, SliceFn};
pub use take::{take, take_into, TakeFn};
pub use to_arrow::*;

mod between;
mod binary_numeric;
mod boolean;
mod cast;
mod compare;
mod fill_forward;
mod fill_null;
mod filter;
mod invert;
mod like;
mod mask;
mod min_max;
mod scalar_at;
mod search_sorted;
mod slice;
mod take;
mod to_arrow;

#[cfg(feature = "test-harness")]
pub mod test_harness {
    pub use crate::compute::binary_numeric::test_harness::test_binary_numeric;
    pub use crate::compute::mask::test_harness::test_mask;
}
