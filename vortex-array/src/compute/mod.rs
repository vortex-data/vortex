//! Compute kernels on top of Vortex Arrays.
//!
//! We aim to provide a basic set of compute kernels that can be used to efficiently index, slice,
//! and filter Vortex Arrays in their encoded forms.
//!
//! Every array encoding has the ability to implement their own efficient implementations of these
//! operators, else we will decode, and perform the equivalent operator from Arrow.

use std::any::Any;
use std::fmt::{Debug, Formatter};

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
pub use filter::*;
pub use invert::{InvertFn, invert};
pub use is_constant::*;
pub use is_sorted::*;
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
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arcref::ArcRef;
use crate::builders::ArrayBuilder;
use crate::{Array, ArrayRef};

#[cfg(feature = "arbitrary")]
mod arbitrary;
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
mod is_sorted;
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

pub trait ComputeFn {
    /// The globally unique identifier for the compute function.
    fn id(&self) -> ArcRef<str>;

    /// Returns the function as the [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

    /// Invokes the compute function entry-point with the given input arguments and options.
    ///
    /// The entry-point logic can short-circuit compute using statistics, update result array
    /// statistics, search for relevant compute kernels, and canonicalize the inputs in order
    /// to successfully compute a result.
    fn invoke<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<Output>;

    /// Computes the return type of the function given the input arguments.
    ///
    /// All kernel implementations will be validated to return the [`DType`] as computed here.
    fn return_type<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<DType>;

    /// Returns whether the function operates elementwise, i.e. the output is the same shape as the
    /// input and no information is shared between elements.
    ///
    /// Examples include `add`, `subtract`, `and`, `cast`, `fill_null` etc.
    /// Examples that are not elementwise include `sum`, `count`, `min`, `fill_forward` etc.
    fn is_elementwise(&self) -> bool;
}

pub type ComputeFnRef = ArcRef<dyn ComputeFn>;

/// Arguments to a compute function invocation.
pub struct InvocationArgs<'a> {
    pub inputs: &'a [Input<'a>],
    pub options: Option<&'a dyn Options>,
}

/// Input to a compute function.
pub enum Input<'a> {
    Scalar(&'a Scalar),
    Array(&'a dyn Array),
    Mask(&'a Mask),
    Builder(&'a mut dyn ArrayBuilder),
}

impl Debug for Input<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Input");
        match self {
            Input::Scalar(scalar) => f.field("Scalar", scalar),
            Input::Array(array) => f.field("Array", array),
            Input::Mask(mask) => f.field("Mask", mask),
            Input::Builder(builder) => f.field("Builder", &builder.len()),
        };
        f.finish()
    }
}

impl<'a> Input<'a> {
    pub fn scalar(&self) -> Option<&'a Scalar> {
        match self {
            Input::Scalar(scalar) => Some(*scalar),
            _ => None,
        }
    }

    pub fn array(&self) -> Option<&'a dyn Array> {
        match self {
            Input::Array(array) => Some(*array),
            _ => None,
        }
    }

    pub fn mask(&self) -> Option<&'a Mask> {
        match self {
            Input::Mask(mask) => Some(*mask),
            _ => None,
        }
    }

    pub fn builder(&'a mut self) -> Option<&'a mut dyn ArrayBuilder> {
        match self {
            Input::Builder(builder) => Some(*builder),
            _ => None,
        }
    }
}

/// Output from a compute function.
#[derive(Debug)]
pub enum Output {
    Scalar(Scalar),
    Array(ArrayRef),
}

impl Output {
    pub fn into_scalar(self) -> Option<Scalar> {
        match self {
            Output::Scalar(scalar) => Some(scalar),
            _ => None,
        }
    }

    pub fn into_array(self) -> Option<ArrayRef> {
        match self {
            Output::Array(array) => Some(array),
            _ => None,
        }
    }
}

impl From<ArrayRef> for Output {
    fn from(value: ArrayRef) -> Self {
        Output::Array(value)
    }
}

impl From<Scalar> for Output {
    fn from(value: Scalar) -> Self {
        Output::Scalar(value)
    }
}

/// Options for a compute function invocation.
pub trait Options {
    fn as_any(&self) -> &dyn Any;
}

/// Compute functions can ask arrays for compute kernels for a given invocation.
///
/// The kernel is invoked with the input arguments and options, and can return `None` if it is
/// unable to compute the result for the given inputs due to missing implementation logic.
/// For example, if kernel doesn't support the `LTE` operator.
///
/// If the kernel fails to compute a result, it should return a `Some` with the error.
pub trait Kernel {
    fn invoke<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<Option<Output>>;
}

pub type KernelRef = ArcRef<dyn Kernel>;

#[cfg(feature = "test-harness")]
pub mod test_harness {
    pub use crate::compute::binary_numeric::test_harness::test_binary_numeric;
    pub use crate::compute::mask::test_harness::test_mask;
}
