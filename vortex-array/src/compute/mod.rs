//! Compute kernels on top of Vortex Arrays.
//!
//! We aim to provide a basic set of compute kernels that can be used to efficiently index, slice,
//! and filter Vortex Arrays in their encoded forms.
//!
//! Every array encoding has the ability to implement their own efficient implementations of these
//! operators, else we will decode, and perform the equivalent operator from Arrow.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::RwLock;

pub use between::*;
pub use boolean::*;
pub use cast::*;
pub use compare::*;
pub use fill_forward::{FillForwardFn, fill_forward};
pub use fill_null::{FillNullFn, fill_null};
pub use filter::*;
pub use invert::{InvertFn, invert};
pub use is_constant::*;
pub use is_sorted::*;
pub use like::{LikeFn, LikeOptions, like};
pub use mask::{MaskFn, mask};
pub use min_max::{MinMaxFn, MinMaxResult, min_max};
pub use numeric::*;
pub use optimize::*;
pub use scalar_at::{ScalarAtFn, scalar_at};
pub use search_sorted::*;
pub use slice::{SliceFn, slice};
pub use sum::*;
pub use take::{TakeFn, take, take_into};
pub use take_from::TakeFromFn;
pub use to_arrow::*;
pub use uncompressed_size::*;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arcref::ArcRef;
use crate::builders::ArrayBuilder;
use crate::{Array, ArrayRef};

#[cfg(feature = "arbitrary")]
mod arbitrary;
mod between;
mod boolean;
mod cast;
mod compare;
#[cfg(feature = "test-harness")]
pub mod conformance;
mod fill_forward;
mod fill_null;
mod filter;
mod invert;
mod is_constant;
mod is_sorted;
mod like;
mod mask;
mod min_max;
mod numeric;
mod optimize;
mod scalar_at;
mod search_sorted;
mod slice;
mod sum;
mod take;
mod take_from;
mod to_arrow;
mod uncompressed_size;

/// An instance of a compute function holding the implementation vtable and a set of registered
/// compute kernels.
pub struct ComputeFn {
    id: ArcRef<str>,
    vtable: ArcRef<dyn ComputeFnVTable>,
    kernels: RwLock<Vec<ArcRef<dyn Kernel>>>,
}

impl ComputeFn {
    /// Create a new compute function from the given [`ComputeFnVTable`].
    pub fn new(id: ArcRef<str>, vtable: ArcRef<dyn ComputeFnVTable>) -> Self {
        Self {
            id,
            vtable,
            kernels: Default::default(),
        }
    }

    /// Returns the string identifier of the compute function.
    pub fn id(&self) -> &ArcRef<str> {
        &self.id
    }

    /// Register a kernel for the compute function.
    pub fn register_kernel(&self, kernel: ArcRef<dyn Kernel>) {
        self.kernels
            .write()
            .vortex_expect("poisoned lock")
            .push(kernel);
    }

    /// Invokes the compute function with the given arguments.
    pub fn invoke<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<Output> {
        let expected_dtype = self.vtable.return_type(args)?;
        let expected_len = self.vtable.return_len(args)?;

        let output = self
            .vtable
            .invoke(args, &self.kernels.read().vortex_expect("poisoned lock"))?;

        if output.dtype() != &expected_dtype {
            vortex_bail!(
                "Internal error: compute function {} returned a result of type {} but expected {}",
                self.id,
                output.dtype(),
                &expected_dtype
            );
        }
        if output.len() != expected_len {
            vortex_bail!(
                "Internal error: compute function {} returned a result of length {} but expected {}",
                self.id,
                output.len(),
                expected_len
            );
        }

        Ok(output)
    }
}

/// VTable for the implementation of a compute function.
pub trait ComputeFnVTable: 'static + Send + Sync {
    /// Invokes the compute function entry-point with the given input arguments and options.
    ///
    /// The entry-point logic can short-circuit compute using statistics, update result array
    /// statistics, search for relevant compute kernels, and canonicalize the inputs in order
    /// to successfully compute a result.
    fn invoke<'a>(
        &self,
        args: &'a InvocationArgs<'a>,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output>;

    /// Computes the return type of the function given the input arguments.
    ///
    /// All kernel implementations will be validated to return the [`DType`] as computed here.
    fn return_type<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<DType>;

    /// Computes the return length of the function given the input arguments.
    ///
    /// All kernel implementations will be validated to return the len as computed here.
    /// Scalars are considered to have length 1.
    fn return_len<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<usize>;

    /// Returns whether the function operates elementwise, i.e. the output is the same shape as the
    /// input and no information is shared between elements.
    ///
    /// Examples include `add`, `subtract`, `and`, `cast`, `fill_null` etc.
    /// Examples that are not elementwise include `sum`, `count`, `min`, `fill_forward` etc.
    fn is_elementwise(&self) -> bool;
}

/// Arguments to a compute function invocation.
pub struct InvocationArgs<'a> {
    pub inputs: &'a [Input<'a>],
    pub options: &'a dyn Options,
}

/// Input to a compute function.
pub enum Input<'a> {
    Scalar(&'a Scalar),
    Array(&'a dyn Array),
    Mask(&'a Mask),
    Builder(&'a mut dyn ArrayBuilder),
    DType(&'a DType),
}

impl Debug for Input<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Input");
        match self {
            Input::Scalar(scalar) => f.field("Scalar", scalar),
            Input::Array(array) => f.field("Array", array),
            Input::Mask(mask) => f.field("Mask", mask),
            Input::Builder(builder) => f.field("Builder", &builder.len()),
            Input::DType(dtype) => f.field("DType", dtype),
        };
        f.finish()
    }
}

impl<'a> From<&'a dyn Array> for Input<'a> {
    fn from(value: &'a dyn Array) -> Self {
        Input::Array(value)
    }
}

impl<'a> From<&'a Scalar> for Input<'a> {
    fn from(value: &'a Scalar) -> Self {
        Input::Scalar(value)
    }
}

impl<'a> From<&'a Mask> for Input<'a> {
    fn from(value: &'a Mask) -> Self {
        Input::Mask(value)
    }
}

impl<'a> From<&'a DType> for Input<'a> {
    fn from(value: &'a DType) -> Self {
        Input::DType(value)
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

    pub fn dtype(&self) -> Option<&'a DType> {
        match self {
            Input::DType(dtype) => Some(*dtype),
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

#[allow(clippy::len_without_is_empty)]
impl Output {
    pub fn dtype(&self) -> &DType {
        match self {
            Output::Scalar(scalar) => scalar.dtype(),
            Output::Array(array) => array.dtype(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Output::Scalar(_) => 1,
            Output::Array(array) => array.len(),
        }
    }

    pub fn unwrap_scalar(self) -> VortexResult<Scalar> {
        match &self {
            Output::Array(_) => vortex_bail!("Expected array output, got Array"),
            Output::Scalar(scalar) => Ok(scalar.clone()),
        }
    }

    pub fn unwrap_array(self) -> VortexResult<ArrayRef> {
        match &self {
            Output::Array(array) => Ok(array.clone()),
            Output::Scalar(_) => vortex_bail!("Expected array output, got Scalar"),
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

impl Options for () {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Compute functions can ask arrays for compute kernels for a given invocation.
///
/// The kernel is invoked with the input arguments and options, and can return `None` if it is
/// unable to compute the result for the given inputs due to missing implementation logic.
/// For example, if kernel doesn't support the `LTE` operator.
///
/// If the kernel fails to compute a result, it should return a `Some` with the error.
pub trait Kernel: 'static + Send + Sync + Debug {
    /// Invokes the kernel with the given input arguments and options.
    fn invoke<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<Option<Output>>;
}

/// Register a kernel for a compute function.
/// See each compute function for the correct type of kernel to register.
#[macro_export]
macro_rules! register_kernel {
    ($T:expr) => {
        $crate::aliases::inventory::submit!($T);
    };
}
