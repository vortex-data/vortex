use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::arcref::ArcRef;
use crate::compute::{ComputeFn, InvocationArgs, Output};

pub trait ComputeFnImpl {
    type Inputs<'a>;
    type Output;

    fn id() -> ArcRef<str>;
    fn invoke(args: Self::Inputs<'_>) -> VortexResult<Self::Output>;
    fn return_type(args: Self::Inputs<'_>) -> VortexResult<DType>;
    fn is_elementwise() -> bool;
}

impl<F> ComputeFn for F
where
    F: ComputeFnImpl,
    for<'a> F::Inputs<'a>: TryFrom<&'a InvocationArgs<'a>, Error = VortexError>,
    F::Output: Into<Output>,
{
    fn id(&self) -> ArcRef<str> {
        <F as ComputeFnImpl>::id()
    }

    fn invoke<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<Output> {
        Ok(<F as ComputeFnImpl>::invoke(args.try_into()?)?.into())
    }

    fn return_type<'a>(&self, args: &'a InvocationArgs<'a>) -> VortexResult<DType> {
        <F as ComputeFnImpl>::return_type(args.try_into()?)
    }

    fn is_elementwise(&self) -> bool {
        <F as ComputeFnImpl>::is_elementwise()
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
