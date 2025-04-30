use std::any::Any;
use std::sync::LazyLock;

use arrow::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};

use crate::arcref::ArcRef;
use crate::arrow::array::ArrowArray;
use crate::compute::{
    BetweenKernelRef, BetweenOptions, ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options,
    Output,
};
use crate::{Array, ArrayRef, Encoding, arrow};

/// Converts a Vortex array to an Arrow array.
pub fn to_arrow(array: &dyn Array, arrow_type: &DataType) -> VortexResult<ArrowArrayRef> {
    let arrow = TO_ARROW_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options: &ToArrowOptions {
                arrow_type: arrow_type.clone(),
            },
        })?
        .unwrap_array()?
        .as_any()
        .downcast_ref::<ArrowArray>()
        .ok_or_else(|| vortex_err!("ToArrow compute kernels must return a Vortex ArrowArray"))?
        .inner()
        .clone();

    if arrow.data_type() != arrow_type {
        vortex_bail!(
            "Arrow array type mismatch: expected {:?}, got {:?}",
            arrow_type,
            arrow.data_type()
        );
    }

    Ok(arrow)
}

struct ToArrowOptions {
    /// The Arrow data type to convert to.
    arrow_type: DataType,
}

impl Options for ToArrowOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct ToArrow;

impl ComputeFnVTable for ToArrow {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        todo!()
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        todo!()
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        todo!()
    }

    fn is_elementwise(&self) -> bool {
        todo!()
    }
}

pub static TO_ARROW_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("to_arrow".into(), ArcRef::new_ref(&ToArrow));
    for kernel in inventory::iter::<ToArrowKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct ToArrowArgs<'a> {
    array: &'a dyn Array,
    arrow_type: &'a DataType,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for ToArrowArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        todo!()
    }
}

pub struct ToArrowKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(ToArrowKernelRef);

pub trait ToArrowKernel: Encoding {
    fn to_arrow(
        &self,
        arr: &Self::Array,
        arrow_type: &DataType,
    ) -> VortexResult<Option<ArrowArrayRef>>;
}

#[derive(Debug)]
pub struct ToArrowKernelAdapter<E: Encoding>(pub E);

impl<E: Encoding + ToArrowKernel> ToArrowKernelAdapter<E> {
    pub const fn lift(&'static self) -> ToArrowKernelRef {
        ToArrowKernelRef(ArcRef::new_ref(self))
    }
}

impl<E: Encoding + ToArrowKernel> Kernel for ToArrowKernelAdapter<E> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = ToArrowArgs::try_from(args)?;
        let Some(array) = inputs.array.as_any().downcast_ref::<E::Array>() else {
            return Ok(None);
        };

        let Some(arrow_array) = E::to_arrow(&self.0, array, inputs.arrow_type)? else {
            return Ok(None);
        }

        Ok(Some(ArrowArray::new(arrow_array, array.dtype().clone()).to_array().into()))
    }
}
