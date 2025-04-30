use std::any::Any;
use std::sync::LazyLock;

use arrow::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::arcref::ArcRef;
use crate::arrow::array::ArrowArray;
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options, Output};
use crate::{Array, arrow};

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

pub static TO_ARROW_FN: LazyLock<ComputeFn> = LazyLock::new(|| todo!());
