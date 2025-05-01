//! Utilities to work with `Arrow` data and types.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

mod array;
pub mod compute;
mod convert;
mod datum;
mod record_batch;

pub use datum::*;

use crate::arrow::compute::ToArrowOptions;

pub trait FromArrowArray<A> {
    fn from_arrow(array: A, nullable: bool) -> Self;
}

pub trait IntoArrowArray {
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef>;

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef>;
}

impl IntoArrowArray for crate::ArrayRef {
    /// Convert this [`crate::ArrayRef`] into an Arrow [`crate::ArrayRef`] by using the array's preferred
    /// Arrow [`DataType`].
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef> {
        compute::to_arrow_opts(&self, &ToArrowOptions { arrow_type: None })
    }

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef> {
        compute::to_arrow_opts(
            &self,
            &ToArrowOptions {
                arrow_type: Some(data_type.clone()),
            },
        )
    }
}
