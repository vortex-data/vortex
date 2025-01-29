//! Utilities to work with `Arrow` data and types.

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

pub use crate::arrow::dtype::{infer_data_type, infer_schema};

mod array;
mod datum;
mod dtype;
mod record_batch;
pub use datum::*;

use crate::compute::to_arrow;
use crate::ArrayData;

pub trait FromArrowArray<A> {
    fn from_arrow(array: A, nullable: bool) -> Self;
}

pub trait FromArrowType<T>: Sized {
    fn from_arrow(value: T) -> Self;
}

pub trait TryFromArrowType<T>: Sized {
    fn try_from_arrow(value: T) -> VortexResult<Self>;
}

pub trait IntoArrowArray {
    fn into_arrow_preferred(self) -> VortexResult<ArrayRef>;

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrayRef>;
}
