//! Utilities to work with `Arrow` data and types.

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

mod array;
mod datum;
mod dtype;
mod record_batch;
pub use datum::*;

pub trait FromArrowArray<A> {
    fn from_arrow(array: A, nullable: bool) -> Self;
}

pub trait IntoArrowArray {
    fn into_arrow_preferred(self) -> VortexResult<ArrayRef>;

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrayRef>;
}
