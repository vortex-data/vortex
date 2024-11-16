//! Utilities to work with `Arrow` data and types.

use vortex_error::VortexResult;

pub use crate::arrow::dtype::{infer_data_type, infer_schema};

mod array;
mod datum;
mod dtype;
mod recordbatch;
pub mod wrappers;
pub use datum::*;

pub trait FromArrowArray<A> {
    fn from_arrow(array: A, nullable: bool) -> Self;
}

pub trait FromArrowType<T>: Sized {
    fn from_arrow(value: T) -> Self;
}

pub trait TryFromArrowType<T>: Sized {
    fn try_from_arrow(value: T) -> VortexResult<Self>;
}
