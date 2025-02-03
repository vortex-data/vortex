use std::pin::Pin;

pub use adapter::*;
pub use ext::*;
use futures_util::Stream;
pub use take_rows::*;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::Array;

mod adapter;
mod ext;
mod take_rows;

/// A stream of array chunks along with a DType.
///
/// Can be thought of as equivalent to Arrow's RecordBatchReader.
pub trait ArrayStream: Stream<Item = VortexResult<Array>> {
    fn dtype(&self) -> &DType;
}

/// Trait for a [`Stream`] of [`Array`]s that can be passed between threads.
pub type SendableArrayStream = Pin<Box<dyn ArrayStream + Send>>;
