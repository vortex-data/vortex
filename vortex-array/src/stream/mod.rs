pub use adapter::*;
pub use ext::*;
use futures_util::Stream;
pub use take_rows::*;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::ArrayData;

mod adapter;
mod ext;
mod take_rows;

/// A stream of array chunks along with a DType.
///
/// Can be thought of as equivalent to Arrow's RecordBatchReader.
pub trait ArrayStream: Stream<Item = VortexResult<ArrayData>> {
    fn dtype(&self) -> &DType;
}
