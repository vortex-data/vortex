pub mod inline;
pub mod tokio;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use vortex_array::ArrayData;
use vortex_error::VortexResult;

/// An execution driver is used to drive the execution of the scan operation.
pub trait ExecDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<ArrayData>>>,
    ) -> BoxStream<'static, VortexResult<ArrayData>>;
}
