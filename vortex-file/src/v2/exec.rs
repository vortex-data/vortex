use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use vortex_array::ArrayData;
use vortex_error::VortexResult;

/// An execution driver is used to drive the execution of the scan operation.
pub trait ExecDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<ArrayData>>>,
    ) -> BoxStream<'static, VortexResult<ArrayData>>;
}

/// An execution driver that awaits the futures inline using the caller's runtime.
pub struct InlineDriver;

impl ExecDriver for InlineDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<ArrayData>>>,
    ) -> BoxStream<'static, VortexResult<ArrayData>> {
        stream.then(|future| future).boxed()
    }
}
