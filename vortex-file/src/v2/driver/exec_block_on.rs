use futures_executor::block_on;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::v2::driver::ExecDriver;

/// An execution driver that runs the futures to completion in the current thread.
pub struct BlockOnDriver;

impl ExecDriver for BlockOnDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<ArrayData>>>,
    ) -> BoxStream<'static, VortexResult<ArrayData>> {
        stream.map(|future| block_on(future)).boxed()
    }
}
