use futures::future::BoxFuture;
use futures::stream::BoxStream;
use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::v2::segments::SegmentRequest;

/// The I/O driver is used to resolve segment requests.
pub trait IoDriver {
    fn drive(&self, stream: BoxStream<'static, SegmentRequest>) -> BoxStream<VortexResult<()>>;
}

/// The execution driver is used to drive the execution of the scan operation.
pub trait ExecDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<ArrayData>>>,
    ) -> BoxStream<VortexResult<ArrayData>>;
}
