mod exec_block_on;
mod io_file;

pub use exec_block_on::*;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
pub use io_file::*;
use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::v2::segments::SegmentRequest;

/// The I/O driver is used to resolve segment requests.
pub trait IoDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, SegmentRequest>,
    ) -> BoxStream<'static, VortexResult<()>>;
}

/// The execution driver is used to drive the execution of the scan operation.
pub trait ExecDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<ArrayData>>>,
    ) -> BoxStream<'static, VortexResult<ArrayData>>;
}
