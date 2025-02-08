mod mode;
pub use mode::*;
pub mod inline;

#[cfg(feature = "tokio")]
pub mod tokio;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use vortex_error::VortexResult;

/// An execution driver is used to drive the execution of the scan operation.
///
/// It is passed a stream of futures that (typically) process a single split of the file.
/// Drivers are able to control the concurrency of the execution with [`futures::StreamExt::buffered`],
/// as well as _where_ the futures are executed by spawning them onto a specific runtime or thread
/// pool.
pub trait ExecDriver: Send + Sync {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<()>>>,
    ) -> BoxStream<'static, VortexResult<()>>;
}
