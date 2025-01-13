use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use vortex_array::ArrayData;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::v2::driver::{ExecutionDriver, IoDriver};

/// The [`ExecutionMode`] describes how the CPU-bound layout evaluation tasks are executed.
/// Typically, there is one task per file split (row-group).
pub enum ExecutionMode {
    /// Executes the tasks inline as part of polling the returned
    /// [`vortex_array::stream::ArrayStream`].
    Inline,
    /// Spawns the tasks onto a provided Rayon thread pool.
    // TODO(ngates): feature-flag this dependency.
    RayonThreadPool(Arc<rayon::ThreadPool>),
    /// Spawns the tasks onto a provided Tokio runtime.
    // TODO(ngates): feature-flag this dependency.
    TokioRuntime(Arc<tokio::runtime::Handle>),
}

impl ExecutionMode {
    pub fn new_driver(&self, segments: Arc<dyn IoDriver>) -> Arc<dyn ExecutionDriver> {
        match self {
            ExecutionMode::Inline => {}
            ExecutionMode::RayonThreadPool(_) => {}
            ExecutionMode::TokioRuntime(_) => {}
        }
    }
}

struct InlineDriver(Arc<dyn IoDriver>);

impl ExecutionDriver for InlineDriver {
    fn drive(
        &self,
        evaluation: &dyn FnOnce(
            Arc<dyn AsyncSegmentReader>,
        )
            -> BoxStream<'static, BoxFuture<'static, VortexResult<ArrayData>>>,
    ) -> BoxStream<VortexResult<ArrayData>> {
        let stream = evaluation(Arc::new(self));
        todo!()
    }
}

impl AsyncSegmentReader for InlineDriver {
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        todo!()
    }
}
