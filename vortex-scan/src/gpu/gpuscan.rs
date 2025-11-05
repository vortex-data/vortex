// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::sync::Arc;

use futures::{Stream, TryStreamExt};
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::Expression;
use vortex_gpu::GpuVector;
use vortex_io::runtime::{BlockingRuntime, Handle};
use vortex_layout::GpuLayoutReaderRef;

use crate::gpu::gputask::{GpuTaskContext, TaskFuture, gpu_split_exec};

pub struct GpuScan<A: 'static + Send> {
    handle: Handle,
    layout_reader: GpuLayoutReaderRef,
    projection: Expression,
    splits: BTreeSet<u64>,
    map_fn: Arc<dyn Fn(Vec<GpuVector>) -> VortexResult<Vec<A>> + Send + Sync>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

impl GpuScan<ArrayRef> {
    pub fn execute_array_iter<B: BlockingRuntime>(
        &self,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream()?;
        let iter = runtime.block_on_stream(stream);
        Ok(ArrayIteratorAdapter::new(dtype, iter))
    }

    pub fn execute_array_stream(&self) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream()?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }
}

impl<A: 'static + Send> GpuScan<A> {
    pub(super) fn new(
        handle: Handle,
        layout_reader: GpuLayoutReaderRef,
        projection: Expression,
        splits: BTreeSet<u64>,
        map_fn: Arc<dyn Fn(Vec<GpuVector>) -> VortexResult<Vec<A>> + Send + Sync>,
        dtype: DType,
    ) -> Self {
        Self {
            handle,
            layout_reader,
            projection,
            splits,
            map_fn,
            dtype,
        }
    }

    pub fn execute(&self) -> VortexResult<Vec<TaskFuture<Option<Vec<A>>>>> {
        let ctx = Arc::new(GpuTaskContext {
            reader: self.layout_reader.clone(),
            projection: self.projection.clone(),
            mapper: self.map_fn.clone(),
        });

        let ranges = self
            .splits
            .iter()
            .copied()
            .tuple_windows()
            .map(|(start, end)| start..end);

        ranges
            .map(|range| gpu_split_exec(ctx.clone(), range))
            .try_collect()
    }

    pub fn execute_stream(
        &self,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        use futures::StreamExt;
        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let concurrency = num_workers;
        let handle = self.handle.clone();

        let stream = futures::stream::iter(self.execute()?).map(move |task| handle.spawn(task));

        let stream = stream.buffered(concurrency).boxed();

        Ok(stream
            .filter_map(|chunk| async move { chunk.transpose() })
            .map(|v| v.map(|a| futures::stream::iter(a.into_iter().map(Ok))))
            .try_flatten())
    }
}
