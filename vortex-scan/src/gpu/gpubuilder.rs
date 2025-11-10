// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::sync::Arc;

use futures::Stream;
use vortex_array::expr::transform::simplify_typed;
use vortex_array::expr::{Expression, root};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_gpu::GpuVector;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::gpu::GpuLayoutReaderRef;
use vortex_session::VortexSession;

use crate::gpu::GpuScan;
use crate::gpu::gputask::TaskFuture;
use crate::scan_builder::filter_and_projection_masks;

pub struct GpuScanBuilder<A> {
    session: VortexSession,
    layout_reader: GpuLayoutReaderRef,
    projection: Expression,
    map_fn: Arc<dyn Fn(Vec<GpuVector>) -> VortexResult<Vec<A>> + Send + Sync>,
}

impl GpuScanBuilder<GpuVector> {
    pub fn new(session: VortexSession, layout_reader: GpuLayoutReaderRef) -> Self {
        Self {
            session,
            layout_reader,
            projection: root(),
            map_fn: Arc::new(Ok),
        }
    }

    /// Returns an [`ArrayStream`] with tasks spawned onto the session's runtime handle.
    ///
    /// See [`ScanBuilder::into_stream`] for more details.
    pub fn into_array_stream(
        self,
    ) -> VortexResult<impl Stream<Item = VortexResult<GpuVector>> + Send + 'static> {
        self.into_stream()
    }

    /// Returns an [`ArrayIterator`] using the given blocking runtime.
    pub fn into_array_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl Iterator<Item = VortexResult<GpuVector>> + 'static> {
        let stream = self.into_array_stream()?;
        Ok(runtime.block_on_stream(stream))
    }
}

impl<A: 'static + Send> GpuScanBuilder<A> {
    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = projection;
        self
    }

    /// The [`DType`] returned by the scan, after applying the projection.
    pub fn dtype(&self) -> VortexResult<DType> {
        self.projection.return_dtype(self.layout_reader.dtype())
    }

    /// Map each split of the scan. The function will be run on the spawned task.
    pub fn map<B: 'static>(
        self,
        map_fn: impl Fn(Vec<A>) -> VortexResult<Vec<B>> + 'static + Send + Sync,
    ) -> GpuScanBuilder<B> {
        let old_map_fn = self.map_fn;
        GpuScanBuilder {
            session: self.session,
            layout_reader: self.layout_reader,
            projection: self.projection,
            map_fn: Arc::new(move |a| old_map_fn(a).and_then(&map_fn)),
        }
    }

    pub fn prepare(self) -> VortexResult<GpuScan<A>> {
        let dtype = self.dtype()?;
        let handle = self.session.handle();

        // Spin up the root layout reader, and wrap it in a FilterLayoutReader to perform
        // conjunction splitting if a filter is provided.
        let layout_reader = self.layout_reader;

        // Normalize and simplify the expressions.
        let projection = simplify_typed(self.projection, layout_reader.dtype())?;

        // Construct field masks and compute the row splits of the scan.
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, None, layout_reader.dtype())?;
        let field_mask: Vec<_> = [filter_mask, projection_mask].concat();

        let mut splits = BTreeSet::<u64>::new();
        splits.insert(0);

        // Register the splits for all the layouts.
        layout_reader.register_splits(&field_mask, 0, &mut splits)?;

        Ok(GpuScan::new(
            handle,
            layout_reader,
            projection,
            splits,
            self.map_fn,
            dtype,
        ))
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    pub fn build(self) -> VortexResult<Vec<TaskFuture<Option<Vec<A>>>>> {
        self.prepare()?.execute()
    }

    /// Returns a [`Stream`] with tasks spawned onto the session's runtime handle.
    pub fn into_stream(
        self,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        self.prepare()?.execute_stream()
    }

    /// Returns an [`Iterator`] using the handle's runtime.
    pub fn into_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl Iterator<Item = VortexResult<A>> + 'static> {
        let stream = self.into_stream()?;
        Ok(runtime.block_on_stream(stream))
    }
}
