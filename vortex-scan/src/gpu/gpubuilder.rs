// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::Stream;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_expr::transform::simplify_typed;
use vortex_expr::{ExprRef, root};
use vortex_io::runtime::{BlockingRuntime, Handle};
use vortex_layout::gpu::GpuLayoutReader;
use vortex_layout::{LayoutReader, LayoutReaderRef};

use crate::SplitBy;
use crate::gpu::GpuScan;
use crate::scan_builder::filter_and_projection_masks;

pub struct GpuScanBuilder<A> {
    handle: Option<Handle>,
    layout_reader: LayoutReaderRef,
    projection: ExprRef,
    split_by: SplitBy,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

impl GpuScanBuilder<ArrayRef> {
    pub fn new(layout_reader: Arc<dyn GpuLayoutReader>) -> Self {
        Self {
            handle: Handle::find(),
            layout_reader,
            projection: root(),
            split_by: SplitBy::Layout,
            map_fn: Arc::new(Ok),
        }
    }

    /// Returns an [`ArrayStream`] with tasks spawned onto the scan's [`Handle`].
    ///
    /// See [`ScanBuilder::into_stream`] for more details.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype()?;
        let stream = self.into_stream()?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }

    /// Returns an [`ArrayIterator`] using the given blocking runtime.
    pub fn into_array_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let stream = self.with_handle(runtime.handle()).into_array_stream()?;
        let dtype = stream.dtype().clone();
        Ok(ArrayIteratorAdapter::new(
            dtype,
            runtime.block_on_stream(|_| stream),
        ))
    }
}

impl<A: 'static + Send> GpuScanBuilder<A> {
    /// Provide a handle to the runtime on which to spawn tasks.
    pub fn with_handle(mut self, handle: Handle) -> Self {
        self.handle = Some(handle);
        self
    }

    pub fn with_projection(mut self, projection: ExprRef) -> Self {
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
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> GpuScanBuilder<B> {
        let old_map_fn = self.map_fn;
        GpuScanBuilder {
            handle: self.handle,
            layout_reader: self.layout_reader,
            projection: self.projection,
            split_by: self.split_by,
            map_fn: Arc::new(move |a| old_map_fn(a).and_then(&map_fn)),
        }
    }

    pub fn prepare(self) -> VortexResult<GpuScan<A>> {
        let dtype = self.dtype()?;

        let Some(handle) = self.handle else {
            vortex_bail!(
                "A runtime handle must be provided to the scan builder using `with_handle`"
            );
        };
        // Spin up the root layout reader, and wrap it in a FilterLayoutReader to perform
        // conjunction splitting if a filter is provided.
        let mut layout_reader = self.layout_reader;

        // Normalize and simplify the expressions.
        let projection = simplify_typed(self.projection, layout_reader.dtype())?;

        // Construct field masks and compute the row splits of the scan.
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, None, layout_reader.dtype())?;
        let field_mask: Vec<_> = [filter_mask, projection_mask].concat();

        let splits = self.split_by.splits(layout_reader.as_ref(), &field_mask)?;

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
    pub fn build(self) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        self.prepare()?.execute()
    }

    /// Returns a [`Stream`] with tasks spawned onto the scan's [`Handle`].
    pub fn into_stream(
        self,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        self.prepare()?.execute_stream(None)
    }

    /// Returns an [`Iterator`] using the given blocking runtime.
    pub fn into_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl Iterator<Item = VortexResult<A>> + 'static> {
        let stream = self.with_handle(runtime.handle()).into_stream()?;
        Ok(runtime.block_on_stream(|_| stream))
    }
}
