//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod filter;
mod pruning;

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use vortex_array::ArrayContext;
use vortex_array::arrays::BinaryView;
use vortex_array::stats::Precision;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{SharedVortexResult, VortexResult};
use vortex_expr::{ExprRef, LikeVTable};

use crate::layouts::view::ViewLayout;
use crate::layouts::view::reader::array::ViewProjection;
use crate::layouts::view::reader::filter::ViewFilter;
use crate::segments::{SegmentId, SegmentSource};
use crate::{
    ArrayEvaluation, LayoutReader, LazyReaderChildren, MaskEvaluation, NoOpPruningEvaluation,
    PruningEvaluation,
};

type BinaryViewFuture = Shared<BoxFuture<'static, SharedVortexResult<Buffer<BinaryView>>>>;
type BufferFuture = Shared<BoxFuture<'static, SharedVortexResult<ByteBuffer>>>;

/// Handle for fetching a data buffer by index.
#[derive(Clone)]
pub(crate) struct FetchBuffers {
    name: Arc<str>,
    segment_source: Arc<dyn SegmentSource>,
    buffer_ids: Arc<[SegmentId]>,
    buffers: Vec<OnceLock<BufferFuture>>,
}

impl FetchBuffers {
    pub fn new(
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        buffer_ids: Arc<[SegmentId]>,
    ) -> Self {
        let buffers = vec![OnceLock::new(); buffer_ids.len()];

        Self {
            name,
            segment_source,
            buffer_ids,
            buffers,
        }
    }

    // For each segment, we can initialize the first segment ID that needs to trigger it to run.
    pub fn fetch_buffer(&self, index: usize) -> BufferFuture {
        let segment_fut = self
            .segment_source
            .request(self.buffer_ids[index], &self.name);

        self.buffers[index]
            .get_or_init(|| {
                segment_fut
                    .map(|result| result.map_err(Arc::new))
                    .boxed()
                    .shared()
            })
            .clone()
    }
}

/// Scan node for extracting arrays out of a `ViewLayout`.
///
/// The node implements the pruning, filtering and projecting tasks. Pruning is able to pushdown
/// certain string operations as a scan over the views buffer, without needing to materialize the
/// string buffers eagerly.
#[allow(unused)]
pub struct ViewReader {
    pub(super) layout: ViewLayout,
    pub(super) name: Arc<str>,
    pub(super) children: LazyReaderChildren,
    pub(super) views: OnceLock<BinaryViewFuture>,
    pub(super) fetch_buffers: FetchBuffers,
    pub(super) segment_source: Arc<dyn SegmentSource>,
    pub(super) ctx: ArrayContext,
}

impl ViewReader {
    pub fn new(
        layout: ViewLayout,
        name: impl Into<Arc<str>>,
        segment_source: Arc<dyn SegmentSource>,
        ctx: ArrayContext,
    ) -> Self {
        let name = name.into();
        let children = LazyReaderChildren::new(layout.children.clone(), segment_source.clone());

        let fetch_buffers = FetchBuffers::new(
            name.clone(),
            segment_source.clone(),
            layout.segment_ids().into(),
        );

        Self {
            layout,
            children,
            segment_source,
            ctx,
            fetch_buffers,
            name,
            views: OnceLock::new(),
        }
    }
}
impl ViewReader {
    /// Get a handle to the future for loading the views, initializing it if it's not already
    /// been defined.
    fn views_future(&self) -> VortexResult<BinaryViewFuture> {
        // Eagerly register the request for the segment. This is important for prefetching.
        let segment_fut = self.segment_source.request(self.layout.views, &self.name);

        Ok(self
            .views
            .get_or_init(|| {
                segment_fut
                    .map(|result| {
                        // Reinterpret the ByteBuffer to a Buffer<BinaryView>
                        // The segment fetcher should load it into a properly aligned buffer, so this
                        // should not result in an extra copy.
                        result
                            .map(Buffer::<BinaryView>::from_byte_buffer)
                            .map_err(Arc::new)
                    })
                    .boxed()
                    .shared()
                    .boxed()
                    .shared()
            })
            .clone())
    }
}

impl LayoutReader for ViewReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.layout.row_count)
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_offset + self.layout.row_count);
        Ok(())
    }

    #[allow(clippy::dbg_macro)]
    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        // Attempt to prune if top-level is `LIKE` or `<>`
        if expr.is::<LikeVTable>() {
            let like_expr = expr.as_::<LikeVTable>();
            dbg!(like_expr);
            dbg!(like_expr.pattern());
        };

        Ok(Box::new(NoOpPruningEvaluation))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;

        Ok(Box::new(ViewFilter {
            row_range,
            name: self.name.clone(),
            expr: expr.clone(),
            dtype: self.dtype().clone(),
            views: self.views_future()?,
            fetch_buffers: self.fetch_buffers.clone(),
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        Ok(Box::new(ViewProjection {
            row_range: row_range.clone(),
            expr: expr.clone(),
        }))
    }
}
