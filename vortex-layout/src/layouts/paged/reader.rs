// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use once_cell::sync::OnceCell;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_error::SharedVortexResult;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_flatbuffers::FlatBuffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::layout_from_flatbuffer;
use crate::layouts::paged::PagedLayout;
use crate::reader::LayoutReader;
use crate::reader::RowSplits;
use crate::reader::SplitRange;
use crate::segments::SegmentSource;

/// A reader for the subtree behind a page, resolved once and shared by every evaluation.
type SharedReaderFuture = Shared<BoxFuture<'static, SharedVortexResult<LayoutReaderRef>>>;

/// A [`LayoutReader`] that fetches and parses its subtree on first use, then delegates to it.
pub struct PagedReader {
    layout: PagedLayout,
    name: Arc<str>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    ctx: LayoutReaderContext,
    child: OnceCell<SharedReaderFuture>,
}

impl PagedReader {
    pub(crate) fn new(
        layout: PagedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> Self {
        Self {
            layout,
            name,
            segment_source,
            session,
            ctx,
            child: OnceCell::new(),
        }
    }

    /// Return a future resolving to the reader for this page's subtree.
    ///
    /// The segment request is only issued once the page is actually evaluated, so a page that is
    /// pruned away, or never intersected by any scanned row range, costs no IO.
    fn child_reader(&self) -> SharedReaderFuture {
        self.child
            .get_or_init(|| {
                let segment_fut = self.segment_source.request(self.layout.segment_id());
                let dtype = self.layout.dtype().clone();
                let layout_ctx = self.layout.layout_ctx().clone();
                let array_ctx = self.layout.array_ctx().clone();
                let segment_source = Arc::clone(&self.segment_source);
                let session = self.session.clone();
                let reader_ctx = self.ctx.clone();
                let name = Arc::clone(&self.name);

                async move {
                    let reader = async {
                        let segment = segment_fut.await?;
                        // The page is a `Layout` flatbuffer root, verified in its own right.
                        let page = FlatBuffer::align_from(segment.to_host_sync());
                        let layout = layout_from_flatbuffer(
                            page,
                            &dtype,
                            &layout_ctx,
                            &array_ctx,
                            &session,
                        )?;
                        layout.new_reader(name, segment_source, &session, &reader_ctx)
                    }
                    .await;
                    reader.map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone()
    }
}

impl LayoutReader for PagedReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // Planning is synchronous and must not read the page, so the boundaries come from the
        // row offsets the writer promoted into the page's metadata.
        split_range.check_bounds(self.layout.row_count())?;

        let row_range = split_range.row_range();
        let boundaries = self.layout.boundaries();
        splits.reserve(boundaries.len());
        for offset in boundaries.offsets() {
            // The range's own end is registered below; anything outside it is another split's.
            if offset > row_range.start && offset < row_range.end {
                splits.push(
                    split_range
                        .row_offset()
                        .checked_add(offset)
                        .vortex_expect("Paged layout split offset overflow"),
                );
            }
        }
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let child = self.child_reader();
        let row_range = row_range.clone();
        let expr = expr.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            let child = child.await?;
            child.pruning_evaluation(&row_range, &expr, mask)?.await
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let child = self.child_reader();
        let row_range = row_range.clone();
        let expr = expr.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            let child = child.await?;
            child.filter_evaluation(&row_range, &expr, mask)?.await
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let child = self.child_reader();
        let row_range = row_range.clone();
        let expr = expr.clone();

        Ok(async move {
            let child = child.await?;
            child.projection_evaluation(&row_range, &expr, mask)?.await
        }
        .boxed())
    }
}
