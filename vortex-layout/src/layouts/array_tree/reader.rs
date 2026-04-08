// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinView;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::layouts::SharedArrayFuture;
use crate::layouts::array_tree::flat::ArrayTreeFlatLayout;
use crate::reader::ArrayFuture;
use crate::reader::SplitRange;
use crate::segments::SegmentSource;

/// Transparent reader for [`super::ArrayTreeLayout`].
///
/// Delegates all operations to the data child reader. The array_trees auxiliary child
/// is consumed at construction time (by [`super::ArrayTreeLayout::new_reader`]) to inject
/// compact flatbuffers into [`ArrayTreeFlatLayout`] descendants.
pub struct ArrayTreeReader {
    name: Arc<str>,
    data_reader: LayoutReaderRef,
}

impl ArrayTreeReader {
    pub fn new(name: Arc<str>, data_reader: LayoutReaderRef) -> Self {
        Self { name, data_reader }
    }
}

impl LayoutReader for ArrayTreeReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.data_reader.dtype()
    }

    fn row_count(&self) -> u64 {
        self.data_reader.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.data_reader
            .register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        self.data_reader.pruning_evaluation(row_range, expr, mask)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.data_reader.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.data_reader
            .projection_evaluation(row_range, expr, mask)
    }
}

/// The threshold of mask density below which we will evaluate the expression only over the
/// selected rows, and above which we evaluate the expression over all rows and then select after.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Reader for [`ArrayTreeFlatLayout`].
///
/// Similar to [`super::super::flat::reader::FlatReader`] but obtains its compact encoding tree
/// from the shared [`super::ArrayTreesSource`] rather than from inline layout metadata or
/// the data segment.
pub struct ArrayTreeFlatReader {
    layout: ArrayTreeFlatLayout,
    name: Arc<str>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
}

impl ArrayTreeFlatReader {
    pub(crate) fn new(
        layout: ArrayTreeFlatLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> Self {
        Self {
            layout,
            name,
            segment_source,
            session,
        }
    }

    /// Register the segment request and return a future that resolves to the deserialized array.
    ///
    /// If a shared [`super::ArrayTreesSource`] has been injected, the compact flatbuffer is
    /// obtained from there (concurrently with the data segment fetch). Otherwise falls back
    /// to parsing from the segment (like FlatReader).
    fn array_future(&self) -> SharedArrayFuture {
        let row_count = usize::try_from(self.layout.inner().row_count())
            .vortex_expect("row count must fit in usize");

        let segment_fut = self
            .segment_source
            .request(self.layout.inner().segment_id());
        let ctx = self.layout.inner().array_ctx().clone();
        let session = self.session.clone();
        let dtype = self.layout.inner().dtype().clone();

        // If a source has been injected, resolve the compact tree from it.
        // Otherwise, fall back to parsing from the segment (like FlatReader).
        let source_future = self.layout.source().map(|s| s.array_future());
        let chunk_idx = self.layout.chunk_idx();

        async move {
            let segment = segment_fut.await?;
            let parts = if let Some(source_future) = source_future {
                // Resolve the VarBin array of compact trees (shared, read once).
                let trees_array = source_future.await?;
                let trees = trees_array.try_downcast::<VarBinView>().map_err(|_| {
                    Arc::new(vortex_error::vortex_err!(
                        "array_trees child is not a VarBinView array"
                    ))
                })?;
                let compact_tree = trees.bytes_at(chunk_idx);
                SerializedArray::from_flatbuffer_and_segment(compact_tree, segment)?
            } else {
                SerializedArray::try_from(segment)?
            };
            parts
                .decode(&dtype, row_count, &ctx, &session)
                .map_err(Arc::new)
        }
        .boxed()
        .shared()
    }
}

impl LayoutReader for ArrayTreeFlatReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.inner().dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.inner().row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        split_range.check_bounds(self.layout.inner().row_count())?;
        splits.insert(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within layout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within layout size");
        let name = Arc::clone(&self.name);
        let array = self.array_future();
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            let mut array = array.clone().await?;
            let mask = mask.await?;

            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }

            let array_mask = if mask.density() < EXPR_EVAL_THRESHOLD {
                let array = array.apply(&expr)?;
                let array = array.filter(mask.clone())?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.execute::<Mask>(&mut ctx)?;
                mask.intersect_by_rank(&array_mask)
            } else {
                let array = array.apply(&expr)?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.execute::<Mask>(&mut ctx)?;
                mask.bitand(&array_mask)
            };

            tracing::debug!(
                "ArrayTreeFlat mask evaluation {} - {} (mask = {}) => {}",
                name,
                expr,
                mask.density(),
                array_mask.density(),
            );

            Ok(array_mask)
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within layout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within layout size");
        let name = Arc::clone(&self.name);
        let array = self.array_future();
        let expr = expr.clone();

        Ok(async move {
            tracing::debug!("ArrayTreeFlat array evaluation {} - {}", name, expr);

            let mut array = array.clone().await?;
            let mask = mask.await?;

            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }

            if !mask.all_true() {
                array = array.filter(mask)?;
            }

            array = array.apply(&expr)?;
            Ok(array)
        }
        .boxed())
    }
}
