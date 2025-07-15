//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod filter;
mod pruning;

use std::collections::BTreeSet;
use std::future::ready;
use std::ops::Range;
use std::sync::Arc;

use futures::future::{BoxFuture, Shared, try_join_all};
use futures::{FutureExt, TryFutureExt};
use vortex_array::ArrayContext;
use vortex_array::arrays::{BinaryView, VarBinViewArray};
use vortex_array::stats::Precision;
use vortex_array::validity::Validity;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, FieldMask, Nullability};
use vortex_error::{SharedVortexResult, VortexExpect, VortexResult, vortex_bail};
use vortex_expr::{ExprRef, LikeVTable, root};
use vortex_mask::{AllOr, Mask};
use vortex_utils::aliases::hash_set::HashSet;

use crate::layouts::SharedByteBufferFuture;
use crate::layouts::view::{ValidityTag, ViewLayout};
use crate::segments::SegmentSource;
use crate::{
    ArrayEvaluation, LayoutChildren, LayoutReader, LazyReaderChildren, MaskEvaluation,
    NoOpPruningEvaluation, PruningEvaluation,
};

type SharedBinaryViewFuture = Shared<BoxFuture<'static, SharedVortexResult<Buffer<BinaryView>>>>;

/// Scan node for reading arrays out of a `ViewLayout`.
///
/// This node implements the pruning, filtering and projecting tasks. Pruning is able to pushdown
/// certain string operations as a scan over the views buffer, without needing to materialize the
/// string buffers eagerly.
#[allow(unused)]
pub struct ViewReader {
    pub(super) layout: ViewLayout,
    pub(super) name: Arc<str>,
    pub(super) children: Option<LazyReaderChildren>,
    pub(super) views: SharedBinaryViewFuture,
    pub(super) buffers: Arc<[SharedByteBufferFuture]>,
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
        let children = (layout.children.nchildren() > 0)
            .then(|| LazyReaderChildren::new(layout.children.clone(), segment_source.clone()));

        // Prefetch the views buffer
        let views: SharedBinaryViewFuture = segment_source
            .request(layout.views, &name)
            .map_ok(Buffer::<BinaryView>::from_byte_buffer)
            .map_err(Arc::new)
            .boxed()
            .shared();

        // Prefetch all of the data buffers.
        // Depending on the pruning/filter stages, we may not need all of them.
        let buffers: Arc<[SharedByteBufferFuture]> = layout
            .buffers
            .iter()
            .map(|&segment| {
                segment_source
                    .request(segment, &name)
                    .map_err(Arc::new)
                    .boxed()
                    .shared()
            })
            .collect();

        Self {
            layout,
            name,
            children,
            views,
            buffers,
            segment_source,
            ctx,
        }
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
        let validity = if let Some(ref children) = self.children {
            let validity_name: Arc<str> = format!("{}.validity", self.name()).into();
            let validity_reader =
                children.get(0, &DType::Bool(Nullability::NonNullable), &validity_name)?;
            let validity_eval = validity_reader.projection_evaluation(row_range, &root())?;
            Some(validity_eval)
        } else {
            None
        };

        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        Ok(Box::new(ViewEvaluation {
            row_range,
            validity,
            validity_tag: self.layout.validity_tag,
            name: self.name.clone(),
            expr: expr.clone(),
            dtype: self.dtype().clone(),
            views: self.views.clone(),
            buffers: self.buffers.clone(),
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        let validity = if let Some(ref children) = self.children {
            let validity_name: Arc<str> = format!("{}.validity", self.name()).into();
            let validity_reader =
                children.get(0, &DType::Bool(Nullability::NonNullable), &validity_name)?;
            let validity_eval = validity_reader.projection_evaluation(row_range, &root())?;
            Some(validity_eval)
        } else {
            None
        };

        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;

        Ok(Box::new(ViewEvaluation {
            row_range,
            validity,
            validity_tag: self.layout.validity_tag,
            name: self.name.clone(),
            expr: expr.clone(),
            dtype: self.dtype().clone(),
            views: self.views.clone(),
            buffers: self.buffers.clone(),
        }))
    }
}

/// Filter execution for ViewLayout.
///
/// Filter evaluation is only needed using a mask over the views buffer, and then the
/// string buffers can be deserialized independently.
pub(crate) struct ViewEvaluation {
    pub(crate) row_range: Range<usize>,
    pub(crate) name: Arc<str>,
    pub(crate) dtype: DType,
    pub(crate) validity_tag: ValidityTag,
    pub(crate) expr: ExprRef,
    pub(crate) views: SharedBinaryViewFuture,
    pub(crate) buffers: Arc<[SharedByteBufferFuture]>,

    // Evaluations for different filter types.
    // In our case: non-nullable.
    pub(crate) validity: Option<Box<dyn ArrayEvaluation>>,
}

impl ViewEvaluation {
    async fn build_validity(&self, mask: &Mask) -> VortexResult<Validity> {
        match self.validity_tag {
            ValidityTag::NonNullable => Ok(Validity::NonNullable),
            ValidityTag::AllValid => Ok(Validity::AllValid),
            ValidityTag::AllInvalid => Ok(Validity::AllInvalid),
            ValidityTag::Array => {
                let Some(validity) = &self.validity else {
                    vortex_bail!("Validity child expected but not present");
                };
                let array = validity.invoke(mask.clone()).await?;
                if let Some(all) = array.as_constant() {
                    if all
                        .as_bool()
                        .value()
                        .vortex_expect("validity must be non-null bool")
                    {
                        Ok(Validity::AllValid)
                    } else {
                        Ok(Validity::AllInvalid)
                    }
                } else {
                    Ok(Validity::Array(array))
                }
            }
        }
    }

    #[allow(clippy::use_debug)]
    async fn build_array(&self, mask: &Mask) -> VortexResult<VarBinViewArray> {
        let mut views_buffer = self.views.clone().await?;
        if self.row_range.start > 0 || self.row_range.end < views_buffer.len() {
            // Slice the views buffer down to this portion of the split
            views_buffer = views_buffer.slice(self.row_range.start..self.row_range.end);
        }

        // Make a set of all of the buffers that we know we need.
        let mut required_buffers = HashSet::<u32>::new();
        match mask.slices() {
            AllOr::All => {
                for &view in views_buffer.iter() {
                    if !view.is_inlined() {
                        required_buffers.insert(view.as_view().buffer_index());
                    }
                }
            }
            // Check only the sliced elements
            AllOr::Some(slices) => {
                for &(start, end) in slices {
                    for &view in &views_buffer[start..end] {
                        if !view.is_inlined() {
                            required_buffers.insert(view.as_view().buffer_index());
                        }
                    }
                }
            }
            // No buffers needed
            AllOr::None => {}
        }

        // Force each required buffer to be loaded before executing the filter.
        println!(
            "VIEW_FILTER.invoke: required buffers: {:?}",
            required_buffers
        );

        let buffer_count = required_buffers.iter().copied().max().unwrap_or(0);

        // Fetch all of the buffers needed to execute the filter operation.
        let mut resolved_buffers = Vec::new();

        for i in 0..buffer_count {
            let idx = i;
            if required_buffers.contains(&idx) {
                resolved_buffers.push(self.buffers[idx as usize].clone());
            } else {
                resolved_buffers.push(ready(Ok(ByteBuffer::empty())).boxed().shared());
            }
        }

        // Poll all of the buffers together, allowing their IO to execute concurrently.
        let resolved_buffers = try_join_all(resolved_buffers).await?;

        let validity = self.build_validity(mask).await?;
        println!("VALIDITY: {validity:?}");

        VarBinViewArray::try_new(views_buffer, resolved_buffers, self.dtype.clone(), validity)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use futures::stream::once;
    use vortex_array::arrays::{BinaryView, VarBinViewArray};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayContext, IntoArray};
    use vortex_buffer::{ByteBuffer, buffer};
    use vortex_dtype::{DType, Nullability};
    use vortex_expr::root;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::view::writer::ViewStrategy;
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::strategy::SequentialStreamExt;
    use crate::{LayoutRef, LayoutStrategy, SequentialStreamAdapter};

    fn view_layout() -> (LayoutRef, Arc<dyn SegmentSource>) {
        let ctx = ArrayContext::empty();
        let strategy = ViewStrategy {
            validity_strategy: Arc::new(FlatLayoutStrategy::default()),
            fallback_strategy: Arc::new(FlatLayoutStrategy::default()),
        };

        let segment_source = TestSegments::default();
        let writer = Box::new(segment_source.clone());
        let writer = SequenceWriter::new(writer);
        let mut sequence_id = SequenceId::root();

        let stream = SequentialStreamAdapter::new(
            DType::Utf8(Nullability::Nullable),
            once(async move {
                // Hand-roll some views and some buffers that are a mix of inlined and
                // outlined strings.
                let views = buffer![
                    BinaryView::new_view(b"long string with its own buffer 0", 0, 0),
                    BinaryView::new_inlined(b"inlined 1"),
                    BinaryView::new_view(b"long string with its own buffer 1", 1, 0),
                    BinaryView::new_inlined(b"inlined 2"),
                    BinaryView::new_view(b"long string with its own buffer 2", 2, 0),
                    BinaryView::new_inlined(b"inlined 3"),
                ];

                let buffers = vec![
                    ByteBuffer::from(b"long string with its own buffer 0".to_vec()),
                    ByteBuffer::from(b"long string with its own buffer 1".to_vec()),
                    ByteBuffer::from(b"long string with its own buffer 2".to_vec()),
                ];

                let array = VarBinViewArray::try_new(
                    views,
                    buffers,
                    DType::Utf8(Nullability::Nullable),
                    Validity::AllValid,
                )
                .unwrap()
                .into_array();

                Ok((sequence_id.advance(), array))
            }),
        )
        .sendable();

        let layout = block_on(strategy.write_stream(&ctx, writer, stream)).unwrap();
        (layout, Arc::new(segment_source))
    }

    #[tokio::test]
    async fn test_read_no_buffers() {
        let (layout, segment_source) = view_layout();

        let reader = layout
            .new_reader("test_reader".into(), segment_source)
            .unwrap();

        let row_range = 0u64..6u64;

        // Get the current value.
        let project = reader.projection_evaluation(&row_range, &root()).unwrap();

        // Project with a mask that will only match the inlined strings.
        let result = project
            .invoke(Mask::from_iter([false, true, false, true, false, true]))
            .await
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(
            result.scalar_at(0).unwrap(),
            Scalar::utf8("inlined 1", Nullability::Nullable)
        );
        assert_eq!(
            result.scalar_at(1).unwrap(),
            Scalar::utf8("inlined 2", Nullability::Nullable)
        );
        assert_eq!(
            result.scalar_at(2).unwrap(),
            Scalar::utf8("inlined 3", Nullability::Nullable)
        );
    }
}
