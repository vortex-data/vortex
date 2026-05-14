// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`Scan`] — entrypoint for the LayoutPlan v2 path.
//!
//! [`Scan::build`] turns a layout, projection, and selection into a
//! [`LayoutPlanRef`] tree by recursing through `Layout::plan`. The
//! initial cut handles the projection-only case (no filter); filter
//! conjuncts and `FilterPlan` arrive in a later PR — see
//! `LAYOUT_PLAN.md` § Scan construction.

use std::sync::Arc;

use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::segments::SegmentSource;
use crate::v2::demand::RowDemand;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PlanArguments;
use crate::v2::plan::PlanContext;

/// Scan request for the LayoutPlan v2 path. Mirrors the inputs to
/// `vortex_layout::scan::scan_builder::ScanBuilder` but produces a
/// [`LayoutPlanRef`] rather than driving execution itself.
///
/// `Scan` always builds a plan covering the layout's full row space.
/// Engines that want to read only part of the file pick the relevant
/// partitions out of the resulting plan — see `LAYOUT_PLAN.md`
/// § Partial scans.
pub struct Scan {
    /// Root layout being scanned.
    pub layout: LayoutRef,
    /// Source of segment bytes for the file under scan.
    pub segment_source: Arc<dyn SegmentSource>,
    /// Session used at plan-construction time.
    pub session: VortexSession,
    /// Projection expression — defaults to `root()` for "all columns".
    pub projection: Expression,
    /// Optional filter. Today only `None` is supported by [`Scan::build`].
    pub filter: Option<Expression>,
    /// Pre-mask selection over the layout's row space.
    pub selection: Selection,
}

impl Scan {
    /// Build the layout plan tree for this scan.
    ///
    /// Projection-only is the only supported shape today. A `Some`
    /// filter returns an error so callers (e.g. the DataFusion opener)
    /// can fall back to the v1 [`crate::scan::scan_builder::ScanBuilder`]
    /// path.
    pub fn build(&self) -> VortexResult<LayoutPlanRef> {
        if self.filter.is_some() {
            vortex_bail!(
                "Scan::build does not yet support filter expressions; \
                 fall back to the v1 ScanBuilder path"
            );
        }

        let demand = Arc::new(RowDemand::empty());
        let ctx = PlanContext {
            demand,
            segment_source: Arc::clone(&self.segment_source),
            session: self.session.clone(),
        };

        self.layout.plan(PlanArguments {
            selection: self.selection.clone(),
            expr: self.projection.clone(),
            ctx,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use futures::stream;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::expr::root;
    use vortex_array::stream::ArrayStream as _;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_scan::selection::Selection;

    use super::Scan;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::scan::scan_builder::ScanBuilder;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt as _;
    use crate::test::SESSION;

    /// Build a `Chunked(Struct(Flat, Flat))` layout with two chunks.
    /// Returns the segment source, the layout, and the array we wrote.
    fn build_chunked_struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef, ArrayRef) {
        let chunk1 = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2, 3].into_array()),
                ("b", buffer![10i32, 20, 30].into_array()),
            ]
            .as_slice(),
        )
        .unwrap()
        .into_array();
        let chunk2 = StructArray::from_fields(
            [
                ("a", buffer![4i32, 5].into_array()),
                ("b", buffer![40i32, 50].into_array()),
            ]
            .as_slice(),
        )
        .unwrap()
        .into_array();
        let dtype = chunk1.dtype().clone();

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let segments_for_strategy = Arc::<TestSegments>::clone(&segments);
        let strategy = ChunkedLayoutStrategy::new(TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        ));

        let (mut sequence_id, eof) = SequenceId::root().split();
        let chunk1_for_write = chunk1.clone();
        let chunk2_for_write = chunk2.clone();
        let dtype_for_write = dtype.clone();

        let layout = block_on(|handle| async move {
            let session = SESSION.clone().with_handle(handle);
            strategy
                .write_stream(
                    ctx,
                    segments_for_strategy,
                    SequentialStreamAdapter::new(
                        dtype_for_write,
                        stream::iter([
                            Ok((sequence_id.advance(), chunk1_for_write)),
                            Ok((sequence_id.advance(), chunk2_for_write)),
                        ]),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        let combined = ChunkedArray::try_new(vec![chunk1, chunk2], dtype)
            .unwrap()
            .into_array();
        (segments, layout, combined)
    }

    /// Drive the legacy [`ScanBuilder`] path to read the layout into a single array.
    fn read_v1(segments: Arc<dyn SegmentSource>, layout: &LayoutRef) -> VortexResult<ArrayRef> {
        read_v1_with(segments, layout, root())
    }

    fn read_v1_with(
        segments: Arc<dyn SegmentSource>,
        layout: &LayoutRef,
        projection: vortex_array::expr::Expression,
    ) -> VortexResult<ArrayRef> {
        let (chunks, dtype) = block_on(|handle| async move {
            let session = SESSION.clone().with_handle(handle);
            let reader = layout.new_reader("v1".into(), segments, &session)?;
            let stream = ScanBuilder::new(session, reader)
                .with_projection(projection)
                .into_array_stream()?;
            let dtype = stream.dtype().clone();
            let mut stream = stream;
            let mut chunks = Vec::new();
            while let Some(chunk) = stream.next().await {
                chunks.push(chunk?);
            }
            VortexResult::Ok((chunks, dtype))
        })?;
        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }

    /// Drive the v2 [`Scan`] / [`crate::v2::plan::LayoutPlan`] path.
    fn read_v2(segments: Arc<dyn SegmentSource>, layout: &LayoutRef) -> VortexResult<ArrayRef> {
        read_v2_with(segments, layout, root())
    }

    fn read_v2_with(
        segments: Arc<dyn SegmentSource>,
        layout: &LayoutRef,
        projection: vortex_array::expr::Expression,
    ) -> VortexResult<ArrayRef> {
        let layout = Arc::clone(layout);
        let row_count = layout.row_count();
        let (chunks, plan_dtype) = block_on(|handle| async move {
            let session = SESSION.clone().with_handle(handle);
            let scan = Scan {
                layout,
                segment_source: segments,
                session: session.clone(),
                projection,
                filter: None,
                selection: Selection::All,
            };
            let plan = scan.build()?;
            let plan_dtype = plan.schema().clone();

            let mut chunks = Vec::new();
            // Single execute call covering the entire layout's row
            // range. The plan internally handles whatever
            // chunking/slicing it needs.
            let mut stream = plan.execute(0..row_count, &session)?;
            while let Some(chunk) = stream.next().await {
                chunks.push(chunk?);
            }
            VortexResult::Ok((chunks, plan_dtype))
        })?;
        Ok(ChunkedArray::try_new(chunks, plan_dtype)?.into_array())
    }

    /// Build a single-chunk `Flat` layout backed by a primitive array.
    fn build_flat_layout() -> (Arc<dyn SegmentSource>, LayoutRef, ArrayRef) {
        let array = buffer![1i32, 2, 3, 4, 5].into_array();
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let segments_for_strategy = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let array_for_write = array.clone();
        let layout = block_on(|handle| async move {
            let session = SESSION.clone().with_handle(handle);
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments_for_strategy,
                    crate::sequence::SequentialArrayStreamExt::sequenced(
                        array_for_write.to_array_stream(),
                        ptr,
                    ),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();
        (segments, layout, array)
    }

    /// V1 and V2 must produce identical output for a projection-only scan.
    #[test]
    fn diff_v1_v2_projection_only_chunked_struct() -> VortexResult<()> {
        let (segments, layout, expected) = build_chunked_struct_layout();

        let v1 = read_v1(Arc::clone(&segments), &layout)?;
        let v2 = read_v2(Arc::clone(&segments), &layout)?;

        // V1 and V2 must agree.
        assert_arrays_eq!(v1, v2);
        // And both must agree with what we wrote.
        assert_arrays_eq!(v2, expected);

        Ok(())
    }

    /// Same diff but for a single-chunk `Flat` — exercises the bare
    /// `FlatLayout::plan` path with no `Chunked`/`Struct` wrappers.
    #[test]
    fn diff_v1_v2_projection_only_flat() -> VortexResult<()> {
        let (segments, layout, expected) = build_flat_layout();

        let v1 = read_v1(Arc::clone(&segments), &layout)?;
        let v2 = read_v2(Arc::clone(&segments), &layout)?;

        assert_arrays_eq!(v1, v2);
        assert_arrays_eq!(v2, expected);

        Ok(())
    }

    /// Project a single field out of the chunked struct via
    /// `get_item("a", root())`. Exercises `StructLayout::plan`'s
    /// partition-based field routing.
    #[test]
    fn diff_v1_v2_field_projection_chunked_struct() -> VortexResult<()> {
        use vortex_array::expr::get_item;
        let (segments, layout, _) = build_chunked_struct_layout();

        let proj = get_item("a", root());
        let v1 = read_v1_with(Arc::clone(&segments), &layout, proj.clone())?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, proj)?;

        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    /// Project a `pack(get_item, get_item)` re-ordering of struct
    /// fields. Exercises the partition + re-assembly `ProjectPlan`
    /// path on top of `StructLayout::plan`.
    #[test]
    fn diff_v1_v2_pack_projection_chunked_struct() -> VortexResult<()> {
        use vortex_array::dtype::Nullability;
        use vortex_array::expr::get_item;
        use vortex_array::expr::pack;
        let (segments, layout, _) = build_chunked_struct_layout();

        let proj = pack(
            [
                ("b_out", get_item("b", root())),
                ("a_out", get_item("a", root())),
            ],
            Nullability::NonNullable,
        );
        let v1 = read_v1_with(Arc::clone(&segments), &layout, proj.clone())?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, proj)?;

        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    /// Build a `Chunked(Dict(values=Flat, codes=Chunked(Flat)))` layout
    /// — strings with dictionary compression. Exercises
    /// `DictLayout::plan` + `DictDecodePlan`.
    fn build_dict_chunked_layout() -> (Arc<dyn SegmentSource>, LayoutRef, ArrayRef) {
        use vortex_array::arrays::VarBinArray;
        use vortex_array::dtype::Nullability::NonNullable;

        // Two chunks of repeated strings — high duplication so the
        // writer should reach for the dict strategy.
        let chunk1 = VarBinArray::from_iter(
            ["alpha", "beta", "alpha", "alpha"].into_iter().map(Some),
            DType::Utf8(NonNullable),
        )
        .into_array();
        let chunk2 = VarBinArray::from_iter(
            ["beta", "gamma", "alpha", "gamma"].into_iter().map(Some),
            DType::Utf8(NonNullable),
        )
        .into_array();
        let dtype = chunk1.dtype().clone();

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let segments_for_strategy = Arc::<TestSegments>::clone(&segments);
        let strategy = ChunkedLayoutStrategy::new(crate::layouts::dict::writer::DictStrategy::new(
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            crate::layouts::dict::writer::DictLayoutOptions::default(),
        ));
        let (mut sequence_id, eof) = SequenceId::root().split();
        let chunk1_for_write = chunk1.clone();
        let chunk2_for_write = chunk2.clone();
        let dtype_for_write = dtype.clone();
        let layout = block_on(|handle| async move {
            let session = SESSION.clone().with_handle(handle);
            strategy
                .write_stream(
                    ctx,
                    segments_for_strategy,
                    SequentialStreamAdapter::new(
                        dtype_for_write,
                        stream::iter([
                            Ok((sequence_id.advance(), chunk1_for_write)),
                            Ok((sequence_id.advance(), chunk2_for_write)),
                        ]),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        let combined = ChunkedArray::try_new(vec![chunk1, chunk2], dtype)
            .unwrap()
            .into_array();
        (segments, layout, combined)
    }

    /// V1/V2 must agree for a dict-encoded chunked Utf8 column with
    /// `root()` projection.
    #[test]
    fn diff_v1_v2_projection_only_dict() -> VortexResult<()> {
        let (segments, layout, expected) = build_dict_chunked_layout();

        let v1 = read_v1(Arc::clone(&segments), &layout)?;
        let v2 = read_v2(Arc::clone(&segments), &layout)?;

        assert_arrays_eq!(v1, v2);
        assert_arrays_eq!(v2, expected);
        Ok(())
    }

    /// `pack([], …)` projection: no field references. V2 must build a
    /// plan that does not recurse into any field's `Layout::plan` and
    /// emit a stream of empty structs of the layout's row count. The
    /// pre-fix behaviour would still plan every field through
    /// `plan_full_struct_with_projection`, paying per-field setup cost
    /// (and surfacing as the `COUNT(*)` regression on wide schemas).
    #[test]
    fn diff_v1_v2_empty_pack_projection_chunked_struct() -> VortexResult<()> {
        use vortex_array::dtype::Nullability;
        use vortex_array::expr::pack;
        let (segments, layout, _) = build_chunked_struct_layout();

        let proj = pack(Vec::<(&str, vortex_array::expr::Expression)>::new(), Nullability::NonNullable);
        let v1 = read_v1_with(Arc::clone(&segments), &layout, proj.clone())?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, proj)?;

        assert_arrays_eq!(v1, v2);
        // Five rows of empty struct.
        assert_eq!(v2.len(), 5);
        Ok(())
    }
}
