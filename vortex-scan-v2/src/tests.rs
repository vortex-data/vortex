// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::checked_add;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::Flat;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::row_idx::row_idx;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::segments::SharedSegmentSource;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

use crate::ScanBuilder;
use crate::SplitBy;

#[derive(Clone)]
struct TrackingSource {
    inner: Arc<TestSegments>,
    requests: Arc<Mutex<Vec<SegmentId>>>,
}

impl TrackingSource {
    fn new(inner: Arc<TestSegments>) -> Self {
        Self {
            inner,
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn requests(&self) -> Vec<SegmentId> {
        self.requests.lock().clone()
    }
}

impl SegmentSource for TrackingSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requests.lock().push(id);
        self.inner.request(id)
    }
}

#[test]
fn scans_layout_through_optimized_plans() -> VortexResult<()> {
    block_on(|handle| async {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        let segments = Arc::new(TestSegments::default());
        let (sequence, eof) = SequenceId::root().split();
        let input = PrimitiveArray::from_iter(0_i32..10).into_array();
        let layout = FlatLayoutStrategy::default()
            .write_stream(
                ArrayContext::empty().into(),
                Arc::<TestSegments>::clone(&segments),
                input.to_array_stream().sequenced(sequence),
                eof,
                &session,
            )
            .await?;

        let actual = ScanBuilder::try_new(&layout, segments, session.clone())?
            .with_filter(gt(root(), lit(4_i32)))
            .with_projection(checked_add(root(), lit(1_i32)))
            .with_split_by(SplitBy::RowCount(3))
            .into_array_stream()?
            .read_all()
            .await?;
        let expected = PrimitiveArray::from_iter(6_i32..11).into_array();

        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
        Ok(())
    })
}

#[test]
fn filter_and_projection_share_flat_segment_request() -> VortexResult<()> {
    block_on(|handle| async {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        let segments = Arc::new(TestSegments::default());
        let (sequence, eof) = SequenceId::root().split();
        let input = PrimitiveArray::from_iter(0_i32..10).into_array();
        let layout = FlatLayoutStrategy::default()
            .write_stream(
                ArrayContext::empty().into(),
                Arc::<TestSegments>::clone(&segments),
                input.to_array_stream().sequenced(sequence),
                eof,
                &session,
            )
            .await?;
        let data_segment = layout.as_::<Flat>().segment_id();
        let tracking = TrackingSource::new(segments);
        let source: Arc<dyn SegmentSource> = Arc::new(SharedSegmentSource::new(tracking.clone()));

        let actual = ScanBuilder::try_new(&layout, source, session.clone())?
            .with_filter(gt(root(), lit(4_i32)))
            .with_projection(checked_add(root(), lit(1_i32)))
            .into_array_stream()?
            .read_all()
            .await?;
        let expected = PrimitiveArray::from_iter(6_i32..11).into_array();

        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
        assert_eq!(
            tracking
                .requests()
                .into_iter()
                .filter(|&segment| segment == data_segment)
                .count(),
            1
        );
        Ok(())
    })
}

#[test]
fn empty_projection_reads_no_struct_fields() -> VortexResult<()> {
    block_on(|handle| async {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        let segments = Arc::new(TestSegments::default());
        let (sequence, eof) = SequenceId::root().split();
        let input = StructArray::from_fields(
            [
                ("a", buffer![1_i32, 2, 3, 4, 5, 6].into_array()),
                ("b", buffer![7_i32, 8, 9, 10, 11, 12].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let layout = TableStrategy::new(Arc::clone(&flat), flat)
            .write_stream(
                ArrayContext::empty().into(),
                Arc::<TestSegments>::clone(&segments),
                input.to_array_stream().sequenced(sequence),
                eof,
                &session,
            )
            .await?;
        let tracking = TrackingSource::new(segments);
        let source: Arc<dyn SegmentSource> = Arc::new(tracking.clone());
        let projection = pack(
            std::iter::empty::<(&str, Expression)>(),
            Nullability::NonNullable,
        );

        let actual = ScanBuilder::try_new(&layout, source, session)?
            .with_projection(projection)
            .into_array_stream()?
            .read_all()
            .await?;

        assert_eq!(actual.len(), 6);
        assert_eq!(
            actual.dtype(),
            &DType::Struct(StructFields::empty(), Nullability::NonNullable)
        );
        assert!(tracking.requests().is_empty());
        Ok(())
    })
}

#[test]
fn zoned_pruning_skips_a_falsified_data_chunk() -> VortexResult<()> {
    block_on(|handle| async {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        let segments = Arc::new(TestSegments::default());
        let (sequence, eof) = SequenceId::root().split();
        let input = ChunkedArray::from_iter([
            buffer![1_i32, 2, 3].into_array(),
            buffer![4_i32, 5, 6].into_array(),
            buffer![7_i32, 8].into_array(),
        ])
        .into_array();
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3)
                    .ok_or_else(|| vortex_err!("zone length is zero"))?,
                ..Default::default()
            },
        );
        let layout = strategy
            .write_stream(
                ArrayContext::empty().into(),
                Arc::<TestSegments>::clone(&segments),
                input.to_array_stream().sequenced(sequence),
                eof,
                &session,
            )
            .await?;

        let data = layout
            .slot(0)?
            .ok_or_else(|| vortex_err!("zoned data child is absent"))?;
        let first_data = data
            .slot(0)?
            .ok_or_else(|| vortex_err!("first data chunk is absent"))?;
        let first_data_segment = first_data.as_::<Flat>().segment_id();
        let zones = layout
            .slot(1)?
            .ok_or_else(|| vortex_err!("zoned stats child is absent"))?;
        let zones_segment = zones.as_::<Flat>().segment_id();

        let tracking = Arc::new(TrackingSource::new(segments));
        let source: Arc<dyn SegmentSource> = Arc::clone(&tracking) as Arc<dyn SegmentSource>;
        let actual = ScanBuilder::try_new(&layout, source, session.clone())?
            .with_filter(gt(root(), lit(5_i32)))
            .into_array_stream()?
            .read_all()
            .await?;
        let expected = PrimitiveArray::from_iter(6_i32..9).into_array();

        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
        let requests = tracking.requests();
        assert!(requests.contains(&zones_segment));
        assert!(!requests.contains(&first_data_segment));
        Ok(())
    })
}

#[test]
fn scans_row_idx_and_struct_expression_partitions() -> VortexResult<()> {
    block_on(|handle| async {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        let segments = Arc::new(TestSegments::default());
        let (sequence, eof) = SequenceId::root().split();
        let input = StructArray::from_fields(
            [
                ("a", buffer![1_i32, 6, 7, 8, 9, 2].into_array()),
                ("b", buffer![10_i32, 20, 3, 40, 5, 60].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let strategy = TableStrategy::new(Arc::clone(&flat), flat);
        let layout = strategy
            .write_stream(
                ArrayContext::empty().into(),
                Arc::<TestSegments>::clone(&segments),
                input.to_array_stream().sequenced(sequence),
                eof,
                &session,
            )
            .await?;

        let filter = and(
            gt(row_idx(), lit(102_u64)),
            and(
                gt(get_item("a", root()), lit(5_i32)),
                gt(get_item("b", root()), lit(10_i32)),
            ),
        );
        let actual = ScanBuilder::try_new(&layout, segments, session.clone())?
            .with_row_offset(100)
            .with_filter(filter)
            .with_projection(row_idx())
            .with_split_by(SplitBy::RowCount(2))
            .into_array_stream()?
            .read_all()
            .await?;
        let expected = PrimitiveArray::from_iter([103_u64]).into_array();

        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
        Ok(())
    })
}

#[test]
fn scans_selected_rows_from_a_list_plan() -> VortexResult<()> {
    block_on(|handle| async {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        let segments = Arc::new(TestSegments::default());
        let (sequence, eof) = SequenceId::root().split();
        let input = ListArray::try_new(
            buffer![1_i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0_u32, 2, 2, 5, 6].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let strategy = TableStrategy::new(Arc::clone(&flat), flat).with_list_layout();
        let layout = strategy
            .write_stream(
                ArrayContext::empty().into(),
                Arc::<TestSegments>::clone(&segments),
                input.to_array_stream().sequenced(sequence),
                eof,
                &session,
            )
            .await?;

        let actual = ScanBuilder::try_new(&layout, segments, session.clone())?
            .with_row_indices(StrictSortedBuffer::try_new(buffer![1_u64, 3])?)
            .into_array_stream()?
            .read_all()
            .await?;
        let expected = ListArray::try_new(
            buffer![6_i32].into_array(),
            buffer![0_u32, 0, 1].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
        Ok(())
    })
}
