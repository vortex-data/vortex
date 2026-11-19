// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::Expression;
use vortex_array::expr::Lambda;
use vortex_array::expr::cast;
use vortex_array::expr::checked_add;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::list_contains;
use vortex_array::expr::list_length;
use vortex_array::expr::list_transform;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::var;
use vortex_buffer::buffer;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::*;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::writer::ListLayoutStrategy;
use crate::layouts::repartition::RepartitionStrategy;
use crate::layouts::repartition::RepartitionWriterOptions;
use crate::layouts::struct_::writer::StructStrategy;
use crate::scan::split_by::SplitBy;
use crate::segments::CountingSegmentSource;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialArrayStreamExt;
use crate::session::LayoutSession;
use crate::test::SESSION;

/// Validity-class projections (`is_null` / `is_not_null` of the list) round-trip through the
/// validity-only read path, for both nullable and non-nullable lists.
#[rstest]
// `create_basic_list_array(true)` has validity `[true, false, true]`.
#[case::nullable(true, vec![true, false, true])]
#[case::non_nullable(false, vec![true, true, true])]
#[tokio::test]
async fn projection_validity_class(
    #[case] nullable: bool,
    #[case] valid: Vec<bool>,
) -> VortexResult<()> {
    let list = create_basic_list_array(nullable);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let not_null_expr = is_not_null(root()).bind(reader.dtype())?;
    let not_null = reader
        .projection_evaluation(&(0..3), &not_null_expr, MaskFuture::new_true(3))?
        .await?;
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(not_null, BoolArray::from_iter(valid.clone()), &mut exec_ctx);

    let is_null_expr = is_null(root()).bind(reader.dtype())?;
    let is_null_res = reader
        .projection_evaluation(&(0..3), &is_null_expr, MaskFuture::new_true(3))?
        .await?;
    assert_arrays_eq!(
        is_null_res,
        BoolArray::from_iter(valid.iter().map(|v| !v).collect::<Vec<_>>()),
        &mut exec_ctx
    );

    Ok(())
}

#[tokio::test]
async fn projection_list_length_reads_offsets() -> VortexResult<()> {
    let list = create_basic_list_array(false);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = list_length(root()).bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
        .await?;

    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, buffer![2u64, 2, 1].into_array(), &mut exec_ctx);
    Ok(())
}

#[tokio::test]
async fn projection_list_length_preserves_validity() -> VortexResult<()> {
    let list = create_basic_list_array(true);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = list_length(root()).bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
        .await?;

    let expected =
        PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(1)]).into_array();
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

#[tokio::test]
async fn projection_list_length_applies_sparse_mask() -> VortexResult<()> {
    let list = create_basic_list_array(true);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let mask = Mask::from_iter([false, true, true]);
    let expr = list_length(root()).bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(0..3), &expr, MaskFuture::ready(mask))?
        .await?;

    let expected = PrimitiveArray::from_option_iter::<u64, _>([None, Some(1)]).into_array();
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

#[tokio::test]
async fn projection_cast_list_length() -> VortexResult<()> {
    let list = create_basic_list_array(true);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = cast(
        list_length(root()),
        DType::Primitive(PType::I64, Nullability::Nullable),
    )
    .bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
        .await?;

    let expected =
        PrimitiveArray::from_option_iter::<i64, _>([Some(2), None, Some(1)]).into_array();
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

#[tokio::test]
async fn filter_evaluation_list_length() -> VortexResult<()> {
    let list = create_basic_list_array(true);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = gt(list_length(root()), lit(1u64)).bind(reader.dtype())?;
    let result = reader
        .filter_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
        .await?;

    assert_eq!(result, Mask::from_iter([true, false, false]));
    Ok(())
}

#[rstest]
#[case::is_not_null_nullable(true, is_not_null(root()), Mask::from_iter([true, false, true]))]
#[case::is_not_null_non_nullable(false, is_not_null(root()), Mask::new_true(3))]
#[case::is_null_nullable(true, is_null(root()), Mask::from_iter([false, true, false]))]
#[case::is_null_non_nullable(false, is_null(root()), Mask::new_false(3))]
#[tokio::test]
async fn filter_evaluation_validity_class(
    #[case] nullable: bool,
    #[case] expr: Expression,
    #[case] expected: Mask,
) -> VortexResult<()> {
    let list = create_basic_list_array(nullable);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = expr.bind(reader.dtype())?;
    let result = reader
        .filter_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
        .await?;

    assert_eq!(result, expected);
    Ok(())
}

#[tokio::test]
async fn filter_evaluation_intersects_with_input_mask() -> VortexResult<()> {
    let list = create_basic_list_array(true);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let input_mask = Mask::from_iter([true, true, false]);
    let expr = is_not_null(root()).bind(reader.dtype())?;
    let result = reader
        .filter_evaluation(&(0..3), &expr, MaskFuture::ready(input_mask))?
        .await?;

    assert_eq!(result, Mask::from_iter([true, false, false]));
    Ok(())
}

#[tokio::test]
async fn filter_evaluation_sparse_mask_maps_by_rank() -> VortexResult<()> {
    let list = create_six_list_array();
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let input_mask = Mask::from_iter([false, false, false, false, true, false]);
    let expr = is_not_null(root()).bind(reader.dtype())?;
    let result = reader
        .filter_evaluation(&(0..6), &expr, MaskFuture::ready(input_mask))?
        .await?;

    assert_eq!(
        result,
        Mask::from_iter([false, false, false, false, true, false])
    );
    Ok(())
}

fn flat_list_strategy() -> ListLayoutStrategy {
    ListLayoutStrategy::default()
}

fn layout_test_session() -> VortexSession {
    VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
}

async fn write_layout<S: LayoutStrategy>(
    strategy: &S,
    array: ArrayRef,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef, VortexSession)> {
    let session = layout_test_session().with_tokio();
    let segments = Arc::new(TestSegments::default());
    let segments_ref: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
    let (ptr, eof) = SequenceId::root().split();
    let stream = array.to_array_stream().sequenced(ptr);
    let layout = strategy
        .write_stream(
            ArrayContext::empty().into(),
            segments,
            stream,
            eof,
            &session,
        )
        .await?;
    Ok((segments_ref, layout, session))
}

fn materialize_u64_array(array: ArrayRef) -> Vec<u64> {
    let mut ctx = SESSION.create_execution_ctx();
    array
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap()
        .as_slice::<u64>()
        .to_vec()
}

fn create_basic_list_array(nullable: bool) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = if nullable {
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array())
    } else {
        Validity::NonNullable
    };

    ListArray::try_new(
        buffer![1i32, 2, 3, 4, 5].into_array(),
        buffer![0u32, 2, 4, 5].into_array(),
        validity,
        &mut ctx,
    )
    .expect("array is valid")
    .into_array()
}

fn create_six_list_array() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let validity =
        Validity::Array(BoolArray::from_iter([true, false, true, false, true, true]).into_array());

    ListArray::try_new(
        buffer![1i32, 2, 3, 4, 5, 6].into_array(),
        buffer![0u32, 1, 2, 3, 4, 5, 6].into_array(),
        validity,
        &mut ctx,
    )
    .expect("array is valid")
    .into_array()
}

#[tokio::test]
async fn fetch_offsets_includes_extra_endpoint() -> VortexResult<()> {
    let list = create_basic_list_array(false);

    let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
    let ctx = LayoutReaderContext::new();
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;
    let reader = reader
        .as_any()
        .downcast_ref::<ListReader>()
        .expect("ListReader");

    let offsets = reader.fetch_raw_offsets(&(1..3))?.await?;
    assert_eq!(materialize_u64_array(offsets), vec![2u64, 4, 5]);

    Ok(())
}

#[rstest]
#[case::full_range(0..3, false)]
#[case::partial_start(0..2, false)]
#[case::partial_end(1..3, false)]
#[case::middle_single(1..2, false)]
#[case::empty_range(1..1, false)]
#[case::full_range_null(0..3, true)]
#[tokio::test]
async fn projection_evaluation_round_trips(
    #[case] row_range: Range<u64>,
    #[case] nullable: bool,
) -> VortexResult<()> {
    let list = create_basic_list_array(nullable);
    let ctx = LayoutReaderContext::new();

    let len = usize::try_from(row_range.end - row_range.start)?;
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = root().bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&row_range, &expr, MaskFuture::new_true(len))?
        .await?;

    let expected =
        list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?;
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

#[tokio::test]
async fn projection_evaluation_applies_mask() -> VortexResult<()> {
    let list = create_basic_list_array(false);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let mask = Mask::from_iter([true, false, true]);
    let expr = root().bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(0..3), &expr, MaskFuture::ready(mask.clone()))?
        .await?;

    let expected = list.filter(mask)?;
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

/// Build a list with 5 rows and lengths [2, 3, 0, 3, 2].
fn create_wider_list_array(nullable: bool) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = if nullable {
        Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array())
    } else {
        Validity::NonNullable
    };
    ListArray::try_new(
        buffer![10i32, 11, 20, 21, 22, 30, 31, 32, 40, 41].into_array(),
        buffer![0u32, 2, 5, 5, 8, 10].into_array(),
        validity,
        &mut ctx,
    )
    .expect("array is valid")
    .into_array()
}

/// Sparse row masks over a whole chunk round-trip: the whole-chunk read is filtered by the
/// mask in memory.
#[rstest]
#[case::single_middle(Mask::from_iter([false, false, false, true, false]), false)]
#[case::two_far_apart(Mask::from_iter([true, false, false, true, false]), false)]
#[case::boundaries(Mask::from_iter([true, false, false, false, true]), false)]
#[case::kept_empty_row(Mask::from_iter([false, false, true, false, false]), false)]
#[case::sparse_nullable(Mask::from_iter([true, false, true, false, true]), true)]
#[case::all_false(Mask::new_false(5), false)]
#[tokio::test]
async fn projection_evaluation_sparse_mask_round_trips(
    #[case] mask: Mask,
    #[case] nullable: bool,
) -> VortexResult<()> {
    let list = create_wider_list_array(nullable);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = root().bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(0..5), &expr, MaskFuture::ready(mask.clone()))?
        .await?;

    let expected = list.filter(mask)?;
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

/// A partial range against a single flat elements segment takes the bounded path. The flat
/// reader may still fetch its whole segment, but the list reader reconstructs only the requested
/// element range.
#[tokio::test]
async fn projection_evaluation_partial_range_bounded() -> VortexResult<()> {
    let list = create_wider_list_array(false);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = root().bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(1..4), &expr, MaskFuture::new_true(3))?
        .await?;

    let expected = list.slice(1..4)?;
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

#[test]
fn maps_element_splits_to_outer_grid() {
    assert_eq!(map_element_split_to_outer_grid(0, 100, 100, 8), None);
    assert_eq!(map_element_split_to_outer_grid(20, 100, 100, 8), Some(25));
    assert_eq!(map_element_split_to_outer_grid(25, 100, 100, 8), Some(25));
    assert_eq!(map_element_split_to_outer_grid(50, 100, 100, 8), Some(50));
    assert_eq!(map_element_split_to_outer_grid(75, 100, 100, 8), Some(75));
    assert_eq!(map_element_split_to_outer_grid(100, 100, 100, 8), None);

    let mut splits = (1..1_000)
        .filter_map(|split| {
            map_element_split_to_outer_grid(split, 1_000, 100_000, MIN_LIST_SPLIT_COUNT)
        })
        .collect::<Vec<_>>();
    splits.dedup();

    let expected = (1..MIN_LIST_SPLIT_COUNT)
        .map(|grid_index| grid_index * 100_000 / MIN_LIST_SPLIT_COUNT)
        .collect::<Vec<_>>();
    assert_eq!(splits, expected);
}

#[rstest]
#[case::empty(0, 64)]
#[case::small(100_000, 64)]
#[case::floor_boundary(64 * 8192, 64)]
#[case::past_floor(64 * 8192 + 1, 128)]
#[case::million(1_000_000, 128)]
#[case::ten_million(10_000_000, 2048)]
fn split_grid_grows_with_row_count(#[case] outer_row_count: u64, #[case] expected: u64) {
    assert_eq!(list_split_grid_size(outer_row_count), expected);
}

#[tokio::test]
async fn nested_list_propagates_element_splits() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = ListArray::try_new(
        PrimitiveArray::from_iter(0..128_i32).into_array(),
        PrimitiveArray::from_iter((0..=8_u32).map(|idx| idx * 16)).into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let outer = ListArray::try_new(
        inner,
        buffer![0u32, 2, 4, 6, 8].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();

    let inner_strategy = ListLayoutStrategy::default().with_elements(chunked_elements_strategy());
    let strategy = ListLayoutStrategy::default().with_elements(Arc::new(inner_strategy));
    let (segments, layout, session) = write_layout(&strategy, outer).await?;
    let reader = layout.new_reader("".into(), segments, &session, &LayoutReaderContext::new())?;

    let splits = SplitBy::Layout.splits(reader.as_ref(), &(0..4), &[FieldMask::All])?;
    assert_eq!(splits, vec![0, 1, 2, 3, 4]);
    Ok(())
}

/// A list strategy whose `elements` child is repartitioned into two-element chunks, so the
/// reader takes the bounded (chunk-skipping) path for strict sub-ranges. Offsets stay flat.
fn chunked_elements_list_strategy() -> ListLayoutStrategy {
    ListLayoutStrategy::default().with_elements(chunked_elements_strategy())
}

fn chunked_elements_strategy() -> Arc<dyn LayoutStrategy> {
    Arc::new(RepartitionStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 2,
            block_size_target: None,
            canonicalize: true,
        },
    ))
}

/// The chunked-elements strategy must actually produce a chunked `elements` layout, otherwise
/// the reader would silently take the whole-chunk path and the bounded read would be untested.
#[tokio::test]
async fn chunked_elements_produces_chunked_layout() -> VortexResult<()> {
    let list = create_wider_list_array(false);
    let (_segments, layout, _session) =
        write_layout(&chunked_elements_list_strategy(), list).await?;
    let tree = layout.display_tree().to_string();
    assert!(
        tree.contains("elements: vortex.chunked"),
        "elements should be chunked:\n{tree}"
    );
    Ok(())
}

/// A sparse mask must remain selective even when the requested row range covers the complete
/// local list layout. The selected first list touches one element chunk plus the offsets
/// segment; reading all five element chunks would make the count six.
#[tokio::test]
async fn full_range_sparse_mask_crops_element_read() -> VortexResult<()> {
    let list = create_wider_list_array(false);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) =
        write_layout(&chunked_elements_list_strategy(), list.clone()).await?;
    let source = Arc::new(CountingSegmentSource::new(segments));
    let reader = layout.new_reader("".into(), Arc::clone(&source) as _, &session, &ctx)?;

    let mask = Mask::from_iter([true, false, false, false, false]);
    let expr = root().bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(0..5), &expr, MaskFuture::ready(mask.clone()))?
        .await?;

    let expected = list.filter(mask)?;
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    assert_eq!(source.requests(), 2);
    Ok(())
}

/// With chunked elements, sub-range projections take the bounded read path. Every
/// range/mask/nullability combination must match the same projection over the ground-truth
/// array (`list.slice(range).filter(mask)`).
#[rstest]
#[case::full_all_true(0..5, Mask::new_true(5), false)]
#[case::subrange_all_true(1..4, Mask::new_true(3), false)]
#[case::subrange_sparse(1..4, Mask::from_iter([true, false, true]), false)]
#[case::partial_start(0..2, Mask::new_true(2), false)]
#[case::partial_end(2..5, Mask::new_true(3), false)]
#[case::single_non_empty(0..1, Mask::new_true(1), false)]
#[case::single_empty_row(2..3, Mask::from_iter([true]), false)]
#[case::empty_range(2..2, Mask::new_true(0), false)]
#[case::subrange_all_false(1..4, Mask::new_false(3), false)]
#[case::subrange_single_interior(0..4, Mask::from_iter([false, true, false, false]), false)]
#[case::subrange_sparse_nullable(1..4, Mask::from_iter([true, false, true]), true)]
#[case::partial_end_nullable(2..5, Mask::new_true(3), true)]
#[tokio::test]
async fn chunked_elements_round_trips(
    #[case] row_range: Range<u64>,
    #[case] mask: Mask,
    #[case] nullable: bool,
) -> VortexResult<()> {
    let list = create_wider_list_array(nullable);
    let ctx = LayoutReaderContext::new();
    let (segments, layout, session) =
        write_layout(&chunked_elements_list_strategy(), list.clone()).await?;
    let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

    let expr = root().bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&row_range, &expr, MaskFuture::ready(mask.clone()))?
        .await?;

    let sliced = list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?;
    let expected = sliced.filter(mask)?;
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

/// Lists `[0, 1, 2]`, `[]`, `[3, 4, 5]`, `[6, 7]`, `[8, 9]`. When nullable, the third is null
/// and keeps its elements.
fn create_ten_element_list_array(nullable: bool) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = if nullable {
        Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array())
    } else {
        Validity::NonNullable
    };
    ListArray::try_new(
        PrimitiveArray::from_iter(0..10i32).into_array(),
        buffer![0u32, 3, 3, 6, 8, 10].into_array(),
        validity,
        &mut ctx,
    )
    .expect("array is valid")
    .into_array()
}

/// Slices of one list share its whole `elements`, so every chunk but the first starts at a
/// nonzero offset and every chunk but the last leaves trailing elements unreferenced.
fn sliced_chunks(array: &ArrayRef, cuts: &[usize]) -> VortexResult<Vec<ArrayRef>> {
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(array.len());
    bounds
        .windows(2)
        .map(|range| array.slice(range[0]..range[1]))
        .collect()
}

fn chunked_flat() -> Arc<dyn LayoutStrategy> {
    Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()))
}

/// A list strategy that keeps every chunk, so a multi-chunk stream reaches the list writer.
fn chunked_list_strategy() -> ListLayoutStrategy {
    ListLayoutStrategy::default()
        .with_elements(chunked_flat())
        .with_offsets(chunked_flat())
        .with_validity(chunked_flat())
}

async fn assert_round_trips<S: LayoutStrategy>(
    strategy: &S,
    chunks: Vec<ArrayRef>,
    expected: ArrayRef,
) -> VortexResult<()> {
    let chunked = ChunkedArray::try_new(chunks, expected.dtype().clone())?.into_array();
    let (segments, layout, session) = write_layout(strategy, chunked).await?;
    let reader = layout.new_reader("".into(), segments, &session, &LayoutReaderContext::new())?;

    let len = expected.len();
    let expr = root().bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&(0..len as u64), &expr, MaskFuture::new_true(len))?
        .await?;
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(result, expected, &mut exec_ctx);
    Ok(())
}

#[rstest]
#[case::one_cut(&[2], false, false)]
#[case::two_cuts(&[1, 3], false, false)]
#[case::null_and_empty_at_cuts(&[1, 3], true, false)]
#[case::list_view_chunks(&[1, 3], true, true)]
#[case::non_nullable_list_view_chunks(&[1, 3], false, true)]
#[tokio::test]
async fn sliced_list_chunks_round_trip(
    #[case] cuts: &[usize],
    #[case] nullable: bool,
    #[case] as_list_view: bool,
) -> VortexResult<()> {
    let list = create_ten_element_list_array(nullable);
    let source = if as_list_view {
        let mut ctx = SESSION.create_execution_ctx();
        list.clone()
            .execute::<ListViewArray>(&mut ctx)?
            .into_array()
    } else {
        list.clone()
    };
    let chunks = sliced_chunks(&source, cuts)?;
    assert_round_trips(&chunked_list_strategy(), chunks, list).await
}

#[tokio::test]
async fn sliced_nested_list_chunks_round_trip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = ListArray::try_new(
        PrimitiveArray::from_iter(0..12i32).into_array(),
        buffer![0u32, 2, 2, 5, 7, 9, 12].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let outer = ListArray::try_new(
        inner,
        buffer![0u32, 2, 3, 5, 6].into_array(),
        Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
        &mut ctx,
    )?
    .into_array();

    let strategy = chunked_list_strategy().with_elements(Arc::new(chunked_list_strategy()));
    assert_round_trips(&strategy, sliced_chunks(&outer, &[1, 2])?, outer).await
}

/// Lists `[{a, 1}, null]`, null (hiding `[{c, i32::MAX}]`), `[]`, `[{null, 3}, {e, 4}]` and
/// `[{f, i32::MAX}]`, of nullable `{sku: utf8?, qty: i32}` elements.
fn create_items_list_array() -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let elements = StructArray::try_new(
        FieldNames::from(["sku", "qty"]),
        vec![
            VarBinViewArray::from_iter_nullable_str([
                Some("a"),
                Some("b"),
                Some("c"),
                None,
                Some("e"),
                Some("f"),
            ])
            .into_array(),
            buffer![1i32, 2, i32::MAX, 3, 4, i32::MAX].into_array(),
        ],
        6,
        Validity::from_iter([true, false, true, true, true, true]),
    )?
    .into_array();
    Ok(ListArray::try_new(
        elements,
        buffer![0u32, 2, 3, 3, 5, 6].into_array(),
        Validity::from_iter([true, false, true, true, true]),
        &mut ctx,
    )?
    .into_array())
}

/// A list strategy that writes struct elements as a struct layout, so each element field and the
/// element validity is a flat layout of one segment.
fn struct_elements_list_strategy() -> ListLayoutStrategy {
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    ListLayoutStrategy::default()
        .with_elements(Arc::new(StructStrategy::new(Arc::clone(&flat), flat)))
}

fn sku_lambda() -> VortexResult<Lambda> {
    Lambda::new(["x"], get_item("sku", var("x")))
}

fn qty_lambda() -> VortexResult<Lambda> {
    Lambda::new(["x"], get_item("qty", var("x")))
}

/// `list_transform(root, x -> x.qty + 1)`, which overflows on `i32::MAX`.
fn qty_plus_one() -> VortexResult<Expression> {
    Ok(list_transform(
        root(),
        Lambda::new(["x"], checked_add(get_item("qty", var("x")), lit(1i32)))?,
    ))
}

/// Projects `expr` over `row_range` under `mask` and counts the segments the read requests.
async fn project_counting(
    list: ArrayRef,
    expr: &Expression,
    row_range: Range<u64>,
    mask: Mask,
) -> VortexResult<(ArrayRef, usize)> {
    let (segments, layout, session) = write_layout(&struct_elements_list_strategy(), list).await?;
    let source = Arc::new(CountingSegmentSource::new(segments));
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&source) as _,
        &session,
        &LayoutReaderContext::new(),
    )?;
    let expr = expr.bind(reader.dtype())?;
    let result = reader
        .projection_evaluation(&row_range, &expr, MaskFuture::ready(mask))?
        .await?;
    // Execute so lazy results surface their errors before the count is read.
    let result = result
        .execute::<RecursiveCanonical>(&mut session.create_execution_ctx())?
        .0
        .into_array();
    Ok((result, source.requests()))
}

/// The expected projection: `expr` over the rows of `list` in `row_range` that `mask` selects.
fn expected_projection(
    list: &ArrayRef,
    expr: &Expression,
    row_range: &Range<u64>,
    mask: Mask,
) -> VortexResult<ArrayRef> {
    list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?
        .filter(mask)?
        .apply(expr)
}

/// A projection that reads the elements only through `list_transform` lambdas reads the element
/// columns those lambdas read, plus the offsets, the list validity and the element validity. The
/// stored list has five segments: offsets, list validity, element validity, `sku` and `qty`.
#[rstest]
#[case::root(root(), 5)]
#[case::sku(list_transform(root(), sku_lambda()?), 4)]
#[case::qty(list_transform(root(), qty_lambda()?), 4)]
#[case::both(
    pack(
        [
            ("skus", list_transform(root(), sku_lambda()?)),
            ("qtys", list_transform(root(), qty_lambda()?)),
        ],
        Nullability::NonNullable,
    ),
    5
)]
#[case::with_length(
    pack(
        [
            ("skus", list_transform(root(), sku_lambda()?)),
            ("len", list_length(root())),
        ],
        Nullability::NonNullable,
    ),
    4
)]
#[case::predicate(list_contains(list_transform(root(), sku_lambda()?), lit("e")), 4)]
#[tokio::test]
async fn element_projection_reads_needed_fields(
    #[case] expr: Expression,
    #[case] segments: usize,
) -> VortexResult<()> {
    let list = create_items_list_array()?;
    let (result, requests) = project_counting(list.clone(), &expr, 0..5, Mask::new_true(5)).await?;
    let expected = list.apply(&expr)?;
    assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    assert_eq!(requests, segments);
    Ok(())
}

/// Pushed-down lambdas keep the semantics of `list_transform` for null lists, empty lists and
/// null elements, over every read path: full, bounded and sparse.
#[rstest]
#[case::full(0..5, Mask::new_true(5))]
#[case::subrange(1..4, Mask::new_true(3))]
#[case::sparse(0..5, Mask::from_iter([true, false, true, false, true]))]
#[case::null_list_only(1..2, Mask::new_true(1))]
#[case::empty_list_only(2..3, Mask::new_true(1))]
#[case::none_selected(0..5, Mask::new_false(5))]
#[case::empty_range(2..2, Mask::new_true(0))]
#[tokio::test]
async fn element_projection_round_trips(
    #[case] row_range: Range<u64>,
    #[case] mask: Mask,
    #[values(
        list_transform(root(), sku_lambda()?),
        list_transform(root(), qty_lambda()?),
        pack(
            [
                ("skus", list_transform(root(), sku_lambda()?)),
                ("valid", is_not_null(root())),
            ],
            Nullability::NonNullable,
        )
    )]
    expr: Expression,
) -> VortexResult<()> {
    let list = create_items_list_array()?;
    let expected = expected_projection(&list, &expr, &row_range, mask.clone())?;
    let (result, _) = project_counting(list, &expr, row_range, mask).await?;
    assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// A lambda that can fail runs only on the elements of the selected valid lists, as in memory:
/// the `i32::MAX` under the null list never reaches it, nor does the one in the last list when the
/// mask leaves that list out. Only `qty` is read.
#[rstest]
#[case::full_without_last(0..4, Mask::new_true(4))]
#[case::masked_last(0..5, Mask::from_iter([true, true, true, true, false]))]
#[case::sparse(0..5, Mask::from_iter([false, true, false, true, false]))]
#[case::null_list_only(1..2, Mask::new_true(1))]
#[tokio::test]
async fn fallible_lambda_runs_on_selected_valid_lists(
    #[case] row_range: Range<u64>,
    #[case] mask: Mask,
) -> VortexResult<()> {
    let list = create_items_list_array()?;
    let expr = qty_plus_one()?;
    let expected = expected_projection(&list, &expr, &row_range, mask.clone())?;
    let (result, requests) = project_counting(list, &expr, row_range, mask).await?;
    assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    assert_eq!(requests, 4);
    Ok(())
}

/// The reader fails where the lambda fails in memory: on the `i32::MAX` of a selected valid list.
#[tokio::test]
async fn fallible_lambda_fails_on_selected_overflow() -> VortexResult<()> {
    let list = create_items_list_array()?;
    let expr = qty_plus_one()?;
    let mut ctx = SESSION.create_execution_ctx();
    assert!(
        list.clone()
            .apply(&expr)?
            .execute::<RecursiveCanonical>(&mut ctx)
            .is_err()
    );
    assert!(
        project_counting(list, &expr, 4..5, Mask::new_true(1))
            .await
            .is_err()
    );
    Ok(())
}

/// A filter through a pushed-down lambda reads only the element column it tests.
#[rstest]
#[case::dense(Mask::new_true(5))]
#[case::sparse(Mask::from_iter([false, false, false, true, false]))]
#[tokio::test]
async fn filter_evaluation_pushes_lambda(#[case] input: Mask) -> VortexResult<()> {
    let list = create_items_list_array()?;
    let (segments, layout, session) =
        write_layout(&struct_elements_list_strategy(), list.clone()).await?;
    let source = Arc::new(CountingSegmentSource::new(segments));
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&source) as _,
        &session,
        &LayoutReaderContext::new(),
    )?;

    let filter = list_contains(list_transform(root(), sku_lambda()?), lit("e"));
    let result = reader
        .filter_evaluation(
            &(0..5),
            &filter.bind(reader.dtype())?,
            MaskFuture::ready(input.clone()),
        )?
        .await?;

    let mut ctx = session.create_execution_ctx();
    let matches = list
        .apply(&filter)?
        .fill_null(false)?
        .execute::<Mask>(&mut ctx)?;
    assert_eq!(result, &input & &matches);
    assert_eq!(source.requests(), 4);
    Ok(())
}

/// Every split sees the same element projection, so the elements reader's identity-keyed caches
/// stay warm.
#[tokio::test]
async fn element_projection_is_cached() -> VortexResult<()> {
    let list = create_items_list_array()?;
    let (segments, layout, session) = write_layout(&struct_elements_list_strategy(), list).await?;
    let reader = layout.new_reader("".into(), segments, &session, &LayoutReaderContext::new())?;
    let reader = reader
        .as_any()
        .downcast_ref::<ListReader>()
        .expect("ListReader");

    let expr = list_transform(root(), sku_lambda()?).bind(reader.dtype())?;
    let first = reader.element_projection(&expr)?;
    assert!(Arc::ptr_eq(&first, &reader.element_projection(&expr)?));
    Ok(())
}
