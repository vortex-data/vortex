// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::stream;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::ext_storage;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_mask::Mask;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::extension::Extension;
use crate::layouts::flat::Flat;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::table::TableStrategy;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::Zoned as ZonedPlanVTable;
use crate::plan::physical::optimize;
use crate::scan::scan_builder::ScanBuilder;
use crate::segments::CountingSegmentSource;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;
use crate::test::SESSION;
use crate::test::TestPoint;

const LEN: usize = 6;

/// Points `{a: i, b: 10 * i}` for `i` in `0..LEN`, null at every third row when nullable.
fn points(nullability: Nullability) -> VortexResult<ArrayRef> {
    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::from_iter((0..LEN).map(|row| row % 3 != 0)),
    };
    let storage = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        [
            buffer![0i32, 1, 2, 3, 4, 5].into_array(),
            buffer![0i64, 10, 20, 30, 40, 50].into_array(),
        ],
        LEN,
        validity,
    )?;
    Ok(ExtensionArray::try_new(TestPoint::dtype(nullability), storage.into_array())?.into_array())
}

/// A table `{id: i32, point: test.point}`.
fn table(nullability: Nullability) -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[
        ("id", buffer![0i32, 1, 2, 3, 4, 5].into_array()),
        ("point", points(nullability)?),
    ])?
    .into_array())
}

fn flat_table() -> TableStrategy {
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    TableStrategy::new(Arc::clone(&flat), flat)
}

/// [`flat_table`] with [`TestPoint`] columns opted into the extension layout.
fn point_table() -> TableStrategy {
    flat_table().with_extension_layout([TestPoint.id()])
}

async fn write(
    strategy: &dyn LayoutStrategy,
    chunks: Vec<ArrayRef>,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
    let segments = Arc::new(TestSegments::default());
    let (mut sequence_id, eof) = SequenceId::root().split();
    let dtype = chunks[0].dtype().clone();
    let chunks = chunks
        .into_iter()
        .map(|chunk| Ok((sequence_id.advance(), chunk)))
        .collect::<Vec<_>>();
    let layout = strategy
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(dtype, stream::iter(chunks)).sendable(),
            eof,
            &SESSION,
        )
        .await?;
    Ok((segments, layout))
}

/// Scans `layout` through its reader, the path of the classic scan.
async fn scan(
    segments: Arc<dyn SegmentSource>,
    layout: &LayoutRef,
    projection: &Expression,
    filter: Option<&Expression>,
) -> VortexResult<ArrayRef> {
    let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
    ScanBuilder::new(SESSION.clone(), reader)
        .with_projection(projection.bind(layout.dtype())?)
        .with_some_filter(
            filter
                .map(|filter| filter.bind(layout.dtype()))
                .transpose()?,
        )
        .into_array_stream()?
        .read_all()
        .await
}

/// The rows of `array` where `filter` holds, projected by `projection`, evaluated in memory.
fn reference(
    array: &ArrayRef,
    projection: &Expression,
    filter: Option<&Expression>,
) -> VortexResult<ArrayRef> {
    let array = match filter {
        Some(filter) => {
            let mask = array
                .clone()
                .apply(filter)?
                .fill_null(false)?
                .execute::<Mask>(&mut SESSION.create_execution_ctx())?;
            array.filter(mask)?
        }
        None => array.clone(),
    };
    array.apply(projection)
}

fn point_storage(field: &str) -> Expression {
    get_item(field, ext_storage(get_item("point", root())))
}

/// The storage struct of an extension column is split into one column per field.
#[tokio::test]
async fn writes_storage_fields_as_columns() -> VortexResult<()> {
    let (_, layout) = write(&point_table(), vec![table(Nullability::Nullable)?]).await?;
    insta::assert_snapshot!(layout.display_tree(), @r"
    vortex.struct, dtype: {id=i32, point=test.point({a=i32, b=i64}?)}, children: 2
    ├── id: vortex.flat, dtype: i32, segment: 0
    └── point: vortex.ext, dtype: test.point({a=i32, b=i64}?), children: 1
        └── storage: vortex.struct, dtype: {a=i32, b=i64}?, children: 3
            ├── validity: vortex.flat, dtype: bool, segment: 1
            ├── a: vortex.flat, dtype: i32, segment: 2
            └── b: vortex.flat, dtype: i64, segment: 3
    ");
    Ok(())
}

/// Extension columns keep their leaf layout unless their type is opted into the extension layout
/// and their storage is a struct.
#[rstest]
#[case::disabled(flat_table(), points(Nullability::NonNullable))]
#[case::other_type(
    flat_table().with_extension_layout([Timestamp.id()]),
    points(Nullability::NonNullable),
)]
#[case::primitive_storage(
    flat_table().with_extension_layout([Timestamp.id()]),
    ExtensionArray::try_new(
        Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        buffer![1i64, 2, 3].into_array(),
    )
    .map(IntoArray::into_array),
)]
#[tokio::test]
async fn keeps_other_extensions_as_leaves(
    #[case] strategy: TableStrategy,
    #[case] array: VortexResult<ArrayRef>,
) -> VortexResult<()> {
    let (_, layout) = write(&strategy, vec![array?]).await?;
    assert!(layout.is::<Flat>(), "{}", layout.display_tree());
    Ok(())
}

/// Chunks of any encoding are written through their storage.
#[tokio::test]
async fn writes_extension_chunks_of_any_encoding() -> VortexResult<()> {
    let points = points(Nullability::Nullable)?;
    let dtype = points.dtype().clone();
    let ext_dtype = TestPoint::dtype(Nullability::Nullable);
    let storage = Scalar::struct_(
        ext_dtype.storage_dtype().clone(),
        vec![Scalar::from(7i32), Scalar::from(70i64)],
    );
    let chunks = vec![
        points.clone(),
        ConstantArray::new(Scalar::extension_ref(ext_dtype.clone(), storage), 2).into_array(),
        ConstantArray::new(Scalar::null(dtype.clone()), 2).into_array(),
        ChunkedArray::try_new(
            vec![points.slice(0..2)?, points.slice(4..LEN)?],
            dtype.clone(),
        )?
        .into_array(),
    ];
    let expected = ChunkedArray::try_new(chunks.clone(), dtype)?.into_array();

    let chunked: Arc<dyn LayoutStrategy> =
        Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
    let strategy =
        TableStrategy::new(Arc::clone(&chunked), chunked).with_extension_layout([TestPoint.id()]);
    let (segments, layout) = write(&strategy, chunks).await?;
    assert!(layout.is::<Extension>(), "{}", layout.display_tree());
    let actual = scan(segments, &layout, &root(), None).await?;
    assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// An extension column nested in a list is decomposed under the list elements.
#[tokio::test]
async fn decomposes_extension_list_elements() -> VortexResult<()> {
    let points = points(Nullability::NonNullable)?;
    let list = ListArray::try_new(
        points,
        buffer![0u32, 2, 2, 6].into_array(),
        Validity::NonNullable,
        &mut SESSION.create_execution_ctx(),
    )?
    .into_array();
    let strategy = point_table().with_list_layout();
    let (segments, layout) = write(&strategy, vec![list.clone()]).await?;
    insta::assert_snapshot!(layout.display_tree(), @r"
    vortex.list, dtype: list(test.point({a=i32, b=i64})), children: 2
    ├── elements: vortex.ext, dtype: test.point({a=i32, b=i64}), children: 1
    │   └── storage: vortex.struct, dtype: {a=i32, b=i64}, children: 2
    │       ├── a: vortex.flat, dtype: i32, segment: 1
    │       └── b: vortex.flat, dtype: i64, segment: 2
    └── offsets: vortex.flat, dtype: u64, segment: 0
    ");

    let actual = scan(segments, &layout, &root(), None).await?;
    assert_arrays_eq!(actual, list, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// Reading through the classic scan returns what the expression returns over the input.
#[rstest]
#[case::field(point_storage("a"), None)]
#[case::storage(ext_storage(get_item("point", root())), None)]
#[case::values(get_item("point", root()), None)]
#[case::table(root(), None)]
#[case::is_null(is_null(get_item("point", root())), None)]
#[case::mixed(
    pack(
        [("a", point_storage("a")), ("point", get_item("point", root()))],
        Nullability::NonNullable,
    ),
    None,
)]
#[case::filter_field(get_item("id", root()), Some(gt(point_storage("a"), lit(2i32))))]
#[case::filter_values(
    get_item("point", root()),
    Some(and(gt(point_storage("a"), lit(1i32)), lt(point_storage("b"), lit(50i64))))
)]
#[case::filter_is_not_null(point_storage("b"), Some(is_not_null(get_item("point", root()))))]
#[tokio::test]
async fn scan_matches_reference(
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
    #[case] projection: Expression,
    #[case] filter: Option<Expression>,
) -> VortexResult<()> {
    let table = table(nullability)?;
    let (segments, layout) = write(&point_table(), vec![table.clone()]).await?;
    let actual = scan(segments, &layout, &projection, filter.as_ref()).await?;
    let expected = reference(&table, &projection, filter.as_ref())?;
    assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// A filter that reads the extension values is evaluated over values rebuilt from the storage.
#[rstest]
#[tokio::test]
async fn filter_over_values_reads_storage(
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let points = points(nullability)?;
    let (segments, layout) = write(&point_table(), vec![points.clone()]).await?;
    let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
    // `get_item` over `pack` uses the extension root outside `ext_storage`.
    let filter = is_not_null(get_item(
        "value",
        pack([("value", root())], Nullability::NonNullable),
    ));
    let input = Mask::from_iter([true, true, false, true, true, true]);

    let actual = reader
        .filter_evaluation(
            &(0..LEN as u64),
            &filter.bind(reader.dtype())?,
            MaskFuture::ready(input.clone()),
        )?
        .await?;

    let valid = points
        .validity()?
        .execute_mask(LEN, &mut SESSION.create_execution_ctx())?;
    assert_eq!(actual, &input & &valid);
    Ok(())
}

/// Projecting storage fields reads only their segments, plus the storage validity when the
/// extension is nullable.
#[rstest]
#[case::field(Nullability::NonNullable, point_storage("a"), 1)]
#[case::nullable_field(Nullability::Nullable, point_storage("a"), 2)]
#[case::is_null(Nullability::Nullable, is_null(get_item("point", root())), 1)]
#[case::values(Nullability::NonNullable, get_item("point", root()), 2)]
#[case::nullable_values(Nullability::Nullable, get_item("point", root()), 3)]
#[tokio::test]
async fn projection_reads_only_needed_fields(
    #[case] nullability: Nullability,
    #[case] projection: Expression,
    #[case] expected_requests: usize,
) -> VortexResult<()> {
    let (segments, layout) = write(&point_table(), vec![table(nullability)?]).await?;
    let source = Arc::new(CountingSegmentSource::new(segments));
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&source) as _,
        &SESSION,
        &Default::default(),
    )?;
    reader
        .projection_evaluation(
            &(0..LEN as u64),
            &projection.bind(reader.dtype())?,
            MaskFuture::new_true(LEN),
        )?
        .await?;
    assert_eq!(source.requests(), expected_requests);
    Ok(())
}

/// The plan of a storage field projection reads only that field, while a projection of the
/// extension values keeps the assembly of the whole storage.
#[tokio::test]
async fn plan_pushes_storage_reads_down() -> VortexResult<()> {
    let (_, layout) = write(&point_table(), vec![table(Nullability::NonNullable)?]).await?;
    let optimized = |projection: Expression| -> VortexResult<_> {
        let plan = EvalPlan::try_new(projection.bind(layout.dtype())?, layout.new_plan()?)?;
        optimize(plan.into_plan(), &SESSION)
    };

    insta::assert_snapshot!(optimized(point_storage("a"))?.display_tree(), @"
    root: vortex.plan.filter(i32, rows=6)
      child: vortex.plan.segment_scan(i32, rows=6)
    ");
    insta::assert_snapshot!(optimized(get_item("point", root()))?.display_tree(), @"
    root: vortex.plan.ext_pack(test.point({a=i32, b=i64}), rows=6)
      storage: vortex.plan.pack({a=i32, b=i64}, rows=6)
        a: vortex.plan.filter(i32, rows=6)
          child: vortex.plan.segment_scan(i32, rows=6)
        b: vortex.plan.filter(i64, rows=6)
          child: vortex.plan.segment_scan(i64, rows=6)
    ");
    Ok(())
}

/// A filter on a storage field plans zone pruning over the zone map of that field, reading only
/// metadata.
#[tokio::test]
async fn pruning_plan_reaches_storage_zones() -> VortexResult<()> {
    let zoned = ZonedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions {
            block_size: NonZeroUsize::new(2).ok_or_else(|| vortex_err!("zero block size"))?,
            ..Default::default()
        },
    );
    let strategy = TableStrategy::new(Arc::new(FlatLayoutStrategy::default()), Arc::new(zoned))
        .with_extension_layout([TestPoint.id()]);
    let (_, layout) = write(&strategy, vec![table(Nullability::NonNullable)?]).await?;

    let falsifier = gt(point_storage("a"), lit(2i32))
        .bind(layout.dtype())?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!("filter has no falsifier"))?;
    let pruning = optimize(
        EvalPlan::try_new(falsifier, layout.new_plan()?)?.into_plan(),
        &SESSION,
    )?;
    let zoned = pruning
        .as_opt::<ZonedPlanVTable>()
        .ok_or_else(|| vortex_err!("not a zone pruning plan: {}", pruning.display_tree()))?;
    assert!(zoned.is_pruning());
    assert_eq!(zoned.zone_len(), 2);
    Ok(())
}
