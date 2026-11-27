// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::stream;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ForeignExtDType;
use vortex_array::field_path;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::fill_null;
use vortex_expr::get_item;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::pack;
use vortex_expr::root;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_functions::expr::ext_storage;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use super::Extension;
use super::ExtensionLayout;
use super::expr::rebind_to_storage;
use crate::LayoutDeserializeArgs;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::Flat;
use crate::layouts::flat::FlatLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::table::TableStrategy;
use crate::layouts::zoned::ZonedLayoutEncoding;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::Zoned;
use crate::plan::physical::optimize;
use crate::segments::CountingSegmentSource;
use crate::segments::SegmentId;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;
use crate::test::SESSION;

static EXTENSION_ID: CachedId = CachedId::new("test.measurement");

/// Writes `chunks` in order with `strategy`, and returns the layout with the segments it wrote.
async fn write(
    strategy: &TableStrategy,
    chunks: Vec<ArrayRef>,
) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
    let dtype = chunks
        .first()
        .vortex_expect("a write needs at least one chunk")
        .dtype()
        .clone();
    let segments = Arc::new(TestSegments::default());
    let (mut ptr, eof) = SequenceId::root().split();
    let chunks: Vec<VortexResult<_>> = chunks
        .into_iter()
        .map(|chunk| Ok((ptr.advance(), chunk)))
        .collect();

    let layout = strategy
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(dtype, stream::iter(chunks)).sendable(),
            eof,
            &SESSION,
        )
        .await?;

    Ok((layout, segments))
}

fn flat_table() -> TableStrategy {
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    TableStrategy::new(Arc::clone(&flat), flat)
}

fn extension_dtype(storage_dtype: &DType) -> VortexResult<ExtDTypeRef> {
    ForeignExtDType::from_parts(*EXTENSION_ID, vec![], storage_dtype.clone())
}

#[tokio::test]
async fn test_storage_reads_use_field_layouts() -> VortexResult<()> {
    let storage = StructArray::try_from_iter([
        ("value", buffer![1i32, 2, 30, 40].into_array()),
        ("other", buffer![10i64, 20, 300, 400].into_array()),
    ])?;
    let ext_dtype = extension_dtype(storage.dtype())?;
    let array = ExtensionArray::try_new(ext_dtype.clone(), storage.into_array())?.into_array();
    let zoned = ZonedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions {
            block_size: NonZeroUsize::new(2).vortex_expect("two rows per zone"),
            ..Default::default()
        },
    );
    let strategy = TableStrategy::new(Arc::new(FlatLayoutStrategy::default()), Arc::new(zoned))
        .with_extension_layout([ext_dtype.id()]);
    let (layout, segments) = write(&strategy, vec![array.slice(0..2)?, array.slice(2..4)?]).await?;

    let source = Arc::new(CountingSegmentSource::new(segments));
    let reader = layout.new_reader(
        "measurement".into(),
        Arc::clone(&source) as _,
        &SESSION,
        &Default::default(),
    )?;
    let field = get_item("value", ext_storage(root()));
    let predicate = gt(field.clone(), lit(20i32)).bind(layout.dtype())?;

    let pruned = reader
        .split_pruning_evaluation(&(0..4), &predicate, Mask::new_true(4))?
        .await?;
    assert_eq!(pruned, Mask::from_iter([false, false, true, true]));
    assert_eq!(source.requests(), 1);

    let values = reader
        .projection_evaluation(
            &(0..4),
            &field.bind(layout.dtype())?,
            MaskFuture::ready(pruned),
        )?
        .await?;
    assert_arrays_eq!(
        values,
        buffer![30i32, 40].into_array(),
        &mut SESSION.create_execution_ctx()
    );
    assert_eq!(source.requests(), 2);

    let falsifier = predicate
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!("storage comparison has no falsifier"))?;
    let pruning = EvalPlan::try_new(falsifier, layout.new_plan()?)?;
    let pruning = optimize(pruning.into_plan(), &SESSION)?;
    let zoned = pruning
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!("expected field zone map, got {}", pruning.display_tree()))?;
    assert!(zoned.is_pruning());

    Ok(())
}

#[tokio::test]
async fn test_opted_in_primitive_storage_uses_the_leaf() -> VortexResult<()> {
    let storage = buffer![1i32, 2].into_array();
    let ext_dtype = extension_dtype(storage.dtype())?;
    let flat_storage = FlatLayout::new(
        2,
        storage.dtype().clone(),
        SegmentId::from(0),
        ReadContext::new([]),
    )
    .into_layout();
    assert!(ExtensionLayout::try_new(ext_dtype.clone(), flat_storage).is_err());

    let array = ExtensionArray::try_new(ext_dtype.clone(), storage)?.into_array();
    let strategy = flat_table().with_extension_layout([ext_dtype.id()]);
    let (layout, _) = write(&strategy, vec![array]).await?;
    assert!(layout.is::<Flat>());

    Ok(())
}

#[tokio::test]
async fn test_field_override_applies_to_storage_field() -> VortexResult<()> {
    let storage = StructArray::try_from_iter([
        ("x", buffer![1i32, 2].into_array()),
        ("y", buffer![10i64, 20].into_array()),
    ])?;
    let ext_dtype = extension_dtype(storage.dtype())?;
    let point = ExtensionArray::try_new(ext_dtype.clone(), storage.into_array())?.into_array();
    let table = StructArray::try_from_iter([("point", point)])?.into_array();
    let zoned = ZonedStrategy::new(
        FlatLayoutStrategy::default(),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions::default(),
    );
    let strategy = flat_table()
        .with_extension_layout([ext_dtype.id()])
        .with_field_writer(field_path!(point.x), Arc::new(zoned));
    let (layout, _) = write(&strategy, vec![table]).await?;

    let point = layout.slot(1)?.vortex_expect("the table has a point field");
    let storage = point.as_::<Extension>().storage()?;
    let x = storage.slot(1)?.vortex_expect("the storage has an x field");
    let y = storage.slot(2)?.vortex_expect("the storage has a y field");
    assert!(x.is::<ZonedLayoutEncoding>());
    assert!(y.is::<Flat>());

    Ok(())
}

#[test]
fn test_mixed_projection_keeps_extension_scope() -> VortexResult<()> {
    let storage = StructArray::try_from_iter([("value", buffer![1i32].into_array())])?;
    let dtype = DType::Extension(extension_dtype(storage.dtype())?);
    let projection = pack(
        [
            ("point", root()),
            ("value", get_item("value", ext_storage(root()))),
        ],
        Nullability::NonNullable,
    )
    .bind(&dtype)?;

    assert!(rebind_to_storage(&projection, storage.dtype())?.is_none());

    Ok(())
}

#[tokio::test]
async fn test_whole_value_filter_and_mixed_projection_fallbacks() -> VortexResult<()> {
    let storage = StructArray::try_from_iter_with_validity(
        [
            ("x", buffer![1i32, 99, 3].into_array()),
            ("y", buffer![10i64, 990, 30].into_array()),
        ],
        Validity::from_iter([true, false, true]),
    )?;
    let ext_dtype = extension_dtype(storage.dtype())?;
    let array = ExtensionArray::try_new(ext_dtype.clone(), storage.into_array())?.into_array();
    let strategy = flat_table().with_extension_layout([ext_dtype.id()]);
    let (layout, segments) = write(&strategy, vec![array.clone()]).await?;

    let reader = layout.new_reader("point".into(), segments, &SESSION, &Default::default())?;
    let non_null_ext = ext_dtype.with_nullability(Nullability::NonNullable);
    let fallback = Scalar::extension_ref(
        non_null_ext.clone(),
        Scalar::struct_(
            non_null_ext.storage_dtype().clone(),
            [Scalar::from(5i32), Scalar::from(50i64)],
        ),
    );
    let filter = gt(
        get_item("x", ext_storage(fill_null(root(), lit(fallback)))),
        lit(2i32),
    )
    .bind(layout.dtype())?;

    let matched = reader
        .filter_evaluation(&(0..3), &filter, MaskFuture::ready(Mask::new_true(3)))?
        .await?;
    assert_eq!(matched, Mask::from_iter([false, true, true]));

    let projection = pack(
        [("point", root()), ("x", get_item("x", ext_storage(root())))],
        Nullability::NonNullable,
    )
    .bind(layout.dtype())?;
    let projected = reader
        .projection_evaluation(&(0..3), &projection, MaskFuture::ready(matched))?
        .await?;
    let expected = StructArray::try_from_iter([
        ("point", array.slice(1..3)?),
        (
            "x",
            PrimitiveArray::from_option_iter([None, Some(3i32)]).into_array(),
        ),
    ])?
    .into_array();
    assert_arrays_eq!(projected, expected, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[rstest]
#[case::non_extension_dtype(
    DType::Primitive(PType::I32, Nullability::NonNullable),
    vec![],
    1,
    2,
    "extension layout requires an extension dtype",
)]
#[case::non_struct_storage(
    DType::Extension(extension_dtype(&DType::Primitive(PType::I32, Nullability::NonNullable))?),
    vec![],
    1,
    2,
    "extension layout requires struct storage",
)]
#[case::direct_segments(
    DType::Extension(extension_dtype(&DType::Struct(
        StructFields::from_iter([("x", DType::Primitive(PType::I32, Nullability::NonNullable))]),
        Nullability::NonNullable,
    ))?),
    vec![SegmentId::from(0)],
    1,
    2,
    "extension layout must have no direct segments",
)]
#[case::zero_children(
    DType::Extension(extension_dtype(&DType::Struct(
        StructFields::from_iter([("x", DType::Primitive(PType::I32, Nullability::NonNullable))]),
        Nullability::NonNullable,
    ))?),
    vec![],
    0,
    2,
    "extension layout must have exactly one storage child",
)]
#[case::two_children(
    DType::Extension(extension_dtype(&DType::Struct(
        StructFields::from_iter([("x", DType::Primitive(PType::I32, Nullability::NonNullable))]),
        Nullability::NonNullable,
    ))?),
    vec![],
    2,
    2,
    "extension layout must have exactly one storage child",
)]
#[case::row_count_mismatch(
    DType::Extension(extension_dtype(&DType::Struct(
        StructFields::from_iter([("x", DType::Primitive(PType::I32, Nullability::NonNullable))]),
        Nullability::NonNullable,
    ))?),
    vec![],
    1,
    4,
    "extension layout storage row count must match the parent",
)]
fn test_deserialize_rejects_invalid_layouts(
    #[case] dtype: DType,
    #[case] segment_ids: Vec<SegmentId>,
    #[case] child_count: usize,
    #[case] row_count: u64,
    #[case] expected_err: &str,
) -> VortexResult<()> {
    let storage_dtype = DType::Struct(
        StructFields::from_iter([("x", DType::Primitive(PType::I32, Nullability::NonNullable))]),
        Nullability::NonNullable,
    );
    let children = (0..child_count)
        .map(|i| {
            FlatLayout::new(
                2,
                storage_dtype.clone(),
                SegmentId::from(u32::try_from(i).vortex_expect("small test index")),
                ReadContext::new([]),
            )
            .into_layout()
        })
        .collect();
    let children = OwnedLayoutChildren::layout_children(children);
    let read_ctx = ReadContext::new([]);
    let args = LayoutDeserializeArgs {
        session: &SESSION,
        array_read_ctx: &read_ctx,
        dtype: &dtype,
        row_count,
        segment_ids,
        children: &children,
    };

    let err = Extension
        .deserialize(&args, &EmptyMetadata)
        .expect_err("invalid extension layout must fail deserialization");
    assert!(
        err.to_string().contains(expected_err),
        "expected error containing {expected_err:?}, got {err}"
    );

    Ok(())
}
