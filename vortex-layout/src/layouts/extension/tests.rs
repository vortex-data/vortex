// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::stream;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ForeignExtDType;
use vortex_array::expr::ext_storage;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::field_path;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use super::Extension;
use super::ExtensionLayout;
use super::expr::rebind_to_storage;
use crate::LayoutRef;
use crate::LayoutStrategy;
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
