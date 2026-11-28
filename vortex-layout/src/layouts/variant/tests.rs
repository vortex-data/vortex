// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::test_harness::check_metadata;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_expr::lit;
use vortex_expr::mask;
use vortex_io::runtime::single::block_on;
use vortex_session::registry::ReadContext;

use super::Variant;
use super::VariantLayout;
use super::storage_dtype;
use crate::LayoutBuildContext;
use crate::LayoutDeserialization;
use crate::LayoutId;
use crate::LayoutRef;
use crate::VTable;
use crate::layout_children;
use crate::layouts::flat::FlatLayout;
use crate::layouts::struct_::Struct;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentId;
use crate::segments::TestSegments;
use crate::session::LayoutSessionExt;
use crate::test::SESSION;

fn flat_child(dtype: DType, segment: u32) -> LayoutRef {
    FlatLayout::new(3, dtype, SegmentId::from(segment), ReadContext::new([])).into_layout()
}

fn layouts() -> VortexResult<(LayoutRef, VariantLayout)> {
    let dtype = DType::Variant(Nullability::Nullable);
    let shredded = DType::Primitive(PType::I64, Nullability::Nullable);
    let storage = StructLayout::new(
        3,
        storage_dtype(&dtype, shredded.clone()),
        vec![
            flat_child(dtype.clone(), 0),
            flat_child(shredded.clone(), 1),
        ],
    )
    .into_layout();
    let variant = VariantLayout::from_storage(dtype, shredded, &storage, true)?;
    Ok((storage, variant))
}

fn read(
    id: LayoutId,
    dtype: &DType,
    metadata: &[u8],
    children: Vec<LayoutRef>,
) -> VortexResult<LayoutRef> {
    let plugin = SESSION
        .layouts()
        .registry()
        .get(&id)
        .vortex_expect("built-in layout");
    plugin.deserialize(
        LayoutDeserialization {
            serialized_id: id,
            dtype,
            row_count: 3,
            metadata,
            segment_ids: vec![],
            children: layout_children(children).as_ref(),
        },
        &LayoutBuildContext {
            session: &SESSION,
            array_read_ctx: &ReadContext::new([]),
        },
    )
}

#[test]
fn constant_projection_keeps_its_declared_dtype_over_variant_storage() -> VortexResult<()> {
    let (_, layout) = layouts()?;
    let reader = layout.new_reader(
        "".into(),
        Arc::new(TestSegments::default()),
        &SESSION,
        &Default::default(),
    )?;
    let expr = mask(lit(1i32), lit(true)).bind(reader.dtype())?;
    let projection = reader.projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?;
    let result = block_on(|_| projection)?;
    assert_eq!(result.dtype(), expr.dtype());
    let expected = PrimitiveArray::from_option_iter([Some(1i32); 3]);
    assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn struct_and_variant_round_trip_as_distinct_layouts() -> VortexResult<()> {
    let (storage, variant) = layouts()?;
    let plain = SESSION.layout_serialize(storage.as_ref())?;
    assert_eq!(plain.serialized_id, Struct.id());
    assert!(plain.metadata.is_empty());
    let plain_roundtrip = read(
        plain.serialized_id,
        storage.dtype(),
        &plain.metadata,
        storage.children()?,
    )?;
    assert!(plain_roundtrip.is::<Struct>());
    assert_eq!(plain_roundtrip.new_plan()?.dtype(), storage.dtype());

    assert_eq!(variant.nchildren(), 2);
    assert_eq!(variant.slot_to_child(0), None);
    let variant = variant.into_layout();
    let written = SESSION.layout_serialize(variant.as_ref())?;
    assert_eq!(written.serialized_id, Variant.id());
    check_metadata("variant.metadata", &written.metadata);
    let mut children = variant.children()?;
    // A trailing child with a different row count is ignored.
    children.push(
        FlatLayout::new(0, DType::Null, SegmentId::from(2), ReadContext::new([])).into_layout(),
    );
    let decoded = read(written.serialized_id, variant.dtype(), &written.metadata, children)?;
    assert!(decoded.is::<Variant>());
    assert_eq!(decoded.dtype(), variant.dtype());
    assert_eq!(decoded.new_plan()?.dtype(), variant.dtype());
    let fields = decoded.children()?;
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].dtype(), variant.dtype());
    assert_eq!(
        fields[1].dtype(),
        &DType::Primitive(PType::I64, Nullability::Nullable)
    );

    assert!(read(Struct.id(), variant.dtype(), &[], variant.children()?).is_err());
    assert!(read(Variant.id(), storage.dtype(), &written.metadata, storage.children()?).is_err());
    assert!(read(Struct.id(), storage.dtype(), &[0x18, 0x01], storage.children()?).is_err());
    Ok(())
}

#[test]
fn variant_validates_both_required_children() -> VortexResult<()> {
    let (_, layout) = layouts()?;
    let layout = layout.into_layout();
    let written = SESSION.layout_serialize(layout.as_ref())?;
    let mut children = layout.children()?;
    let shredded = children.pop().vortex_expect("shredded child");
    let read_children = |children| {
        read(
            written.serialized_id,
            layout.dtype(),
            &written.metadata,
            children,
        )
    };
    let err = read_children(children.clone()).unwrap_err();
    assert!(err.to_string().contains("needs core and shredded children"));
    children.push(
        FlatLayout::new(
            1,
            shredded.dtype().clone(),
            SegmentId::from(1),
            ReadContext::new([]),
        )
        .into_layout(),
    );
    let err = read_children(children).unwrap_err();
    assert!(
        err.to_string()
            .contains("Column 1 row count does not match parent")
    );
    Ok(())
}
