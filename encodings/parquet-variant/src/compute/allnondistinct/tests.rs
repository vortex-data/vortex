// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use parquet_variant::VariantBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use vortex_array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ParquetVariant;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

fn metadata(len: usize) -> ArrayRef {
    VarBinViewArray::from_iter_bin(vec![b"\x01\x00\x00"; len]).into_array()
}

fn binary<T: AsRef<[u8]>>(values: impl IntoIterator<Item = T>) -> ArrayRef {
    VarBinViewArray::from_iter_bin(values).into_array()
}

fn pv(
    validity: Validity,
    meta: ArrayRef,
    value: Option<ArrayRef>,
    typed_value: Option<ArrayRef>,
) -> VortexResult<ArrayRef> {
    Ok(ParquetVariant::try_new(validity, meta, value, typed_value)?.into_array())
}

fn single_field_object(field: &str, value: i32) -> (Vec<u8>, Vec<u8>) {
    let mut builder = VariantBuilder::new();
    let mut obj = builder.new_object();
    obj.insert(field, value);
    obj.finish();
    builder.finish()
}

fn shredded_ab_object(
    validity: Validity,
    meta: ArrayRef,
    root_value: Option<ArrayRef>,
    a_vals: ArrayRef,
    raw_b: ArrayRef,
    swap_fields: bool,
) -> VortexResult<ArrayRef> {
    let len = a_vals.len();
    let field_a = StructArray::try_new(
        FieldNames::from(["typed_value"]),
        vec![a_vals],
        len,
        Validity::NonNullable,
    )?
    .into_array();
    let field_b = StructArray::try_new(
        FieldNames::from(["value"]),
        vec![raw_b],
        len,
        Validity::NonNullable,
    )?
    .into_array();
    let (names, fields) = if swap_fields {
        (FieldNames::from(["b", "a"]), vec![field_b, field_a])
    } else {
        (FieldNames::from(["a", "b"]), vec![field_a, field_b])
    };
    let typed = StructArray::try_new(names, fields, len, Validity::NonNullable)?.into_array();
    pv(validity, meta, root_value, Some(typed))
}

fn accumulate_pair(
    lhs: ArrayRef,
    rhs: ArrayRef,
    batch_validity: Validity,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<bool> {
    let len = lhs.len();
    let batch = StructArray::try_new(
        FieldNames::from(["lhs", "rhs"]),
        vec![lhs, rhs],
        len,
        batch_validity,
    )?
    .into_array();
    let mut acc = Accumulator::try_new(AllNonDistinct, EmptyOptions, batch.dtype().clone())?;
    acc.accumulate(&batch, ctx)?;
    Ok(acc.finish()?.as_bool().value().unwrap_or(false))
}

#[test]
fn all_non_distinct_basic_and_empty() -> VortexResult<()> {
    let (_, v1) = VariantBuilder::new().with_value(1i32).finish();
    let (_, v2) = VariantBuilder::new().with_value(2i32).finish();
    let (_, v3) = VariantBuilder::new().with_value(3i32).finish();
    let mut ctx = SESSION.create_execution_ctx();

    let empty = pv(
        Validity::NonNullable,
        metadata(0),
        Some(binary(Vec::<&[u8]>::new())),
        None,
    )?;
    assert!(all_non_distinct(&empty, &empty, &mut ctx)?);

    let lhs = pv(
        Validity::NonNullable,
        metadata(2),
        Some(binary([v1.as_slice(), v2.as_slice()])),
        None,
    )?;
    let rhs_diff = pv(
        Validity::NonNullable,
        metadata(2),
        Some(binary([v1.as_slice(), v3.as_slice()])),
        None,
    )?;
    assert!(all_non_distinct(&lhs, &lhs, &mut ctx)?);
    assert!(!all_non_distinct(&lhs, &rhs_diff, &mut ctx)?);

    let both = pv(
        Validity::NonNullable,
        metadata(2),
        Some(binary([v1.as_slice(), v2.as_slice()])),
        Some(buffer![1i32, 2].into_array()),
    )?;
    assert!(all_non_distinct(&both, &both, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_mixed_shredded_and_unshredded() -> VortexResult<()> {
    let (m1, v1) = VariantBuilder::new().with_value(1i32).finish();
    let (m2, v2) = VariantBuilder::new().with_value(2i32).finish();
    let meta = binary([m1.as_slice(), m2.as_slice()]);

    let unshredded = pv(
        Validity::NonNullable,
        meta.clone(),
        Some(binary([v1.as_slice(), v2.as_slice()])),
        None,
    )?;
    let shredded_eq = pv(
        Validity::NonNullable,
        meta.clone(),
        None,
        Some(buffer![1i32, 2].into_array()),
    )?;
    let shredded_ne = pv(
        Validity::NonNullable,
        meta,
        None,
        Some(buffer![1i32, 3].into_array()),
    )?;

    let mut ctx = SESSION.create_execution_ctx();
    assert!(all_non_distinct(&shredded_eq, &unshredded, &mut ctx)?);
    assert!(!all_non_distinct(&shredded_ne, &unshredded, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_metadata_dictionaries() -> VortexResult<()> {
    let (meta_a, val_a) = single_field_object("a", 1);
    let (meta_b, val_b) = single_field_object("b", 1);
    assert_eq!(val_a, val_b);

    let mut ctx = SESSION.create_execution_ctx();
    let a = pv(
        Validity::NonNullable,
        binary([meta_a.as_slice()]),
        Some(binary([val_a.as_slice()])),
        None,
    )?;
    let b = pv(
        Validity::NonNullable,
        binary([meta_b.as_slice()]),
        Some(binary([val_b.as_slice()])),
        None,
    )?;
    assert!(!all_non_distinct(&a, &b, &mut ctx)?);

    // Logically equal `{"x": 1}` with dictionary `["y", "x"]` so `"x"` is field ID 1.
    let (meta_x, val_x) = single_field_object("x", 1);
    let mut val_x_id1 = val_x.clone();
    val_x_id1[2] = 1;
    let x0 = pv(
        Validity::NonNullable,
        binary([meta_x.as_slice()]),
        Some(binary([val_x.as_slice()])),
        None,
    )?;
    let x1 = pv(
        Validity::NonNullable,
        binary([b"\x01\x02\x00\x01\x02yx".as_slice()]),
        Some(binary([val_x_id1.as_slice()])),
        None,
    )?;
    assert!(all_non_distinct(&x0, &x1, &mut ctx)?);

    // Fully shredded leaf ignores differing metadata dictionaries.
    let leaf_a = pv(
        Validity::NonNullable,
        binary([meta_a.as_slice(), meta_a.as_slice()]),
        None,
        Some(buffer![1i32, 2].into_array()),
    )?;
    let leaf_b_eq = pv(
        Validity::NonNullable,
        binary([meta_b.as_slice(), meta_b.as_slice()]),
        None,
        Some(buffer![1i32, 2].into_array()),
    )?;
    let leaf_b_ne = pv(
        Validity::NonNullable,
        binary([meta_b.as_slice(), meta_b.as_slice()]),
        None,
        Some(buffer![1i32, 3].into_array()),
    )?;
    assert!(all_non_distinct(&leaf_a, &leaf_b_eq, &mut ctx)?);
    assert!(!all_non_distinct(&leaf_a, &leaf_b_ne, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_raw_only_nested_fields_and_reordered_schema() -> VortexResult<()> {
    let (meta_c, val_c) = single_field_object("c", 3);
    let (_, val_x) = VariantBuilder::new().with_value("x").finish();
    let (_, val_y) = VariantBuilder::new().with_value("y").finish();
    let mut ctx = SESSION.create_execution_ctx();

    // Mixed root `value` presence with a raw-only nested field `b`.
    let mut empty_obj = VariantBuilder::new();
    empty_obj.new_object().finish();
    let (meta_empty, val_empty) = empty_obj.finish();
    let with_root = shredded_ab_object(
        Validity::NonNullable,
        binary([meta_empty.as_slice()]),
        Some(binary([val_empty.as_slice()])),
        buffer![1i32].into_array(),
        binary([val_x.as_slice()]),
        false,
    )?;
    let no_root_eq = shredded_ab_object(
        Validity::NonNullable,
        binary([meta_empty.as_slice()]),
        None,
        buffer![1i32].into_array(),
        binary([val_x.as_slice()]),
        false,
    )?;
    let no_root_ne = shredded_ab_object(
        Validity::NonNullable,
        binary([meta_empty.as_slice()]),
        None,
        buffer![1i32].into_array(),
        binary([val_y.as_slice()]),
        false,
    )?;
    assert!(all_non_distinct(&with_root, &no_root_eq, &mut ctx)?);
    assert!(!all_non_distinct(&with_root, &no_root_ne, &mut ctx)?);

    // Reordered shredded schema (`["a", "b"]` vs `["b", "a"]`) with a raw-only field.
    let lhs_ord = shredded_ab_object(
        Validity::NonNullable,
        binary([meta_c.as_slice()]),
        Some(binary([val_c.as_slice()])),
        buffer![1i32].into_array(),
        binary([val_x.as_slice()]),
        false,
    )?;
    let rhs_ord = shredded_ab_object(
        Validity::NonNullable,
        binary([meta_c.as_slice()]),
        Some(binary([val_c.as_slice()])),
        buffer![1i32].into_array(),
        binary([val_x.as_slice()]),
        true,
    )?;
    assert!(all_non_distinct(&lhs_ord, &rhs_ord, &mut ctx)?);

    // Differing root `value` nullability with a masked null row and differing raw-only field.
    let (_, masked_l) = VariantBuilder::new().with_value("ignored_l").finish();
    let (_, masked_r) = VariantBuilder::new().with_value("ignored_r").finish();
    let meta3 = binary([meta_c.as_slice(), meta_c.as_slice(), meta_c.as_slice()]);
    let val3_non_null = binary([val_c.as_slice(), val_c.as_slice(), val_c.as_slice()]);
    let val3_nullable = VarBinViewArray::from_iter_nullable_bin([
        Some(val_c.as_slice()),
        Some(val_c.as_slice()),
        Some(val_c.as_slice()),
    ])
    .into_array();
    let lhs_null = shredded_ab_object(
        Validity::from_iter([true, false, true]),
        meta3.clone(),
        Some(val3_non_null),
        buffer![1i32, 99, 2].into_array(),
        binary([val_x.as_slice(), masked_l.as_slice(), val_x.as_slice()]),
        false,
    )?;
    let rhs_null_eq = shredded_ab_object(
        Validity::from_iter([true, false, true]),
        meta3.clone(),
        Some(val3_nullable.clone()),
        buffer![1i32, 99, 2].into_array(),
        binary([val_x.as_slice(), masked_r.as_slice(), val_x.as_slice()]),
        false,
    )?;
    let rhs_null_ne = shredded_ab_object(
        Validity::from_iter([true, false, true]),
        meta3,
        Some(val3_nullable),
        buffer![1i32, 99, 2].into_array(),
        binary([val_x.as_slice(), masked_r.as_slice(), val_y.as_slice()]),
        false,
    )?;
    assert!(all_non_distinct(&lhs_null, &rhs_null_eq, &mut ctx)?);
    assert!(!all_non_distinct(&lhs_null, &rhs_null_ne, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_physical_representation_fallbacks() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    // Same metadata, value in `typed_value` on lhs vs raw `value` on rhs.
    let (meta, val_42) = VariantBuilder::new().with_value(42i32).finish();
    let (_, val_43) = VariantBuilder::new().with_value(43i32).finish();
    let typed_42 = pv(
        Validity::NonNullable,
        binary([meta.as_slice()]),
        Some(VarBinViewArray::from_iter_nullable_bin([None::<&[u8]>]).into_array()),
        Some(PrimitiveArray::from_option_iter([Some(42i32)]).into_array()),
    )?;
    let raw_42 = pv(
        Validity::NonNullable,
        binary([meta.as_slice()]),
        Some(VarBinViewArray::from_iter_nullable_bin([Some(val_42.as_slice())]).into_array()),
        Some(PrimitiveArray::from_option_iter([None::<i32>]).into_array()),
    )?;
    let raw_43 = pv(
        Validity::NonNullable,
        binary([meta.as_slice()]),
        Some(VarBinViewArray::from_iter_nullable_bin([Some(val_43.as_slice())]).into_array()),
        Some(PrimitiveArray::from_option_iter([None::<i32>]).into_array()),
    )?;
    assert!(all_non_distinct(&typed_42, &raw_42, &mut ctx)?);
    assert!(!all_non_distinct(&typed_42, &raw_43, &mut ctx)?);

    // ShortString("x") vs LongString("x").
    let short_str = pv(
        Validity::NonNullable,
        metadata(1),
        Some(binary([[0x05u8, b'x']])),
        None,
    )?;
    let long_str = pv(
        Validity::NonNullable,
        metadata(1),
        Some(binary([[0x40u8, 0x01, 0x00, 0x00, 0x00, b'x']])),
        None,
    )?;
    assert!(all_non_distinct(&short_str, &long_str, &mut ctx)?);

    // Unshredded UUID vs same-text string remain distinct.
    let uuid_bytes: &[u8] = &[
        0x50, 0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
        0x00, 0x00,
    ];
    let (uuid_meta, uuid_str) = VariantBuilder::new()
        .with_value("550e8400-e29b-41d4-a716-446655440000")
        .finish();
    let uuid_arr = pv(
        Validity::NonNullable,
        binary([uuid_meta.as_slice()]),
        Some(binary([uuid_bytes])),
        None,
    )?;
    let str_arr = pv(
        Validity::NonNullable,
        binary([uuid_meta.as_slice()]),
        Some(binary([uuid_str.as_slice()])),
        None,
    )?;
    assert!(!all_non_distinct(&uuid_arr, &str_arr, &mut ctx)?);

    // Missing active raw `value` matches encoded `Variant::Null`.
    let missing_val = pv(
        Validity::NonNullable,
        metadata(1),
        Some(VarBinViewArray::from_iter_nullable_bin([None::<&[u8]>]).into_array()),
        None,
    )?;
    let encoded_null = pv(
        Validity::NonNullable,
        metadata(1),
        Some(VarBinViewArray::from_iter_nullable_bin([Some([0x00u8].as_slice())]).into_array()),
        None,
    )?;
    assert!(all_non_distinct(&missing_val, &encoded_null, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_outer_and_batch_validity() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (meta, val) = VariantBuilder::new().with_value(42i32).finish();

    // Differing outer validity on an active row returns false.
    let valid_1 = pv(
        Validity::from_iter([true]),
        binary([meta.as_slice()]),
        Some(binary([val.as_slice()])),
        None,
    )?;
    let null_1 = pv(
        Validity::from_iter([false]),
        binary([meta.as_slice()]),
        Some(binary([val.as_slice()])),
        None,
    )?;
    assert!(!accumulate_pair(
        valid_1,
        null_1,
        Validity::NonNullable,
        &mut ctx
    )?);

    // Garbage under shared outer nulls is ignored.
    let (meta_a, val_a) = single_field_object("a", 1);
    let (meta_b, val_b) = single_field_object("b", 2);
    let shared_lhs = pv(
        Validity::from_iter([true, false]),
        binary([meta_a.as_slice(), meta_a.as_slice()]),
        Some(binary([val_a.as_slice(), val_a.as_slice()])),
        None,
    )?;
    let shared_rhs = pv(
        Validity::from_iter([true, false]),
        binary([meta_a.as_slice(), meta_b.as_slice()]),
        Some(binary([val_a.as_slice(), val_b.as_slice()])),
        None,
    )?;
    assert!(all_non_distinct(&shared_lhs, &shared_rhs, &mut ctx)?);
    assert!(accumulate_pair(
        shared_lhs,
        shared_rhs,
        Validity::NonNullable,
        &mut ctx
    )?);

    // Batch StructArray validity masks out row 1 where outer validity differs.
    let batch_lhs = pv(
        Validity::from_iter([true, true]),
        binary([meta.as_slice(), meta.as_slice()]),
        Some(binary([val.as_slice(), val.as_slice()])),
        None,
    )?;
    let batch_rhs = pv(
        Validity::from_iter([true, false]),
        binary([meta.as_slice(), meta.as_slice()]),
        Some(binary([val.as_slice(), val.as_slice()])),
        None,
    )?;
    assert!(accumulate_pair(
        batch_lhs,
        batch_rhs,
        Validity::from_iter([true, false]),
        &mut ctx
    )?);
    Ok(())
}
