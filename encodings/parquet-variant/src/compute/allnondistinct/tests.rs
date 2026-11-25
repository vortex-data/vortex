// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use parquet_variant::VariantBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use vortex_array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::AllNonDistinctParquetVariant;
use crate::ParquetVariant;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

/// Non-nullable, minimally-valid metadata column of `len` rows.
fn metadata(len: usize) -> ArrayRef {
    VarBinViewArray::from_iter_bin(vec![b"\x01\x00"; len]).into_array()
}

/// Non-nullable binary `value` column.
fn binary<T: AsRef<[u8]>>(values: impl IntoIterator<Item = T>) -> ArrayRef {
    VarBinViewArray::from_iter_bin(values).into_array()
}

fn parquet_variant(
    len: usize,
    value: Option<ArrayRef>,
    typed_value: Option<ArrayRef>,
) -> VortexResult<ArrayRef> {
    Ok(
        ParquetVariant::try_new(Validity::NonNullable, metadata(len), value, typed_value)?
            .into_array(),
    )
}

#[test]
fn all_non_distinct_matches_equal_unshredded() -> VortexResult<()> {
    let lhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), None)?;
    let rhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), None)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_detects_distinct_unshredded() -> VortexResult<()> {
    let lhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), None)?;
    let rhs = parquet_variant(2, Some(binary([b"\x10", b"\x12"])), None)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert!(!all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_matches_equal_value_and_typed() -> VortexResult<()> {
    let typed = || buffer![1i32, 2].into_array();
    let lhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), Some(typed()))?;
    let rhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), Some(typed()))?;
    let mut ctx = SESSION.create_execution_ctx();
    assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_empty_is_true() -> VortexResult<()> {
    let lhs = parquet_variant(0, Some(binary(Vec::<&[u8]>::new())), None)?;
    let rhs = parquet_variant(0, Some(binary(Vec::<&[u8]>::new())), None)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_declines_when_only_one_side_is_shredded() -> VortexResult<()> {
    let shredded = parquet_variant(
        2,
        Some(binary([b"\x10", b"\x11"])),
        Some(buffer![1i32, 2].into_array()),
    )?;
    let unshredded = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), None)?;
    let batch = StructArray::try_new(
        FieldNames::from(["lhs", "rhs"]),
        vec![shredded, unshredded],
        2,
        Validity::NonNullable,
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let aggregate = AllNonDistinct.bind(EmptyOptions);
    let result = AllNonDistinctParquetVariant.aggregate(&aggregate, &batch, &mut ctx)?;
    assert!(
        result.is_none(),
        "a shredded value cannot be compared to an unshredded one from `value` alone, so the \
         kernel must decline rather than report them identical; got {result:?}"
    );
    Ok(())
}

fn single_field_object(field: &str, value: i32) -> (Vec<u8>, Vec<u8>) {
    let mut builder = VariantBuilder::new();
    let mut obj = builder.new_object();
    obj.insert(field, value);
    obj.finish();
    builder.finish()
}

#[test]
fn all_non_distinct_detects_different_metadata_dictionaries() -> VortexResult<()> {
    let (meta_a, val_a) = single_field_object("a", 1);
    let (meta_b, val_b) = single_field_object("b", 1);
    // Both objects reference field ID 0 with value 1, so their value bytes are identical
    // while their metadata dictionaries differ.
    assert_eq!(val_a, val_b);
    assert_ne!(meta_a, meta_b);

    let lhs = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_a.as_slice()]),
        Some(binary([val_a.as_slice()])),
        None,
    )?
    .into_array();
    let rhs = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_b.as_slice()]),
        Some(binary([val_b.as_slice()])),
        None,
    )?
    .into_array();

    let batch = StructArray::try_new(
        FieldNames::from(["lhs", "rhs"]),
        vec![lhs.clone(), rhs.clone()],
        1,
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let aggregate = AllNonDistinct.bind(EmptyOptions);
    assert_eq!(
        AllNonDistinctParquetVariant.aggregate(&aggregate, &batch, &mut ctx)?,
        Some(Scalar::bool(false, Nullability::NonNullable)),
    );
    assert!(!all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_matches_logically_equal_with_different_metadata() -> VortexResult<()> {
    let (meta_x, val_x) = single_field_object("x", 1);
    // Same logical object `{"x": 1}` with dictionary `["y", "x"]` (version=1, unsorted,
    // dict_size=2, offsets=[0, 1, 2], strings="yx"), so `"x"` is field ID 1 instead of 0.
    let meta_reordered: &[u8] = b"\x01\x02\x00\x01\x02yx";
    let mut val_x_id1 = val_x.clone();
    // Single-field object layout: [header, num_elements=1, field_id, offset_0, offset_1, ...].
    assert_eq!(val_x_id1[2], 0);
    val_x_id1[2] = 1;

    let lhs = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_x.as_slice()]),
        Some(binary([val_x.as_slice()])),
        None,
    )?
    .into_array();
    let rhs = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_reordered]),
        Some(binary([val_x_id1.as_slice()])),
        None,
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(
        lhs.execute_scalar(0, &mut ctx)?,
        rhs.execute_scalar(0, &mut ctx)?
    );
    assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn test_all_non_distinct_detects_different_raw_only_nested_field_with_different_metadata()
-> VortexResult<()> {
    let (meta_empty, val_x) = VariantBuilder::new().with_value("x").finish();
    let (_, val_y) = VariantBuilder::new().with_value("y").finish();
    let (meta_extra, _) = single_field_object("unused", 0);
    assert_ne!(meta_empty, meta_extra);

    let make_shredded_object = |meta: &[u8], raw_b: &[u8]| -> VortexResult<ArrayRef> {
        let field_a = StructArray::try_new(
            FieldNames::from(["typed_value"]),
            vec![buffer![1i32].into_array()],
            1,
            Validity::NonNullable,
        )?
        .into_array();
        let field_b = StructArray::try_new(
            FieldNames::from(["value"]),
            vec![binary([raw_b])],
            1,
            Validity::NonNullable,
        )?
        .into_array();
        let typed_value = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![field_a, field_b],
            1,
            Validity::NonNullable,
        )?
        .into_array();
        Ok(ParquetVariant::try_new(
            Validity::NonNullable,
            binary([meta]),
            None,
            Some(typed_value),
        )?
        .into_array())
    };

    let lhs = make_shredded_object(&meta_empty, &val_x)?;
    let rhs = make_shredded_object(&meta_extra, &val_y)?;

    let mut ctx = SESSION.create_execution_ctx();
    assert_ne!(
        lhs.execute_scalar(0, &mut ctx)?,
        rhs.execute_scalar(0, &mut ctx)?
    );
    assert!(!all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn test_all_non_distinct_detects_different_raw_only_nested_field_with_differing_value_nullability()
-> VortexResult<()> {
    let (meta_c, val_c) = single_field_object("c", 3);
    let (_, val_x) = VariantBuilder::new().with_value("x").finish();
    let (_, val_y) = VariantBuilder::new().with_value("y").finish();
    let (_, val_masked_lhs) = VariantBuilder::new().with_value("ignored_lhs").finish();
    let (_, val_masked_rhs) = VariantBuilder::new().with_value("ignored_rhs").finish();

    let make_shredded_object =
        |root_value: ArrayRef, raw_b_rows: [&[u8]; 3]| -> VortexResult<ArrayRef> {
            let field_a = StructArray::try_new(
                FieldNames::from(["typed_value"]),
                vec![buffer![1i32, 99i32, 2i32].into_array()],
                3,
                Validity::NonNullable,
            )?
            .into_array();
            let field_b = StructArray::try_new(
                FieldNames::from(["value"]),
                vec![binary(raw_b_rows)],
                3,
                Validity::NonNullable,
            )?
            .into_array();
            let typed_value = StructArray::try_new(
                FieldNames::from(["a", "b"]),
                vec![field_a, field_b],
                3,
                Validity::NonNullable,
            )?
            .into_array();
            Ok(ParquetVariant::try_new(
                Validity::from_iter([true, false, true]),
                binary([meta_c.as_slice(), meta_c.as_slice(), meta_c.as_slice()]),
                Some(root_value),
                Some(typed_value),
            )?
            .into_array())
        };

    let lhs_root = binary([val_c.as_slice(), val_c.as_slice(), val_c.as_slice()]);
    let rhs_root = VarBinViewArray::from_iter_nullable_bin([
        Some(val_c.as_slice()),
        Some(val_c.as_slice()),
        Some(val_c.as_slice()),
    ])
    .into_array();
    assert_ne!(lhs_root.dtype(), rhs_root.dtype());

    // Row 0 matches, row 1 is masked out (shared outer null), row 2 differs in raw-only field `b`.
    let lhs = make_shredded_object(lhs_root, [&val_x, &val_masked_lhs, &val_x])?;
    let rhs = make_shredded_object(rhs_root, [&val_x, &val_masked_rhs, &val_y])?;

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(
        lhs.execute_scalar(0, &mut ctx)?,
        rhs.execute_scalar(0, &mut ctx)?
    );
    assert_ne!(
        lhs.execute_scalar(2, &mut ctx)?,
        rhs.execute_scalar(2, &mut ctx)?
    );
    assert!(!all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn test_all_non_distinct_matches_reordered_shredded_fields_with_raw_only_field() -> VortexResult<()>
{
    let (meta_c, val_c) = single_field_object("c", 3);
    let (_, val_x) = VariantBuilder::new().with_value("x").finish();

    let make_field_a = || -> VortexResult<ArrayRef> {
        Ok(StructArray::try_new(
            FieldNames::from(["typed_value"]),
            vec![buffer![1i32].into_array()],
            1,
            Validity::NonNullable,
        )?
        .into_array())
    };
    let make_field_b = || -> VortexResult<ArrayRef> {
        Ok(StructArray::try_new(
            FieldNames::from(["value"]),
            vec![binary([val_x.as_slice()])],
            1,
            Validity::NonNullable,
        )?
        .into_array())
    };

    let lhs_typed = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![make_field_a()?, make_field_b()?],
        1,
        Validity::NonNullable,
    )?
    .into_array();
    let rhs_typed = StructArray::try_new(
        FieldNames::from(["b", "a"]),
        vec![make_field_b()?, make_field_a()?],
        1,
        Validity::NonNullable,
    )?
    .into_array();

    let lhs = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_c.as_slice()]),
        Some(binary([val_c.as_slice()])),
        Some(lhs_typed),
    )?
    .into_array();
    let rhs = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_c.as_slice()]),
        Some(binary([val_c.as_slice()])),
        Some(rhs_typed),
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn test_all_non_distinct_compares_fully_shredded_leaf_despite_different_metadata()
-> VortexResult<()> {
    let (meta_a, _) = single_field_object("a", 1);
    let (meta_b, _) = single_field_object("b", 1);

    let lhs = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_a.as_slice(), meta_a.as_slice()]),
        None,
        Some(buffer![1i32, 2].into_array()),
    )?
    .into_array();
    let rhs_equal = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_b.as_slice(), meta_b.as_slice()]),
        None,
        Some(buffer![1i32, 2].into_array()),
    )?
    .into_array();
    let rhs_distinct = ParquetVariant::try_new(
        Validity::NonNullable,
        binary([meta_b.as_slice(), meta_b.as_slice()]),
        None,
        Some(buffer![1i32, 3].into_array()),
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    assert!(all_non_distinct(&lhs, &rhs_equal, &mut ctx)?);
    assert!(!all_non_distinct(&lhs, &rhs_distinct, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_accumulator_detects_differing_outer_validity() -> VortexResult<()> {
    let (meta, val) = VariantBuilder::new().with_value(42i32).finish();
    let lhs = ParquetVariant::try_new(
        Validity::from_iter([true]),
        binary([meta.as_slice()]),
        Some(binary([val.as_slice()])),
        None,
    )?
    .into_array();
    let rhs = ParquetVariant::try_new(
        Validity::from_iter([false]),
        binary([meta.as_slice()]),
        Some(binary([val.as_slice()])),
        None,
    )?
    .into_array();

    let batch = StructArray::try_new(
        FieldNames::from(["lhs", "rhs"]),
        vec![lhs.clone(), rhs.clone()],
        1,
        Validity::NonNullable,
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let mut acc = Accumulator::try_new(AllNonDistinct, EmptyOptions, batch.dtype().clone())?;
    acc.accumulate(&batch, &mut ctx)?;
    let result = acc.finish()?;
    assert_eq!(result.as_bool().value(), Some(false));
    assert!(!all_non_distinct(&lhs, &rhs, &mut ctx)?);
    Ok(())
}

#[test]
fn all_non_distinct_ignores_garbage_under_shared_outer_nulls() -> VortexResult<()> {
    let (meta_a, val_a) = single_field_object("a", 1);
    let (meta_b, val_b) = single_field_object("b", 2);
    let validity = Validity::from_iter([true, false]);

    let lhs = ParquetVariant::try_new(
        validity.clone(),
        binary([meta_a.as_slice(), meta_a.as_slice()]),
        Some(binary([val_a.as_slice(), val_a.as_slice()])),
        None,
    )?
    .into_array();
    let rhs = ParquetVariant::try_new(
        validity,
        binary([meta_a.as_slice(), meta_b.as_slice()]),
        Some(binary([val_a.as_slice(), val_b.as_slice()])),
        None,
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);

    let batch = StructArray::try_new(
        FieldNames::from(["lhs", "rhs"]),
        vec![lhs, rhs],
        2,
        Validity::NonNullable,
    )?
    .into_array();
    let mut acc = Accumulator::try_new(AllNonDistinct, EmptyOptions, batch.dtype().clone())?;
    acc.accumulate(&batch, &mut ctx)?;
    assert_eq!(acc.finish()?.as_bool().value(), Some(true));
    Ok(())
}

#[test]
fn all_non_distinct_respects_batch_struct_validity() -> VortexResult<()> {
    let (meta, val) = VariantBuilder::new().with_value(42i32).finish();
    let lhs = ParquetVariant::try_new(
        Validity::from_iter([true, true]),
        binary([meta.as_slice(), meta.as_slice()]),
        Some(binary([val.as_slice(), val.as_slice()])),
        None,
    )?
    .into_array();
    let rhs = ParquetVariant::try_new(
        Validity::from_iter([true, false]),
        binary([meta.as_slice(), meta.as_slice()]),
        Some(binary([val.as_slice(), val.as_slice()])),
        None,
    )?
    .into_array();

    // Row 1 is masked out by the outer StructArray validity, so only row 0 counts.
    let batch = StructArray::try_new(
        FieldNames::from(["lhs", "rhs"]),
        vec![lhs, rhs],
        2,
        Validity::from_iter([true, false]),
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let mut acc = Accumulator::try_new(AllNonDistinct, EmptyOptions, batch.dtype().clone())?;
    acc.accumulate(&batch, &mut ctx)?;
    assert_eq!(acc.finish()?.as_bool().value(), Some(true));
    Ok(())
}
