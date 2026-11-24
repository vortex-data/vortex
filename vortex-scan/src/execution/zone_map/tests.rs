// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::bounded_max::BOUNDED_MAX_BOUND;
use vortex_array::aggregate_fn::fns::bounded_max::BOUNDED_MAX_UNKNOWN;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::stats::Stat;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_expr::Expression;
use vortex_expr::gt;
use vortex_expr::gt_eq;
use vortex_expr::lit;
use vortex_expr::lt;
use vortex_expr::root;
use vortex_expr::stats::all_nan;
use vortex_expr::stats::all_non_nan;
use vortex_expr::stats::all_non_null;
use vortex_expr::stats::all_null;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_expr::stats::stat;
use vortex_functions::expr::dynamic;
use vortex_functions::expr::is_null;
use vortex_mask::Mask;

use super::ZoneMap;
use crate::test::SESSION;

fn falsify(expr: &Expression, dtype: DType) -> BoundExpression {
    expr.bind(&dtype)
        .unwrap()
        .falsify(&SESSION)
        .unwrap()
        .unwrap()
}

fn prune(zone_map: &ZoneMap, predicate: &Expression) -> VortexResult<Mask> {
    zone_map.prune(&predicate.bind(&zone_map.column_dtype)?, &SESSION)
}

fn default_bounded_stat_max_bytes() -> NonZeroUsize {
    // SAFETY: 64 is non-zero.
    unsafe { NonZeroUsize::new_unchecked(64) }
}

#[test]
fn test_zone_map_prunes() {
    // Construct a zone map with 3 zones:
    //
    // +----------+----------+
    // |  a_min   |  a_max   |
    // +----------+----------+
    // |  1       |  5       |
    // +----------+----------+
    // |  2       |  6       |
    // +----------+----------+
    // |  3       |  7       |
    // +----------+----------+
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let zone_map = ZoneMap::try_new(
        PType::I32.into(),
        StructArray::from_fields(&[
            (
                max.to_string(),
                PrimitiveArray::new(buffer![5i32, 6i32, 7i32], Validity::AllValid).into_array(),
            ),
            (
                min.to_string(),
                PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::AllValid).into_array(),
            ),
        ])
        .unwrap(),
        Arc::new([max, min]),
        3,
        10,
    )
    .unwrap();
    let ctx = &mut SESSION.create_execution_ctx();

    // A >= 6
    // => A.max < 6
    let expr = gt_eq(root(), lit(6i32));
    let pruning_expr = falsify(&expr, PType::I32.into());
    let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, false, false]),
        ctx
    );

    // A > 5
    // => A.max <= 5
    let expr = gt(root(), lit(5i32));
    let pruning_expr = falsify(&expr, PType::I32.into());
    let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, false, false]),
        ctx
    );

    // A < 2
    // => A.min >= 2
    let expr = lt(root(), lit(2i32));
    let pruning_expr = falsify(&expr, PType::I32.into());
    let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([false, true, true]),
        ctx
    );
}

#[test]
fn bounded_display_names_satisfy_min_max_rewrites() {
    let bounded_max = BoundedMax.bind(BoundedMaxOptions {
        max_bytes: default_bounded_stat_max_bytes(),
    });
    let bounded_min = BoundedMin.bind(BoundedMinOptions {
        max_bytes: default_bounded_stat_max_bytes(),
    });
    let zone_map = ZoneMap::try_new(
        PType::I32.into(),
        StructArray::from_fields(&[
            (
                bounded_max.to_string(),
                StructArray::try_new(
                    [BOUNDED_MAX_BOUND, BOUNDED_MAX_UNKNOWN].into(),
                    vec![
                        PrimitiveArray::new(buffer![5i32, 6i32, 7i32], Validity::AllValid)
                            .into_array(),
                        BoolArray::from_iter([false, false, false]).into_array(),
                    ],
                    3,
                    Validity::AllValid,
                )
                .unwrap()
                .into_array(),
            ),
            (
                bounded_min.to_string(),
                PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::AllValid).into_array(),
            ),
        ])
        .unwrap(),
        Arc::new([bounded_max, bounded_min]),
        3,
        10,
    )
    .unwrap();
    let ctx = &mut SESSION.create_execution_ctx();

    let expr = gt(root(), lit(5i32));
    let pruning_expr = falsify(&expr, PType::I32.into());
    let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, false, false]),
        ctx
    );

    let expr = lt(root(), lit(2i32));
    let pruning_expr = falsify(&expr, PType::I32.into());
    let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([false, true, true]),
        ctx
    );
}

#[test]
fn all_null_stat_fn_lowers_to_null_count_field() {
    let null_count = NullCount.bind(EmptyOptions);
    let zone_map = ZoneMap::try_new(
        PType::U64.into(),
        StructArray::from_fields(&[(
            null_count.to_string(),
            PrimitiveArray::new(buffer![4u64, 0, 2], Validity::AllValid).into_array(),
        )])
        .unwrap(),
        Arc::new([null_count]),
        4,
        10,
    )
    .unwrap();
    let ctx = &mut SESSION.create_execution_ctx();

    let mask = prune(&zone_map, &all_null(root())).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, false, true]),
        ctx
    );

    let mask = prune(&zone_map, &all_non_null(root())).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([false, true, false]),
        ctx
    );
}

#[test]
fn all_nan_stat_fn_lowers_to_nan_count_field() {
    let nan_count = NanCount.bind(EmptyOptions);
    let zone_map = ZoneMap::try_new(
        PType::F32.into(),
        StructArray::from_fields(&[(
            nan_count.to_string(),
            PrimitiveArray::new(buffer![4u64, 0, 2], Validity::AllValid).into_array(),
        )])
        .unwrap(),
        Arc::new([nan_count]),
        4,
        10,
    )
    .unwrap();
    let ctx = &mut SESSION.create_execution_ctx();

    let mask = prune(&zone_map, &all_nan(root())).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, false, true]),
        ctx
    );

    let mask = prune(&zone_map, &all_non_nan(root())).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([false, true, false]),
        ctx
    );
}

#[test]
fn non_float_nan_stat_fns_fail_to_bind() {
    let dtype = DType::from(PType::I32);
    for expr in [all_nan(root()), all_non_nan(root())] {
        let error = expr.bind(&dtype).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not support input dtype i32"),
            "{error}"
        );
    }
}

#[test]
fn unavailable_stat_fn_lowers_to_unknown_mask() {
    let zone_map = ZoneMap::try_new(
        PType::U64.into(),
        StructArray::try_new(FieldNames::empty(), vec![], 3, Validity::NonNullable).unwrap(),
        Arc::new([]),
        4,
        10,
    )
    .unwrap();
    let ctx = &mut SESSION.create_execution_ctx();

    let mask = prune(&zone_map, &all_non_null(root())).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([false, false, false]),
        ctx
    );

    let expr = gt(root(), lit(5u64));
    let pruning_expr = falsify(&expr, PType::U64.into());
    let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([false, false, false]),
        ctx
    );
}

#[test]
fn float_min_max_stat_fn_requires_nan_count() {
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let zone_map = ZoneMap::try_new(
        PType::F32.into(),
        StructArray::from_fields(&[(
            max.to_string(),
            PrimitiveArray::new(buffer![5.0f32, 6.0, 7.0], Validity::AllValid).into_array(),
        )])
        .unwrap(),
        Arc::new([max.clone()]),
        4,
        12,
    )
    .unwrap();
    let ctx = &mut SESSION.create_execution_ctx();

    let expr = gt(root(), lit(5.0f32));
    let pruning_expr = falsify(&expr, PType::F32.into());
    let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([false, false, false]),
        ctx
    );

    let nan_count = NanCount.bind(EmptyOptions);
    let zone_map = ZoneMap::try_new(
        PType::F32.into(),
        StructArray::from_fields(&[
            (
                max.to_string(),
                PrimitiveArray::new(buffer![5.0f32, 6.0, 7.0], Validity::AllValid).into_array(),
            ),
            (
                nan_count.to_string(),
                PrimitiveArray::new(buffer![0u64, 0, 0], Validity::AllValid).into_array(),
            ),
        ])
        .unwrap(),
        Arc::new([max, nan_count]),
        4,
        12,
    )
    .unwrap();

    let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, false, false]),
        ctx
    );
}

#[test]
fn fixed_size_list_min_max_stat_fn_lowers_to_unknown_mask() {
    // Regression test for issue #8189: Min/Max is defined for FixedSizeList<T>
    // when T is orderable. If the zone map does not carry the requested stat,
    // lowering should produce an unknown typed null rather than rejecting the dtype.
    let elem_dtype = Arc::new(DType::Decimal(
        DecimalDType::new(10, 2),
        Nullability::Nullable,
    ));
    let column_dtype = DType::FixedSizeList(elem_dtype, 1, Nullability::Nullable);

    let zone_map = ZoneMap::try_new(
        column_dtype,
        StructArray::try_new(FieldNames::empty(), vec![], 3, Validity::NonNullable).unwrap(),
        Arc::new([]),
        4,
        10,
    )
    .unwrap();

    let max_fn = Stat::Max
        .aggregate_fn()
        .expect("max should have an aggregate function");
    let predicate = is_null(stat(root(), max_fn));

    // Missing StatFn lowers to a nullable null literal, so `is_null(...)` is true for every zone.
    let mask = prune(&zone_map, &predicate).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, true, true]),
        &mut SESSION.create_execution_ctx()
    );
}

#[test]
fn unsupported_aggregate_input_dtype_errors() {
    let zone_map = ZoneMap::try_new(
        DType::Null,
        StructArray::try_new(FieldNames::empty(), vec![], 3, Validity::NonNullable).unwrap(),
        Arc::new([]),
        4,
        10,
    )
    .unwrap();

    let max_fn = Stat::Max
        .aggregate_fn()
        .expect("max should have an aggregate function");
    let predicate = is_null(stat(root(), max_fn));
    let error = prune(&zone_map, &predicate).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("Aggregate function vortex.max() does not support input dtype null"),
        "{error}"
    );
}

#[test]
fn all_null_stat_fn_lowers_to_aggregate_field() {
    let all_null_agg = AllNull.bind(EmptyOptions);
    let zone_map = ZoneMap::try_new(
        PType::U64.into(),
        StructArray::from_fields(&[(
            all_null_agg.to_string(),
            BoolArray::from_iter([Some(false), Some(true), Some(true)]).into_array(),
        )])
        .unwrap(),
        Arc::new([all_null_agg]),
        4,
        10,
    )
    .unwrap();

    let mask = prune(&zone_map, &all_null(root())).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([false, true, true]),
        &mut SESSION.create_execution_ctx()
    );
}

#[test]
fn all_non_null_stat_fn_lowers_to_aggregate_field() {
    let all_non_null_agg = AllNonNull.bind(EmptyOptions);
    let zone_map = ZoneMap::try_new(
        PType::U64.into(),
        StructArray::from_fields(&[(
            all_non_null_agg.to_string(),
            BoolArray::from_iter([Some(true), Some(false), Some(false)]).into_array(),
        )])
        .unwrap(),
        Arc::new([all_non_null_agg]),
        4,
        10,
    )
    .unwrap();

    let mask = prune(&zone_map, &all_non_null(root())).unwrap();
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, false, false]),
        &mut SESSION.create_execution_ctx()
    );
}

#[test]
fn lowered_dynamic_predicate_tracks_rhs_updates() -> VortexResult<()> {
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let zone_map = ZoneMap::try_new(
        PType::I32.into(),
        StructArray::from_fields(&[
            (
                max.to_string(),
                PrimitiveArray::new(buffer![5i32, 6i32, 7i32], Validity::AllValid).into_array(),
            ),
            (
                min.to_string(),
                PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::AllValid).into_array(),
            ),
        ])?,
        Arc::new([max, min]),
        3,
        10,
    )?;
    let ctx = &mut SESSION.create_execution_ctx();

    let threshold = Arc::new(AtomicI32::new(5));
    let rhs = Arc::clone(&threshold);
    let expr = dynamic(
        CompareOperator::Gt,
        move || Some(ScalarValue::from(rhs.load(Ordering::Relaxed))),
        PType::I32.into(),
        true,
        root(),
    );

    // Lower once, as the zoned reader does, and evaluate the lowered predicate after each
    // update of the dynamic value.
    let lowered = zone_map.lower_stats(falsify(&expr, PType::I32.into()))?;

    // A > 5 => A.max <= 5
    let mask = zone_map.prune_lowered(&lowered, &SESSION)?;
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, false, false]),
        ctx
    );

    // A > 6 => A.max <= 6
    threshold.store(6, Ordering::Relaxed);
    let mask = zone_map.prune_lowered(&lowered, &SESSION)?;
    assert_arrays_eq!(
        mask.into_array(),
        BoolArray::from_iter([true, true, false]),
        ctx
    );

    Ok(())
}
