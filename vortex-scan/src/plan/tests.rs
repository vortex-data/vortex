// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::Expression;
use vortex_expr::and;
use vortex_expr::bound::and as bound_and;
use vortex_expr::cast;
use vortex_expr::checked_add;
use vortex_expr::get_item;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::pack;
use vortex_expr::root;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_functions::expr::byte_length;
use vortex_functions::expr::dynamic;
use vortex_functions::expr::is_null;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::execution::pipeline::Scan;
use crate::plan::Filter;
use crate::plan::*;
use crate::row_idx::row_idx;
use crate::source::SegmentId;
use crate::test::SESSION;
use crate::test::concat;
use crate::test::dict;
use crate::test::flat;
use crate::test::struct_plan;

fn primitive(ptype: PType, nullability: Nullability) -> DType {
    DType::Primitive(ptype, nullability)
}

fn make_eval(expression: Expression, child: PlanRef) -> VortexResult<EvalPlan> {
    let expression = expression.bind(child.dtype())?.optimize_recursive()?;
    EvalPlan::try_new(expression, child)
}

fn make_row_idx_plan(expression: Expression, child: PlanRef) -> VortexResult<PlanRef> {
    let expression = expression.bind(child.dtype())?.optimize_recursive()?;
    plan_row_idx_expression(expression, child, &SESSION)
}

fn child_of(plan: &PlanRef, slot: usize) -> VortexResult<PlanRef> {
    plan.slot_required(slot)
}

/// The aggregate zone table of a max over `dtype`, and the max aggregate.
fn max_zones_dtype(dtype: &DType) -> VortexResult<(AggregateFnRef, DType)> {
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let max_dtype = max
        .state_dtype(dtype)
        .ok_or_else(|| vortex_err!("max does not support {dtype}"))?;
    let zones_dtype = DType::Struct(
        StructFields::from_iter([(max.to_string(), max_dtype.as_nullable())]),
        Nullability::NonNullable,
    );
    Ok((max, zones_dtype))
}

fn zoned_plan(
    data: PlanRef,
    zones: PlanRef,
    zone_len: NonZeroUsize,
    aggregates: Arc<[AggregateFnRef]>,
) -> PlanRef {
    ZonedPlan::new(data, zones, zone_len.get() as u64, aggregates).into_plan()
}

#[test]
fn optimize_preserves_unvisited_concat_children() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let slots = vec![
        Some(LazyPlanRef::ready(flat(2, dtype.clone(), 0))),
        Some(LazyPlanRef::lazy(|| {
            Err(vortex_err!("unvisited child initialized"))
        })),
    ]
    .into_iter()
    .collect();
    // SAFETY: Each logical child has two rows of the same dtype. The second initializer
    // deliberately fails so the test observes whether optimization visits it eagerly.
    let plan =
        unsafe { ConcatPlan::from_slots_unchecked(dtype, 4, vec![0, 2].into(), slots) }.into_plan();
    let plan = optimize(plan, &SESSION)?;
    assert_eq!(child_of(&plan, 0)?.row_count(), 2);
    let error = child_of(&plan, 1).expect_err("the second initializer must fail when visited");
    assert!(error.to_string().contains("unvisited child initialized"));
    Ok(())
}

#[test]
fn pack_validity_child_must_match_nullability() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let fields = StructFields::from_iter([("a", field_dtype.clone())]);
    let plan = struct_plan(
        3,
        DType::Struct(fields, Nullability::NonNullable),
        vec![flat(3, field_dtype, 0)],
    );
    let validity = flat(3, DType::Bool(Nullability::NonNullable), 1);

    let error = plan
        .with_slots(vec![validity, child_of(&plan, Pack::FIELDS_OFFSET)?])
        .err()
        .ok_or_else(|| vortex_err!("validity on a non-nullable Pack unexpectedly succeeded"))?;
    assert!(
        error
            .to_string()
            .contains("Non-nullable Pack has a validity child"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn with_slots_rejects_mismatched_arity() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = concat(
        3,
        dtype.clone(),
        vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)],
    );
    let plan = layout;

    let error = plan
        .with_slots(vec![child_of(&plan, 0)?])
        .err()
        .ok_or_else(|| vortex_err!("mismatched arity unexpectedly succeeded"))?;
    assert!(
        error
            .to_string()
            .contains("Concat expects 2 slots but got 1"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn with_slots_replaces_children_in_order() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = concat(
        3,
        dtype.clone(),
        vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)],
    );
    let plan = layout;

    let swapped = plan.with_slots(vec![child_of(&plan, 1)?, child_of(&plan, 0)?])?;
    assert_eq!(child_of(&swapped, 0)?.row_count(), 1);
    assert_eq!(child_of(&swapped, 1)?.row_count(), 2);
    assert_eq!(swapped.as_::<Concat>().row_offsets(), &[0, 1]);
    Ok(())
}

#[test]
fn eval_try_new_validates_expression_root_dtype() -> VortexResult<()> {
    let expression = root().bind(&primitive(PType::I32, Nullability::NonNullable))?;
    let child = flat(3, primitive(PType::I64, Nullability::NonNullable), 0);

    let error = EvalPlan::try_new(expression, child)
        .err()
        .ok_or_else(|| vortex_err!("mismatched Eval root dtype unexpectedly succeeded"))?;
    assert!(
        error
            .to_string()
            .contains("Eval expression is not bound to child dtype i64"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn optimize_drops_identity_expressions() -> VortexResult<()> {
    let child = flat(3, primitive(PType::I32, Nullability::NonNullable), 0);
    let expression = root().bind(child.dtype())?;
    let plan: PlanRef = EvalPlan::try_new(expression, child)?.into_plan();

    assert!(optimize(plan, &SESSION)?.is::<Filter>());
    Ok(())
}

#[test]
fn optimize_rewrites_nested_children() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = concat(
        3,
        dtype.clone(),
        vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)],
    );
    let chunked = layout;

    // Wrap the first chunk in an identity expression, which optimization should remove without
    // any chunked-specific rule.
    let chunk = child_of(&chunked, 0)?;
    let identity: PlanRef = EvalPlan::try_new(root().bind(chunk.dtype())?, chunk)?.into_plan();
    let wrapped = chunked.with_slots(vec![identity, child_of(&chunked, 1)?])?;

    let optimized = optimize(wrapped, &SESSION)?;
    assert!(optimized.is::<Concat>());
    assert!(child_of(&optimized, 0)?.is::<Filter>());
    assert!(child_of(&optimized, 1)?.is::<Filter>());
    Ok(())
}

#[test]
fn plan_display_matches_array_tree_display_shape() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(3, field_dtype.clone(), 0), flat(3, field_dtype, 1)],
    );
    let child = layout;
    let expression = get_item("a", root()).bind(child.dtype())?;
    let plan = EvalPlan::try_new(expression, child)?;

    assert_eq!(plan.to_string(), "vortex.plan.eval(i32, rows=3) expr=$.a");
    let plan: PlanRef = plan.into_plan();

    assert_eq!(plan.to_string(), "vortex.plan.eval(i32, rows=3) expr=$.a");
    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.eval(i32, rows=3) expr=$.a
      child: vortex.plan.pack({a=i32, b=i32}, rows=3)
        a: vortex.plan.filter(i32, rows=3)
          child: vortex.test.source(i32, rows=3)
        b: vortex.plan.filter(i32, rows=3)
          child: vortex.test.source(i32, rows=3)
    ");

    struct DepthExtractor;

    impl PlanTreeExtractor<PlanRef, PlanTreeContext> for DepthExtractor {
        fn write_header(
            &self,
            _plan: &PlanRef,
            context: &PlanTreeContext,
            formatter: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(formatter, " depth={}", context.depth())
        }
    }

    insta::assert_snapshot!(plan.tree_display_builder().with(DepthExtractor), @r"
    root: depth=0
      child: depth=1
        a: depth=2
          child: depth=3
        b: depth=2
          child: depth=3
    ");

    let nullable_fields = StructFields::from_iter([
        ("a", primitive(PType::I32, Nullability::NonNullable)),
        ("b", primitive(PType::I32, Nullability::NonNullable)),
    ]);
    let nullable_layout = struct_plan(
        3,
        DType::Struct(nullable_fields, Nullability::Nullable),
        vec![
            flat(3, DType::Bool(Nullability::NonNullable), 2),
            flat(3, primitive(PType::I32, Nullability::NonNullable), 3),
            flat(3, primitive(PType::I32, Nullability::NonNullable), 4),
        ],
    );
    let nullable = nullable_layout;
    insta::assert_snapshot!(nullable.tree_display_builder(), @r"
    root:
      validity:
        child:
      a:
        child:
      b:
        child:
    ");
    Ok(())
}

#[test]
fn chunked_plan_display_names_chunks() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = concat(
        3,
        dtype.clone(),
        vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)],
    );
    let plan = layout;

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.concat(i32, rows=3)
      chunks[0]: vortex.plan.filter(i32, rows=2)
        child: vortex.test.source(i32, rows=2)
      chunks[1]: vortex.plan.filter(i32, rows=1)
        child: vortex.test.source(i32, rows=1)
    ");
    Ok(())
}

#[test]
fn dict_plan_display_names_logical_children() -> VortexResult<()> {
    let layout = dict(
        flat(2, primitive(PType::I32, Nullability::NonNullable), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    );
    let plan = layout;

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.take(i32, rows=3)
      codes: vortex.plan.filter(u8, rows=3)
        child: vortex.test.source(u8, rows=3)
      values: vortex.plan.filter(i32, rows=2)
        child: vortex.test.source(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn list_plan_display_handles_optional_validity() -> VortexResult<()> {
    let element_dtype = primitive(PType::I32, Nullability::NonNullable);
    let offsets_dtype = primitive(PType::U32, Nullability::NonNullable);
    let non_nullable_layout = list_plan(
        DType::List(Arc::new(element_dtype.clone()), Nullability::NonNullable),
        flat(4, element_dtype.clone(), 0),
        flat(3, offsets_dtype.clone(), 1),
        None,
    );
    let non_nullable = non_nullable_layout;

    insta::assert_snapshot!(non_nullable.display_tree(), @r"
    root: vortex.plan.list_pack(list(i32), rows=2)
      elements: vortex.plan.filter(i32, rows=4)
        child: vortex.test.source(i32, rows=4)
      offsets: vortex.plan.filter(u32, rows=3)
        child: vortex.test.source(u32, rows=3)
    ");

    let nullable_layout = list_plan(
        DType::List(Arc::new(element_dtype.clone()), Nullability::Nullable),
        flat(4, element_dtype, 2),
        flat(3, offsets_dtype, 3),
        Some(flat(2, DType::Bool(Nullability::NonNullable), 4)),
    );
    let nullable = nullable_layout;

    insta::assert_snapshot!(nullable.display_tree(), @r"
    root: vortex.plan.list_pack(list(i32)?, rows=2)
      elements: vortex.plan.filter(i32, rows=4)
        child: vortex.test.source(i32, rows=4)
      offsets: vortex.plan.filter(u32, rows=3)
        child: vortex.test.source(u32, rows=3)
      validity: vortex.plan.filter(bool, rows=2)
        child: vortex.test.source(bool, rows=2)
    ");
    Ok(())
}

#[test]
fn row_idx_only_expression_uses_row_idx_source() -> VortexResult<()> {
    let layout = flat(3, primitive(PType::I32, Nullability::NonNullable), 0);
    let plan = make_row_idx_plan(row_idx(), layout)?;

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(u64, rows=3) expr=$
      child: vortex.plan.row_idx(u64, rows=3)
    ");

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @"root: vortex.plan.row_idx(u64, rows=3)");
    assert!(optimized.is::<RowIdx>());
    Ok(())
}

#[test]
fn expression_partitions_across_row_idx_and_struct() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = dict(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    );
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 2)],
    );
    let expression = and(
        gt(row_idx(), lit(11_u64)),
        and(
            gt(get_item("a", root()), lit(5_i32)),
            gt(get_item("b", root()), lit(7_i32)),
        ),
    );
    let plan = make_row_idx_plan(expression, layout)?;

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.0 and $.1)
      child: vortex.plan.pack({0=bool, 1=bool}, rows=3)
        0: vortex.plan.eval(bool, rows=3) expr=($ > 11u64)
          child: vortex.plan.row_idx(u64, rows=3)
        1: vortex.plan.eval(bool, rows=3) expr=($.0 and $.1)
          child: vortex.plan.pack({0=bool, 1=bool}, rows=3)
            0: vortex.plan.take(bool, rows=3)
              codes: vortex.plan.filter(u8, rows=3)
                child: vortex.test.source(u8, rows=3)
              values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
                child: vortex.plan.filter(i32, rows=2)
                  child: vortex.test.source(i32, rows=2)
            1: vortex.plan.eval(bool, rows=3) expr=($ > 7i32)
              child: vortex.plan.filter(i32, rows=3)
                child: vortex.test.source(i32, rows=3)
    ");
    Ok(())
}

#[test]
fn empty_projection_prunes_row_idx_child_fields() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(3, value_dtype.clone(), 0), flat(3, value_dtype, 1)],
    );
    let projection = pack(
        std::iter::empty::<(&str, Expression)>(),
        Nullability::NonNullable,
    );
    let plan = make_row_idx_plan(projection, layout)?;

    let optimized = optimize(plan, &SESSION)?;
    let projection = optimized
        .as_opt::<Eval>()
        .ok_or_else(|| vortex_err!("optimized plan has no projection expression"))?;
    let child = projection.child_plan()?;
    let empty_struct = child
        .as_opt::<Pack>()
        .ok_or_else(|| vortex_err!("empty projection did not prune the RowIdx child"))?;

    assert_eq!(empty_struct.nfields(), 0);
    assert_eq!(empty_struct.nslots(), Pack::FIELDS_OFFSET);
    assert_eq!(empty_struct.children().count(), 0);
    Ok(())
}

#[test]
fn row_idx_and_data_expression_pushes_data_into_chunks() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = concat(
        3,
        dtype.clone(),
        vec![flat(1, dtype.clone(), 0), flat(2, dtype, 1)],
    );
    let child = layout;
    let expression = and(gt(row_idx(), lit(10_u64)), gt(root(), lit(5_i32)));
    let plan = optimize(make_row_idx_plan(expression, child)?, &SESSION)?;

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.0 and $.1)
      child: vortex.plan.pack({0=bool, 1=bool}, rows=3)
        0: vortex.plan.eval(bool, rows=3) expr=($ > 10u64)
          child: vortex.plan.row_idx(u64, rows=3)
        1: vortex.plan.concat(bool, rows=3)
          chunks[0]: vortex.plan.eval(bool, rows=1) expr=($ > 5i32)
            child: vortex.plan.filter(i32, rows=1)
              child: vortex.test.source(i32, rows=1)
          chunks[1]: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
            child: vortex.plan.filter(i32, rows=2)
              child: vortex.test.source(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_struct_field_and_dictionary_values() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let dictionary = dict(flat(2, value_dtype.clone(), 0), flat(3, codes_dtype, 1));
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 2)],
    );
    let plan = make_eval(gt(get_item("a", root()), lit(5_i32)), layout)?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.a > 5i32)
      child: vortex.plan.pack({a=i32, b=i32}, rows=3)
        a: vortex.plan.take(i32, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.test.source(u8, rows=3)
          values: vortex.plan.filter(i32, rows=2)
            child: vortex.test.source(i32, rows=2)
        b: vortex.plan.filter(i32, rows=3)
          child: vortex.test.source(i32, rows=3)
    ");

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.take(bool, rows=3)
      codes: vortex.plan.filter(u8, rows=3)
        child: vortex.test.source(u8, rows=3)
      values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
        child: vortex.plan.filter(i32, rows=2)
          child: vortex.test.source(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_struct_field_with_heterogeneous_chunks() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = dict(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    );
    let chunks = concat(
        5,
        value_dtype.clone(),
        vec![dictionary, flat(2, value_dtype.clone(), 2)],
    );
    let layout = struct_plan(
        5,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![chunks, flat(5, value_dtype, 3)],
    );
    let plan = make_eval(gt(get_item("a", root()), lit(5_i32)), layout)?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(bool, rows=5) expr=($.a > 5i32)
      child: vortex.plan.pack({a=i32, b=i32}, rows=5)
        a: vortex.plan.concat(i32, rows=5)
          chunks[0]: vortex.plan.take(i32, rows=3)
            codes: vortex.plan.filter(u8, rows=3)
              child: vortex.test.source(u8, rows=3)
            values: vortex.plan.filter(i32, rows=2)
              child: vortex.test.source(i32, rows=2)
          chunks[1]: vortex.plan.filter(i32, rows=2)
            child: vortex.test.source(i32, rows=2)
        b: vortex.plan.filter(i32, rows=5)
          child: vortex.test.source(i32, rows=5)
    ");

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.concat(bool, rows=5)
      chunks[0]: vortex.plan.take(bool, rows=3)
        codes: vortex.plan.filter(u8, rows=3)
          child: vortex.test.source(u8, rows=3)
        values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
          child: vortex.plan.filter(i32, rows=2)
            child: vortex.test.source(i32, rows=2)
      chunks[1]: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
        child: vortex.plan.filter(i32, rows=2)
          child: vortex.test.source(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_nested_struct_fields_in_one_pass() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let nested_dtype = DType::Struct(
        StructFields::from_iter([("x", value_dtype.clone()), ("y", value_dtype.clone())]),
        Nullability::NonNullable,
    );
    let nested = struct_plan(
        3,
        nested_dtype.clone(),
        vec![
            flat(3, value_dtype.clone(), 0),
            flat(3, value_dtype.clone(), 1),
        ],
    );
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("nested", nested_dtype), ("z", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![nested, flat(3, value_dtype, 2)],
    );
    let expression = gt(get_item("x", get_item("nested", root())), lit(5_i32));
    let plan = make_eval(expression, layout)?.into_plan();

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($ > 5i32)
      child: vortex.plan.filter(i32, rows=3)
        child: vortex.test.source(i32, rows=3)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_single_field_nested_structs() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let inner_dtype = DType::Struct(
        StructFields::from_iter([("b", value_dtype.clone())]),
        Nullability::NonNullable,
    );
    let inner = struct_plan(3, inner_dtype.clone(), vec![flat(3, value_dtype, 0)]);
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", inner_dtype)]),
            Nullability::NonNullable,
        ),
        vec![inner],
    );
    let expression = gt(get_item("b", get_item("a", root())), lit(5_i32));
    let plan = make_eval(expression, layout)?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($.a.b > 5i32)
      child: vortex.plan.pack({a={b=i32}}, rows=3)
        a: vortex.plan.pack({b=i32}, rows=3)
          b: vortex.plan.filter(i32, rows=3)
            child: vortex.test.source(i32, rows=3)
    ");

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($ > 5i32)
      child: vortex.plan.filter(i32, rows=3)
        child: vortex.test.source(i32, rows=3)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_three_single_field_structs() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let c_dtype = DType::Struct(
        StructFields::from_iter([("c", value_dtype.clone())]),
        Nullability::NonNullable,
    );
    let b_dtype = DType::Struct(
        StructFields::from_iter([("b", c_dtype.clone())]),
        Nullability::NonNullable,
    );
    let c = struct_plan(3, c_dtype, vec![flat(3, value_dtype, 0)]);
    let b = struct_plan(3, b_dtype.clone(), vec![c]);
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", b_dtype)]),
            Nullability::NonNullable,
        ),
        vec![b],
    );
    let expression = gt(
        get_item("c", get_item("b", get_item("a", root()))),
        lit(5_i32),
    );
    let plan = make_eval(expression, layout)?.into_plan();

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($ > 5i32)
      child: vortex.plan.filter(i32, rows=3)
        child: vortex.test.source(i32, rows=3)
    ");
    Ok(())
}

#[test]
fn compound_expression_pushes_through_nested_struct_and_dictionary() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let dictionary = dict(flat(2, value_dtype.clone(), 0), flat(3, codes_dtype, 1));
    let inner_dtype = DType::Struct(
        StructFields::from_iter([
            ("b", value_dtype.clone()),
            ("c", value_dtype.clone()),
            ("unused", value_dtype.clone()),
        ]),
        Nullability::NonNullable,
    );
    let inner = struct_plan(
        3,
        inner_dtype.clone(),
        vec![
            dictionary,
            flat(3, value_dtype.clone(), 2),
            flat(3, value_dtype.clone(), 3),
        ],
    );
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", inner_dtype), ("unused", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![inner, flat(3, value_dtype, 4)],
    );
    let expression = and(
        gt(get_item("b", get_item("a", root())), lit(5_i32)),
        gt(get_item("c", get_item("a", root())), lit(7_i32)),
    );
    let plan = make_eval(expression, layout)?.into_plan();

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.0 and $.1)
      child: vortex.plan.pack({0=bool, 1=bool}, rows=3)
        0: vortex.plan.take(bool, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.test.source(u8, rows=3)
          values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
            child: vortex.plan.filter(i32, rows=2)
              child: vortex.test.source(i32, rows=2)
        1: vortex.plan.eval(bool, rows=3) expr=($ > 7i32)
          child: vortex.plan.filter(i32, rows=3)
            child: vortex.test.source(i32, rows=3)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_dictionary_of_struct_values() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let values_dtype = DType::Struct(
        StructFields::from_iter([("b", value_dtype.clone()), ("unused", value_dtype.clone())]),
        Nullability::NonNullable,
    );
    let values = struct_plan(
        2,
        values_dtype.clone(),
        vec![
            flat(2, value_dtype.clone(), 0),
            flat(2, value_dtype.clone(), 1),
        ],
    );
    let dictionary = dict(
        values,
        flat(3, primitive(PType::U8, Nullability::NonNullable), 2),
    );
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", values_dtype), ("unused", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 3)],
    );
    let expression = gt(get_item("b", get_item("a", root())), lit(5_i32));
    let plan = make_eval(expression, layout)?.into_plan();

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.take(bool, rows=3)
      codes: vortex.plan.filter(u8, rows=3)
        child: vortex.test.source(u8, rows=3)
      values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
        child: vortex.plan.filter(i32, rows=2)
          child: vortex.test.source(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn multi_field_struct_expression_pushes_into_each_field() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = dict(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    );
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([
                ("a", value_dtype.clone()),
                ("b", value_dtype.clone()),
                ("c", value_dtype.clone()),
            ]),
            Nullability::NonNullable,
        ),
        vec![
            dictionary,
            flat(3, value_dtype.clone(), 2),
            flat(3, value_dtype, 3),
        ],
    );
    let expression = and(
        gt(get_item("a", root()), lit(5_i32)),
        gt(get_item("b", root()), lit(7_i32)),
    );
    let plan = make_eval(expression, layout)?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=(($.a > 5i32) and ($.b > 7i32))
      child: vortex.plan.pack({a=i32, b=i32, c=i32}, rows=3)
        a: vortex.plan.take(i32, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.test.source(u8, rows=3)
          values: vortex.plan.filter(i32, rows=2)
            child: vortex.test.source(i32, rows=2)
        b: vortex.plan.filter(i32, rows=3)
          child: vortex.test.source(i32, rows=3)
        c: vortex.plan.filter(i32, rows=3)
          child: vortex.test.source(i32, rows=3)
    ");

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.0 and $.1)
      child: vortex.plan.pack({0=bool, 1=bool}, rows=3)
        0: vortex.plan.take(bool, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.test.source(u8, rows=3)
          values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
            child: vortex.plan.filter(i32, rows=2)
              child: vortex.test.source(i32, rows=2)
        1: vortex.plan.eval(bool, rows=3) expr=($ > 7i32)
          child: vortex.plan.filter(i32, rows=3)
            child: vortex.test.source(i32, rows=3)
    ");
    let reoptimized = optimize(optimized.clone(), &SESSION)?;
    assert!(PlanRef::ptr_eq(&optimized, &reoptimized));
    Ok(())
}

#[test]
fn repeated_cross_field_expressions_reach_a_fixed_point() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(3, value_dtype.clone(), 0), flat(3, value_dtype, 1)],
    );
    let a = get_item("a", root());
    let b = get_item("b", root());
    let expression = and(
        gt(checked_add(a.clone(), b.clone()), lit(10_i32)),
        gt(checked_add(a, b), lit(20_i32)),
    );
    let plan = make_eval(expression, layout)?.into_plan();

    let optimized = optimize(plan, &SESSION)?;
    let reoptimized = optimize(optimized.clone(), &SESSION)?;
    assert!(PlanRef::ptr_eq(&optimized, &reoptimized));
    Ok(())
}

#[test]
fn multi_field_struct_expression_keeps_cross_field_refinement() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = dict(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    );
    let layout = struct_plan(
        3,
        DType::Struct(
            StructFields::from_iter([
                ("a", value_dtype.clone()),
                ("b", value_dtype.clone()),
                ("c", value_dtype.clone()),
            ]),
            Nullability::NonNullable,
        ),
        vec![
            dictionary,
            flat(3, value_dtype.clone(), 2),
            flat(3, value_dtype, 3),
        ],
    );
    let expression = gt(
        checked_add(get_item("a", root()), get_item("b", root())),
        lit(10_i32),
    );
    let plan = make_eval(expression, layout)?.into_plan();

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=(($.0 + $.1) > 10i32)
      child: vortex.plan.pack({0=i32, 1=i32}, rows=3)
        0: vortex.plan.take(i32, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.test.source(u8, rows=3)
          values: vortex.plan.filter(i32, rows=2)
            child: vortex.test.source(i32, rows=2)
        1: vortex.plan.filter(i32, rows=3)
          child: vortex.test.source(i32, rows=3)
    ");
    assert_eq!(
        optimized.display_tree().to_string(),
        optimize(optimized.clone(), &SESSION)?
            .display_tree()
            .to_string()
    );
    Ok(())
}

#[test]
fn negative_cost_part_of_expression_pushes_into_dictionary_values() -> VortexResult<()> {
    let dictionary = dict(
        flat(2, DType::Utf8(Nullability::NonNullable), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    );
    let expression = cast(
        byte_length(root()),
        primitive(PType::I64, Nullability::NonNullable),
    );
    let optimized = optimize(make_eval(expression, dictionary)?.into_plan(), &SESSION)?;

    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(i64, rows=3) expr=cast($ as i64)
      child: vortex.plan.take(u64, rows=3)
        codes: vortex.plan.filter(u8, rows=3)
          child: vortex.test.source(u8, rows=3)
        values: vortex.plan.eval(u64, rows=2) expr=vortex.byte_length($)
          child: vortex.plan.filter(utf8, rows=2)
            child: vortex.test.source(utf8, rows=2)
    ");
    Ok(())
}

#[test]
fn dictionary_pushdown_rejects_unsafe_expressions() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = dict(
        flat(2, value_dtype, 0),
        flat(3, primitive(PType::U8, Nullability::Nullable), 1),
    );

    for expression in [
        lit(false),
        is_null(root()),
        gt(checked_add(root(), lit(1_i32)), lit(5_i32)),
    ] {
        let plan = make_eval(expression.clone(), dictionary.clone())?.into_plan();
        let optimized = optimize(plan, &SESSION)?;
        let eval = optimized.as_opt::<Eval>().ok_or_else(|| {
            vortex_err!("Expression unexpectedly pushed into dictionary: {expression}")
        })?;
        assert!(eval.child_plan()?.is::<Take>());
    }
    Ok(())
}

#[test]
fn non_strict_expression_pushes_into_dictionary_without_null_codes() -> VortexResult<()> {
    let dictionary = dict(
        flat(2, primitive(PType::I32, Nullability::Nullable), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    );
    let optimized = optimize(
        make_eval(is_null(root()), dictionary)?.into_plan(),
        &SESSION,
    )?;
    let take = optimized
        .as_opt::<Take>()
        .ok_or_else(|| vortex_err!("is_null was not pushed into the dictionary: {optimized}"))?;
    assert!(take.values()?.is::<Eval>());
    Ok(())
}

#[test]
fn stats_expression_rewrites_to_zoned_pruning_plan() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let max_dtype = max
        .state_dtype(&dtype)
        .ok_or_else(|| vortex_err!("max does not support {dtype}"))?;
    let zones_dtype = DType::Struct(
        StructFields::from_iter([(max.to_string(), max_dtype.as_nullable())]),
        Nullability::NonNullable,
    );
    let zone_len = NonZeroUsize::new(3).ok_or_else(|| vortex_err!("zone length is zero"))?;
    let layout = zoned_plan(
        flat(5, dtype.clone(), 0),
        flat(2, zones_dtype, 1),
        zone_len,
        vec![max].into(),
    );
    let session = VortexSession::empty();
    let filter = gt(root(), lit(5_i32));
    let falsifier = filter
        .bind(&dtype)?
        .falsify(&session)?
        .ok_or_else(|| vortex_err!("filter has no falsifier"))?;
    let source = layout;
    let plan = EvalPlan::try_new(falsifier.clone(), source.clone())?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.eval(bool?, rows=5) expr=(stat($, vortex.max()) <= 5i32)
      child: vortex.plan.zoned(i32, rows=5)
        data: vortex.plan.filter(i32, rows=5)
          child: vortex.test.source(i32, rows=5)
        zones: vortex.plan.filter({vortex.max()=i32?}, rows=2)
          child: vortex.test.source({vortex.max()=i32?}, rows=2)
    ");

    let optimized = optimize(plan, &SESSION)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.zoned(bool?, rows=5) prune=(stat($, vortex.max()) <= 5i32)
      zones: vortex.plan.filter({vortex.max()=i32?}, rows=2)
        child: vortex.test.source({vortex.max()=i32?}, rows=2)
    ");
    let zoned = optimized
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!("optimized pruning plan is not zoned"))?;
    assert!(zoned.is_pruning());
    assert_eq!(zoned.pruning_expression(), Some(&falsifier));
    assert!(zoned.data_plan()?.is_none());
    assert_eq!(zoned.nslots(), 2);
    assert_eq!(zoned.children().count(), 1);

    let mixed_expression = bound_and(falsifier, gt(root(), lit(0_i32)).bind(&dtype)?);
    let mixed = optimize(
        EvalPlan::try_new(mixed_expression, source)?.into_plan(),
        &SESSION,
    )?;
    let mixed = mixed
        .as_opt::<Eval>()
        .ok_or_else(|| vortex_err!("expression with a data reference was pushed into zones"))?;
    assert!(mixed.child_plan()?.is::<Filter>());
    Ok(())
}

/// A plan that reads a zoned column's data is optimized to the data alone: the zone table is
/// neither a child of the plan nor among the segments it reads.
#[test]
fn data_expression_strips_zoned_to_its_data() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let zone_len = NonZeroUsize::new(3).ok_or_else(|| vortex_err!("zone length is zero"))?;
    let layout = zoned_plan(
        flat(5, dtype.clone(), 0),
        flat(2, zones_dtype, 1),
        zone_len,
        vec![max].into(),
    );
    let source = layout;

    let plan = EvalPlan::try_new(gt(root(), lit(4_i32)).bind(&dtype)?, source.clone())?.into_plan();
    insta::assert_snapshot!(optimize(plan, &SESSION)?.display_tree(), @"
    root: vortex.plan.eval(bool, rows=5) expr=($ > 4i32)
      child: vortex.plan.filter(i32, rows=5)
        child: vortex.test.source(i32, rows=5)
    ");

    let identity = optimize(source, &SESSION)?;
    assert!(identity.is::<Filter>());
    let segments = Scan::segments(&SESSION, &identity, 0..5, &Mask::new_true(5))?;
    assert_eq!(segments, vec![SegmentId::from(0)]);
    Ok(())
}

#[test]
fn pruning_expression_partitions_across_row_idx_and_zoned_struct_field() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let max_dtype = max
        .state_dtype(&value_dtype)
        .ok_or_else(|| vortex_err!("max does not support {value_dtype}"))?;
    let zones_dtype = DType::Struct(
        StructFields::from_iter([(max.to_string(), max_dtype.as_nullable())]),
        Nullability::NonNullable,
    );
    let zone_len = NonZeroUsize::new(3).ok_or_else(|| vortex_err!("zone length is zero"))?;
    let zoned = zoned_plan(
        flat(5, value_dtype.clone(), 0),
        flat(2, zones_dtype, 1),
        zone_len,
        vec![max].into(),
    );
    let struct_dtype = DType::Struct(
        StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
        Nullability::NonNullable,
    );
    let layout = struct_plan(
        5,
        struct_dtype.clone(),
        vec![zoned, flat(5, value_dtype, 2)],
    );
    let source = layout;
    let session = VortexSession::empty();
    let filter = and(
        gt(row_idx(), lit(11_u64)),
        gt(get_item("a", root()), lit(5_i32)),
    );
    let falsifier = filter
        .bind(&struct_dtype)?
        .falsify(&session)?
        .ok_or_else(|| vortex_err!("filter has no falsifier"))?;

    let optimized = optimize(
        plan_row_idx_expression(falsifier, source, &SESSION)?,
        &SESSION,
    )?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool?, rows=5) expr=($.0 or $.1)
      child: vortex.plan.pack({0=bool?, 1=bool?}, rows=5)
        0: vortex.plan.eval(bool?, rows=5) expr=(stat($, vortex.max()) <= 11u64)
          child: vortex.plan.row_idx(u64, rows=5)
        1: vortex.plan.zoned(bool?, rows=5) prune=(stat($, vortex.max()) <= 5i32)
          zones: vortex.plan.filter({vortex.max()=i32?}, rows=2)
            child: vortex.test.source({vortex.max()=i32?}, rows=2)
    ");
    Ok(())
}

/// A predicate holding a dynamic comparison is never pushed onto dictionary values, alone or
/// alongside static conjuncts: it stays above the take, so the values never change.
#[test]
fn dynamic_predicates_stay_above_dictionary_values() -> VortexResult<()> {
    let values_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let layout = dict(flat(2, values_dtype.clone(), 0), flat(3, codes_dtype, 1));
    let threshold = || {
        dynamic(
            CompareOperator::Gt,
            || Some(ScalarValue::from(5_i32)),
            values_dtype.clone(),
            true,
            root(),
        )
    };

    for predicate in [threshold(), and(gt(root(), lit(0_i32)), threshold())] {
        let optimized = optimize(make_eval(predicate, layout.clone())?.into_plan(), &SESSION)?;
        let eval = optimized
            .as_opt::<Eval>()
            .ok_or_else(|| vortex_err!("dynamic predicate was pushed down"))?;
        let take = eval
            .child_plan()?
            .as_opt::<Take>()
            .ok_or_else(|| vortex_err!("dynamic predicate is not over the take"))?
            .clone();
        assert!(!take.values()?.is::<Eval>(), "{}", optimized.display_tree());
    }
    Ok(())
}

/// Codes that may be null make the take nullable while its values are not; the predicate is
/// rebound to the values when pushed onto them. Dictionary layouts keep nulls in their values,
/// so such a take is built rather than lowered.
#[test]
fn expression_pushes_onto_values_less_nullable_than_the_take() -> VortexResult<()> {
    let values_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::Nullable);
    let take = TakePlan::new(flat(3, codes_dtype, 1), flat(2, values_dtype.clone(), 0)).into_plan();
    assert!(take.dtype().is_nullable());
    let plan = make_eval(gt(root(), lit(5_i32)), take)?.into_plan();
    let dtype = plan.dtype().clone();

    let optimized = optimize(plan, &SESSION)?;
    assert_eq!(optimized.dtype(), &dtype);
    let take = optimized
        .as_opt::<Take>()
        .ok_or_else(|| vortex_err!("predicate was not pushed onto the values"))?;
    let values = take.values()?;
    let eval = values
        .as_opt::<Eval>()
        .ok_or_else(|| vortex_err!("values are not evaluated"))?;
    assert!(eval.expression().is_root_bound_to(&values_dtype));
    Ok(())
}

/// Eval($ > 5, Eval($.a, (...))) -> Eval($.a > 5, (...))
#[test]
fn test_eval_of_eval() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let values = vec![flat(3, dtype.clone(), 2)];
    let fields = StructFields::from_iter([("a", dtype)]);
    let struct_dtype = DType::Struct(fields, Nullability::NonNullable);
    let layout = struct_plan(3, struct_dtype, values);
    let plan = make_eval(get_item("a", root()), layout)?.into_plan();
    let plan = make_eval(gt(root(), lit(5_i32)), plan)?.into_plan();
    insta::assert_snapshot!(optimize(plan, &SESSION)?.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($ > 5i32)
      child: vortex.plan.filter(i32, rows=3)
        child: vortex.test.source(i32, rows=3)
    ");
    Ok(())
}

/// Eval(pack({0: $ > 0}, (...))) -> Pack({0: $ > 0}, (...))
#[test]
fn test_eval_of_pack() -> VortexResult<()> {
    let expr = pack([("0", gt(root(), lit(0_i32)))], Nullability::NonNullable);
    let plan = flat(3, primitive(PType::I32, Nullability::NonNullable), 0);
    let plan = make_eval(expr, plan)?.into_plan();
    insta::assert_snapshot!(optimize(plan, &SESSION)?.display_tree(), @"
    root: vortex.plan.pack({0=bool}, rows=3)
      0: vortex.plan.eval(bool, rows=3) expr=($ > 0i32)
        child: vortex.plan.filter(i32, rows=3)
          child: vortex.test.source(i32, rows=3)
    ");
    Ok(())
}

fn list_plan(
    dtype: DType,
    elements: PlanRef,
    offsets: PlanRef,
    validity: Option<PlanRef>,
) -> PlanRef {
    ListPackPlan::try_new(
        dtype.nullability(),
        offsets.row_count() - 1,
        elements,
        offsets,
        validity,
    )
    .expect("valid test list")
    .into_plan()
}
