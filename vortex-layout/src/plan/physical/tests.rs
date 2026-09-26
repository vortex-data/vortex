// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::bound::and as bound_and;
use vortex_array::expr::byte_length;
use vortex_array::expr::cast;
use vortex_array::expr::checked_add;
use vortex_array::expr::dynamic;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::stats::as_stat_bitset_bytes;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use super::*;
use crate::LayoutBuildContext;
use crate::LayoutEncoding;
use crate::LayoutRef;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::foreign::new_foreign_layout;
use crate::layouts::list::ListLayout;
use crate::layouts::row_idx::row_idx;
use crate::layouts::struct_::StructLayout;
use crate::layouts::zoned::LegacyStatsLayoutEncoding;
use crate::layouts::zoned::MAX_IS_TRUNCATED;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::legacy_stats_to_aggregates;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::segments::SegmentId;
use crate::test::SESSION;

fn primitive(ptype: PType, nullability: Nullability) -> DType {
    DType::Primitive(ptype, nullability)
}

fn flat(row_count: u64, dtype: DType, segment: u32) -> LayoutRef {
    FlatLayout::new(
        row_count,
        dtype,
        SegmentId::from(segment),
        ReadContext::new([]),
    )
    .into_layout()
}

fn unsupported(row_count: u64, dtype: DType) -> LayoutRef {
    static ID: CachedId = CachedId::new("vortex.test.unsupported");
    new_foreign_layout(*ID, dtype, row_count, Vec::new(), Vec::new(), Vec::new())
}

fn make_plan(layout: LayoutRef) -> VortexResult<PlanRef> {
    lower(&layout)
}

fn make_eval(expression: Expression, child: PlanRef) -> VortexResult<EvalPlan> {
    let expression = expression.bind(child.dtype())?.optimize_recursive()?;
    EvalPlan::try_new(expression, child)
}

fn make_row_idx_plan(expression: Expression, child: PlanRef) -> VortexResult<PlanRef> {
    let expression = expression.bind(child.dtype())?.optimize_recursive()?;
    plan_row_idx_expression(expression, child)
}

fn child_of(plan: &PlanRef, index: usize) -> VortexResult<PlanRef> {
    plan.child(index)?
        .ok_or_else(|| vortex_err!(NotFound: "missing child {index}"))
}

fn assert_unsupported(error: vortex_error::VortexError) {
    assert!(
        error
            .to_string()
            .contains("No physical plan implementation for layout 'vortex.test.unsupported'"),
        "unexpected error: {error}"
    );
}

#[test]
fn unsupported_layout_has_no_plan() -> VortexResult<()> {
    let layout = unsupported(3, DType::Null);

    assert_unsupported(lower(&layout).err().ok_or_else(
        || vortex_err!(InvalidArgument: "unsupported layout unexpectedly produced a plan"),
    )?);
    Ok(())
}

#[test]
fn flat_plan_is_a_filtered_segment_scan_without_children() -> VortexResult<()> {
    let plan = make_plan(flat(3, primitive(PType::I32, Nullability::NonNullable), 0))?;

    assert!(plan.is::<Filter>());
    let scan = child_of(&plan, 0)?;
    assert!(scan.is::<SegmentScan>());
    assert_eq!(scan.child_count(), 0);
    assert!(scan.child(0)?.is_none());
    Ok(())
}

#[test]
fn chunked_plan_exposes_chunks() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)]),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    assert!(plan.is::<Concat>());
    assert_eq!(plan.child_count(), 2);
    assert_eq!(child_of(&plan, 0)?.row_count(), 2);
    assert_eq!(child_of(&plan, 1)?.row_count(), 1);
    Ok(())
}

#[test]
fn chunked_plan_lowers_each_chunk_on_access() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        2,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![
            flat(1, dtype.clone(), 0),
            unsupported(1, dtype),
        ]),
    )
    .into_layout();

    let plan = make_plan(layout)?;
    assert_eq!(child_of(&plan, 0)?.row_count(), 1);
    assert_unsupported(child_of(&plan, 1).err().ok_or_else(
        || vortex_err!(InvalidArgument: "unsupported chunk unexpectedly produced a plan"),
    )?);
    Ok(())
}

#[test]
fn struct_plan_lowers_each_field_on_access() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        1,
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(1, field_dtype.clone(), 0), unsupported(1, field_dtype)],
    )
    .into_layout();

    let plan = make_plan(layout)?;
    assert_eq!(child_of(&plan, 0)?.row_count(), 1);
    assert_unsupported(child_of(&plan, 1).err().ok_or_else(
        || vortex_err!(InvalidArgument: "unsupported field unexpectedly produced a plan"),
    )?);
    Ok(())
}

#[test]
fn dict_plan_orders_codes_before_values() -> VortexResult<()> {
    let values_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let layout = DictLayout::new(
        flat(2, values_dtype.clone(), 0),
        flat(3, codes_dtype.clone(), 1),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    assert!(plan.is::<Take>());
    assert_eq!(plan.child_count(), 2);
    assert_eq!(child_of(&plan, 0)?.dtype(), &codes_dtype);
    assert_eq!(child_of(&plan, 1)?.dtype(), &values_dtype);
    Ok(())
}

#[test]
fn list_plan_appends_validity_when_nullable() -> VortexResult<()> {
    let element_dtype = primitive(PType::I32, Nullability::NonNullable);
    let offsets_dtype = primitive(PType::U32, Nullability::NonNullable);
    let non_nullable = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::NonNullable),
        flat(3, element_dtype.clone(), 0),
        flat(3, offsets_dtype.clone(), 1),
        None,
    )
    .into_layout();
    let plan = make_plan(non_nullable)?;

    assert!(plan.is::<ListPack>());
    assert_eq!(plan.child_count(), 2);
    assert_eq!(child_of(&plan, 0)?.dtype(), &element_dtype);
    assert_eq!(child_of(&plan, 1)?.dtype(), &offsets_dtype);

    let nullable = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::Nullable),
        flat(3, element_dtype, 2),
        flat(3, offsets_dtype, 3),
        Some(flat(2, DType::Bool(Nullability::NonNullable), 4)),
    )
    .into_layout();
    let nullable_plan = make_plan(nullable)?;
    assert_eq!(nullable_plan.child_count(), 3);
    assert_eq!(
        child_of(&nullable_plan, 2)?.dtype(),
        &DType::Bool(Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn struct_plan_appends_validity_when_nullable() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let fields = StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]);
    let non_nullable = StructLayout::new(
        3,
        DType::Struct(fields.clone(), Nullability::NonNullable),
        vec![
            flat(3, field_dtype.clone(), 0),
            flat(3, field_dtype.clone(), 1),
        ],
    )
    .into_layout();
    let plan = make_plan(non_nullable)?;

    assert!(plan.is::<Pack>());
    assert_eq!(plan.child_count(), 2);
    assert_eq!(child_of(&plan, 0)?.dtype(), &field_dtype);
    assert_eq!(child_of(&plan, 1)?.dtype(), &field_dtype);

    let nullable = StructLayout::new(
        3,
        DType::Struct(fields, Nullability::Nullable),
        vec![
            flat(3, DType::Bool(Nullability::NonNullable), 2),
            flat(3, field_dtype.clone(), 3),
            flat(3, field_dtype, 4),
        ],
    )
    .into_layout();
    let nullable_plan = make_plan(nullable)?;
    assert_eq!(nullable_plan.child_count(), 3);
    assert_eq!(
        child_of(&nullable_plan, 2)?.dtype(),
        &DType::Bool(Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn with_children_rejects_mismatched_arity() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)]),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    let error = plan
        .with_children(vec![child_of(&plan, 0)?])
        .err()
        .ok_or_else(|| vortex_err!(MismatchedTypes: "mismatched arity unexpectedly succeeded"))?;
    assert!(
        error
            .to_string()
            .contains("Concat expects 2 children but got 1"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn with_children_replaces_children_in_order() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)]),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    let swapped = plan.with_children(vec![child_of(&plan, 1)?, child_of(&plan, 0)?])?;
    assert_eq!(child_of(&swapped, 0)?.row_count(), 1);
    assert_eq!(child_of(&swapped, 1)?.row_count(), 2);
    assert_eq!(swapped.as_::<Concat>().row_offsets(), &[0, 1]);
    Ok(())
}

#[test]
fn eval_try_new_validates_expression_root_dtype() -> VortexResult<()> {
    let expression = root().bind(&primitive(PType::I32, Nullability::NonNullable))?;
    let child = make_plan(flat(3, primitive(PType::I64, Nullability::NonNullable), 0))?;

    let error = EvalPlan::try_new(expression, child).err().ok_or_else(
        || vortex_err!(MismatchedTypes: "mismatched Eval root dtype unexpectedly succeeded"),
    )?;
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
    let child = make_plan(flat(3, primitive(PType::I32, Nullability::NonNullable), 0))?;
    let expression = root().bind(child.dtype())?;
    let plan: PlanRef = EvalPlan::try_new(expression, child)?.into_plan();

    assert!(optimize(plan)?.is::<Filter>());
    Ok(())
}

#[test]
fn optimize_rewrites_nested_children() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)]),
    )
    .into_layout();
    let chunked = make_plan(layout)?;

    // Wrap the first chunk in an identity expression, which optimization should remove without
    // any chunked-specific rule.
    let chunk = child_of(&chunked, 0)?;
    let identity: PlanRef = EvalPlan::try_new(root().bind(chunk.dtype())?, chunk)?.into_plan();
    let wrapped = chunked.with_children(vec![identity, child_of(&chunked, 1)?])?;

    let optimized = optimize(wrapped)?;
    assert!(optimized.is::<Concat>());
    assert!(child_of(&optimized, 0)?.is::<Filter>());
    assert!(child_of(&optimized, 1)?.is::<Filter>());
    Ok(())
}

#[test]
fn plan_display_matches_array_tree_display_shape() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(3, field_dtype.clone(), 0), flat(3, field_dtype, 1)],
    )
    .into_layout();
    let child = make_plan(layout)?;
    let expression = get_item("a", root()).bind(child.dtype())?;
    let plan = EvalPlan::try_new(expression, child)?;

    assert_eq!(plan.to_string(), "vortex.plan.eval(i32, rows=3) expr=$.a");
    let plan: PlanRef = plan.into_plan();

    assert_eq!(plan.to_string(), "vortex.plan.eval(i32, rows=3) expr=$.a");
    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.eval(i32, rows=3) expr=$.a
      child: vortex.plan.pack({a=i32, b=i32}, rows=3)
        a: vortex.plan.filter(i32, rows=3)
          child: vortex.plan.segment_scan(i32, rows=3)
        b: vortex.plan.filter(i32, rows=3)
          child: vortex.plan.segment_scan(i32, rows=3)
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
    let nullable_layout = StructLayout::new(
        3,
        DType::Struct(nullable_fields, Nullability::Nullable),
        vec![
            flat(3, DType::Bool(Nullability::NonNullable), 2),
            flat(3, primitive(PType::I32, Nullability::NonNullable), 3),
            flat(3, primitive(PType::I32, Nullability::NonNullable), 4),
        ],
    )
    .into_layout();
    let nullable = make_plan(nullable_layout)?;
    insta::assert_snapshot!(nullable.tree_display_builder(), @r"
    root:
      a:
        child:
      b:
        child:
      validity:
        child:
    ");
    Ok(())
}

#[test]
fn chunked_plan_display_names_chunks() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)]),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.concat(i32, rows=3)
      chunks[0]: vortex.plan.filter(i32, rows=2)
        child: vortex.plan.segment_scan(i32, rows=2)
      chunks[1]: vortex.plan.filter(i32, rows=1)
        child: vortex.plan.segment_scan(i32, rows=1)
    ");
    Ok(())
}

#[test]
fn dict_plan_display_names_logical_children() -> VortexResult<()> {
    let layout = DictLayout::new(
        flat(2, primitive(PType::I32, Nullability::NonNullable), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.take(i32, rows=3)
      codes: vortex.plan.filter(u8, rows=3)
        child: vortex.plan.segment_scan(u8, rows=3)
      values: vortex.plan.filter(i32, rows=2)
        child: vortex.plan.segment_scan(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn list_plan_display_handles_optional_validity() -> VortexResult<()> {
    let element_dtype = primitive(PType::I32, Nullability::NonNullable);
    let offsets_dtype = primitive(PType::U32, Nullability::NonNullable);
    let non_nullable_layout = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::NonNullable),
        flat(4, element_dtype.clone(), 0),
        flat(3, offsets_dtype.clone(), 1),
        None,
    )
    .into_layout();
    let non_nullable = make_plan(non_nullable_layout)?;

    insta::assert_snapshot!(non_nullable.display_tree(), @r"
    root: vortex.plan.list_pack(list(i32), rows=2)
      elements: vortex.plan.filter(i32, rows=4)
        child: vortex.plan.segment_scan(i32, rows=4)
      offsets: vortex.plan.filter(u32, rows=3)
        child: vortex.plan.segment_scan(u32, rows=3)
    ");

    let nullable_layout = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::Nullable),
        flat(4, element_dtype, 2),
        flat(3, offsets_dtype, 3),
        Some(flat(2, DType::Bool(Nullability::NonNullable), 4)),
    )
    .into_layout();
    let nullable = make_plan(nullable_layout)?;

    insta::assert_snapshot!(nullable.display_tree(), @r"
    root: vortex.plan.list_pack(list(i32)?, rows=2)
      elements: vortex.plan.filter(i32, rows=4)
        child: vortex.plan.segment_scan(i32, rows=4)
      offsets: vortex.plan.filter(u32, rows=3)
        child: vortex.plan.segment_scan(u32, rows=3)
      validity: vortex.plan.filter(bool, rows=2)
        child: vortex.plan.segment_scan(bool, rows=2)
    ");
    Ok(())
}

#[test]
fn row_idx_only_expression_uses_row_idx_source() -> VortexResult<()> {
    let layout = flat(3, primitive(PType::I32, Nullability::NonNullable), 0);
    let plan = make_row_idx_plan(row_idx(), make_plan(layout)?)?;

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(u64, rows=3) expr=$
      child: vortex.plan.row_idx(u64, rows=3)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"root: vortex.plan.row_idx(u64, rows=3)");
    assert!(optimized.is::<RowIdx>());
    Ok(())
}

#[test]
fn expression_partitions_across_row_idx_and_struct() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 2)],
    )
    .into_layout();
    let expression = and(
        gt(row_idx(), lit(11_u64)),
        and(
            gt(get_item("a", root()), lit(5_i32)),
            gt(get_item("b", root()), lit(7_i32)),
        ),
    );
    let plan = make_row_idx_plan(expression, make_plan(layout)?)?;

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.row_idx and $.child)
      child: vortex.plan.pack({row_idx=bool, child=bool}, rows=3)
        row_idx: vortex.plan.eval(bool, rows=3) expr=($ > 11u64)
          child: vortex.plan.row_idx(u64, rows=3)
        child: vortex.plan.eval(bool, rows=3) expr=($.a and $.b)
          child: vortex.plan.pack({a=bool, b=bool}, rows=3)
            a: vortex.plan.take(bool, rows=3)
              codes: vortex.plan.filter(u8, rows=3)
                child: vortex.plan.segment_scan(u8, rows=3)
              values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
                child: vortex.plan.filter(i32, rows=2)
                  child: vortex.plan.segment_scan(i32, rows=2)
            b: vortex.plan.eval(bool, rows=3) expr=($ > 7i32)
              child: vortex.plan.filter(i32, rows=3)
                child: vortex.plan.segment_scan(i32, rows=3)
    ");
    Ok(())
}

#[test]
fn empty_projection_prunes_row_idx_child_fields() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(3, value_dtype.clone(), 0), flat(3, value_dtype, 1)],
    )
    .into_layout();
    let projection = pack(
        std::iter::empty::<(&str, Expression)>(),
        Nullability::NonNullable,
    );
    let plan = make_row_idx_plan(projection, make_plan(layout)?)?;

    let optimized = optimize(plan)?;
    let projection = optimized.as_opt::<Eval>().ok_or_else(
        || vortex_err!(AssertionFailed: "optimized plan has no projection expression"),
    )?;
    let child = projection.child_plan()?;
    let empty_struct = child.as_opt::<Pack>().ok_or_else(
        || vortex_err!(AssertionFailed: "empty projection did not prune the RowIdx child"),
    )?;

    assert_eq!(empty_struct.nfields(), 0);
    assert_eq!(empty_struct.children().len(), 0);
    Ok(())
}

#[test]
fn row_idx_and_data_expression_pushes_data_into_chunks() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(1, dtype.clone(), 0), flat(2, dtype, 1)]),
    )
    .into_layout();
    let child = make_plan(layout)?;
    let expression = and(gt(row_idx(), lit(10_u64)), gt(root(), lit(5_i32)));
    let plan = optimize(make_row_idx_plan(expression, child)?)?;

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($.row_idx and $.child)
      child: vortex.plan.pack({row_idx=bool, child=bool}, rows=3)
        row_idx: vortex.plan.eval(bool, rows=3) expr=($ > 10u64)
          child: vortex.plan.row_idx(u64, rows=3)
        child: vortex.plan.concat(bool, rows=3)
          chunks[0]: vortex.plan.eval(bool, rows=1) expr=($ > 5i32)
            child: vortex.plan.filter(i32, rows=1)
              child: vortex.plan.segment_scan(i32, rows=1)
          chunks[1]: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
            child: vortex.plan.filter(i32, rows=2)
              child: vortex.plan.segment_scan(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_struct_field_and_dictionary_values() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let dictionary =
        DictLayout::new(flat(2, value_dtype.clone(), 0), flat(3, codes_dtype, 1)).into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 2)],
    )
    .into_layout();
    let plan = make_eval(gt(get_item("a", root()), lit(5_i32)), make_plan(layout)?)?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.a > 5i32)
      child: vortex.plan.pack({a=i32, b=i32}, rows=3)
        a: vortex.plan.take(i32, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.filter(i32, rows=2)
            child: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.filter(i32, rows=3)
          child: vortex.plan.segment_scan(i32, rows=3)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.take(bool, rows=3)
      codes: vortex.plan.filter(u8, rows=3)
        child: vortex.plan.segment_scan(u8, rows=3)
      values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
        child: vortex.plan.filter(i32, rows=2)
          child: vortex.plan.segment_scan(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_struct_field_with_heterogeneous_chunks() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let chunks = ChunkedLayout::new(
        5,
        value_dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![dictionary, flat(2, value_dtype.clone(), 2)]),
    )
    .into_layout();
    let layout = StructLayout::new(
        5,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![chunks, flat(5, value_dtype, 3)],
    )
    .into_layout();
    let plan = make_eval(gt(get_item("a", root()), lit(5_i32)), make_plan(layout)?)?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(bool, rows=5) expr=($.a > 5i32)
      child: vortex.plan.pack({a=i32, b=i32}, rows=5)
        a: vortex.plan.concat(i32, rows=5)
          chunks[0]: vortex.plan.take(i32, rows=3)
            codes: vortex.plan.filter(u8, rows=3)
              child: vortex.plan.segment_scan(u8, rows=3)
            values: vortex.plan.filter(i32, rows=2)
              child: vortex.plan.segment_scan(i32, rows=2)
          chunks[1]: vortex.plan.filter(i32, rows=2)
            child: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.filter(i32, rows=5)
          child: vortex.plan.segment_scan(i32, rows=5)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.concat(bool, rows=5)
      chunks[0]: vortex.plan.take(bool, rows=3)
        codes: vortex.plan.filter(u8, rows=3)
          child: vortex.plan.segment_scan(u8, rows=3)
        values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
          child: vortex.plan.filter(i32, rows=2)
            child: vortex.plan.segment_scan(i32, rows=2)
      chunks[1]: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
        child: vortex.plan.filter(i32, rows=2)
          child: vortex.plan.segment_scan(i32, rows=2)
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
    let nested = StructLayout::new(
        3,
        nested_dtype.clone(),
        vec![
            flat(3, value_dtype.clone(), 0),
            flat(3, value_dtype.clone(), 1),
        ],
    )
    .into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("nested", nested_dtype), ("z", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![nested, flat(3, value_dtype, 2)],
    )
    .into_layout();
    let expression = gt(get_item("x", get_item("nested", root())), lit(5_i32));
    let plan = make_eval(expression, make_plan(layout)?)?.into_plan();

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($ > 5i32)
      child: vortex.plan.filter(i32, rows=3)
        child: vortex.plan.segment_scan(i32, rows=3)
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
    let inner =
        StructLayout::new(3, inner_dtype.clone(), vec![flat(3, value_dtype, 0)]).into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", inner_dtype)]),
            Nullability::NonNullable,
        ),
        vec![inner],
    )
    .into_layout();
    let expression = gt(get_item("b", get_item("a", root())), lit(5_i32));
    let plan = make_eval(expression, make_plan(layout)?)?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($.a.b > 5i32)
      child: vortex.plan.pack({a={b=i32}}, rows=3)
        a: vortex.plan.pack({b=i32}, rows=3)
          b: vortex.plan.filter(i32, rows=3)
            child: vortex.plan.segment_scan(i32, rows=3)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($ > 5i32)
      child: vortex.plan.filter(i32, rows=3)
        child: vortex.plan.segment_scan(i32, rows=3)
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
    let c = StructLayout::new(3, c_dtype, vec![flat(3, value_dtype, 0)]).into_layout();
    let b = StructLayout::new(3, b_dtype.clone(), vec![c]).into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", b_dtype)]),
            Nullability::NonNullable,
        ),
        vec![b],
    )
    .into_layout();
    let expression = gt(
        get_item("c", get_item("b", get_item("a", root()))),
        lit(5_i32),
    );
    let plan = make_eval(expression, make_plan(layout)?)?.into_plan();

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($ > 5i32)
      child: vortex.plan.filter(i32, rows=3)
        child: vortex.plan.segment_scan(i32, rows=3)
    ");
    Ok(())
}

#[test]
fn compound_expression_pushes_through_nested_struct_and_dictionary() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let dictionary =
        DictLayout::new(flat(2, value_dtype.clone(), 0), flat(3, codes_dtype, 1)).into_layout();
    let inner_dtype = DType::Struct(
        StructFields::from_iter([
            ("b", value_dtype.clone()),
            ("c", value_dtype.clone()),
            ("unused", value_dtype.clone()),
        ]),
        Nullability::NonNullable,
    );
    let inner = StructLayout::new(
        3,
        inner_dtype.clone(),
        vec![
            dictionary,
            flat(3, value_dtype.clone(), 2),
            flat(3, value_dtype.clone(), 3),
        ],
    )
    .into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", inner_dtype), ("unused", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![inner, flat(3, value_dtype, 4)],
    )
    .into_layout();
    let expression = and(
        gt(get_item("b", get_item("a", root())), lit(5_i32)),
        gt(get_item("c", get_item("a", root())), lit(7_i32)),
    );
    let plan = make_eval(expression, make_plan(layout)?)?.into_plan();

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.eval(bool, rows=3) expr=($.b and $.c)
      child: vortex.plan.pack({b=bool, c=bool}, rows=3)
        b: vortex.plan.take(bool, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
            child: vortex.plan.filter(i32, rows=2)
              child: vortex.plan.segment_scan(i32, rows=2)
        c: vortex.plan.eval(bool, rows=3) expr=($ > 7i32)
          child: vortex.plan.filter(i32, rows=3)
            child: vortex.plan.segment_scan(i32, rows=3)
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
    let values = StructLayout::new(
        2,
        values_dtype.clone(),
        vec![
            flat(2, value_dtype.clone(), 0),
            flat(2, value_dtype.clone(), 1),
        ],
    )
    .into_layout();
    let dictionary = DictLayout::new(
        values,
        flat(3, primitive(PType::U8, Nullability::NonNullable), 2),
    )
    .into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", values_dtype), ("unused", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 3)],
    )
    .into_layout();
    let expression = gt(get_item("b", get_item("a", root())), lit(5_i32));
    let plan = make_eval(expression, make_plan(layout)?)?.into_plan();

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.take(bool, rows=3)
      codes: vortex.plan.filter(u8, rows=3)
        child: vortex.plan.segment_scan(u8, rows=3)
      values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
        child: vortex.plan.filter(i32, rows=2)
          child: vortex.plan.segment_scan(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn multi_field_struct_expression_pushes_into_each_field() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let layout = StructLayout::new(
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
    )
    .into_layout();
    let expression = and(
        gt(get_item("a", root()), lit(5_i32)),
        gt(get_item("b", root()), lit(7_i32)),
    );
    let plan = make_eval(expression, make_plan(layout)?)?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=(($.a > 5i32) and ($.b > 7i32))
      child: vortex.plan.pack({a=i32, b=i32, c=i32}, rows=3)
        a: vortex.plan.take(i32, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.filter(i32, rows=2)
            child: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.filter(i32, rows=3)
          child: vortex.plan.segment_scan(i32, rows=3)
        c: vortex.plan.filter(i32, rows=3)
          child: vortex.plan.segment_scan(i32, rows=3)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.a and $.b)
      child: vortex.plan.pack({a=bool, b=bool}, rows=3)
        a: vortex.plan.take(bool, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
            child: vortex.plan.filter(i32, rows=2)
              child: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.eval(bool, rows=3) expr=($ > 7i32)
          child: vortex.plan.filter(i32, rows=3)
            child: vortex.plan.segment_scan(i32, rows=3)
    ");
    let reoptimized = optimize(optimized.clone())?;
    assert!(PlanRef::ptr_eq(&optimized, &reoptimized));
    Ok(())
}

#[test]
fn repeated_cross_field_expressions_reach_a_fixed_point() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(3, value_dtype.clone(), 0), flat(3, value_dtype, 1)],
    )
    .into_layout();
    let a = get_item("a", root());
    let b = get_item("b", root());
    let expression = and(
        gt(checked_add(a.clone(), b.clone()), lit(10_i32)),
        gt(checked_add(a, b), lit(20_i32)),
    );
    let plan = make_eval(expression, make_plan(layout)?)?.into_plan();

    let optimized = optimize(plan)?;
    let reoptimized = optimize(optimized.clone())?;
    assert!(PlanRef::ptr_eq(&optimized, &reoptimized));
    Ok(())
}

#[test]
fn multi_field_struct_expression_keeps_cross_field_refinement() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let layout = StructLayout::new(
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
    )
    .into_layout();
    let expression = gt(
        checked_add(get_item("a", root()), get_item("b", root())),
        lit(10_i32),
    );
    let plan = make_eval(expression, make_plan(layout)?)?.into_plan();

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=(($.a + $.b) > 10i32)
      child: vortex.plan.pack({a=i32, b=i32}, rows=3)
        a: vortex.plan.take(i32, rows=3)
          codes: vortex.plan.filter(u8, rows=3)
            child: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.filter(i32, rows=2)
            child: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.filter(i32, rows=3)
          child: vortex.plan.segment_scan(i32, rows=3)
    ");
    assert_eq!(
        optimized.display_tree().to_string(),
        optimize(optimized.clone())?.display_tree().to_string()
    );
    Ok(())
}

#[test]
fn negative_cost_part_of_expression_pushes_into_dictionary_values() -> VortexResult<()> {
    let dictionary = DictLayout::new(
        flat(2, DType::Utf8(Nullability::NonNullable), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let expression = cast(
        byte_length(root()),
        primitive(PType::I64, Nullability::NonNullable),
    );
    let optimized = optimize(make_eval(expression, make_plan(dictionary)?)?.into_plan())?;

    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(i64, rows=3) expr=cast($ as i64)
      child: vortex.plan.take(u64, rows=3)
        codes: vortex.plan.filter(u8, rows=3)
          child: vortex.plan.segment_scan(u8, rows=3)
        values: vortex.plan.eval(u64, rows=2) expr=vortex.byte_length($)
          child: vortex.plan.filter(utf8, rows=2)
            child: vortex.plan.segment_scan(utf8, rows=2)
    ");
    Ok(())
}

#[test]
fn dictionary_pushdown_rejects_unsafe_expressions() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype, 0),
        flat(3, primitive(PType::U8, Nullability::Nullable), 1),
    )
    .into_layout();

    for expression in [
        lit(false),
        is_null(root()),
        gt(checked_add(root(), lit(1_i32)), lit(5_i32)),
    ] {
        let plan = make_eval(expression.clone(), make_plan(Arc::clone(&dictionary))?)?.into_plan();
        let optimized = optimize(plan)?;
        let eval = optimized.as_opt::<Eval>().ok_or_else(|| {
            vortex_err!(AssertionFailed: "Expression unexpectedly pushed into dictionary: {expression}")
        })?;
        assert!(eval.child_plan()?.is::<Take>());
    }
    Ok(())
}

#[test]
fn non_strict_expression_pushes_into_dictionary_without_null_codes() -> VortexResult<()> {
    let dictionary = DictLayout::new(
        flat(2, primitive(PType::I32, Nullability::Nullable), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let optimized = optimize(make_eval(is_null(root()), make_plan(dictionary)?)?.into_plan())?;
    let take = optimized.as_opt::<Take>().ok_or_else(
        || vortex_err!(AssertionFailed: "is_null was not pushed into the dictionary: {optimized}"),
    )?;
    assert!(take.values()?.is::<Eval>());
    Ok(())
}

#[test]
fn nullable_struct_keeps_expression_above_parent_validity() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone())]),
            Nullability::Nullable,
        ),
        vec![
            flat(3, DType::Bool(Nullability::NonNullable), 0),
            flat(3, field_dtype, 1),
        ],
    )
    .into_layout();
    let plan = make_eval(gt(get_item("a", root()), lit(5_i32)), make_plan(layout)?)?.into_plan();

    let optimized = optimize(plan)?;
    let eval = optimized.as_opt::<Eval>().ok_or_else(
        || vortex_err!(AssertionFailed: "Nullable struct expression unexpectedly pushed down"),
    )?;
    assert!(eval.child_plan()?.is::<Pack>());
    Ok(())
}

#[test]
fn zoned_plan_exposes_data_and_zones() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let zone_len =
        NonZeroUsize::new(3).ok_or_else(|| vortex_err!(AssertionFailed: "zone length is zero"))?;
    let layout = ZonedLayout::try_new(
        flat(5, dtype, 0),
        flat(2, zones_dtype, 1),
        zone_len,
        vec![max].into(),
    )?
    .into_layout();

    let plan = make_plan(layout)?;
    assert!(plan.is::<Zoned>());
    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.zoned(i32, rows=5)
      data: vortex.plan.filter(i32, rows=5)
        child: vortex.plan.segment_scan(i32, rows=5)
      zones: vortex.plan.filter({vortex.max()=i32?}, rows=2)
        child: vortex.plan.segment_scan({vortex.max()=i32?}, rows=2)
    ");
    Ok(())
}

#[test]
fn stats_expression_rewrites_to_zoned_pruning_plan() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let max_dtype = max
        .state_dtype(&dtype)
        .ok_or_else(|| vortex_err!(AssertionFailed: "max does not support {dtype}"))?;
    let zones_dtype = DType::Struct(
        StructFields::from_iter([(max.to_string(), max_dtype.as_nullable())]),
        Nullability::NonNullable,
    );
    let zone_len =
        NonZeroUsize::new(3).ok_or_else(|| vortex_err!(AssertionFailed: "zone length is zero"))?;
    let layout = ZonedLayout::try_new(
        flat(5, dtype.clone(), 0),
        flat(2, zones_dtype, 1),
        zone_len,
        vec![max].into(),
    )?
    .into_layout();
    let session = vortex_array::array_session();
    let filter = gt(root(), lit(5_i32));
    let falsifier = filter
        .bind(&dtype)?
        .falsify(&session)?
        .ok_or_else(|| vortex_err!(AssertionFailed: "filter has no falsifier"))?;
    let source = make_plan(layout)?;
    let plan = EvalPlan::try_new(falsifier.clone(), source.clone())?.into_plan();

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: vortex.plan.eval(bool?, rows=5) expr=(stat($, vortex.max()) <= 5i32)
      child: vortex.plan.zoned(i32, rows=5)
        data: vortex.plan.filter(i32, rows=5)
          child: vortex.plan.segment_scan(i32, rows=5)
        zones: vortex.plan.filter({vortex.max()=i32?}, rows=2)
          child: vortex.plan.segment_scan({vortex.max()=i32?}, rows=2)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.zoned(bool?, rows=5) prune=(stat($, vortex.max()) <= 5i32)
      zones: vortex.plan.filter({vortex.max()=i32?}, rows=2)
        child: vortex.plan.segment_scan({vortex.max()=i32?}, rows=2)
    ");
    let zoned = optimized
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!(AssertionFailed: "optimized pruning plan is not zoned"))?;
    assert!(zoned.is_pruning());
    assert_eq!(zoned.pruning_expression(), Some(&falsifier));
    assert!(zoned.data_plan()?.is_none());
    assert_eq!(zoned.children().len(), 1);

    let mixed_expression = bound_and(falsifier, gt(root(), lit(0_i32)).bind(&dtype)?);
    let mixed = optimize(EvalPlan::try_new(mixed_expression, source)?.into_plan())?;
    let mixed = mixed.as_opt::<Eval>().ok_or_else(
        || vortex_err!(AssertionFailed: "expression with a data reference was pushed into zones"),
    )?;
    let mixed_child = mixed.child_plan()?;
    assert!(mixed_child.is::<Zoned>());
    assert!(
        !mixed_child
            .as_opt::<Zoned>()
            .ok_or_else(|| vortex_err!(AssertionFailed: "mixed expression child is not zoned"))?
            .is_pruning()
    );
    Ok(())
}

#[test]
fn pruning_expression_partitions_across_row_idx_and_zoned_struct_field() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let max_dtype = max
        .state_dtype(&value_dtype)
        .ok_or_else(|| vortex_err!(AssertionFailed: "max does not support {value_dtype}"))?;
    let zones_dtype = DType::Struct(
        StructFields::from_iter([(max.to_string(), max_dtype.as_nullable())]),
        Nullability::NonNullable,
    );
    let zone_len =
        NonZeroUsize::new(3).ok_or_else(|| vortex_err!(AssertionFailed: "zone length is zero"))?;
    let zoned = ZonedLayout::try_new(
        flat(5, value_dtype.clone(), 0),
        flat(2, zones_dtype, 1),
        zone_len,
        vec![max].into(),
    )?
    .into_layout();
    let struct_dtype = DType::Struct(
        StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
        Nullability::NonNullable,
    );
    let layout = StructLayout::new(
        5,
        struct_dtype.clone(),
        vec![zoned, flat(5, value_dtype, 2)],
    )
    .into_layout();
    let source = make_plan(layout)?;
    let session = vortex_array::array_session();
    let filter = and(
        gt(row_idx(), lit(11_u64)),
        gt(get_item("a", root()), lit(5_i32)),
    );
    let falsifier = filter
        .bind(&struct_dtype)?
        .falsify(&session)?
        .ok_or_else(|| vortex_err!(AssertionFailed: "filter has no falsifier"))?;

    let optimized = optimize(plan_row_idx_expression(falsifier, source)?)?;
    insta::assert_snapshot!(optimized.display_tree(), @r"
    root: vortex.plan.eval(bool?, rows=5) expr=($.row_idx or $.child)
      child: vortex.plan.pack({row_idx=bool?, child=bool?}, rows=5)
        row_idx: vortex.plan.eval(bool?, rows=5) expr=(stat($, vortex.max()) <= 11u64)
          child: vortex.plan.row_idx(u64, rows=5)
        child: vortex.plan.zoned(bool?, rows=5) prune=(stat($, vortex.max()) <= 5i32)
          zones: vortex.plan.filter({vortex.max()=i32?}, rows=2)
            child: vortex.plan.segment_scan({vortex.max()=i32?}, rows=2)
    ");
    Ok(())
}

/// A legacy stats layout over 5 rows of `dtype` in zones of `zone_len` rows, whose zone table of
/// `zones_dtype` holds `stats`.
fn legacy_layout(
    dtype: &DType,
    zone_len: u32,
    stats: &[Stat],
    zones_dtype: DType,
) -> VortexResult<LayoutRef> {
    let children = OwnedLayoutChildren::layout_children(vec![
        flat(5, dtype.clone(), 0),
        flat(2, zones_dtype, 1),
    ]);
    let session = vortex_array::array_session();
    let read_ctx = ReadContext::new([]);
    let build_ctx = LayoutBuildContext {
        session: &session,
        array_read_ctx: &read_ctx,
    };
    let mut metadata = zone_len.to_le_bytes().to_vec();
    metadata.extend(as_stat_bitset_bytes(stats));
    LayoutEncoding::build(
        &LegacyStatsLayoutEncoding,
        dtype,
        5,
        &metadata,
        Vec::new(),
        children.as_ref(),
        &build_ctx,
    )
}

#[test]
fn legacy_stats_layout_lowers_to_legacy_stats_plan() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = legacy_layout(&dtype, 3, &[Stat::Max], legacy_max_zones_dtype(&dtype))?;

    let plan = make_plan(layout)?;
    assert!(plan.is::<LegacyStats>());
    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.legacy_stats(i32, rows=5)
      data: vortex.plan.filter(i32, rows=5)
        child: vortex.plan.segment_scan(i32, rows=5)
      zones: vortex.plan.filter({max=i32?, max_is_truncated=bool}, rows=2)
        child: vortex.plan.segment_scan({max=i32?, max_is_truncated=bool}, rows=2)
    ");
    Ok(())
}

/// Zoned layouts with no zones, or a zone table holding no statistics, lower to their data.
#[test]
fn zoned_layouts_without_statistics_lower_to_their_data() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let empty = DType::Struct(StructFields::empty(), Nullability::NonNullable);
    let zone_len =
        NonZeroUsize::new(3).ok_or_else(|| vortex_err!(AssertionFailed: "zone length is zero"))?;
    let layouts = [
        legacy_layout(&dtype, 0, &[Stat::Max], legacy_max_zones_dtype(&dtype))?,
        legacy_layout(&dtype, 3, &[], empty.clone())?,
        ZonedLayout::try_new(
            flat(5, dtype.clone(), 0),
            flat(2, empty, 1),
            zone_len,
            Vec::new().into(),
        )?
        .into_layout(),
    ];
    for layout in layouts {
        let plan = make_plan(layout)?;
        assert!(plan.is::<Filter>(), "{}", plan.display_tree());
    }
    Ok(())
}

/// A legacy zone table holding the max of a `dtype` column.
fn legacy_max_zones_dtype(dtype: &DType) -> DType {
    DType::Struct(
        StructFields::from_iter([
            (Stat::Max.name(), dtype.as_nullable()),
            (MAX_IS_TRUNCATED, DType::Bool(Nullability::NonNullable)),
        ]),
        Nullability::NonNullable,
    )
}

/// Legacy stats over a 5-row i32 column in 3-row zones, holding each zone's max.
fn legacy_max_plan(dtype: &DType) -> VortexResult<PlanRef> {
    Ok(LegacyStatsPlan::new(
        make_plan(flat(5, dtype.clone(), 0))?,
        make_plan(flat(2, legacy_max_zones_dtype(dtype), 1))?,
        3,
        vec![Stat::Max].into(),
    )
    .into_plan())
}

/// The aggregate zone table of a max over `dtype`, and the max aggregate.
fn max_zones_dtype(dtype: &DType) -> VortexResult<(AggregateFnRef, DType)> {
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let max_dtype = max
        .state_dtype(dtype)
        .ok_or_else(|| vortex_err!(AssertionFailed: "max does not support {dtype}"))?;
    let zones_dtype = DType::Struct(
        StructFields::from_iter([(max.to_string(), max_dtype.as_nullable())]),
        Nullability::NonNullable,
    );
    Ok((max, zones_dtype))
}

#[test]
fn legacy_stats_rewrite_to_zoned_over_converted_zones() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let legacy = legacy_max_plan(&dtype)?;

    let optimized = optimize(legacy.clone())?;
    let zoned = optimized
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!(AssertionFailed: "legacy stats did not become zoned"))?;
    assert!(!zoned.is_pruning());
    assert_eq!(zoned.zone_len(), 3);
    assert_eq!(
        zoned
            .aggregate_fns()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        vec![max.to_string()]
    );
    let zones = zoned.zones_plan()?;
    assert!(zones.is::<Eval>());
    assert_eq!(zones.dtype(), &zones_dtype);

    let falsifier = gt(root(), lit(5_i32))
        .bind(&dtype)?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!(AssertionFailed: "filter has no falsifier"))?;
    let pruning = optimize(EvalPlan::try_new(falsifier, legacy)?.into_plan())?;
    let pruning = pruning.as_opt::<Zoned>().ok_or_else(
        || vortex_err!(AssertionFailed: "legacy stats predicate did not become a pruning plan"),
    )?;
    assert!(pruning.is_pruning());
    assert!(pruning.zones_plan()?.is::<Eval>());
    Ok(())
}

#[test]
fn converted_legacy_zones_prune() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let table_dtype = legacy_max_zones_dtype(&dtype);
    let (aggregate_fns, conversion) =
        legacy_stats_to_aggregates(&dtype, &table_dtype, &[Stat::Max]);
    let table = StructArray::from_fields(&[
        (
            Stat::Max.name(),
            PrimitiveArray::from_option_iter([Some(3_i32), Some(9)]).into_array(),
        ),
        (
            MAX_IS_TRUNCATED,
            BoolArray::from_iter([false, true]).into_array(),
        ),
    ])?;
    assert_eq!(table.dtype(), &table_dtype);

    let converted = table
        .into_array()
        .apply_bound(&conversion.bind(&table_dtype)?)?
        .execute::<StructArray>(&mut SESSION.create_execution_ctx())?;
    let zone_map = ZoneMap::try_new(dtype.clone(), converted, aggregate_fns, 3, 5)?;
    let falsifier = gt(root(), lit(5_i32))
        .bind(&dtype)?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!(AssertionFailed: "filter has no falsifier"))?;
    // Only the first zone's max proves no row exceeds 5.
    assert_eq!(
        zone_map.prune(&falsifier, &SESSION)?,
        Mask::from_iter([true, false])
    );
    Ok(())
}

#[test]
fn legacy_stats_under_zoned_are_dropped() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let zones = make_plan(flat(2, zones_dtype, 2))?;
    let plan =
        ZonedPlan::new(legacy_max_plan(&dtype)?, zones.clone(), 3, vec![max].into()).into_plan();

    let optimized = optimize(plan)?;
    let zoned = optimized
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!(AssertionFailed: "zoned plan was rewritten away"))?;
    assert!(zoned.data_plan()?.is_some_and(|data| data.is::<Filter>()));
    assert!(PlanRef::ptr_eq(&zoned.zones_plan()?, &zones));
    Ok(())
}

#[test]
fn legacy_stats_over_zoned_are_dropped() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let zoned = ZonedPlan::new(
        make_plan(flat(5, dtype.clone(), 0))?,
        make_plan(flat(2, zones_dtype, 2))?,
        3,
        vec![max].into(),
    )
    .into_plan();
    let legacy = LegacyStatsPlan::new(
        zoned.clone(),
        make_plan(flat(2, legacy_max_zones_dtype(&dtype), 1))?,
        3,
        vec![Stat::Max].into(),
    )
    .into_plan();

    assert!(PlanRef::ptr_eq(&optimize(legacy)?, &zoned));
    Ok(())
}

#[test]
fn legacy_stats_with_zero_zone_length_keep_only_data() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let data = make_plan(flat(5, dtype.clone(), 0))?;
    let legacy = LegacyStatsPlan::new(
        data.clone(),
        make_plan(flat(2, legacy_max_zones_dtype(&dtype), 1))?,
        0,
        vec![Stat::Max].into(),
    )
    .into_plan();

    assert!(PlanRef::ptr_eq(&optimize(legacy)?, &data));
    Ok(())
}

/// A predicate holding a dynamic comparison is never pushed onto dictionary values, alone or
/// alongside static conjuncts: it stays above the take, so the values never change.
#[test]
fn dynamic_predicates_stay_above_dictionary_values() -> VortexResult<()> {
    let values_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let layout =
        DictLayout::new(flat(2, values_dtype.clone(), 0), flat(3, codes_dtype, 1)).into_layout();
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
        let optimized =
            optimize(make_eval(predicate, make_plan(Arc::clone(&layout))?)?.into_plan())?;
        let eval = optimized
            .as_opt::<Eval>()
            .ok_or_else(|| vortex_err!(AssertionFailed: "dynamic predicate was pushed down"))?;
        let take = eval
            .child_plan()?
            .as_opt::<Take>()
            .ok_or_else(|| vortex_err!(AssertionFailed: "dynamic predicate is not over the take"))?
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
    let take = TakePlan::new(
        make_plan(flat(3, codes_dtype, 1))?,
        make_plan(flat(2, values_dtype.clone(), 0))?,
    )
    .into_plan();
    assert!(take.dtype().is_nullable());
    let plan = make_eval(gt(root(), lit(5_i32)), take)?.into_plan();
    let dtype = plan.dtype().clone();

    let optimized = optimize(plan)?;
    assert_eq!(optimized.dtype(), &dtype);
    let take = optimized
        .as_opt::<Take>()
        .ok_or_else(|| vortex_err!(AssertionFailed: "predicate was not pushed onto the values"))?;
    let values = take.values()?;
    let eval = values
        .as_opt::<Eval>()
        .ok_or_else(|| vortex_err!(AssertionFailed: "values are not evaluated"))?;
    assert!(eval.expression().is_root_bound_to(&values_dtype));
    Ok(())
}

/// Rebuilding a pruning plan over new zones discards the proof over the old ones.
#[test]
fn zoned_pruning_plan_with_new_zones_proves_them_again() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let zones = make_plan(flat(2, zones_dtype, 1))?;
    let source = ZonedPlan::new(
        make_plan(flat(5, dtype.clone(), 0))?,
        zones.clone(),
        3,
        vec![max].into(),
    )
    .into_plan();
    let falsifier = gt(root(), lit(5_i32))
        .bind(&dtype)?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!(AssertionFailed: "filter has no falsifier"))?;
    let pruning = optimize(EvalPlan::try_new(falsifier, source)?.into_plan())?;
    let pruning = pruning
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!(AssertionFailed: "predicate did not become a pruning plan"))?;
    let proof = pruning
        .pruned_zones()
        .ok_or_else(|| vortex_err!(AssertionFailed: "pruning plan has no proof"))?;

    let rebuilt = pruning.clone().into_plan().with_children(vec![zones])?;
    let rebuilt_proof = rebuilt
        .as_opt::<Zoned>()
        .and_then(|zoned| zoned.pruned_zones())
        .ok_or_else(|| vortex_err!(AssertionFailed: "rebuilt plan has no proof"))?;
    assert!(!Arc::ptr_eq(proof, rebuilt_proof));
    Ok(())
}
