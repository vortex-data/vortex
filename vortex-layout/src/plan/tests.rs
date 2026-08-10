// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::checked_add;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use super::*;
use crate::LayoutRef;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::foreign::new_foreign_layout;
use crate::layouts::list::ListLayout;
use crate::layouts::row_idx::row_idx;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentId;

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
    let expression = expression
        .optimize_recursive(child.dtype())?
        .bind(child.dtype())?;
    EvalPlan::try_new(expression, child)
}

fn child_of(plan: &PlanRef, index: usize) -> VortexResult<PlanRef> {
    plan.child(index)?
        .ok_or_else(|| vortex_err!("missing child {index}"))
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

    assert_unsupported(
        lower(&layout)
            .err()
            .ok_or_else(|| vortex_err!("unsupported layout unexpectedly produced a plan"))?,
    );
    Ok(())
}

#[test]
fn flat_plan_has_no_children() -> VortexResult<()> {
    let plan = make_plan(flat(3, primitive(PType::I32, Nullability::NonNullable), 0))?;

    assert!(plan.is::<SegmentScan>());
    assert_eq!(plan.child_count(), 0);
    assert!(plan.child(0)?.is_none());
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
    assert_unsupported(
        child_of(&plan, 1)
            .err()
            .ok_or_else(|| vortex_err!("unsupported chunk unexpectedly produced a plan"))?,
    );
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
    assert_unsupported(
        child_of(&plan, 1)
            .err()
            .ok_or_else(|| vortex_err!("unsupported field unexpectedly produced a plan"))?,
    );
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
        .ok_or_else(|| vortex_err!("mismatched arity unexpectedly succeeded"))?;
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
    let child = make_plan(flat(3, primitive(PType::I32, Nullability::NonNullable), 0))?;
    let expression = root().bind(child.dtype())?;
    let plan: PlanRef = EvalPlan::try_new(expression, child)?.into_plan();

    assert!(optimize(plan)?.is::<SegmentScan>());
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
    assert!(child_of(&optimized, 0)?.is::<SegmentScan>());
    assert!(child_of(&optimized, 1)?.is::<SegmentScan>());
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
        a: vortex.plan.segment_scan(i32, rows=3)
        b: vortex.plan.segment_scan(i32, rows=3)
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
        b: depth=2
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
      b:
      validity:
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
      chunks[0]: vortex.plan.segment_scan(i32, rows=2)
      chunks[1]: vortex.plan.segment_scan(i32, rows=1)
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
      codes: vortex.plan.segment_scan(u8, rows=3)
      values: vortex.plan.segment_scan(i32, rows=2)
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
      elements: vortex.plan.segment_scan(i32, rows=4)
      offsets: vortex.plan.segment_scan(u32, rows=3)
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
      elements: vortex.plan.segment_scan(i32, rows=4)
      offsets: vortex.plan.segment_scan(u32, rows=3)
      validity: vortex.plan.segment_scan(bool, rows=2)
    ");
    Ok(())
}

#[test]
fn row_idx_plan_preserves_row_index_expressions() -> VortexResult<()> {
    let layout = flat(3, primitive(PType::I32, Nullability::NonNullable), 0);
    let plan = RowIdxPlan::new(10, make_plan(layout)?).into_plan();
    let bound_expression = row_idx().bind(plan.dtype())?;
    let plan = optimize(EvalPlan::try_new(bound_expression.clone(), plan)?.into_plan())?;
    let expression = plan
        .as_opt::<Eval>()
        .ok_or_else(|| vortex_err!("optimized plan is not an expression plan"))?;

    assert_eq!(expression.expression(), &bound_expression);
    assert!(expression.child_plan()?.is::<RowIdx>());
    assert_eq!(expression.row_count(), 3);
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
    let plan = make_eval(
        projection,
        RowIdxPlan::new(0, make_plan(layout)?).into_plan(),
    )?
    .into_plan();

    let optimized = optimize(plan)?;
    let projection = optimized
        .as_opt::<Eval>()
        .ok_or_else(|| vortex_err!("optimized plan has no projection expression"))?;
    let child = projection.child_plan()?;
    let empty_struct = child
        .as_opt::<Pack>()
        .ok_or_else(|| vortex_err!("empty projection did not prune the RowIdx child"))?;

    assert_eq!(empty_struct.nfields(), 0);
    assert_eq!(empty_struct.children().len(), 0);
    Ok(())
}

#[test]
fn chunked_plan_preserves_global_row_index_expressions() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(1, dtype.clone(), 0), flat(2, dtype, 1)]),
    )
    .into_layout();
    let child = make_plan(layout)?;
    let bound_expression = row_idx().bind(child.dtype())?;
    let plan = optimize(EvalPlan::try_new(bound_expression.clone(), child)?.into_plan())?;
    let expression = plan
        .as_opt::<Eval>()
        .ok_or_else(|| vortex_err!("Row-index expression unexpectedly pushed into chunks"))?;

    assert_eq!(expression.expression(), &bound_expression);
    assert!(expression.child_plan()?.is::<Concat>());
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
          codes: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.segment_scan(i32, rows=3)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.take(bool, rows=3)
      codes: vortex.plan.segment_scan(u8, rows=3)
      values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
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
            codes: vortex.plan.segment_scan(u8, rows=3)
            values: vortex.plan.segment_scan(i32, rows=2)
          chunks[1]: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.segment_scan(i32, rows=5)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.concat(bool, rows=5)
      chunks[0]: vortex.plan.take(bool, rows=3)
        codes: vortex.plan.segment_scan(u8, rows=3)
        values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
          child: vortex.plan.segment_scan(i32, rows=2)
      chunks[1]: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
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
          codes: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.segment_scan(i32, rows=3)
        c: vortex.plan.segment_scan(i32, rows=3)
    ");

    let optimized = optimize(plan)?;
    insta::assert_snapshot!(optimized.display_tree(), @"
    root: vortex.plan.eval(bool, rows=3) expr=($.a and $.b)
      child: vortex.plan.pack({a=bool, b=bool}, rows=3)
        a: vortex.plan.take(bool, rows=3)
          codes: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.eval(bool, rows=2) expr=($ > 5i32)
            child: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.eval(bool, rows=3) expr=($ > 7i32)
          child: vortex.plan.segment_scan(i32, rows=3)
    ");
    assert_eq!(
        optimized.display_tree().to_string(),
        optimize(optimized.clone())?.display_tree().to_string()
    );
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
          codes: vortex.plan.segment_scan(u8, rows=3)
          values: vortex.plan.segment_scan(i32, rows=2)
        b: vortex.plan.segment_scan(i32, rows=3)
    ");
    assert_eq!(
        optimized.display_tree().to_string(),
        optimize(optimized.clone())?.display_tree().to_string()
    );
    Ok(())
}

#[test]
fn dictionary_pushdown_rejects_unsafe_expressions() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype, 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
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
            vortex_err!("Expression unexpectedly pushed into dictionary: {expression}")
        })?;
        assert!(eval.child_plan()?.is::<Take>());
    }
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
    let eval = optimized
        .as_opt::<Eval>()
        .ok_or_else(|| vortex_err!("Nullable struct expression unexpectedly pushed down"))?;
    assert!(eval.child_plan()?.is::<Pack>());
    Ok(())
}
