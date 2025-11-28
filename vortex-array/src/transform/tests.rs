// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::FieldNames;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArraySession;
use crate::array::ArrayRef;
use crate::array::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::expr::session::ExprSession;
use crate::expr::transform::ExprOptimizer;
use crate::transform::ArrayParentReduceRule;
use crate::transform::ArrayReduceRule;
use crate::transform::ArrayRuleContext;
use crate::validity::Validity;

/// Test rule that unwraps single-chunk ChunkedArrays
#[derive(Debug, Default)]
struct UnwrapSingleChunkRule;

impl ArrayReduceRule<ChunkedVTable> for UnwrapSingleChunkRule {
    fn reduce(
        &self,
        array: &ChunkedArray,
        _ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>> {
        if array.nchunks() == 1 {
            return Ok(Some(array.chunk(0).clone()));
        }
        Ok(None)
    }
}

#[test]
fn test_unwrap_single_chunk_rule() -> VortexResult<()> {
    let expr_session = ExprSession::default();
    let expr_optimizer = ExprOptimizer::new(&expr_session);
    let ctx = ArrayRuleContext::new(expr_optimizer);

    let primitive = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let chunked = ChunkedArray::from_iter([primitive.clone()]);

    let result = UnwrapSingleChunkRule
        .reduce(&chunked, &ctx)?
        .vortex_expect("transformed");

    assert!(Arc::ptr_eq(&primitive, &result));
    Ok(())
}

#[test]
fn test_unwrap_single_chunk_rule_no_op() -> VortexResult<()> {
    let expr_session = ExprSession::default();
    let expr_optimizer = ExprOptimizer::new(&expr_session);
    let ctx = ArrayRuleContext::new(expr_optimizer);

    let chunked = ChunkedArray::from_iter([
        PrimitiveArray::from_iter([1i32, 2]).into_array(),
        PrimitiveArray::from_iter([3i32, 4]).into_array(),
    ]);

    let result = UnwrapSingleChunkRule.reduce(&chunked, &ctx)?;

    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_reduce_rules_traverse_whole_tree() -> VortexResult<()> {
    let array_session = ArraySession::default();
    let expr_session = ExprSession::default();

    array_session.register_reduce_rule::<ChunkedVTable, UnwrapSingleChunkRule>(
        &ChunkedVTable,
        UnwrapSingleChunkRule,
    );

    let expr_optimizer = ExprOptimizer::new(&expr_session);
    let optimizer = array_session.optimizer(expr_optimizer);

    let inner_field1 = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let inner_field1_chunked = ChunkedArray::from_iter([inner_field1.clone()]);

    let inner_field2 = PrimitiveArray::from_iter([4i32, 5, 6]).into_array();
    let inner_field2_chunked = ChunkedArray::from_iter([inner_field2.clone()]);

    let inner_struct = StructArray::try_new(
        FieldNames::from(["field1", "field2"]),
        vec![
            inner_field1_chunked.into_array(),
            inner_field2_chunked.into_array(),
        ],
        3,
        Validity::NonNullable,
    )?;

    let outer_field = PrimitiveArray::from_iter([100i64, 200, 300]).into_array();
    let outer_field_chunked = ChunkedArray::from_iter([outer_field.clone()]);

    let outer_struct = StructArray::try_new(
        FieldNames::from(["inner_struct", "outer_field"]),
        vec![inner_struct.into_array(), outer_field_chunked.into_array()],
        3,
        Validity::NonNullable,
    )?;

    let optimized = optimizer.optimize_array(outer_struct.into_array())?;

    let optimized_outer = optimized.as_opt::<StructVTable>().unwrap();
    let optimized_inner_struct = optimized_outer.field_by_name("inner_struct")?;
    let optimized_outer_field = optimized_outer.field_by_name("outer_field")?;

    assert!(Arc::ptr_eq(&outer_field, optimized_outer_field));

    let inner_struct_view = optimized_inner_struct.as_opt::<StructVTable>().unwrap();
    let optimized_field1 = inner_struct_view.field_by_name("field1")?;
    let optimized_field2 = inner_struct_view.field_by_name("field2")?;

    assert!(Arc::ptr_eq(&inner_field1, optimized_field1));
    assert!(Arc::ptr_eq(&inner_field2, optimized_field2));
    Ok(())
}

// Odd rule for testing
#[derive(Debug, Default)]
struct ConstantInStructRule;

impl ArrayParentReduceRule<ConstantVTable, StructVTable> for ConstantInStructRule {
    fn reduce_parent(
        &self,
        array: &ConstantArray,
        parent: &StructArray,
        _child_idx: usize,
        _ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>> {
        StructArray::try_from_iter(
            parent
                .names()
                .iter()
                .zip(parent.fields().iter())
                .enumerate()
                .map(|(idx, (name, field))| {
                    if field.is::<ConstantVTable>() {
                        (
                            name,
                            ConstantArray::new(
                                i32::try_from(idx).vortex_expect("must fit"),
                                array.len(),
                            )
                            .into_array(),
                        )
                    } else {
                        (name, field.clone())
                    }
                }),
        )
        .map(|s| Some(s.to_array()))
    }
}

#[test]
fn test_parent_rules_traverse_whole_tree() -> VortexResult<()> {
    let array_session = ArraySession::default();
    let expr_session = ExprSession::default();

    array_session.register_parent_rule::<ConstantVTable, StructVTable, ConstantInStructRule>(
        &ConstantVTable,
        &StructVTable,
        ConstantInStructRule,
    );

    let expr_optimizer = ExprOptimizer::new(&expr_session);
    let optimizer = array_session.optimizer(expr_optimizer);

    let deep_field1 = ConstantArray::new(100i32, 5);
    let deep_field2 = ConstantArray::new(200i32, 5);

    let inner_struct = StructArray::try_new(
        FieldNames::from(["deep_field1", "deep_field2"]),
        vec![deep_field1.into_array(), deep_field2.into_array()],
        5,
        Validity::NonNullable,
    )?;

    let outer_field = ConstantArray::new(999i32, 5);

    let outer_struct = StructArray::from_fields(&[
        ("inner_struct", inner_struct.into_array()),
        ("outer_field", outer_field.into_array()),
    ])?
    .into_array();

    let optimized = optimizer.optimize_array(outer_struct.clone())?;

    let optimized_outer = optimized.as_opt::<StructVTable>().unwrap();
    let inner_struct = optimized_outer.field_by_name("inner_struct")?;
    let outer_field = optimized_outer.field_by_name("outer_field")?;

    let outer_field_const = outer_field.as_constant().vortex_expect("is constant");
    assert_eq!(
        i32::try_from(outer_field_const)?,
        1,
        "outer_field at depth 1 should have child_idx=1 from parent rule"
    );

    let inner_struct_view = inner_struct.as_opt::<StructVTable>().unwrap();
    let deep_field1 = inner_struct_view.field_by_name("deep_field1")?;
    let deep_field2 = inner_struct_view.field_by_name("deep_field2")?;

    let deep_field1_const = deep_field1.as_constant().vortex_expect("is constant");
    let deep_field2_const = deep_field2.as_constant().vortex_expect("is constant");

    assert_eq!(
        i32::try_from(deep_field1_const)?,
        0,
        "deep_field1 at depth 2 should have child_idx=0 from parent rule"
    );
    assert_eq!(
        i32::try_from(deep_field2_const)?,
        1,
        "deep_field2 at depth 2 should have child_idx=1 from parent rule"
    );

    Ok(())
}
