// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::FieldNames;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::array::ArrayRef;
use crate::array::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::Exact;
use crate::validity::Validity;

/// Test rule that unwraps single-chunk ChunkedArrays
#[derive(Debug, Default)]
struct UnwrapSingleChunkRule;

impl ArrayReduceRule<Exact<ChunkedVTable>> for UnwrapSingleChunkRule {
    fn matcher(&self) -> Exact<ChunkedVTable> {
        Exact::from(&ChunkedVTable)
    }

    fn reduce(&self, array: &ChunkedArray) -> VortexResult<Option<ArrayRef>> {
        if array.nchunks() == 1 {
            return Ok(Some(array.chunk(0).clone()));
        }
        Ok(None)
    }
}

#[test]
fn test_unwrap_single_chunk_rule() -> VortexResult<()> {
    let primitive = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let chunked = ChunkedArray::from_iter([primitive.clone()]);

    let result = UnwrapSingleChunkRule
        .reduce(&chunked)?
        .vortex_expect("transformed");

    assert!(Arc::ptr_eq(&primitive, &result));
    Ok(())
}

#[test]
fn test_unwrap_single_chunk_rule_no_op() -> VortexResult<()> {
    let chunked = ChunkedArray::from_iter([
        PrimitiveArray::from_iter([1i32, 2]).into_array(),
        PrimitiveArray::from_iter([3i32, 4]).into_array(),
    ]);

    let result = UnwrapSingleChunkRule.reduce(&chunked)?;

    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_reduce_rules_traverse_whole_tree() -> VortexResult<()> {
    let mut optimizer = ArrayOptimizer::default();
    optimizer.register_reduce_rule(UnwrapSingleChunkRule);

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

    println!("PRE: {}", outer_struct.display_tree());
    let optimized = optimizer.optimize_recursive(outer_struct.into_array())?;
    println!("POS: {}", optimized.display_tree());

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
