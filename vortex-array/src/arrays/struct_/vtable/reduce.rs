// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Public API for expression partitioning over struct arrays - may be unused in this crate
// but is intended for external use (e.g., in vortex-layout)

use std::sync::Arc;

use vortex_dtype::{FieldName, FieldNames};
use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ExprArray, StructArray};
use crate::expr::session::ExprSession;
use crate::expr::transform::immediate_access::annotate_scope_access;
use crate::expr::transform::{
    ExprOptimizer, PartitionedExpr, partition, replace, replace_root_fields,
};
use crate::expr::{Expression, col, root};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray};

/// Result of partitioning an expression over a struct.
#[derive(Debug)]
pub(crate) enum Partitioned {
    /// An expression which only operates over a single field
    Single(FieldName, Expression),
    /// An expression which operates over multiple fields
    Multi(Arc<PartitionedExpr<FieldName>>),
}

/// Partition an expression over the fields of a struct array.
///
/// This is used to optimize expression evaluation by splitting expressions that access
/// multiple struct fields into per-field sub-expressions that can be evaluated independently.
///
/// # Arguments
/// * `struct_array` - The struct array whose fields the expression accesses
/// * `expr` - The expression to partition
/// * `session` - The expression session containing registered expressions and rules
///
/// # Returns
/// A `PartitionedStructExpr` indicating whether the expression accesses a single field
/// or multiple fields, along with the partitioned sub-expressions.
pub(crate) fn partition_struct_expr(
    struct_array: &StructArray,
    expr: Expression,
    session: &ExprSession,
) -> VortexResult<Partitioned> {
    let struct_fields = struct_array.struct_fields();

    // First, expand the root scope into the fields of the struct to ensure
    // that partitioning works correctly.
    let expanded_expr = replace(expr, &root(), replace_root_fields(root(), struct_fields));

    // Get optimizer from session
    let opt = ExprOptimizer::new(session);

    let expanded_expr = opt
        .optimize_typed(expanded_expr, struct_array.dtype())
        .vortex_expect("Failed to optimize expression over struct fields");

    // Partition the expression into expressions that can be evaluated over individual fields
    let mut partitioned = partition(
        expanded_expr.clone(),
        struct_array.dtype(),
        annotate_scope_access(struct_fields),
        &opt,
    )
    .vortex_expect("Failed to partition expression over struct fields");

    if partitioned.partitions.len() == 1 {
        // If there's only one partition, we step into the field scope of the original
        // expression by replacing any `$.a` with `$`.
        return Ok(Partitioned::Single(
            partitioned.partition_names[0].clone(),
            replace(
                expanded_expr,
                &col(partitioned.partition_names[0].clone()),
                root(),
            ),
        ));
    }

    // We now need to process the partitioned expressions to rewrite the root scope
    // to be that of the field, rather than the struct. In other words, "stepping in"
    // to the field scope.
    partitioned.partitions = partitioned
        .partitions
        .iter()
        .zip(partitioned.partition_names.iter())
        .map(|(e, name)| replace(e.clone(), &col(name.clone()), root()))
        .collect();

    Ok(Partitioned::Multi(Arc::new(partitioned)))
}

/// Apply a partitioned expression to a struct array by wrapping each field in an ExprArray.
///
/// This creates a new StructArray where each field has its corresponding partitioned
/// expression applied to it.
pub(crate) fn apply_partitioned_expr(
    struct_array: &StructArray,
    partitioned: Partitioned,
) -> VortexResult<ArrayRef> {
    match partitioned {
        Partitioned::Single(field_name, expr) => {
            // Only one field is accessed - optimize by only including that field
            let field_idx = struct_array
                .struct_fields()
                .find(&field_name)
                .vortex_expect("Field should exist in struct");

            let field = &struct_array.fields()[field_idx];
            let dtype = expr
                .return_dtype(field.dtype())
                .vortex_expect("Expression should have valid return dtype");
            Ok(ExprArray::try_new(field.clone(), expr, dtype)?.into_array())
        }
        Partitioned::Multi(partitioned) => {
            // Multiple fields accessed - only include fields that are used in the expression
            let fields_and_names: Vec<(FieldName, ArrayRef)> = struct_array
                .fields()
                .iter()
                .enumerate()
                .filter_map(|(idx, field)| {
                    let field_name = &struct_array.names()[idx];

                    // Find if this field has a partition
                    partitioned
                        .partition_names
                        .iter()
                        .position(|name| name == field_name)
                        .map(|partition_idx| {
                            let expr = &partitioned.partitions[partition_idx];
                            ExprArray::try_new(
                                field.clone(),
                                expr.clone(),
                                partitioned.partition_dtypes[partition_idx].clone(),
                            )
                            .map(|e| (field_name.clone(), e.into_array()))
                        })
                })
                .collect::<VortexResult<_>>()?;

            let (field_names, new_fields): (Vec<_>, Vec<_>) = fields_and_names.into_iter().unzip();

            let child = StructArray::try_new(
                FieldNames::from(field_names),
                new_fields,
                struct_array.len(),
                struct_array.validity().clone(),
            )?
            .into_array();

            Ok(ExprArray::new_infer_dtype(child, partitioned.root.clone())?.into_array())
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::op_ref)]

    use vortex_dtype::FieldNames;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::expr::{and, eq, get_item, gt, lit, lt, root};
    use crate::validity::Validity;

    fn make_test_struct() -> StructArray {
        // Create a struct with fields "a" and "b"
        let a_field = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let b_field = PrimitiveArray::from_iter([10i32, 20, 30, 40, 50]);

        StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![a_field.into_array(), b_field.into_array()],
            5,
            Validity::NonNullable,
        )
        .unwrap()
    }

    #[test]
    fn test_partition_single_field_simple() {
        // Test: get($, "a") > 2
        let struct_array = make_test_struct();
        let expr = gt(get_item("a", root()), lit(2));
        let session = ExprSession::default();

        let partitioned = partition_struct_expr(&struct_array, expr, &session).unwrap();

        match partitioned {
            Partitioned::Single(field_name, _expr) => {
                assert_eq!(field_name.as_ref(), "a");
            }
            Partitioned::Multi(_) => {
                panic!("Expected single partition for expression accessing only field 'a'");
            }
        }
    }

    #[test]
    fn test_partition_single_field_compound() {
        // Test: get($, "a") > 2 & get($, "a") < 5
        let struct_array = make_test_struct();
        let expr = and(
            gt(get_item("a", root()), lit(2)),
            lt(get_item("a", root()), lit(5)),
        );
        let session = ExprSession::default();

        let partitioned = partition_struct_expr(&struct_array, expr, &session).unwrap();

        match partitioned {
            Partitioned::Single(field_name, _expr) => {
                assert_eq!(field_name.as_ref(), "a");
            }
            Partitioned::Multi(_) => {
                panic!("Expected single partition for expression accessing only field 'a'");
            }
        }
    }

    #[test]
    fn test_partition_multi_field() {
        // Test: get($, "a") > 2 & get($, "b") == 10
        let struct_array = make_test_struct();
        let expr = and(
            gt(get_item("a", root()), lit(2)),
            eq(get_item("b", root()), lit(10)),
        );
        let session = ExprSession::default();

        let partitioned = partition_struct_expr(&struct_array, expr, &session).unwrap();

        match partitioned {
            Partitioned::Single(..) => {
                panic!("Expected multi partition for expression accessing fields 'a' and 'b'");
            }
            Partitioned::Multi(partitioned) => {
                // Should have partitions for both "a" and "b"
                let a_name: FieldName = "a".into();
                let b_name: FieldName = "b".into();
                assert!(partitioned.partition_names.iter().any(|n| n == &a_name));
                assert!(partitioned.partition_names.iter().any(|n| n == &b_name));
                assert_eq!(partitioned.partitions.len(), 2);
            }
        }
    }

    #[test]
    fn test_partition_multi_field_with_field_expr() {
        // Test: get($, "a") > 2 & get($, "b") == 10 & get($, "a")
        // This accesses "a" twice and "b" once
        let struct_array = make_test_struct();
        let expr = and(
            and(
                gt(get_item("a", root()), lit(2)),
                eq(get_item("b", root()), lit(10)),
            ),
            get_item("a", root()),
        );
        let session = ExprSession::default();

        let partitioned = partition_struct_expr(&struct_array, expr, &session).unwrap();

        match partitioned {
            Partitioned::Single(..) => {
                panic!("Expected multi partition for expression accessing fields 'a' and 'b'");
            }
            Partitioned::Multi(partitioned) => {
                // Should have partitions for both "a" and "b"
                let a_name: FieldName = "a".into();
                let b_name: FieldName = "b".into();
                assert!(partitioned.partition_names.iter().any(|n| n == &a_name));
                assert!(partitioned.partition_names.iter().any(|n| n == &b_name));
            }
        }
    }

    #[test]
    fn test_partition_constant_expr() {
        // Test: 1 == 2 (no field access)
        let struct_array = make_test_struct();
        let expr = eq(lit(1), lit(2));
        let session = ExprSession::default();

        let partitioned = partition_struct_expr(&struct_array, expr, &session).unwrap();

        // A constant expression might still create partitions, but they won't reference fields
        // The behavior here depends on how the optimizer handles constant expressions
        match partitioned {
            Partitioned::Single(..) | Partitioned::Multi(_) => {
                // Either outcome is acceptable for a constant expression
            }
        }
    }
}
