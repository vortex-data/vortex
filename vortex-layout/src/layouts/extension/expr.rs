// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rebinding expressions over extension values to their storage.

use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::ext_storage::ExtStorage;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_error::VortexResult;

/// Rebinds `expr`, bound to an extension scope, to the extension's storage.
///
/// Every `ext_storage(root)` becomes the storage root. `is_null(root)` and `is_not_null(root)` read
/// the storage validity, because an extension value is null exactly when its storage value is null.
/// Returns `None` when `expr` uses the root in any other way, which needs the extension values
/// themselves.
pub(super) fn rebind_to_storage(
    expr: &BoundExpression,
    storage_dtype: &DType,
) -> VortexResult<Option<BoundExpression>> {
    let storage_root = BoundExpression::new_root(storage_dtype.clone());
    let mut reads_values = false;
    let rebound = expr
        .clone()
        .transform_down(|node| {
            if node.is_root() {
                reads_values = true;
                return Ok(Transformed::no(node));
            }
            if !node
                .children()
                .first()
                .is_some_and(BoundExpression::is_root)
            {
                return Ok(Transformed::no(node));
            }
            let value = if node.is::<ExtStorage>() {
                storage_root.clone()
            } else if node.is::<IsNull>() || node.is::<IsNotNull>() {
                node.with_children([storage_root.clone()])?
            } else {
                return Ok(Transformed::no(node));
            };
            Ok(Transformed {
                value,
                changed: true,
                order: TraversalOrder::Skip,
            })
        })?
        .into_inner();
    Ok((!reads_values).then_some(rebound))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::ext_storage;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;

    use super::rebind_to_storage;
    use crate::test::TestPoint;

    fn extension_dtype() -> DType {
        DType::Extension(TestPoint::dtype(Nullability::Nullable))
    }

    fn storage_dtype() -> DType {
        TestPoint::dtype(Nullability::Nullable)
            .storage_dtype()
            .clone()
    }

    #[rstest]
    #[case::storage(ext_storage(root()), "$")]
    #[case::field(get_item("a", ext_storage(root())), "$.a")]
    #[case::predicate(
        and(
            gt(get_item("a", ext_storage(root())), lit(1i32)),
            gt(get_item("b", ext_storage(root())), lit(2i64)),
        ),
        "(($.a > 1i32) and ($.b > 2i64))"
    )]
    #[case::is_null(is_null(root()), "vortex.is_null($)")]
    #[case::is_not_null(is_not_null(root()), "is_not_null($)")]
    fn rebinds_storage_reads(#[case] expr: Expression, #[case] expected: &str) -> VortexResult<()> {
        let rebound = rebind_to_storage(&expr.bind(&extension_dtype())?, &storage_dtype())?
            .expect("reads only the storage");
        assert_eq!(rebound.to_string(), expected);
        assert!(rebound.is_root_bound_to(&storage_dtype()));
        Ok(())
    }

    #[rstest]
    #[case::root(root())]
    #[case::mixed(pack(
        [("a", get_item("a", ext_storage(root()))), ("value", root())],
        Nullability::NonNullable,
    ))]
    fn keeps_value_reads(#[case] expr: Expression) -> VortexResult<()> {
        let bound = expr.bind(&extension_dtype())?;
        assert!(rebind_to_storage(&bound, &storage_dtype())?.is_none());
        Ok(())
    }
}
