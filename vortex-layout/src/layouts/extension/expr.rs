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
/// Returns `None` if any use of the root requires whole extension values. A null check can use
/// the storage because the extension and its storage share the same validity.
pub(super) fn rebind_to_storage(
    expr: &BoundExpression,
    storage_dtype: &DType,
) -> VortexResult<Option<BoundExpression>> {
    let storage_root = BoundExpression::new_root(storage_dtype.clone());
    let mut needs_extension_value = false;
    let rebound = expr
        .clone()
        .transform_down(|node| {
            if node.is_root() {
                needs_extension_value = true;
                return Ok(Transformed::no(node));
            }

            if !matches!(node.children(), [child] if child.is_root()) {
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

    Ok((!needs_extension_value).then_some(rebound))
}
