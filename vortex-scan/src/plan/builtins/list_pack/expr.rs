// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What an expression over a list reads of the list.

use vortex_array::expr::BoundExpression;
use vortex_array::expr::BoundLambda;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::expr::traversal::pre_order_visit_down;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_array::scalar_fn::fns::list_transform::ListTransform;
use vortex_array::scalar_fn::fns::list_transform::ListTransformOptions;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// What an expression over a list reads of the list.
pub(crate) enum ListRead {
    /// At most the validity, through `is_null($)` and `is_not_null($)`.
    Validity,
    /// At most the offsets and the validity, through these and `list_length($)`.
    Lengths,
    /// The elements, through the lambdas of `list_transform($, lambda)` nodes alone.
    Transforms(ListTransforms),
    /// The elements in any other way, such as the list itself.
    Elements,
}

impl ListRead {
    /// What `expr`, an expression over a list, reads of the list.
    pub(crate) fn of(expr: &BoundExpression) -> VortexResult<Self> {
        let mut transforms: Vec<BoundExpression> = Vec::new();
        let mut reads_lengths = false;
        let mut reads_elements = false;
        pre_order_visit_down(expr, |node: &BoundExpression| {
            if is_root_transform(node) {
                if !transforms.contains(node) {
                    transforms.push(node.clone());
                }
                return Ok(TraversalOrder::Skip);
            }
            if is_validity_read(node) {
                return Ok(TraversalOrder::Skip);
            }
            if is_root_read::<ListLength>(node) {
                reads_lengths = true;
                return Ok(TraversalOrder::Skip);
            }
            if node.is_root() {
                reads_elements = true;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        if reads_elements {
            return Ok(Self::Elements);
        }
        if transforms.is_empty() {
            return Ok(if reads_lengths {
                Self::Lengths
            } else {
                Self::Validity
            });
        }
        Ok(ListTransforms::try_new(transforms)?.map_or(Self::Elements, Self::Transforms))
    }
}

/// The `list_transform($, lambda)` nodes through which an expression over a list reads the list
/// elements, with the body of each lambda as an expression over the elements.
///
/// A lambda that reads one field of struct elements, as in `list_transform($, x -> x.sku)`, has
/// the body `$.sku`, so a reader can evaluate it over the elements and read the column of that
/// field alone.
#[derive(Debug)]
pub struct ListTransforms {
    transforms: Vec<BoundExpression>,
    bodies: Vec<BoundExpression>,
    fallible: bool,
}

impl ListTransforms {
    /// The `list_transform($, lambda)` nodes of `expr`, an expression over a list.
    ///
    /// Returns `None` when `expr` has no such node, or reads the list in another way than through
    /// these nodes, `is_null($)`, `is_not_null($)` and `list_length($)`. The last three need only
    /// the offsets and the validity of the list.
    pub fn of(expr: &BoundExpression) -> VortexResult<Option<Self>> {
        Ok(match ListRead::of(expr)? {
            ListRead::Transforms(transforms) => Some(transforms),
            ListRead::Validity | ListRead::Lengths | ListRead::Elements => None,
        })
    }

    fn try_new(transforms: Vec<BoundExpression>) -> VortexResult<Option<Self>> {
        let mut bodies = Vec::with_capacity(transforms.len());
        let mut fallible = false;
        for transform in &transforms {
            let ListTransformOptions::Bound(lambda) = transform.as_::<ListTransform>() else {
                return Ok(None);
            };
            let Some(body) = body_over_elements(lambda)? else {
                return Ok(None);
            };
            fallible |= !lambda.is_infallible();
            bodies.push(body);
        }
        Ok(Some(Self {
            transforms,
            bodies,
            fallible,
        }))
    }

    /// The distinct `list_transform($, lambda)` nodes, in the order they first occur.
    pub fn transforms(&self) -> &[BoundExpression] {
        &self.transforms
    }

    /// The body of the lambda of each node, as an expression over the list elements.
    pub fn bodies(&self) -> &[BoundExpression] {
        &self.bodies
    }

    /// Whether a lambda can fail.
    ///
    /// `list_transform` runs such a lambda only on the elements of valid lists, so a reader must
    /// not evaluate its body over the elements of null or unselected lists either.
    pub fn is_fallible(&self) -> bool {
        self.fallible
    }

    /// Rewrites `expr`, from which these nodes were taken, to read the results of the lambda
    /// bodies instead of the elements.
    ///
    /// Node `i` becomes `transformed(i)`, and the list in `is_null($)`, `is_not_null($)` and
    /// `list_length($)` becomes `list`, which must have the offsets and the validity of the list.
    pub fn rewrite(
        &self,
        expr: &BoundExpression,
        transformed: impl Fn(usize) -> VortexResult<BoundExpression>,
        list: &BoundExpression,
    ) -> VortexResult<BoundExpression> {
        Ok(expr
            .clone()
            .transform_down(|node| {
                let value = if is_root_transform(&node) {
                    let index = self
                        .transforms
                        .iter()
                        .position(|transform| *transform == node)
                        .ok_or_else(|| vortex_err!("{node} is not a transform of the list"))?;
                    transformed(index)?
                } else if is_validity_read(&node) || is_root_read::<ListLength>(&node) {
                    node.with_children([list.clone()])?
                } else {
                    return Ok(Transformed::no(node));
                };
                Ok(Transformed {
                    value,
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            })?
            .into_inner())
    }
}

/// Whether `expr` is `is_null($)` or `is_not_null($)`.
pub(crate) fn is_validity_read(expr: &BoundExpression) -> bool {
    is_root_read::<IsNull>(expr) || is_root_read::<IsNotNull>(expr)
}

/// Whether `expr` applies the function `F` to the root alone.
fn is_root_read<F: ScalarFnVTable>(expr: &BoundExpression) -> bool {
    expr.is::<F>() && matches!(expr.children(), [child] if child.is_root())
}

fn is_root_transform(expr: &BoundExpression) -> bool {
    is_root_read::<ListTransform>(expr)
}

/// The body of a `list_transform` lambda as an expression over the list elements.
///
/// Binding made the lambda's parameter `x` the field `$.x` of the struct of its inputs, so every
/// `$.x` becomes the elements root. Returns `None` when the body reads the inputs in any other
/// way, which a lambda cannot do.
fn body_over_elements(lambda: &BoundLambda) -> VortexResult<Option<BoundExpression>> {
    let params = lambda.params();
    let ([param], Some(element_dtype)) = (params.names().as_ref(), params.field_by_index(0)) else {
        return Ok(None);
    };
    let elements_root = BoundExpression::new_root(element_dtype);
    let mut reads_inputs = false;
    let body = lambda
        .body()
        .clone()
        .transform_down(|node| {
            if node.as_opt::<GetItem>() == Some(param) && node.child(0).is_root() {
                return Ok(Transformed {
                    value: elements_root.clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }
            reads_inputs |= node.is_root();
            Ok(Transformed::no(node))
        })?
        .into_inner();
    Ok((!reads_inputs).then_some(body))
}
