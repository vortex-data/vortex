// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Moves strict functions that cannot fail inside the validity masks of a nullable struct's fields.
//!
//! The partition rules of a nullable [`Pack`](super::Pack) read a field as `mask(field, validity)`,
//! so a function of the field stays above the mask and the field's plan never sees it. For
//! `list_transform($.items, x -> x.sku)`, the list would then read every element column. Moving
//! the function inside the mask lets the next pass push it into the field's plan, where a list
//! reads only the `sku` elements.

use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound;
use vortex_array::expr::traversal::NodeRewriter;
use vortex_array::expr::traversal::Transformed;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_error::VortexResult;

/// Rewrites `f(mask(x, v))` as `mask(f(x), v)` for a function `f` of one argument that is strict
/// and cannot fail.
///
/// A strict function is null wherever its argument is, so the mask keeps the same rows null. A
/// function that cannot fail may run on the values that `v` hides. A function that can fail, such
/// as a lambda that can overflow, stays above the mask, so it never runs on a hidden value. A
/// function that is not strict, such as `is_null`, also stays above it, because its result is not
/// null where the struct is.
///
/// Field reads are strict and cannot fail, so this also moves a field read that a lifted function
/// exposes, as in `ext_storage(mask(p, v)).a`.
pub(super) struct ApplyStrictFunctionsInsideMasks;

impl NodeRewriter for ApplyStrictFunctionsInsideMasks {
    type NodeTy = BoundExpression;

    fn visit_up(&mut self, node: BoundExpression) -> VortexResult<Transformed<BoundExpression>> {
        let [input] = node.children() else {
            return Ok(Transformed::no(node));
        };
        if !input.is::<Mask>() || !is_strict_and_infallible(&node) {
            return Ok(Transformed::no(node));
        }

        let applied = apply_inside_masks(&node, input)?;
        // A strict function of a nullable input returns a nullable dtype, which the masks give the
        // result too. Should a function break that contract, the expressions above it would change
        // type, so it stays above the masks.
        if applied.dtype() != node.dtype() {
            return Ok(Transformed::no(node));
        }

        Ok(Transformed::yes(applied))
    }
}

/// Whether `node` is a function that is null wherever its argument is, and cannot fail.
fn is_strict_and_infallible(node: &BoundExpression) -> bool {
    node.as_scalar().is_some_and(|scalar_fn| {
        let signature = scalar_fn.signature();
        signature.is_strict() && signature.is_infallible()
    })
}

/// Applies the function of `node` to the input of the masks in `input`, beneath all of them.
fn apply_inside_masks(
    node: &BoundExpression,
    input: &BoundExpression,
) -> VortexResult<BoundExpression> {
    if !input.is::<Mask>() {
        return node.clone().with_children([input.clone()]);
    }

    let applied = apply_inside_masks(node, input.child(0))?;
    Ok(bound::mask(applied, input.child(1).clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::Lambda;
    use vortex_array::expr::checked_add;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::list_length;
    use vortex_array::expr::list_transform;
    use vortex_array::expr::lit;
    use vortex_array::expr::mask;
    use vortex_array::expr::not;
    use vortex_array::expr::root;
    use vortex_array::expr::traversal::NodeExt;
    use vortex_array::expr::var;
    use vortex_error::VortexResult;

    use super::ApplyStrictFunctionsInsideMasks;

    /// The list `l` of a nullable `{l: list(i64)}`, read through the struct's validity.
    fn masked_list() -> Expression {
        mask(get_item("l", root()), is_not_null(root()))
    }

    /// Rewrites `expr` over a nullable `{l: list(i64)}`, returning the result in display form.
    fn rewritten(expr: Expression) -> VortexResult<String> {
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "l",
                DType::List(
                    Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
                    Nullability::NonNullable,
                ),
            )]),
            Nullability::Nullable,
        );
        let expr = expr.bind(&dtype)?;
        Ok(expr
            .rewrite(&mut ApplyStrictFunctionsInsideMasks)?
            .into_inner()
            .to_string())
    }

    /// A strict function of one argument that cannot fail moves inside the masks, so the list's
    /// plan can push it into the elements. A function that can fail would run on the lists that
    /// the mask hides, and one that is not strict is not null where the struct is, so both stay
    /// above it.
    #[rstest]
    #[case::transform(
        list_transform(masked_list(), Lambda::new(["x"], not(eq(var("x"), lit(0i64))))?),
        "vortex.mask(vortex.list.transform($.l, (x) -> vortex.not(($.x = 0i64))), is_not_null($))"
    )]
    #[case::length(
        not(eq(list_length(masked_list()), lit(0u64))),
        "vortex.not((vortex.mask(vortex.list.length($.l), is_not_null($)) = 0u64))"
    )]
    #[case::nested_masks(
        list_length(mask(masked_list(), is_not_null(get_item("l", root())))),
        "vortex.mask(vortex.mask(vortex.list.length($.l), is_not_null($)), is_not_null($.l))"
    )]
    #[case::fallible(
        list_transform(masked_list(), Lambda::new(["x"], checked_add(var("x"), lit(1i64)))?),
        "vortex.list.transform(vortex.mask($.l, is_not_null($)), (x) -> ($.x + 1i64))"
    )]
    #[case::not_strict(
        is_null(masked_list()),
        "vortex.is_null(vortex.mask($.l, is_not_null($)))"
    )]
    fn applies_strict_functions_that_cannot_fail_inside_masks(
        #[case] expr: Expression,
        #[case] expected: &str,
    ) -> VortexResult<()> {
        assert_eq!(rewritten(expr)?, expected);
        Ok(())
    }
}
