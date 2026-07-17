// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::not;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_error::VortexResult;

/// The minimal set of list children an expression needs for evaluation.
///
/// For example:
///     - `is_null(root())` only needs the validity child.
///     - `list_length(root())` only needs the offsets and validity children.
///     - `root()` needs elements, offsets, and validity children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ListChildrenNeeded {
    /// Only the validity child is needed (`is_null` / `is_not_null`).
    Validity,
    /// Only the offsets and validity children are needed (`list_length`).
    OffsetsAndValidity,
    /// All children are needed.
    All,
}

/// The minimal set of list children needed to evaluate `expr`, where `root()` is a field with list dtype.
pub(super) fn get_necessary_list_children(expr: &Expression) -> ListChildrenNeeded {
    if is_null_root(expr) {
        return ListChildrenNeeded::Validity;
    }

    if is_list_length_root(expr) {
        return ListChildrenNeeded::OffsetsAndValidity;
    }

    if is_root(expr) {
        return ListChildrenNeeded::All;
    }

    // Otherwise the requirement is the max over the operands. Childless expressions that never
    // touch the list, such as literals, fall back to the cheapest usable child.
    expr.children()
        .iter()
        .map(get_necessary_list_children)
        .max()
        .unwrap_or(ListChildrenNeeded::Validity)
}

fn is_null_root(expr: &Expression) -> bool {
    (expr.is::<IsNull>() || expr.is::<IsNotNull>())
        && expr.children().len() == 1
        && is_root(expr.child(0))
}

fn is_list_length_root(expr: &Expression) -> bool {
    expr.is::<ListLength>() && expr.children().len() == 1 && is_root(expr.child(0))
}

/// Rewrite a validity-class expression so it can be evaluated against the list's validity bool
/// array (`true` == valid row): `is_not_null(root())` becomes `root()` and `is_null(root())`
/// becomes `not(root())`. All other nodes are rebuilt with rewritten children.
pub(super) fn rewrite_validity_expr(expr: &Expression) -> VortexResult<Expression> {
    if expr.is::<IsNotNull>() && expr.children().len() == 1 && is_root(expr.child(0)) {
        return Ok(root());
    }
    if expr.is::<IsNull>() && expr.children().len() == 1 && is_root(expr.child(0)) {
        return Ok(not(root()));
    }
    let children = expr
        .children()
        .iter()
        .map(rewrite_validity_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

/// Rewrite an offsets-class expression so it can be evaluated against an array of list lengths.
/// `list_length(root())` becomes `root()`. Other references to `root()` are left intact: for
/// offsets-class expressions they can only be validity checks, and the lengths array carries the
/// same validity as the original list.
pub(super) fn rewrite_offsets_expr(expr: &Expression) -> VortexResult<Expression> {
    if is_list_length_root(expr) {
        return Ok(root());
    }

    let children = expr
        .children()
        .iter()
        .map(rewrite_offsets_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::cast;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::list_length;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::root;

    use super::*;

    /// `get_necessary_list_children` keys off the deepest list child an expression touches; `All`
    /// is the always-correct default for anything not specifically recognized.
    #[rstest]
    // `is_null` / `is_not_null` of the list itself need only validity.
    #[case::is_null(is_null(root()), ListChildrenNeeded::Validity)]
    #[case::is_not_null(is_not_null(root()), ListChildrenNeeded::Validity)]
    // Compound over validity-only operands stays validity.
    #[case::not_is_null(not(is_null(root())), ListChildrenNeeded::Validity)]
    // A list-independent (constant) expression falls to the cheapest usable child.
    #[case::constant(lit(5), ListChildrenNeeded::Validity)]
    // `list_length(root())` needs offsets and validity, but not elements.
    #[case::list_length(list_length(root()), ListChildrenNeeded::OffsetsAndValidity)]
    // Compound over offsets-only operands stays offsets.
    #[case::list_length_filter(
        gt(list_length(root()), lit(1u64)),
        ListChildrenNeeded::OffsetsAndValidity
    )]
    #[case::cast_list_length(
        cast(
            list_length(root()),
            DType::Primitive(PType::I64, Nullability::Nullable),
        ),
        ListChildrenNeeded::OffsetsAndValidity
    )]
    // A bare list reference needs the elements.
    #[case::bare_root(root(), ListChildrenNeeded::All)]
    // Any other fn over the list needs the elements.
    #[case::not_root(not(root()), ListChildrenNeeded::All)]
    // `is_null` only short-circuits to validity when its argument is the list itself.
    #[case::is_null_of_derived(is_null(not(root())), ListChildrenNeeded::All)]
    // Max over operands: validity + elements => elements.
    #[case::validity_and_elements(eq(is_null(root()), root()), ListChildrenNeeded::All)]
    fn classify_expr_class(#[case] expr: Expression, #[case] expected: ListChildrenNeeded) {
        assert_eq!(get_necessary_list_children(&expr), expected);
    }
}
