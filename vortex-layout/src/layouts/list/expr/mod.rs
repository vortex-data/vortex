// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Which list children an expression reads, and the rewrites that evaluate it over them.

#[cfg(test)]
mod tests;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::not;
use vortex_array::expr::bound::pack;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_scan::plan::ListTransforms;

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

/// The minimal set of list children needed to evaluate a bound expression.
pub(super) fn get_necessary_bound_list_children(expr: &BoundExpression) -> ListChildrenNeeded {
    if is_bound_null_root(expr) {
        return ListChildrenNeeded::Validity;
    }

    if is_bound_list_length_root(expr) {
        return ListChildrenNeeded::OffsetsAndValidity;
    }

    if expr.is_root() {
        return ListChildrenNeeded::All;
    }

    expr.children()
        .iter()
        .map(get_necessary_bound_list_children)
        .max()
        .unwrap_or(ListChildrenNeeded::Validity)
}

fn is_bound_null_root(expr: &BoundExpression) -> bool {
    (expr.as_scalar().is_some_and(|f| f.is::<IsNull>())
        || expr.as_scalar().is_some_and(|f| f.is::<IsNotNull>()))
        && expr.children().len() == 1
        && expr.children()[0].is_root()
}

fn is_bound_list_length_root(expr: &BoundExpression) -> bool {
    expr.as_scalar().is_some_and(|f| f.is::<ListLength>())
        && expr.children().len() == 1
        && expr.children()[0].is_root()
}

/// Rewrite a validity-class expression so it can be evaluated against the list's validity bool
/// array (`true` == valid row): `is_not_null(root())` becomes `root()` and `is_null(root())`
/// becomes `not(root())`. All other nodes are rebuilt with rewritten children.
pub(super) fn rewrite_validity_expr(expr: &BoundExpression) -> VortexResult<BoundExpression> {
    let validity_dtype = DType::Bool(Nullability::NonNullable);
    rewrite_validity_expr_with_root(expr, &validity_dtype)
}

fn rewrite_validity_expr_with_root(
    expr: &BoundExpression,
    root_dtype: &DType,
) -> VortexResult<BoundExpression> {
    if expr.as_scalar().is_some_and(|f| f.is::<IsNotNull>())
        && expr.children().len() == 1
        && expr.children()[0].is_root()
    {
        return Ok(BoundExpression::new_root(root_dtype.clone()));
    }
    if expr.as_scalar().is_some_and(|f| f.is::<IsNull>())
        && expr.children().len() == 1
        && expr.children()[0].is_root()
    {
        return Ok(not(BoundExpression::new_root(root_dtype.clone())));
    }
    if expr.is_root() {
        return Ok(BoundExpression::new_root(root_dtype.clone()));
    }

    let children = expr
        .children()
        .iter()
        .map(|child| rewrite_validity_expr_with_root(child, root_dtype))
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

/// Rewrite an offsets-class expression so it can be evaluated against an array of list lengths.
/// `list_length(root())` becomes `root()`. Other references to `root()` are left intact: for
/// offsets-class expressions they can only be validity checks, and the lengths array carries the
/// same validity as the original list.
pub(super) fn rewrite_offsets_expr(
    expr: &BoundExpression,
    lengths_dtype: &DType,
) -> VortexResult<BoundExpression> {
    if is_bound_list_length_root(expr) || expr.is_root() {
        return Ok(BoundExpression::new_root(lengths_dtype.clone()));
    }

    let children = expr
        .children()
        .iter()
        .map(|child| rewrite_offsets_expr(child, lengths_dtype))
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

/// An expression over a list that reads the list elements only through the lambdas of
/// `list_transform(root, lambda)` nodes, rewritten so that those lambdas run over the elements
/// child.
///
/// The reader evaluates [`Self::elements`] over the elements child, rebuilds lists over the
/// projected elements with [`Self::rebuild`], and evaluates [`Self::lists`] over them. A lambda
/// that reads one field of struct elements, as in `list_transform(root, x -> x.sku)`, reaches the
/// elements layout as `$.sku`, so only the column of that field is read.
///
/// The identity projection reads the elements as stored: its elements expression is the elements
/// root and its list expression is the original expression.
#[derive(Debug)]
pub(super) struct ElementProjection {
    elements: BoundExpression,
    /// The names of the struct of lists that [`Self::rebuild`] builds, one list per lambda, or
    /// `None` when it builds a single list.
    lists_fields: Option<FieldNames>,
    lists: BoundExpression,
    fallible: bool,
}

impl ElementProjection {
    /// Reads the elements as stored and evaluates `expr` over the rebuilt list.
    pub(super) fn identity(expr: &BoundExpression, elements_dtype: &DType) -> Self {
        Self {
            elements: BoundExpression::new_root(elements_dtype.clone()),
            lists_fields: None,
            lists: expr.clone(),
            fallible: false,
        }
    }

    /// Pushes the lambdas of the `list_transform(root, lambda)` nodes of `expr` into the elements.
    ///
    /// Each lambda body becomes an expression over the elements, with its parameter rebound to
    /// the elements root. The bodies of several distinct lambdas are packed into a struct, which
    /// rebuilds into a struct of lists that share the offsets and validity.
    ///
    /// Returns `None` when `expr` has no such node, or reads the list root in another way than
    /// through these nodes, `is_null(root)`, `is_not_null(root)` and `list_length(root)`. The last
    /// three need only the offsets and validity, which every rebuilt list keeps.
    pub(super) fn push_down(expr: &BoundExpression) -> VortexResult<Option<Self>> {
        let Some(transforms) = ListTransforms::of(expr)? else {
            return Ok(None);
        };

        let (elements, lists_fields, lists_root) =
            if let ([transform], [body]) = (transforms.transforms(), transforms.bodies()) {
                let lists_root = BoundExpression::new_root(transform.dtype().clone());
                (body.clone(), None, lists_root)
            } else {
                let names: FieldNames = (0..transforms.bodies().len())
                    .map(|idx| idx.to_string())
                    .collect();
                let elements = pack(
                    names
                        .iter()
                        .cloned()
                        .zip(transforms.bodies().iter().cloned()),
                    Nullability::NonNullable,
                );
                let lists_dtype = DType::Struct(
                    StructFields::new(
                        names.clone(),
                        transforms
                            .transforms()
                            .iter()
                            .map(|transform| transform.dtype().clone())
                            .collect(),
                    ),
                    Nullability::NonNullable,
                );
                (
                    elements,
                    Some(names),
                    BoundExpression::new_root(lists_dtype),
                )
            };

        let list_at = |idx: usize| match &lists_fields {
            Some(names) => get_item(names[idx].clone(), lists_root.clone()),
            None => lists_root.clone(),
        };
        let lists = transforms.rewrite(expr, |idx| Ok(list_at(idx)), &list_at(0))?;

        Ok(Some(Self {
            elements,
            lists_fields,
            lists,
            fallible: transforms.is_fallible(),
        }))
    }

    /// The expression over the elements child.
    pub(super) fn elements(&self) -> &BoundExpression {
        &self.elements
    }

    /// The expression over what [`Self::rebuild`] builds. It has the dtype of the original
    /// expression.
    pub(super) fn lists(&self) -> &BoundExpression {
        &self.lists
    }

    /// Whether evaluating [`Self::elements`] can fail.
    ///
    /// `list_transform` runs such a lambda only on the elements of valid lists, so the reader must
    /// not evaluate it over the elements of null or unselected lists either.
    pub(super) fn is_fallible(&self) -> bool {
        self.fallible
    }

    /// Rebuilds what [`Self::lists`] reads from the projected `elements` and the `offsets` and
    /// `validity` of the lists.
    ///
    /// # Safety
    ///
    /// `elements` must hold the projection of every element that `offsets` references, at the
    /// position of that element, and `offsets` and `validity` must describe valid lists over them.
    pub(super) unsafe fn rebuild(
        &self,
        elements: ArrayRef,
        offsets: ArrayRef,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        let Some(names) = &self.lists_fields else {
            // SAFETY: the caller guarantees the list invariants.
            return Ok(
                unsafe { ListArray::new_unchecked(elements, offsets, validity) }.into_array(),
            );
        };
        let len = offsets.len().saturating_sub(1);
        let lists = names
            .iter()
            .map(|name| {
                let field = elements.get_item(name.clone())?;
                // SAFETY: every field of the packed projection has one value per element, so the
                // caller's guarantee holds for each field.
                Ok(
                    unsafe { ListArray::new_unchecked(field, offsets.clone(), validity.clone()) }
                        .into_array(),
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_new(names.clone(), lists, len, Validity::NonNullable)?.into_array())
    }
}
