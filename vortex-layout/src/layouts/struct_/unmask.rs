// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pruning predicates over the fields of a nullable struct, read past the struct's validity.

use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::Transformed;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

/// Reads the fields of `predicate` past the struct `validity` wherever the predicate cannot hold
/// in a row where the struct is null.
///
/// A struct reader reads a field of a nullable struct as `mask($.field, validity)`, which is null
/// wherever the struct is. A predicate that reaches the mask only through strict functions is null
/// there too, and AND and OR above it hold only where such an operand holds. Where the struct is
/// valid, the mask returns the field itself. So the predicate over the unmasked field holds in
/// every row where the predicate over the masked field holds, and pruning with it keeps every row
/// the predicate keeps. Reading the field alone, it reaches the field's reader and its zone map.
///
/// A predicate that can hold where the struct is null, such as `is_null($.field)`, keeps the mask.
pub(super) fn unmask_fields(
    predicate: BoundExpression,
    validity: &BoundExpression,
) -> VortexResult<BoundExpression> {
    Ok(unmask(predicate, validity, true)?.into_inner())
}

/// `connective` holds while every function from the predicate root down to `expr` is AND or OR.
fn unmask(
    expr: BoundExpression,
    validity: &BoundExpression,
    connective: bool,
) -> VortexResult<Transformed<BoundExpression>> {
    if expr.is::<Mask>() && expr.child(1) == validity {
        return Ok(Transformed::yes(expr.child(0).clone()));
    }
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(Transformed::no(expr));
    };
    let connective = connective
        && expr
            .as_opt::<Binary>()
            .is_some_and(|operator| matches!(operator, Operator::And | Operator::Or));
    if !connective && !scalar_fn.signature().is_strict() {
        return Ok(Transformed::no(expr));
    }

    let mut changed = false;
    let children = expr
        .children()
        .iter()
        .map(|child| {
            let child = unmask(child.clone(), validity, connective)?;
            changed |= child.changed;
            Ok(child.into_inner())
        })
        .collect::<VortexResult<Vec<_>>>()?;
    if !changed {
        return Ok(Transformed::no(expr));
    }
    Ok(Transformed::yes(expr.with_children(children)?))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::mask;
    use vortex_array::expr::not;
    use vortex_array::expr::or;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;

    use super::unmask_fields;

    /// `$.name` as a struct reader reads it from a nullable struct.
    fn field(name: &str) -> Expression {
        mask(get_item(name, root()), is_not_null(root()))
    }

    /// Unmasks `predicate` over a nullable `{a: i32, b: i32}`, returning both in display form.
    fn unmasked(predicate: Expression) -> VortexResult<String> {
        let dtype = DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ]),
            Nullability::Nullable,
        );
        let validity = is_not_null(root()).bind(&dtype)?;
        Ok(unmask_fields(predicate.bind(&dtype)?, &validity)?.to_string())
    }

    #[rstest]
    #[case::comparison(gt(field("a"), lit(5i32)), "($.a > 5i32)")]
    #[case::not(not(gt(field("a"), lit(5i32))), "vortex.not(($.a > 5i32))")]
    #[case::connectives(
        or(
            gt(field("a"), lit(5i32)),
            and(lt(field("b"), lit(1i32)), is_not_null(root()))
        ),
        "(($.a > 5i32) or (($.b < 1i32) and is_not_null($)))"
    )]
    #[case::is_null(
        is_null(field("a")),
        "vortex.is_null(vortex.mask($.a, is_not_null($)))"
    )]
    #[case::not_and(
        not(and(gt(field("a"), lit(5i32)), lt(field("b"), lit(1i32)))),
        "vortex.not(((vortex.mask($.a, is_not_null($)) > 5i32) and \
         (vortex.mask($.b, is_not_null($)) < 1i32)))"
    )]
    fn unmasks_only_where_struct_nulls_cannot_hold(
        #[case] predicate: Expression,
        #[case] expected: &str,
    ) -> VortexResult<()> {
        assert_eq!(unmasked(predicate)?, expected);
        Ok(())
    }
}
