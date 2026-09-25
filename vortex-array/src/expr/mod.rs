// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex's expression language: scalar operations over [arrays](crate::ArrayRef).
//!
//! An [`Expression`] is a tree of scalar operations rooted at a scope (see [`root`]). Expressions
//! are the common currency of scans: a scan takes a *filter* expression that resolves to a boolean
//! and a *projection* expression that shapes the output. All expressions are serializable and own
//! their own wire format, so they can be pushed down to remote sources and reconstructed on workers.
//!
//! # Scalar functions
//!
//! Each node references a scalar function defined by a
//! [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable). The vtable declares the function signature,
//! properties such as strictness, and the logic that executes it over input arrays. Built-in
//! functions live in [`crate::scalar_fn`]; integration and plugin crates supply additional,
//! use-case-specific functions.
//!
//! # Deferred execution
//!
//! Applying an expression to an array does not compute the result eagerly. Instead it builds a
//! [`ScalarFnArray`](crate::arrays::ScalarFnArray) representing the deferred application, letting
//! downstream encodings push the computation into compressed data, or fuse several expressions
//! together, before any data is materialized. The deferred tree is executed toward canonical form
//! only when a result is actually required.
//!
//! # Type checking
//!
//! Expressions are strictly typed: an input array's dtype must match the function signature exactly,
//! so callers perform any required casts themselves before building the expression. The one
//! relaxation is nullability—for example, equality may compare a `u32` against a `u32?`, but never
//! a `u32` against an `i32`.
//!
//! Scans decompose bound filter expressions into independent conjuncts so that they can evaluate
//! and reorder the most selective predicates first.
//!
//! The implementation takes inspiration from [Postgres] and [Apache Datafusion].
//!
//! [Postgres]: https://www.postgresql.org/docs/current/sql-expressions.html
//! [Apache Datafusion]: https://github.com/apache/datafusion/tree/5fac581efbaffd0e6a9edf931182517524526afd/datafusion/expr

pub mod aliases;
pub mod analysis;
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod bound_expression;
pub mod display;
pub(crate) mod expression;
mod exprs;
pub(crate) mod field;
pub mod forms;
mod optimize;
pub mod proto;
pub mod scope;
pub mod stats;
pub mod transform;
pub mod traversal;

pub use analysis::*;
pub use bound_expression::*;
pub use expression::*;
pub use exprs::and;
pub use exprs::and_collect;
pub use exprs::between;
pub use exprs::binary;
pub use exprs::bound;
pub use exprs::byte_length;
pub use exprs::case_when;
pub use exprs::case_when_no_else;
pub use exprs::cast;
pub use exprs::checked_add;
pub use exprs::col;
pub use exprs::dynamic;
pub use exprs::dynamic_with_options;
pub use exprs::eq;
pub use exprs::ext_storage;
pub use exprs::fill_null;
pub use exprs::get_item;
pub use exprs::gt;
pub use exprs::gt_eq;
pub use exprs::ilike;
pub use exprs::in_list;
pub use exprs::is_nan;
pub use exprs::is_not_null;
pub use exprs::is_null;
pub use exprs::is_root;
pub use exprs::like;
pub use exprs::list_contains;
pub use exprs::list_contains_opts;
pub use exprs::list_length;
pub use exprs::list_sum;
pub use exprs::list_sum_opts;
pub use exprs::lit;
pub use exprs::lt;
pub use exprs::lt_eq;
pub use exprs::mask;
pub use exprs::merge;
pub use exprs::merge_opts;
pub use exprs::nested_case_when;
pub use exprs::not;
pub use exprs::not_eq;
pub use exprs::not_ilike;
pub use exprs::not_like;
pub use exprs::or;
pub use exprs::or_collect;
pub use exprs::pack;
pub use exprs::root;
pub use exprs::select;
pub use exprs::select_exclude;
pub use exprs::union_child_validities;
pub use exprs::variant_get;
pub use exprs::zip_expr;
pub use scope::*;

#[cfg(feature = "_test-harness")]
pub mod test_harness {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;

    pub fn struct_dtype() -> DType {
        DType::Struct(
            StructFields::new(
                ["a", "col1", "col2", "bool1", "bool2"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::U16, Nullability::Nullable),
                    DType::Primitive(PType::U16, Nullability::Nullable),
                    DType::Bool(Nullability::NonNullable),
                    DType::Bool(Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::and;
    use crate::expr::bound;
    use crate::expr::case_when;
    use crate::expr::col;
    use crate::expr::gt;
    use crate::expr::gt_eq;
    use crate::expr::lt;
    use crate::expr::lt_eq;
    use crate::expr::not;
    use crate::expr::not_eq;
    use crate::expr::or;
    use crate::expr::select;
    use crate::expr::select_exclude;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::literal::Literal;

    #[test]
    fn bound_constructors_preserve_order_and_types() -> vortex_error::VortexResult<()> {
        let value_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scope = DType::Struct(
            StructFields::from_iter([("value", value_dtype.clone())]),
            Nullability::NonNullable,
        );

        let root = bound::root(scope.clone());
        let value = bound::get_item("value", root);
        let literal = bound::lit(5i32);
        let condition = bound::gt(value.clone(), literal.clone());
        assert_eq!(condition.dtype(), &DType::Bool(Nullability::NonNullable));
        assert_eq!(condition.children(), &[value.clone(), literal.clone()]);

        let case = bound::case_when(condition.clone(), value.clone(), literal.clone());
        assert_eq!(case.dtype(), &value_dtype);
        assert_eq!(case.children(), &[condition.clone(), value, literal]);

        let packed = bound::pack(
            [("condition", condition.clone()), ("value", case.clone())],
            Nullability::NonNullable,
        );
        assert_eq!(packed.children(), &[condition, case.clone()]);
        assert_eq!(
            packed.dtype(),
            &DType::Struct(
                StructFields::from_iter([
                    ("condition", DType::Bool(Nullability::NonNullable)),
                    ("value", value_dtype),
                ]),
                Nullability::NonNullable,
            )
        );

        let unbound = case_when(gt(col("value"), lit(5i32)), col("value"), lit(5i32));
        assert_eq!(unbound.bind(&scope)?, case);
        Ok(())
    }

    #[test]
    fn expr_display() {
        assert_eq!(col("a").to_string(), "$.a");
        assert_eq!(root().to_string(), "$");

        let col1: Expression = col("col1");
        let col2: Expression = col("col2");
        assert_eq!(
            and(col1.clone(), col2.clone()).to_string(),
            "($.col1 and $.col2)"
        );
        assert_eq!(
            or(col1.clone(), col2.clone()).to_string(),
            "($.col1 or $.col2)"
        );
        assert_eq!(
            eq(col1.clone(), col2.clone()).to_string(),
            "($.col1 = $.col2)"
        );
        assert_eq!(
            not_eq(col1.clone(), col2.clone()).to_string(),
            "($.col1 != $.col2)"
        );
        assert_eq!(
            gt(col1.clone(), col2.clone()).to_string(),
            "($.col1 > $.col2)"
        );
        assert_eq!(
            gt_eq(col1.clone(), col2.clone()).to_string(),
            "($.col1 >= $.col2)"
        );
        assert_eq!(
            lt(col1.clone(), col2.clone()).to_string(),
            "($.col1 < $.col2)"
        );
        assert_eq!(
            lt_eq(col1.clone(), col2.clone()).to_string(),
            "($.col1 <= $.col2)"
        );

        assert_eq!(
            or(lt(col1.clone(), col2.clone()), not_eq(col1.clone(), col2),).to_string(),
            "(($.col1 < $.col2) or ($.col1 != $.col2))"
        );

        assert_eq!(not(col1).to_string(), "vortex.not($.col1)");

        assert_eq!(
            select(vec![FieldName::from("col1")], root()).to_string(),
            "${col1}"
        );
        assert_eq!(
            select(
                vec![FieldName::from("col1"), FieldName::from("col2")],
                root()
            )
            .to_string(),
            "${col1, col2}"
        );
        assert_eq!(
            select_exclude(
                vec![FieldName::from("col1"), FieldName::from("col2")],
                root()
            )
            .to_string(),
            "${~ col1, col2}"
        );

        assert_eq!(lit(Scalar::from(0u8)).to_string(), "0u8");
        assert_eq!(lit(Scalar::from(0.0f32)).to_string(), "0f32");
        assert_eq!(
            lit(Scalar::from(i64::MAX)).to_string(),
            "9223372036854775807i64"
        );
        assert_eq!(lit(Scalar::from(true)).to_string(), "true");
        assert_eq!(
            lit(Scalar::null(DType::Bool(Nullability::Nullable))).to_string(),
            "null"
        );

        assert_eq!(
            lit(Scalar::struct_(
                DType::Struct(
                    StructFields::new(
                        FieldNames::from(["dog", "cat"]),
                        vec![
                            DType::Primitive(PType::U32, Nullability::NonNullable),
                            DType::Utf8(Nullability::NonNullable)
                        ],
                    ),
                    Nullability::NonNullable
                ),
                vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
            ))
            .to_string(),
            "{dog: 32u32, cat: \"rufus\"}"
        );
    }

    #[test]
    fn expr_contains() {
        let expression = &eq(root(), lit(3u64));
        assert!(expression.contains::<Literal>().unwrap());
        let expression = root();
        assert!(!expression.contains::<Literal>().unwrap());
    }
}
