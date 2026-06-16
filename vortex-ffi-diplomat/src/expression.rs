// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for the Vortex expression DSL.
//!
//! The hand-written C ABI exposed `vx_expression` (a `box_wrapper!` over the core
//! [`Expression`]) plus a family of free constructor functions: `vx_expression_root`,
//! `vx_expression_literal`, `vx_expression_select`, `vx_expression_and`, `vx_expression_or`,
//! `vx_expression_binary`, `vx_expression_not`, `vx_expression_is_null`,
//! `vx_expression_get_item`, and `vx_expression_list_contains`, alongside the
//! `vx_binary_operator` C enum.
//!
//! Under Diplomat the opaque type is `VxExpr`; every constructor becomes a method that returns
//! an owned `Box<VxExpr>`. The destructor is generated automatically (no `vx_expression_free`),
//! and instead of returning NULL on a null input, the methods take `&VxExpr` references that
//! Diplomat guarantees to be valid.
//!
//! ## Operator handling
//!
//! The C `vx_binary_operator` enum is preserved as the Diplomat enum [`VxBinaryOperator`] so
//! callers that want a single generic `binary(op, lhs, rhs)` entry point keep it. In addition,
//! the common operators are exposed as their own named/operator-attributed constructors
//! (`eq`, `not_eq`, `gt`, `gte`, `lt`, `lte`, `and`, `or`, `add`, `sub`, `mul`, `div`) so each
//! target language gets ergonomic, idiomatic builders. Diplomat's `comparison`/`add`/`sub`
//! operator attributes are applied where a host-language operator overload reads naturally.

pub use ffi::VxBinaryOperator;

#[diplomat::bridge]
pub mod ffi {
    use std::sync::Arc;

    use diplomat_runtime::DiplomatStr;
    use vortex::dtype::FieldName;
    use vortex::expr::Expression;
    use vortex::expr::and_collect;
    use vortex::expr::get_item;
    use vortex::expr::is_null;
    use vortex::expr::list_contains;
    use vortex::expr::lit;
    use vortex::expr::not;
    use vortex::expr::or_collect;
    use vortex::expr::root;
    use vortex::expr::select;
    use vortex::scalar_fn::ScalarFnVTableExt;
    use vortex::scalar_fn::fns::binary::Binary;
    use vortex::scalar_fn::fns::operators::Operator;

    use crate::error::ffi::VortexFfiError;
    use crate::scalar::ffi::VxScalar;

    /// A node in a Vortex expression tree.
    ///
    /// Expressions represent scalar computations performed on data. Each node carries an
    /// encoding (vtable), heap-allocated metadata, and child expressions. Expressions are
    /// reference-counted internally, so cloning a node is cheap.
    ///
    /// Replaces the C `vx_expression` opaque. Every returned expression is owned by the caller;
    /// Diplomat generates the destructor automatically (the C ABI required `vx_expression_free`).
    #[diplomat::opaque]
    pub struct VxExpr(pub(crate) Expression);

    /// Equalities, inequalities, and boolean/arithmetic operations over possibly-null values.
    ///
    /// For most operations, if either side is null the result is null. `KleeneAnd` and
    /// `KleeneOr` obey Kleene (three-valued) logic. Mirrors the C `vx_binary_operator` enum.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VxBinaryOperator {
        /// Expressions are equal.
        Eq,
        /// Expressions are not equal.
        NotEq,
        /// Left is greater than right.
        Gt,
        /// Left is greater than or equal to right.
        Gte,
        /// Left is less than right.
        Lt,
        /// Left is less than or equal to right.
        Lte,
        /// Kleene (three-valued) boolean AND.
        KleeneAnd,
        /// Kleene (three-valued) boolean OR.
        KleeneOr,
        /// Sum of the arguments. Errors at runtime on overflow/underflow.
        Add,
        /// Difference of the arguments. Errors at runtime on overflow/underflow.
        Sub,
        /// Product of the arguments.
        Mul,
        /// Left divided by right.
        Div,
    }

    impl VxExpr {
        /// Create a root expression.
        ///
        /// A root expression, applied to an array, takes the array itself (as opposed to
        /// `column`/`select`, which take the array's parts). Replaces `vx_expression_root`.
        #[diplomat::attr(auto, named_constructor = "root")]
        pub fn root() -> Box<VxExpr> {
            Box::new(VxExpr(root()))
        }

        /// Create a literal (constant) expression from a scalar.
        ///
        /// Useful for constants in expression trees, especially scan predicates. Replaces
        /// `vx_expression_literal`; the scalar out-parameter error is now a `Result`.
        #[diplomat::attr(auto, named_constructor = "literal")]
        pub fn literal(scalar: &VxScalar) -> Result<Box<VxExpr>, Box<VortexFfiError>> {
            Ok(Box::new(VxExpr(lit(scalar.0.clone()))))
        }

        /// Extract a named field from a struct expression.
        ///
        /// The child must produce a struct-typed value. Replaces `vx_expression_get_item`; the
        /// field name arrives as a UTF-8 string rather than a null-terminated `*const c_char`.
        #[diplomat::attr(auto, named_constructor = "column")]
        pub fn column(item: &DiplomatStr, child: &VxExpr) -> Result<Box<VxExpr>, Box<VortexFfiError>> {
            let item = std::str::from_utf8(item)
                .map_err(|e| VortexFfiError::new(format!("invalid utf-8 field name: {e}")))?;
            let item: FieldName = Arc::<str>::from(item).into();
            Ok(Box::new(VxExpr(get_item(item, child.0.clone()))))
        }

        /// Select (include) specific fields from a struct child expression.
        ///
        /// Produces a struct value with only the named fields. Replaces `vx_expression_select`;
        /// the `(names, len)` C pair becomes a slice of strings.
        #[diplomat::attr(auto, named_constructor = "select")]
        pub fn select(
            names: &[&DiplomatStr],
            child: &VxExpr,
        ) -> Result<Box<VxExpr>, Box<VortexFfiError>> {
            let names: Vec<FieldName> = names
                .iter()
                .map(|name| {
                    std::str::from_utf8(name)
                        .map(|s| FieldName::from(Arc::<str>::from(s)))
                        .map_err(|e| VortexFfiError::new(format!("invalid utf-8 field name: {e}")))
                })
                .collect::<Result<_, _>>()?;
            Ok(Box::new(VxExpr(select(names, child.0.clone()))))
        }

        /// Create a binary expression `lhs OP rhs` from an explicit operator.
        ///
        /// Replaces `vx_expression_binary`. Comparison/arithmetic operators also have their own
        /// dedicated constructors below; this method is the generic fallback.
        #[diplomat::attr(auto, named_constructor = "binary")]
        pub fn binary(operator: VxBinaryOperator, lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            let op: Operator = operator.into();
            Box::new(VxExpr(Binary.new_expr(op, [lhs.0.clone(), rhs.0.clone()])))
        }

        /// `lhs == rhs`.
        #[diplomat::attr(auto, comparison)]
        pub fn eq(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Eq, lhs, rhs)
        }

        /// `lhs != rhs`.
        #[diplomat::attr(auto, named_constructor = "not_eq")]
        pub fn not_eq(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::NotEq, lhs, rhs)
        }

        /// `lhs > rhs`.
        #[diplomat::attr(auto, named_constructor = "gt")]
        pub fn gt(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Gt, lhs, rhs)
        }

        /// `lhs >= rhs`.
        #[diplomat::attr(auto, named_constructor = "gte")]
        pub fn gte(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Gte, lhs, rhs)
        }

        /// `lhs < rhs`.
        #[diplomat::attr(auto, named_constructor = "lt")]
        pub fn lt(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Lt, lhs, rhs)
        }

        /// `lhs <= rhs`.
        #[diplomat::attr(auto, named_constructor = "lte")]
        pub fn lte(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Lte, lhs, rhs)
        }

        /// `lhs + rhs`. Errors at runtime on overflow.
        #[diplomat::attr(auto, add)]
        pub fn add(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Add, lhs, rhs)
        }

        /// `lhs - rhs`. Errors at runtime on underflow.
        #[diplomat::attr(auto, sub)]
        pub fn sub(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Sub, lhs, rhs)
        }

        /// `lhs * rhs`.
        #[diplomat::attr(auto, mul)]
        pub fn mul(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Mul, lhs, rhs)
        }

        /// `lhs / rhs`.
        #[diplomat::attr(auto, div)]
        pub fn div(lhs: &VxExpr, rhs: &VxExpr) -> Box<VxExpr> {
            Self::binary(VxBinaryOperator::Div, lhs, rhs)
        }

        /// Conjunction of the given child expressions.
        ///
        /// Replaces `vx_expression_and`. The C ABI returned NULL for an empty input; here an
        /// empty slice is an error since a Diplomat method cannot return a null `Box`.
        #[diplomat::attr(auto, named_constructor = "and")]
        pub fn and(expressions: &[Box<VxExpr>]) -> Result<Box<VxExpr>, Box<VortexFfiError>> {
            and_collect(expressions.iter().map(|e| e.0.clone()))
                .map(|e| Box::new(VxExpr(e)))
                .ok_or_else(|| VortexFfiError::new("and() requires at least one expression"))
        }

        /// Disjunction of the given child expressions.
        ///
        /// Replaces `vx_expression_or`. As with `and`, an empty input is an error.
        #[diplomat::attr(auto, named_constructor = "or")]
        pub fn or(expressions: &[Box<VxExpr>]) -> Result<Box<VxExpr>, Box<VortexFfiError>> {
            or_collect(expressions.iter().map(|e| e.0.clone()))
                .map(|e| Box::new(VxExpr(e)))
                .ok_or_else(|| VortexFfiError::new("or() requires at least one expression"))
        }

        /// Logical NOT of a boolean child expression. Replaces `vx_expression_not`.
        #[diplomat::attr(auto, named_constructor = "not")]
        pub fn not(child: &VxExpr) -> Box<VxExpr> {
            Box::new(VxExpr(not(child.0.clone())))
        }

        /// A boolean expression that is true where the child is null.
        ///
        /// Replaces `vx_expression_is_null`.
        #[diplomat::attr(auto, named_constructor = "is_null")]
        pub fn is_null(child: &VxExpr) -> Box<VxExpr> {
            Box::new(VxExpr(is_null(child.0.clone())))
        }

        /// A boolean expression that is true where `value` is contained in `list`.
        ///
        /// Replaces `vx_expression_list_contains`.
        #[diplomat::attr(auto, named_constructor = "list_contains")]
        pub fn list_contains(list: &VxExpr, value: &VxExpr) -> Box<VxExpr> {
            Box::new(VxExpr(list_contains(list.0.clone(), value.0.clone())))
        }
    }

    impl From<VxBinaryOperator> for Operator {
        fn from(operator: VxBinaryOperator) -> Self {
            match operator {
                VxBinaryOperator::Eq => Operator::Eq,
                VxBinaryOperator::NotEq => Operator::NotEq,
                VxBinaryOperator::Gt => Operator::Gt,
                VxBinaryOperator::Gte => Operator::Gte,
                VxBinaryOperator::Lt => Operator::Lt,
                VxBinaryOperator::Lte => Operator::Lte,
                VxBinaryOperator::KleeneAnd => Operator::And,
                VxBinaryOperator::KleeneOr => Operator::Or,
                VxBinaryOperator::Add => Operator::Add,
                VxBinaryOperator::Sub => Operator::Sub,
                VxBinaryOperator::Mul => Operator::Mul,
                VxBinaryOperator::Div => Operator::Div,
            }
        }
    }
}

impl ffi::VxExpr {
    /// Borrow the underlying [`vortex::expr::Expression`].
    ///
    /// A building block for the `scan` bridge, which consumes expressions as projection and
    /// filter inputs. Replaces the C ABI `vx_expression::as_ref` helper.
    pub(crate) fn inner(&self) -> &vortex::expr::Expression {
        &self.0
    }
}
