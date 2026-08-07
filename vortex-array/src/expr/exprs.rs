// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Factory functions for creating [`Expression`]s from scalar function vtables.

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::aggregate_fn::NumericalAggregateOpts;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::byte_length::ByteLength;
use crate::scalar_fn::fns::case_when::CaseWhen;
use crate::scalar_fn::fns::case_when::CaseWhenOptions;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::dynamic::DynamicComparison;
use crate::scalar_fn::fns::dynamic::DynamicComparisonExpr;
use crate::scalar_fn::fns::dynamic::Rhs;
use crate::scalar_fn::fns::ext_storage::ExtStorage;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::like::LikeOptions;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_length::ListLength;
use crate::scalar_fn::fns::list_sum::ListSum;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::merge::DuplicateHandling;
use crate::scalar_fn::fns::merge::Merge;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::fns::pack::Pack;
use crate::scalar_fn::fns::pack::PackOptions;
use crate::scalar_fn::fns::root::Root;
use crate::scalar_fn::fns::select::FieldSelection;
use crate::scalar_fn::fns::select::Select;
use crate::scalar_fn::fns::variant_get::VariantGet;
use crate::scalar_fn::fns::variant_get::VariantGetOptions;
use crate::scalar_fn::fns::variant_get::VariantPath;
use crate::scalar_fn::fns::zip::Zip;

static ROOT: LazyLock<Expression> = LazyLock::new(|| {
    Root.try_new_expr(EmptyOptions, vec![])
        .vortex_expect("Creating root() shouldn't fail")
});

/// Creates an expression that references the root scope.
///
/// Returns the entire input array as passed to the expression evaluator.
/// This is commonly used as the starting point for field access and other operations.
pub fn root() -> Expression {
    ROOT.clone()
}

/// Creates a bound expression that references a root scope with the given dtype.
pub fn bound_root(dtype: DType) -> BoundExpression {
    BoundExpression::new_root(dtype)
}

/// Return whether the expression is a root expression.
pub fn is_root(expr: &Expression) -> bool {
    // root doesn't have any children, and scalar_fns have distinct ids
    // so we should almost always hit this eq check
    (expr.scalar_fn().id() == ROOT.scalar_fn().id()) || expr.is::<Root>()
}

// ---- Literal ----

/// Create a new `Literal` expression from a type that coerces to `Scalar`.
///
///
/// ## Example usage
///
/// ```
/// use vortex_array::arrays::PrimitiveArray;
/// use vortex_array::dtype::Nullability;
/// use vortex_array::expr::lit;
/// use vortex_array::scalar_fn::fns::literal::Literal;
/// use vortex_array::scalar::Scalar;
///
/// let number = lit(34i32);
///
/// let scalar = number.as_::<Literal>();
/// assert_eq!(scalar, &Scalar::primitive(34i32, Nullability::NonNullable));
/// ```
pub fn lit(value: impl Into<Scalar>) -> Expression {
    Literal.new_expr(value.into(), [])
}

/// Creates a bound literal expression.
pub fn bound_lit(value: impl Into<Scalar>) -> BoundExpression {
    Literal
        .try_new_bound_expr(value.into(), [])
        .vortex_expect("literal expressions are always well-typed")
}

// ---- GetItem / Col ----

/// Creates an expression that accesses a field from the root array.
///
/// Equivalent to `get_item(field, root())` - extracts a named field from the input array.
///
/// ```rust
/// # use vortex_array::expr::col;
/// let expr = col("name");
/// ```
pub fn col(field: impl Into<FieldName>) -> Expression {
    GetItem.new_expr(field.into(), vec![root()])
}

/// Creates a bound expression that accesses a field from a root scope with the given dtype.
pub fn bound_col(field: impl Into<FieldName>, scope: DType) -> BoundExpression {
    bound_get_item(field, bound_root(scope))
}

/// Creates an expression that extracts a named field from a struct expression.
///
/// Accesses the specified field from the result of the child expression.
///
/// ```rust
/// # use vortex_array::expr::{get_item, root};
/// let expr = get_item("user_id", root());
/// ```
pub fn get_item(field: impl Into<FieldName>, child: Expression) -> Expression {
    GetItem.new_expr(field.into(), vec![child])
}

/// Creates a bound expression that extracts a named field from a struct expression.
pub fn bound_get_item(field: impl Into<FieldName>, child: BoundExpression) -> BoundExpression {
    GetItem
        .try_new_bound_expr(field.into(), [child])
        .vortex_expect("get-item expressions must reference a field in the child dtype")
}

// ---- VariantGet ----

/// Creates an expression that extracts a path from a Variant expression.
///
/// Missing paths, traversal mismatches, and failed casts return null. When `dtype` is `None`,
/// results are nullable Variant values; otherwise results are nullable values of `dtype`.
pub fn variant_get(
    child: Expression,
    path: impl Into<VariantPath>,
    dtype: Option<DType>,
) -> Expression {
    VariantGet.new_expr(VariantGetOptions::new(path.into(), dtype), vec![child])
}

/// Creates a bound expression that extracts a path from a Variant expression.
pub fn bound_variant_get(
    child: BoundExpression,
    path: impl Into<VariantPath>,
    dtype: Option<DType>,
) -> BoundExpression {
    VariantGet
        .try_new_bound_expr(VariantGetOptions::new(path.into(), dtype), [child])
        .vortex_expect("variant-get expressions require a Variant child")
}

// ---- CaseWhen ----

/// Creates a CASE WHEN expression with one WHEN/THEN pair and an ELSE value.
pub fn case_when(
    condition: Expression,
    then_value: Expression,
    else_value: Expression,
) -> Expression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: true,
    };
    CaseWhen.new_expr(options, [condition, then_value, else_value])
}

/// Creates a bound CASE WHEN expression with one WHEN/THEN pair and an ELSE value.
pub fn bound_case_when(
    condition: BoundExpression,
    then_value: BoundExpression,
    else_value: BoundExpression,
) -> BoundExpression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: true,
    };
    CaseWhen
        .try_new_bound_expr(options, [condition, then_value, else_value])
        .vortex_expect("case expressions must have boolean conditions and matching branch dtypes")
}

/// Creates a CASE WHEN expression with one WHEN/THEN pair and no ELSE value.
pub fn case_when_no_else(condition: Expression, then_value: Expression) -> Expression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: false,
    };
    CaseWhen.new_expr(options, [condition, then_value])
}

/// Creates a bound CASE WHEN expression with one WHEN/THEN pair and no ELSE value.
pub fn bound_case_when_no_else(
    condition: BoundExpression,
    then_value: BoundExpression,
) -> BoundExpression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: false,
    };
    CaseWhen
        .try_new_bound_expr(options, [condition, then_value])
        .vortex_expect("case expressions must have boolean conditions")
}

/// Creates an n-ary CASE WHEN expression from WHEN/THEN pairs and an optional ELSE value.
pub fn nested_case_when(
    when_then_pairs: Vec<(Expression, Expression)>,
    else_value: Option<Expression>,
) -> Expression {
    assert!(
        !when_then_pairs.is_empty(),
        "nested_case_when requires at least one when/then pair"
    );

    let has_else = else_value.is_some();
    let mut children = Vec::with_capacity(when_then_pairs.len() * 2 + usize::from(has_else));
    for (condition, then_value) in &when_then_pairs {
        children.push(condition.clone());
        children.push(then_value.clone());
    }
    if let Some(else_expr) = else_value {
        children.push(else_expr);
    }

    let Ok(num_when_then_pairs) = u32::try_from(when_then_pairs.len()) else {
        vortex_panic!("nested_case_when has too many when/then pairs");
    };
    let options = CaseWhenOptions {
        num_when_then_pairs,
        has_else,
    };
    CaseWhen.new_expr(options, children)
}

/// Creates a bound n-ary CASE WHEN expression from WHEN/THEN pairs and an optional ELSE value.
pub fn bound_nested_case_when(
    when_then_pairs: Vec<(BoundExpression, BoundExpression)>,
    else_value: Option<BoundExpression>,
) -> BoundExpression {
    assert!(
        !when_then_pairs.is_empty(),
        "nested_case_when requires at least one when/then pair"
    );

    let Ok(num_when_then_pairs) = u32::try_from(when_then_pairs.len()) else {
        vortex_panic!("nested_case_when has too many when/then pairs");
    };
    let has_else = else_value.is_some();
    let mut children = Vec::with_capacity(when_then_pairs.len() * 2 + usize::from(has_else));
    for (condition, then_value) in when_then_pairs {
        children.push(condition);
        children.push(then_value);
    }
    if let Some(else_expr) = else_value {
        children.push(else_expr);
    }

    let options = CaseWhenOptions {
        num_when_then_pairs,
        has_else,
    };
    CaseWhen
        .try_new_bound_expr(options, children)
        .vortex_expect("case expressions must have boolean conditions and matching branch dtypes")
}

// ---- Binary operators ----

/// Creates a binary expression with the given operator.
pub fn binary(operator: Operator, lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(operator, [lhs, rhs])
        .vortex_expect("Failed to create binary expression")
}

/// Creates a bound binary expression with the given operator.
pub fn bound_binary(
    operator: Operator,
    lhs: BoundExpression,
    rhs: BoundExpression,
) -> BoundExpression {
    Binary
        .try_new_bound_expr(operator, [lhs, rhs])
        .vortex_expect("binary expressions must have compatible operand dtypes")
}

/// Create a new [`Binary`] using the [`Eq`](Operator::Eq) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&eq(root(), lit(3))).unwrap();
/// let mut ctx = array_session().create_execution_ctx();
///
/// assert_eq!(
///     result.execute::<BoolArray>(&mut ctx).unwrap().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Eq, [lhs, rhs])
        .vortex_expect("Failed to create Eq binary expression")
}

/// Creates a bound equality expression.
pub fn bound_eq(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::Eq, lhs, rhs)
}

/// Create a new [`Binary`] using the [`NotEq`](Operator::NotEq) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, not_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&not_eq(root(), lit(3))).unwrap();
/// let mut ctx = array_session().create_execution_ctx();
///
/// assert_eq!(
///     result.execute::<BoolArray>(&mut ctx).unwrap().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn not_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::NotEq, [lhs, rhs])
        .vortex_expect("Failed to create NotEq binary expression")
}

/// Creates a bound inequality expression.
pub fn bound_not_eq(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::NotEq, lhs, rhs)
}

/// Create a new [`Binary`] using the [`Gte`](Operator::Gte) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{gt_eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&gt_eq(root(), lit(3))).unwrap();
/// let mut ctx = array_session().create_execution_ctx();
///
/// assert_eq!(
///     result.execute::<BoolArray>(&mut ctx).unwrap().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn gt_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Gte, [lhs, rhs])
        .vortex_expect("Failed to create Gte binary expression")
}

/// Creates a bound greater-than-or-equal expression.
pub fn bound_gt_eq(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::Gte, lhs, rhs)
}

/// Create a new [`Binary`] using the [`Gt`](Operator::Gt) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{gt, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&gt(root(), lit(2))).unwrap();
/// let mut ctx = array_session().create_execution_ctx();
///
/// assert_eq!(
///     result.execute::<BoolArray>(&mut ctx).unwrap().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn gt(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Gt, [lhs, rhs])
        .vortex_expect("Failed to create Gt binary expression")
}

/// Creates a bound greater-than expression.
pub fn bound_gt(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::Gt, lhs, rhs)
}

/// Create a new [`Binary`] using the [`Lte`](Operator::Lte) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, lt_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&lt_eq(root(), lit(2))).unwrap();
/// let mut ctx = array_session().create_execution_ctx();
///
/// assert_eq!(
///     result.execute::<BoolArray>(&mut ctx).unwrap().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn lt_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Lte, [lhs, rhs])
        .vortex_expect("Failed to create Lte binary expression")
}

/// Creates a bound less-than-or-equal expression.
pub fn bound_lt_eq(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::Lte, lhs, rhs)
}

/// Create a new [`Binary`] using the [`Lt`](Operator::Lt) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, lt};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&lt(root(), lit(3))).unwrap();
/// let mut ctx = array_session().create_execution_ctx();
///
/// assert_eq!(
///     result.execute::<BoolArray>(&mut ctx).unwrap().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn lt(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Lt, [lhs, rhs])
        .vortex_expect("Failed to create Lt binary expression")
}

/// Creates a bound less-than expression.
pub fn bound_lt(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::Lt, lhs, rhs)
}

/// Create a new [`Binary`] using the [`Or`](Operator::Or) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_array::expr::{root, lit, or};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = xs.into_array().apply(&or(root(), lit(false))).unwrap();
/// let mut ctx = array_session().create_execution_ctx();
///
/// assert_eq!(
///     result.execute::<BoolArray>(&mut ctx).unwrap().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn or(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Or, [lhs, rhs])
        .vortex_expect("Failed to create Or binary expression")
}

/// Creates a bound boolean OR expression.
pub fn bound_or(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::Or, lhs, rhs)
}

/// Collects a list of `or`ed values into a single expression using a balanced tree.
///
/// This creates a balanced binary tree to avoid deep nesting that could cause
/// stack overflow during drop or evaluation.
///
/// [a, b, c, d] => or(or(a, b), or(c, d))
pub fn or_collect<I>(iter: I) -> Option<Expression>
where
    I: IntoIterator<Item = Expression>,
{
    iter.into_iter().reduce_balanced(or)
}

/// Collects bound expressions into a balanced tree of boolean OR expressions.
pub fn bound_or_collect<I>(iter: I) -> Option<BoundExpression>
where
    I: IntoIterator<Item = BoundExpression>,
{
    iter.into_iter().reduce_balanced(bound_or)
}

/// Create a new [`Binary`] using the [`And`](Operator::And) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_array::expr::{and, root, lit};
/// let xs = BoolArray::from_iter(vec![true, false, true]).into_array();
/// let result = xs.apply(&and(root(), lit(true))).unwrap();
/// let mut ctx = array_session().create_execution_ctx();
///
/// assert_eq!(
///     result.execute::<BoolArray>(&mut ctx).unwrap().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn and(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::And, [lhs, rhs])
        .vortex_expect("Failed to create And binary expression")
}

/// Creates a bound boolean AND expression.
pub fn bound_and(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::And, lhs, rhs)
}

/// Collects a list of `and`ed values into a single expression using a balanced tree.
///
/// This creates a balanced binary tree to avoid deep nesting that could cause
/// stack overflow during drop or evaluation.
///
/// [a, b, c, d] => and(and(a, b), and(c, d))
pub fn and_collect<I>(iter: I) -> Option<Expression>
where
    I: IntoIterator<Item = Expression>,
{
    iter.into_iter().reduce_balanced(and)
}

/// Collects bound expressions into a balanced tree of boolean AND expressions.
pub fn bound_and_collect<I>(iter: I) -> Option<BoundExpression>
where
    I: IntoIterator<Item = BoundExpression>,
{
    iter.into_iter().reduce_balanced(bound_and)
}

/// The conjunction of an expression's child validities — i.e. the validity of a scalar function
/// whose result is null exactly when any operand is null.
///
/// This is the `ScalarFnVTable::validity` for kernels that propagate nulls and never produce a
/// null from non-null inputs (comparisons, arithmetic, most spatial and tensor operations). Returning it lets
/// the planner derive the output's null mask without executing the kernel. Yields `None` when the
/// expression has no children.
pub fn union_child_validities(expression: &Expression) -> VortexResult<Option<Expression>> {
    let child_validities = expression
        .children()
        .iter()
        .map(Expression::validity)
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(and_collect(child_validities))
}

/// Create a new [`Binary`] using the [`Add`](Operator::Add) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_array::arrays::PrimitiveArray;
/// # use vortex_array::builtins::ArrayBuiltins;
/// # use vortex_array::{VortexSessionExecute, array_session};
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{checked_add, lit, root};
/// let xs = buffer![1, 2, 3].into_array();
/// let result = xs.apply(&checked_add(root(), lit(5))).unwrap();
///
/// let mut ctx = array_session().create_execution_ctx();
/// let result = result.execute::<PrimitiveArray>(&mut ctx).unwrap();
/// assert_eq!(result.as_slice::<i32>(), [6, 7, 8]);
/// ```
pub fn checked_add(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Add, [lhs, rhs])
        .vortex_expect("Failed to create Add binary expression")
}

/// Creates a bound checked-add expression.
pub fn bound_checked_add(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    bound_binary(Operator::Add, lhs, rhs)
}

// ---- Not ----

/// Creates an expression that logically inverts boolean values.
///
/// Returns the logical negation of the input boolean expression.
///
/// ```rust
/// # use vortex_array::expr::{not, root};
/// let expr = not(root());
/// ```
pub fn not(operand: Expression) -> Expression {
    Not.new_expr(EmptyOptions, vec![operand])
}

/// Creates a bound expression that logically inverts boolean values.
pub fn bound_not(operand: BoundExpression) -> BoundExpression {
    Not.try_new_bound_expr(EmptyOptions, [operand])
        .vortex_expect("not expressions require a boolean operand")
}

// ---- Between ----

/// Creates an expression that checks if values are between two bounds.
///
/// Returns a boolean array indicating which values fall within the specified range.
/// The comparison strictness is controlled by the options parameter.
///
/// ```rust
/// # use vortex_array::scalar_fn::fns::between::BetweenOptions;
/// # use vortex_array::scalar_fn::fns::between::StrictComparison;
/// # use vortex_array::expr::{between, lit, root};
/// let opts = BetweenOptions {
///     lower_strict: StrictComparison::NonStrict,
///     upper_strict: StrictComparison::NonStrict,
/// };
/// let expr = between(root(), lit(10), lit(20), opts);
/// ```
pub fn between(
    arr: Expression,
    lower: Expression,
    upper: Expression,
    options: BetweenOptions,
) -> Expression {
    Between
        .try_new_expr(options, [arr, lower, upper])
        .vortex_expect("Failed to create Between expression")
}

/// Creates a bound expression that checks if values are between two bounds.
pub fn bound_between(
    arr: BoundExpression,
    lower: BoundExpression,
    upper: BoundExpression,
    options: BetweenOptions,
) -> BoundExpression {
    Between
        .try_new_bound_expr(options, [arr, lower, upper])
        .vortex_expect("between expressions require compatible operand dtypes")
}

// ---- Select ----

/// Creates an expression that selects (includes) specific fields from an array.
///
/// Projects only the specified fields from the child expression, which must be of DType struct.
/// ```rust
/// # use vortex_array::expr::{select, root};
/// let expr = select(["name", "age"], root());
/// ```
pub fn select(field_names: impl Into<FieldNames>, child: Expression) -> Expression {
    Select
        .try_new_expr(FieldSelection::Include(field_names.into()), [child])
        .vortex_expect("Failed to create Select expression")
}

/// Creates a bound expression that selects specific fields from a struct expression.
pub fn bound_select(field_names: impl Into<FieldNames>, child: BoundExpression) -> BoundExpression {
    Select
        .try_new_bound_expr(FieldSelection::Include(field_names.into()), [child])
        .vortex_expect("select expressions require fields from a struct child")
}

/// Creates an expression that excludes specific fields from an array.
///
/// Projects all fields except the specified ones from the input struct expression.
///
/// ```rust
/// # use vortex_array::expr::{select_exclude, root};
/// let expr = select_exclude(["internal_id", "metadata"], root());
/// ```
pub fn select_exclude(fields: impl Into<FieldNames>, child: Expression) -> Expression {
    Select
        .try_new_expr(FieldSelection::Exclude(fields.into()), [child])
        .vortex_expect("Failed to create Select expression")
}

/// Creates a bound expression that excludes specific fields from a struct expression.
pub fn bound_select_exclude(
    fields: impl Into<FieldNames>,
    child: BoundExpression,
) -> BoundExpression {
    Select
        .try_new_bound_expr(FieldSelection::Exclude(fields.into()), [child])
        .vortex_expect("select expressions require fields from a struct child")
}

// ---- Pack ----

/// Creates an expression that packs values into a struct with named fields.
///
/// ```rust
/// # use vortex_array::dtype::Nullability;
/// # use vortex_array::expr::{pack, col, lit};
/// let expr = pack([("id", col("user_id")), ("constant", lit(42))], Nullability::NonNullable);
/// ```
pub fn pack(
    elements: impl IntoIterator<Item = (impl Into<FieldName>, Expression)>,
    nullability: Nullability,
) -> Expression {
    let (names, values): (Vec<_>, Vec<_>) = elements
        .into_iter()
        .map(|(name, value)| (name.into(), value))
        .unzip();
    Pack.new_expr(
        PackOptions {
            names: names.into(),
            nullability,
        },
        values,
    )
}

/// Creates a bound expression that packs values into a struct with named fields.
pub fn bound_pack(
    elements: impl IntoIterator<Item = (impl Into<FieldName>, BoundExpression)>,
    nullability: Nullability,
) -> BoundExpression {
    let (names, values): (Vec<_>, Vec<_>) = elements
        .into_iter()
        .map(|(name, value)| (name.into(), value))
        .unzip();
    Pack.try_new_bound_expr(
        PackOptions {
            names: names.into(),
            nullability,
        },
        values,
    )
    .vortex_expect("pack expressions must have one name per child")
}

// ---- Cast ----

/// Creates an expression that casts values to a target data type.
///
/// Converts the input expression's values to the specified target type.
///
/// ```rust
/// # use vortex_array::dtype::{DType, Nullability, PType};
/// # use vortex_array::expr::{cast, root};
/// let expr = cast(root(), DType::Primitive(PType::I64, Nullability::NonNullable));
/// ```
pub fn cast(child: Expression, target: DType) -> Expression {
    Cast.try_new_expr(target, [child])
        .vortex_expect("Failed to create Cast expression")
}

/// Creates a bound expression that casts values to a target dtype.
pub fn bound_cast(child: BoundExpression, target: DType) -> BoundExpression {
    Cast.try_new_bound_expr(target, [child])
        .vortex_expect("cast expressions require a supported source and target dtype")
}

// ---- FillNull ----

/// Creates an expression that replaces null values with a fill value.
///
/// ```rust
/// # use vortex_array::expr::{fill_null, root, lit};
/// let expr = fill_null(root(), lit(0i32));
/// ```
pub fn fill_null(child: Expression, fill_value: Expression) -> Expression {
    FillNull.new_expr(EmptyOptions, [child, fill_value])
}

/// Creates a bound expression that replaces null values with a fill value.
pub fn bound_fill_null(child: BoundExpression, fill_value: BoundExpression) -> BoundExpression {
    FillNull
        .try_new_bound_expr(EmptyOptions, [child, fill_value])
        .vortex_expect("fill-null expressions require compatible child and fill dtypes")
}

// ---- IsNull ----

/// Creates an expression that checks for null values.
///
/// Returns a boolean array indicating which positions contain null values.
///
/// ```rust
/// # use vortex_array::expr::{is_null, root};
/// let expr = is_null(root());
/// ```
pub fn is_null(child: Expression) -> Expression {
    IsNull.new_expr(EmptyOptions, vec![child])
}

/// Creates a bound expression that checks for null values.
pub fn bound_is_null(child: BoundExpression) -> BoundExpression {
    IsNull
        .try_new_bound_expr(EmptyOptions, [child])
        .vortex_expect("is-null expressions are always well-typed")
}

// ---- IsNotNull ----

/// Creates an expression that checks for non-null values.
///
/// Returns a boolean array indicating which positions contain non-null values.
///
/// ```rust
/// # use vortex_array::expr::{is_not_null, root};
/// let expr = is_not_null(root());
/// ```
pub fn is_not_null(child: Expression) -> Expression {
    IsNotNull.new_expr(EmptyOptions, vec![child])
}

/// Creates a bound expression that checks for non-null values.
pub fn bound_is_not_null(child: BoundExpression) -> BoundExpression {
    IsNotNull
        .try_new_bound_expr(EmptyOptions, [child])
        .vortex_expect("is-not-null expressions are always well-typed")
}

// ---- Like ----

/// Creates a SQL LIKE expression.
pub fn like(child: Expression, pattern: Expression) -> Expression {
    Like.new_expr(
        LikeOptions {
            negated: false,
            case_insensitive: false,
        },
        [child, pattern],
    )
}

/// Creates a bound SQL LIKE expression.
pub fn bound_like(child: BoundExpression, pattern: BoundExpression) -> BoundExpression {
    bound_like_with_options(child, pattern, false, false)
}

/// Creates a case-insensitive SQL ILIKE expression.
pub fn ilike(child: Expression, pattern: Expression) -> Expression {
    Like.new_expr(
        LikeOptions {
            negated: false,
            case_insensitive: true,
        },
        [child, pattern],
    )
}

/// Creates a bound case-insensitive SQL ILIKE expression.
pub fn bound_ilike(child: BoundExpression, pattern: BoundExpression) -> BoundExpression {
    bound_like_with_options(child, pattern, false, true)
}

/// Creates a negated SQL NOT LIKE expression.
pub fn not_like(child: Expression, pattern: Expression) -> Expression {
    Like.new_expr(
        LikeOptions {
            negated: true,
            case_insensitive: false,
        },
        [child, pattern],
    )
}

/// Creates a bound negated SQL NOT LIKE expression.
pub fn bound_not_like(child: BoundExpression, pattern: BoundExpression) -> BoundExpression {
    bound_like_with_options(child, pattern, true, false)
}

/// Creates a negated case-insensitive SQL NOT ILIKE expression.
pub fn not_ilike(child: Expression, pattern: Expression) -> Expression {
    Like.new_expr(
        LikeOptions {
            negated: true,
            case_insensitive: true,
        },
        [child, pattern],
    )
}

/// Creates a bound negated case-insensitive SQL NOT ILIKE expression.
pub fn bound_not_ilike(child: BoundExpression, pattern: BoundExpression) -> BoundExpression {
    bound_like_with_options(child, pattern, true, true)
}

fn bound_like_with_options(
    child: BoundExpression,
    pattern: BoundExpression,
    negated: bool,
    case_insensitive: bool,
) -> BoundExpression {
    Like.try_new_bound_expr(
        LikeOptions {
            negated,
            case_insensitive,
        },
        [child, pattern],
    )
    .vortex_expect("like expressions require UTF-8 or binary operands")
}

// ---- Mask ----

/// Creates a mask expression that applies the given boolean mask to the input array.
pub fn mask(array: Expression, mask: Expression) -> Expression {
    Mask.new_expr(EmptyOptions, [array, mask])
}

/// Creates a bound mask expression.
pub fn bound_mask(array: BoundExpression, mask: BoundExpression) -> BoundExpression {
    Mask.try_new_bound_expr(EmptyOptions, [array, mask])
        .vortex_expect("mask expressions require a boolean mask")
}

// ---- Merge ----

/// Creates an expression that merges struct expressions into a single struct.
///
/// Combines fields from all input expressions. If field names are duplicated,
/// later expressions win. Fields are not recursively merged.
///
/// ```rust
/// # use vortex_array::dtype::Nullability;
/// # use vortex_array::expr::{merge, get_item, root};
/// let expr = merge([get_item("a", root()), get_item("b", root())]);
/// ```
pub fn merge(elements: impl IntoIterator<Item = impl Into<Expression>>) -> Expression {
    use itertools::Itertools as _;
    let values = elements.into_iter().map(|value| value.into()).collect_vec();
    Merge.new_expr(DuplicateHandling::default(), values)
}

/// Creates a bound expression that merges struct expressions.
pub fn bound_merge(elements: impl IntoIterator<Item = BoundExpression>) -> BoundExpression {
    bound_merge_opts(elements, DuplicateHandling::default())
}

/// Creates a merge expression with explicit duplicate handling.
pub fn merge_opts(
    elements: impl IntoIterator<Item = impl Into<Expression>>,
    duplicate_handling: DuplicateHandling,
) -> Expression {
    use itertools::Itertools as _;
    let values = elements.into_iter().map(|value| value.into()).collect_vec();
    Merge.new_expr(duplicate_handling, values)
}

/// Creates a bound merge expression with explicit duplicate handling.
pub fn bound_merge_opts(
    elements: impl IntoIterator<Item = BoundExpression>,
    duplicate_handling: DuplicateHandling,
) -> BoundExpression {
    Merge
        .try_new_bound_expr(duplicate_handling, elements)
        .vortex_expect("merge expressions require non-nullable struct children")
}

// ---- Zip ----

/// Creates a zip expression that conditionally selects between two arrays.
///
/// ```rust
/// # use vortex_array::expr::{zip_expr, root, lit};
/// let expr = zip_expr(lit(true), root(), lit(0i32));
/// ```
pub fn zip_expr(mask: Expression, if_true: Expression, if_false: Expression) -> Expression {
    Zip.new_expr(EmptyOptions, [if_true, if_false, mask])
}

/// Creates a bound zip expression that conditionally selects between two arrays.
pub fn bound_zip_expr(
    mask: BoundExpression,
    if_true: BoundExpression,
    if_false: BoundExpression,
) -> BoundExpression {
    Zip.try_new_bound_expr(EmptyOptions, [if_true, if_false, mask])
        .vortex_expect("zip expressions require a boolean mask and compatible value dtypes")
}

// ---- Dynamic ----

/// Creates a dynamic comparison expression from its complete options.
pub fn dynamic_with_options(options: DynamicComparisonExpr, lhs: Expression) -> Expression {
    DynamicComparison.new_expr(options, [lhs])
}

/// Creates a bound dynamic comparison expression from its complete options.
pub fn bound_dynamic_with_options(
    options: DynamicComparisonExpr,
    lhs: BoundExpression,
) -> BoundExpression {
    DynamicComparison
        .try_new_bound_expr(options, [lhs])
        .vortex_expect("dynamic comparisons require a compatible left-hand dtype")
}

/// Creates a dynamic comparison expression.
pub fn dynamic(
    operator: CompareOperator,
    rhs_value: impl Fn() -> Option<ScalarValue> + Send + Sync + 'static,
    rhs_dtype: DType,
    default: bool,
    lhs: Expression,
) -> Expression {
    dynamic_with_options(
        DynamicComparisonExpr {
            operator,
            rhs: Arc::new(Rhs {
                value: Arc::new(rhs_value),
                dtype: rhs_dtype,
            }),
            default,
        },
        lhs,
    )
}

/// Creates a bound dynamic comparison expression.
pub fn bound_dynamic(
    operator: CompareOperator,
    rhs_value: impl Fn() -> Option<ScalarValue> + Send + Sync + 'static,
    rhs_dtype: DType,
    default: bool,
    lhs: BoundExpression,
) -> BoundExpression {
    bound_dynamic_with_options(
        DynamicComparisonExpr {
            operator,
            rhs: Arc::new(Rhs {
                value: Arc::new(rhs_value),
                dtype: rhs_dtype,
            }),
            default,
        },
        lhs,
    )
}

// ---- ListContains ----

/// Creates an expression that checks if a value is contained in a list.
///
/// Returns a boolean array indicating whether the value appears in each list.
///
/// ```rust
/// # use vortex_array::expr::{list_contains, lit, root};
/// let expr = list_contains(root(), lit(42));
/// ```
pub fn list_contains(list: Expression, value: Expression) -> Expression {
    ListContains.new_expr(EmptyOptions, [list, value])
}

/// Creates a bound expression that checks if a value is contained in a list.
pub fn bound_list_contains(list: BoundExpression, value: BoundExpression) -> BoundExpression {
    ListContains
        .try_new_bound_expr(EmptyOptions, [list, value])
        .vortex_expect("list-contains expressions require a compatible list and value dtype")
}

// ---- ByteLength ----

/// Creates an expression that computes the byte length of each element.
/// This is akin to ANSI SQL OCTET_LENGTH(), or DuckDB's strlen().
///
/// ```rust
/// # use vortex_array::expr::{byte_length, root};
/// let expr = byte_length(root());
/// ```
pub fn byte_length(input: Expression) -> Expression {
    ByteLength.new_expr(EmptyOptions, [input])
}

/// Creates a bound expression that computes each element's byte length.
pub fn bound_byte_length(input: BoundExpression) -> BoundExpression {
    ByteLength
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("byte-length expressions require a variable-length binary child")
}

// ---- ExtStorage ----

/// Creates an expression that extracts the storage values from an extension array.
///
/// ```rust
/// # use vortex_array::expr::{ext_storage, root};
/// let expr = ext_storage(root());
/// ```
pub fn ext_storage(input: Expression) -> Expression {
    ExtStorage.new_expr(EmptyOptions, [input])
}

/// Creates a bound expression that extracts an extension array's storage values.
pub fn bound_ext_storage(input: BoundExpression) -> BoundExpression {
    ExtStorage
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("extension-storage expressions require an extension child")
}

// ---- ListLength ----

/// Creates an expression that computes the number of elements in each list
/// for `List` and `FixedSizeList` inputs. This is akin to ANSI SQL `CARDINALITY()`,
/// or DuckDB's `len()`/`array_length()`.
///
/// ```rust
/// # use vortex_array::expr::{list_length, root};
/// let expr = list_length(root());
/// ```
pub fn list_length(input: Expression) -> Expression {
    ListLength.new_expr(EmptyOptions, [input])
}

/// Creates a bound expression that computes the number of elements in each list.
pub fn bound_list_length(input: BoundExpression) -> BoundExpression {
    ListLength
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("list-length expressions require a list child")
}

// ---- ListSum ----

/// Creates an expression that sums the elements of each list for `List` and
/// `FixedSizeList` inputs, akin to DuckDB's `list_sum()`.
///
/// Follows SQL `SUM` semantics per list: null lists, empty lists, and lists whose elements are
/// all null yield null; null elements are skipped; integer and decimal overflow yields a null
/// value. The result dtype follows `sum`'s widening rules and is always nullable. NaN float
/// elements are skipped by default; see [`list_sum_opts`] for the NaN-including variant.
///
/// ```rust
/// # use vortex_array::expr::{list_sum, root};
/// let expr = list_sum(root());
/// ```
pub fn list_sum(input: Expression) -> Expression {
    ListSum.new_expr(NumericalAggregateOpts::default(), [input])
}

/// Creates a bound expression that sums the elements of each list.
pub fn bound_list_sum(input: BoundExpression) -> BoundExpression {
    ListSum
        .try_new_bound_expr(NumericalAggregateOpts::default(), [input])
        .vortex_expect("list-sum expressions require a numeric list child")
}

/// Creates a [`list_sum`] expression with explicit [`NumericalAggregateOpts`], controlling
/// whether NaN float elements are skipped (the default) or poison the list's sum to NaN.
pub fn list_sum_opts(input: Expression, options: NumericalAggregateOpts) -> Expression {
    ListSum.new_expr(options, [input])
}

/// Creates a bound list-sum expression with explicit aggregate options.
pub fn bound_list_sum_opts(
    input: BoundExpression,
    options: NumericalAggregateOpts,
) -> BoundExpression {
    ListSum
        .try_new_bound_expr(options, [input])
        .vortex_expect("list-sum expressions require a numeric list child")
}

/// Constructors for expressions whose children have already been bound and type-checked.
///
/// These mirror the constructors in [`crate::expr`] and panic when the supplied children do not
/// form a well-typed expression. Use [`BoundExpression::try_new`] when construction must be
/// fallible.
pub mod bound {
    pub use super::bound_and as and;
    pub use super::bound_and_collect as and_collect;
    pub use super::bound_between as between;
    pub use super::bound_binary as binary;
    pub use super::bound_byte_length as byte_length;
    pub use super::bound_case_when as case_when;
    pub use super::bound_case_when_no_else as case_when_no_else;
    pub use super::bound_cast as cast;
    pub use super::bound_checked_add as checked_add;
    pub use super::bound_col as col;
    pub use super::bound_dynamic as dynamic;
    pub use super::bound_dynamic_with_options as dynamic_with_options;
    pub use super::bound_eq as eq;
    pub use super::bound_ext_storage as ext_storage;
    pub use super::bound_fill_null as fill_null;
    pub use super::bound_get_item as get_item;
    pub use super::bound_gt as gt;
    pub use super::bound_gt_eq as gt_eq;
    pub use super::bound_ilike as ilike;
    pub use super::bound_is_not_null as is_not_null;
    pub use super::bound_is_null as is_null;
    pub use super::bound_like as like;
    pub use super::bound_list_contains as list_contains;
    pub use super::bound_list_length as list_length;
    pub use super::bound_list_sum as list_sum;
    pub use super::bound_list_sum_opts as list_sum_opts;
    pub use super::bound_lit as lit;
    pub use super::bound_lt as lt;
    pub use super::bound_lt_eq as lt_eq;
    pub use super::bound_mask as mask;
    pub use super::bound_merge as merge;
    pub use super::bound_merge_opts as merge_opts;
    pub use super::bound_nested_case_when as nested_case_when;
    pub use super::bound_not as not;
    pub use super::bound_not_eq as not_eq;
    pub use super::bound_not_ilike as not_ilike;
    pub use super::bound_not_like as not_like;
    pub use super::bound_or as or;
    pub use super::bound_or_collect as or_collect;
    pub use super::bound_pack as pack;
    pub use super::bound_root as root;
    pub use super::bound_select as select;
    pub use super::bound_select_exclude as select_exclude;
    pub use super::bound_variant_get as variant_get;
    pub use super::bound_zip_expr as zip_expr;
}
