// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Factory functions for creating [`Expression`]s from scalar function vtables.

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::builtin::binary::Binary;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::fill_null::FillNull;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::scalar_fn::builtin::is_not_null::IsNotNull;
use vortex_array::scalar_fn::builtin::is_null::IsNull;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::builtin::mask::Mask;
use vortex_array::scalar_fn::builtin::not::Not;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_array::scalar_fn::builtin::pack::Pack;
use vortex_array::scalar_fn::builtin::pack::PackOptions;
use vortex_array::scalar_fn::builtin::select::FieldSelection;
use vortex_array::scalar_fn::builtin::select::Select;
use vortex_error::VortexExpect;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::BoundExpression;
use crate::Expression;
use crate::ScalarFnExprExt;

/// Creates an expression that references the root scope.
///
/// Returns the entire input array as passed to the expression evaluator.
/// This is commonly used as the starting point for field access and other operations.
pub fn root() -> Expression {
    Expression::Root
}

/// Creates a bound expression that references a root scope with the given dtype.
pub fn bound_root(dtype: DType) -> BoundExpression {
    BoundExpression::new_root(dtype)
}

/// Return whether the expression is a root expression.
pub fn is_root(expr: &Expression) -> bool {
    expr.is_root()
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
/// use vortex_expr::lit;
/// use vortex_array::scalar_fn::builtin::literal::Literal;
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
/// # use vortex_expr::col;
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
/// # use vortex_expr::{get_item, root};
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

// ---- Variable ----

/// Creates an expression that reads the parameter `name` of the enclosing lambda.
///
/// It is only meaningful in the body of a [`Lambda`](crate::Lambda). Anywhere else it fails
/// to bind.
///
/// ```rust
/// # use vortex_array::dtype::{DType, Nullability, PType};
/// # use vortex_expr::{checked_add, lit, var};
/// let expr = checked_add(var("x"), lit(1_i32));
/// assert_eq!(expr.to_string(), "(@x + 1i32)");
/// // Outside a lambda body, nothing gives `x` a value.
/// assert!(expr.bind(&DType::Primitive(PType::I32, Nullability::NonNullable)).is_err());
/// ```
pub fn var(name: impl Into<FieldName>) -> Expression {
    Expression::Variable(name.into())
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
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&eq(root(), lit(3))).unwrap();
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{root, lit, not_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&not_eq(root(), lit(3))).unwrap();
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{gt_eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&gt_eq(root(), lit(3))).unwrap();
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{gt, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&gt(root(), lit(2))).unwrap();
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{root, lit, lt_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&lt_eq(root(), lit(2))).unwrap();
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{root, lit, lt};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&lt(root(), lit(3))).unwrap();
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_expr::{root, lit, or};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = xs.into_array().apply(&or(root(), lit(false))).unwrap();
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::IntoArray;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_expr::{and, root, lit};
/// let xs = BoolArray::from_iter(vec![true, false, true]).into_array();
/// let result = xs.apply(&and(root(), lit(true))).unwrap();
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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

/// Create a new [`Binary`] using the [`Add`](Operator::Add) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_expr::ArrayExprExt;
/// # use vortex_array::arrays::PrimitiveArray;
/// # use vortex_array::builtins::ArrayBuiltins;
/// # use vortex_array::{VortexSessionExecute};
/// # use vortex_session::VortexSession;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{checked_add, lit, root};
/// let xs = buffer![1, 2, 3].into_array();
/// let result = xs.apply(&checked_add(root(), lit(5))).unwrap();
///
/// let mut ctx = VortexSession::empty().create_execution_ctx();
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
/// # use vortex_expr::{not, root};
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

// ---- Pack ----

/// Creates an expression that packs values into a struct with named fields.
///
/// ```rust
/// # use vortex_array::dtype::Nullability;
/// # use vortex_expr::{pack, col, lit};
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
/// # use vortex_expr::{cast, root};
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
/// # use vortex_expr::{fill_null, root, lit};
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
/// # use vortex_expr::{is_null, root};
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
/// # use vortex_expr::{is_not_null, root};
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

/// Constructors for expressions whose children have already been bound and type-checked.
///
/// These mirror the unbound constructors in this crate and panic when the supplied children do not
/// form a well-typed expression. Use [`BoundExpression::try_new`] when construction must be
/// fallible.
/// Creates an expression that selects (includes) specific fields from an array.
///
/// Projects only the specified fields from the child expression, which must be of DType struct.
/// ```rust
/// # use vortex_expr::select;
/// # use vortex_expr::root;
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
/// # use vortex_expr::select_exclude;
/// # use vortex_expr::root;
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

pub mod bound {
    pub use super::bound_and as and;
    pub use super::bound_and_collect as and_collect;
    pub use super::bound_binary as binary;
    pub use super::bound_cast as cast;
    pub use super::bound_checked_add as checked_add;
    pub use super::bound_col as col;
    pub use super::bound_eq as eq;
    pub use super::bound_fill_null as fill_null;
    pub use super::bound_get_item as get_item;
    pub use super::bound_gt as gt;
    pub use super::bound_gt_eq as gt_eq;
    pub use super::bound_is_not_null as is_not_null;
    pub use super::bound_is_null as is_null;
    pub use super::bound_lit as lit;
    pub use super::bound_lt as lt;
    pub use super::bound_lt_eq as lt_eq;
    pub use super::bound_mask as mask;
    pub use super::bound_not as not;
    pub use super::bound_not_eq as not_eq;
    pub use super::bound_or as or;
    pub use super::bound_or_collect as or_collect;
    pub use super::bound_pack as pack;
    pub use super::bound_root as root;
    pub use super::bound_select as select;
    pub use super::bound_select_exclude as select_exclude;
}
