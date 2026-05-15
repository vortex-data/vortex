// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Factory functions for creating [`Expression`]s from scalar function vtables.

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::case_when::CaseWhen;
use crate::scalar_fn::fns::case_when::CaseWhenOptions;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::dynamic::DynamicComparison;
use crate::scalar_fn::fns::dynamic::DynamicComparisonExpr;
use crate::scalar_fn::fns::dynamic::Rhs;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::like::LikeOptions;
use crate::scalar_fn::fns::list_contains::ListContains;
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

// ---- Root ----

/// Creates an expression that references the root scope.
///
/// Returns the entire input array as passed to the expression evaluator.
/// This is commonly used as the starting point for field access and other operations.
pub fn root() -> Expression {
    Root.try_new_expr(EmptyOptions, vec![])
        .vortex_expect("Failed to create Root expression")
}

/// Return whether the expression is a root expression.
pub fn is_root(expr: &Expression) -> bool {
    expr.is::<Root>()
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

/// Creates a CASE WHEN expression with one WHEN/THEN pair and no ELSE value.
pub fn case_when_no_else(condition: Expression, then_value: Expression) -> Expression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: false,
    };
    CaseWhen.new_expr(options, [condition, then_value])
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

// ---- Binary operators ----

/// Create a new [`Binary`] using the [`Eq`](Operator::Eq) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&eq(root(), lit(3))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Eq, [lhs, rhs])
        .vortex_expect("Failed to create Eq binary expression")
}

/// Create a new [`Binary`] using the [`NotEq`](Operator::NotEq) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::{ IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, not_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&not_eq(root(), lit(3))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn not_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::NotEq, [lhs, rhs])
        .vortex_expect("Failed to create NotEq binary expression")
}

/// Create a new [`Binary`] using the [`Gte`](Operator::Gte) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{gt_eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&gt_eq(root(), lit(3))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn gt_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Gte, [lhs, rhs])
        .vortex_expect("Failed to create Gte binary expression")
}

/// Create a new [`Binary`] using the [`Gt`](Operator::Gt) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{gt, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&gt(root(), lit(2))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn gt(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Gt, [lhs, rhs])
        .vortex_expect("Failed to create Gt binary expression")
}

/// Create a new [`Binary`] using the [`Lte`](Operator::Lte) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, lt_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&lt_eq(root(), lit(2))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn lt_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Lte, [lhs, rhs])
        .vortex_expect("Failed to create Lte binary expression")
}

/// Create a new [`Binary`] using the [`Lt`](Operator::Lt) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, lt};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.into_array().apply(&lt(root(), lit(3))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn lt(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Lt, [lhs, rhs])
        .vortex_expect("Failed to create Lt binary expression")
}

/// Create a new [`Binary`] using the [`Or`](Operator::Or) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::expr::{root, lit, or};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = xs.into_array().apply(&or(root(), lit(false))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn or(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Or, [lhs, rhs])
        .vortex_expect("Failed to create Or binary expression")
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

/// Create a new [`Binary`] using the [`And`](Operator::And) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::arrays::bool::BoolArrayExt;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::expr::{and, root, lit};
/// let xs = BoolArray::from_iter(vec![true, false, true]).into_array();
/// let result = xs.apply(&and(root(), lit(true))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn and(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::And, [lhs, rhs])
        .vortex_expect("Failed to create And binary expression")
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

/// Create a new [`Binary`] using the [`Add`](Operator::Add) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_array::arrow::ArrowArrayExecutor;
/// # use vortex_array::{LEGACY_SESSION, VortexSessionExecute};
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{checked_add, lit, root};
/// let xs = buffer![1, 2, 3].into_array();
/// let result = xs.apply(&checked_add(root(), lit(5))).unwrap();
///
/// let mut ctx = LEGACY_SESSION.create_execution_ctx();
/// assert_eq!(
///     &result.execute_arrow(None, &mut ctx).unwrap(),
///     &buffer![6, 7, 8]
///         .into_array()
///         .execute_arrow(None, &mut ctx)
///         .unwrap()
/// );
/// ```
pub fn checked_add(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Add, [lhs, rhs])
        .vortex_expect("Failed to create Add binary expression")
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

// ---- Mask ----

/// Creates a mask expression that applies the given boolean mask to the input array.
pub fn mask(array: Expression, mask: Expression) -> Expression {
    Mask.new_expr(EmptyOptions, [array, mask])
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

/// Creates a merge expression with explicit duplicate handling.
pub fn merge_opts(
    elements: impl IntoIterator<Item = impl Into<Expression>>,
    duplicate_handling: DuplicateHandling,
) -> Expression {
    use itertools::Itertools as _;
    let values = elements.into_iter().map(|value| value.into()).collect_vec();
    Merge.new_expr(duplicate_handling, values)
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

// ---- Dynamic ----

/// Creates a dynamic comparison expression.
pub fn dynamic(
    operator: CompareOperator,
    rhs_value: impl Fn() -> Option<ScalarValue> + Send + Sync + 'static,
    rhs_dtype: DType,
    default: bool,
    lhs: Expression,
) -> Expression {
    DynamicComparison.new_expr(
        DynamicComparisonExpr {
            operator,
            rhs: Arc::new(Rhs {
                value: Arc::new(rhs_value),
                dtype: rhs_dtype,
            }),
            default,
        },
        [lhs],
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
