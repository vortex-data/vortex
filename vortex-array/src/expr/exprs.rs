// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constructors for bound scalar-function expressions.

use std::sync::Arc;

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
use crate::scalar_fn::fns::is_nan::IsNan;
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
use crate::scalar_fn::fns::select::FieldSelection;
use crate::scalar_fn::fns::select::Select;
use crate::scalar_fn::fns::variant_get::VariantGet;
use crate::scalar_fn::fns::variant_get::VariantGetOptions;
use crate::scalar_fn::fns::variant_get::VariantPath;
use crate::scalar_fn::fns::zip::Zip;

/// Creates a bound expression that references a root scope with the given dtype.
pub fn root(dtype: DType) -> BoundExpression {
    BoundExpression::new_root(dtype)
}

/// Creates a bound literal expression.
pub fn lit(value: impl Into<Scalar>) -> BoundExpression {
    Literal
        .try_new_bound_expr(value.into(), [])
        .vortex_expect("literal expressions are always well-typed")
}

/// Creates a bound expression that accesses a field from a root scope with the given dtype.
pub fn col(field: impl Into<FieldName>, scope: DType) -> BoundExpression {
    get_item(field, root(scope))
}

/// Creates a bound expression that extracts a named field from a struct expression.
pub fn get_item(field: impl Into<FieldName>, child: BoundExpression) -> BoundExpression {
    GetItem
        .try_new_bound_expr(field.into(), [child])
        .vortex_expect("get-item expressions must reference a field in the child dtype")
}

/// Creates a bound expression that extracts a path from a Variant expression.
pub fn variant_get(
    child: BoundExpression,
    path: impl Into<VariantPath>,
    dtype: Option<DType>,
) -> BoundExpression {
    VariantGet
        .try_new_bound_expr(VariantGetOptions::new(path.into(), dtype), [child])
        .vortex_expect("variant-get expressions require a Variant child")
}

/// Creates a bound CASE WHEN expression with one WHEN/THEN pair and an ELSE value.
pub fn case_when(
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
        .vortex_expect(
            "case expressions must have boolean conditions and matching branch dtypes",
        )
}

/// Creates a bound CASE WHEN expression with one WHEN/THEN pair and no ELSE value.
pub fn case_when_no_else(
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

/// Creates a bound n-ary CASE WHEN expression from WHEN/THEN pairs and an optional ELSE value.
pub fn nested_case_when(
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
        .vortex_expect(
            "case expressions must have boolean conditions and matching branch dtypes",
        )
}

/// Creates a bound binary expression with the given operator.
pub fn binary(
    operator: Operator,
    lhs: BoundExpression,
    rhs: BoundExpression,
) -> BoundExpression {
    Binary
        .try_new_bound_expr(operator, [lhs, rhs])
        .vortex_expect("binary expressions must have compatible operand dtypes")
}

/// Creates a bound equality expression.
pub fn eq(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::Eq, lhs, rhs)
}

/// Creates a bound inequality expression.
pub fn not_eq(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::NotEq, lhs, rhs)
}

/// Creates a bound greater-than-or-equal expression.
pub fn gt_eq(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::Gte, lhs, rhs)
}

/// Creates a bound greater-than expression.
pub fn gt(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::Gt, lhs, rhs)
}

/// Creates a bound less-than-or-equal expression.
pub fn lt_eq(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::Lte, lhs, rhs)
}

/// Creates a bound less-than expression.
pub fn lt(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::Lt, lhs, rhs)
}

/// Creates a bound boolean OR expression.
pub fn or(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::Or, lhs, rhs)
}

/// Collects bound expressions into a balanced tree of boolean OR expressions.
pub fn or_collect<I>(iter: I) -> Option<BoundExpression>
where
    I: IntoIterator<Item = BoundExpression>,
{
    iter.into_iter().reduce_balanced(or)
}

/// Creates a bound boolean AND expression.
pub fn and(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::And, lhs, rhs)
}

/// Collects bound expressions into a balanced tree of boolean AND expressions.
pub fn and_collect<I>(iter: I) -> Option<BoundExpression>
where
    I: IntoIterator<Item = BoundExpression>,
{
    iter.into_iter().reduce_balanced(and)
}

/// The conjunction of an expression's child validities — i.e. the validity of a scalar function
/// whose result is null exactly when any operand is null.
///
/// This is the `ScalarFnVTable::validity` for kernels that propagate nulls and never produce a
/// null from non-null inputs (comparisons, arithmetic, most spatial and tensor operations). Returning it lets
/// the planner derive the output's null mask without executing the kernel. Yields `None` when the
/// expression has no children.
pub fn union_child_validities(
    expression: &BoundExpression,
) -> VortexResult<Option<BoundExpression>> {
    let child_validities = expression
        .children()
        .iter()
        .map(BoundExpression::validity)
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(and_collect(child_validities))
}

/// Creates a bound checked-add expression.
pub fn checked_add(lhs: BoundExpression, rhs: BoundExpression) -> BoundExpression {
    binary(Operator::Add, lhs, rhs)
}

/// Creates a bound expression that logically inverts boolean values.
pub fn not(operand: BoundExpression) -> BoundExpression {
    Not.try_new_bound_expr(EmptyOptions, [operand])
        .vortex_expect("not expressions require a boolean operand")
}

/// Creates a bound expression that checks if values are between two bounds.
pub fn between(
    arr: BoundExpression,
    lower: BoundExpression,
    upper: BoundExpression,
    options: BetweenOptions,
) -> BoundExpression {
    Between
        .try_new_bound_expr(options, [arr, lower, upper])
        .vortex_expect("between expressions require compatible operand dtypes")
}

/// Creates a bound expression that selects specific fields from a struct expression.
pub fn select(field_names: impl Into<FieldNames>, child: BoundExpression) -> BoundExpression {
    Select
        .try_new_bound_expr(FieldSelection::Include(field_names.into()), [child])
        .vortex_expect("select expressions require fields from a struct child")
}

/// Creates a bound expression that excludes specific fields from a struct expression.
pub fn select_exclude(
    fields: impl Into<FieldNames>,
    child: BoundExpression,
) -> BoundExpression {
    Select
        .try_new_bound_expr(FieldSelection::Exclude(fields.into()), [child])
        .vortex_expect("select expressions require fields from a struct child")
}

/// Creates a bound expression that packs values into a struct with named fields.
pub fn pack(
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

/// Creates a bound expression that casts values to a target dtype.
pub fn cast(child: BoundExpression, target: DType) -> BoundExpression {
    Cast.try_new_bound_expr(target, [child])
        .vortex_expect("cast expressions require a supported source and target dtype")
}

/// Creates a bound expression that replaces null values with a fill value.
pub fn fill_null(child: BoundExpression, fill_value: BoundExpression) -> BoundExpression {
    FillNull
        .try_new_bound_expr(EmptyOptions, [child, fill_value])
        .vortex_expect("fill-null expressions require compatible child and fill dtypes")
}

/// Creates a bound expression that checks for null values.
pub fn is_null(child: BoundExpression) -> BoundExpression {
    IsNull
        .try_new_bound_expr(EmptyOptions, [child])
        .vortex_expect("is-null expressions are always well-typed")
}

/// Creates a bound expression that checks for NaN values.
pub fn is_nan(child: BoundExpression) -> BoundExpression {
    IsNan
        .try_new_bound_expr(EmptyOptions, [child])
        .vortex_expect("is-nan expressions are always well-typed")
}

/// Creates a bound expression that checks for non-null values.
pub fn is_not_null(child: BoundExpression) -> BoundExpression {
    IsNotNull
        .try_new_bound_expr(EmptyOptions, [child])
        .vortex_expect("is-not-null expressions are always well-typed")
}

/// Creates a bound SQL LIKE expression.
pub fn like(child: BoundExpression, pattern: BoundExpression) -> BoundExpression {
    like_with_options(child, pattern, false, false)
}

/// Creates a bound case-insensitive SQL ILIKE expression.
pub fn ilike(child: BoundExpression, pattern: BoundExpression) -> BoundExpression {
    like_with_options(child, pattern, false, true)
}

/// Creates a bound negated SQL NOT LIKE expression.
pub fn not_like(child: BoundExpression, pattern: BoundExpression) -> BoundExpression {
    like_with_options(child, pattern, true, false)
}

/// Creates a bound negated case-insensitive SQL NOT ILIKE expression.
pub fn not_ilike(child: BoundExpression, pattern: BoundExpression) -> BoundExpression {
    like_with_options(child, pattern, true, true)
}

fn like_with_options(
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

/// Creates a bound mask expression.
pub fn mask(array: BoundExpression, mask: BoundExpression) -> BoundExpression {
    Mask.try_new_bound_expr(EmptyOptions, [array, mask])
        .vortex_expect("mask expressions require a boolean mask")
}

/// Creates a bound expression that merges struct expressions.
pub fn merge(elements: impl IntoIterator<Item = BoundExpression>) -> BoundExpression {
    merge_opts(elements, DuplicateHandling::default())
}

/// Creates a bound merge expression with explicit duplicate handling.
pub fn merge_opts(
    elements: impl IntoIterator<Item = BoundExpression>,
    duplicate_handling: DuplicateHandling,
) -> BoundExpression {
    Merge
        .try_new_bound_expr(duplicate_handling, elements)
        .vortex_expect("merge expressions require non-nullable struct children")
}

/// Creates a bound zip expression that conditionally selects between two arrays.
pub fn zip_expr(
    mask: BoundExpression,
    if_true: BoundExpression,
    if_false: BoundExpression,
) -> BoundExpression {
    Zip.try_new_bound_expr(EmptyOptions, [if_true, if_false, mask])
        .vortex_expect("zip expressions require a boolean mask and compatible value dtypes")
}

/// Creates a bound dynamic comparison expression from its complete options.
pub fn dynamic_with_options(
    options: DynamicComparisonExpr,
    lhs: BoundExpression,
) -> BoundExpression {
    DynamicComparison
        .try_new_bound_expr(options, [lhs])
        .vortex_expect("dynamic comparisons require a compatible left-hand dtype")
}

/// Creates a bound dynamic comparison expression.
pub fn dynamic(
    operator: CompareOperator,
    rhs_value: impl Fn() -> Option<ScalarValue> + Send + Sync + 'static,
    rhs_dtype: DType,
    default: bool,
    lhs: BoundExpression,
) -> BoundExpression {
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

/// Creates a bound expression that checks if a value is contained in a list.
pub fn list_contains(list: BoundExpression, value: BoundExpression) -> BoundExpression {
    ListContains
        .try_new_bound_expr(EmptyOptions, [list, value])
        .vortex_expect("list-contains expressions require a compatible list and value dtype")
}

/// Creates a bound expression that computes each element's byte length.
pub fn byte_length(input: BoundExpression) -> BoundExpression {
    ByteLength
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("byte-length expressions require a variable-length binary child")
}

/// Creates a bound expression that extracts an extension array's storage values.
pub fn ext_storage(input: BoundExpression) -> BoundExpression {
    ExtStorage
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("extension-storage expressions require an extension child")
}

/// Creates a bound expression that computes the number of elements in each list.
pub fn list_length(input: BoundExpression) -> BoundExpression {
    ListLength
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("list-length expressions require a list child")
}

/// Creates a bound expression that sums the elements of each list.
pub fn list_sum(input: BoundExpression) -> BoundExpression {
    ListSum
        .try_new_bound_expr(NumericalAggregateOpts::default(), [input])
        .vortex_expect("list-sum expressions require a numeric list child")
}

/// Creates a bound list-sum expression with explicit aggregate options.
pub fn list_sum_opts(
    input: BoundExpression,
    options: NumericalAggregateOpts,
) -> BoundExpression {
    ListSum
        .try_new_bound_expr(options, [input])
        .vortex_expect("list-sum expressions require a numeric list child")
}
