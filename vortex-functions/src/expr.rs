// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression constructors for this crate's scalar functions.
//!
//! These complement the constructors in `vortex_expr`.

use std::sync::Arc;

use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_expr::BoundExpression;
use vortex_expr::Expression;
use vortex_expr::ScalarFnExprExt;
use vortex_expr::lit;

use crate::between::Between;
use crate::between::BetweenOptions;
use crate::between::StrictComparison;
use crate::byte_length::ByteLength;
use crate::case_when::CaseWhen;
use crate::case_when::CaseWhenOptions;
use crate::dynamic::DynamicComparison;
use crate::dynamic::DynamicComparisonExpr;
use crate::dynamic::Rhs;
use crate::ext_storage::ExtStorage;
use crate::is_nan::IsNan;
use crate::like::Like;
use crate::like::LikeOptions;
use crate::list_contains::ListContains;
use crate::list_contains::ListContainsOptions;
use crate::list_length::ListLength;
use crate::list_sum::ListSum;
use crate::merge::DuplicateHandling;
use crate::merge::Merge;
use crate::replace_time_zone::ReplaceTimeZone;
use crate::replace_time_zone::ReplaceTimeZoneOptions;
use crate::variant_get::VariantGet;
use crate::variant_get::VariantGetOptions;
use crate::variant_get::VariantPath;
use crate::zip::Zip;

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

/// Creates an expression that merges struct expressions into a single struct.
///
/// Combines fields from all input expressions. If field names are duplicated,
/// later expressions win. Fields are not recursively merged.
///
/// ```rust
/// # use vortex_array::dtype::Nullability;
/// # use vortex_functions::expr::merge;
/// # use vortex_expr::{get_item, root};
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

/// Creates an expression that computes the byte length of each element.
/// This is akin to ANSI SQL OCTET_LENGTH(), or DuckDB's strlen().
///
/// ```rust
/// # use vortex_functions::expr::byte_length;
/// # use vortex_expr::root;
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

/// Creates an expression that extracts the storage values from an extension array.
///
/// ```rust
/// # use vortex_functions::expr::ext_storage;
/// # use vortex_expr::root;
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

/// Creates an expression that computes the number of elements in each list
/// for `List` and `FixedSizeList` inputs. This is akin to ANSI SQL `CARDINALITY()`,
/// or DuckDB's `len()`/`array_length()`.
///
/// ```rust
/// # use vortex_functions::expr::list_length;
/// # use vortex_expr::root;
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

/// Creates an expression that sums the elements of each list for `List` and
/// `FixedSizeList` inputs, akin to DuckDB's `list_sum()`.
///
/// Follows SQL `SUM` semantics per list: null lists, empty lists, and lists whose elements are
/// all null yield null; null elements are skipped; integer and decimal overflow yields a null
/// value. The result dtype follows `sum`'s widening rules and is always nullable. NaN float
/// elements are skipped by default; see [`list_sum_opts`] for the NaN-including variant.
///
/// ```rust
/// # use vortex_functions::expr::list_sum;
/// # use vortex_expr::root;
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

/// Reinterpret timestamp wall times in `options.time_zone`, preserving the time unit.
///
/// `ambiguous` is a UTF-8 expression containing `raise`, `earliest`, `latest`, or `null`.
/// Removing the timezone preserves wall time in a timezone-naive timestamp.
pub fn replace_time_zone(
    input: Expression,
    ambiguous: Expression,
    options: ReplaceTimeZoneOptions,
) -> Expression {
    ReplaceTimeZone.new_expr(options, [input, ambiguous])
}

/// Create a bound timezone replacement expression.
pub fn bound_replace_time_zone(
    input: BoundExpression,
    ambiguous: BoundExpression,
    options: ReplaceTimeZoneOptions,
) -> BoundExpression {
    ReplaceTimeZone
        .try_new_bound_expr(options, [input, ambiguous])
        .vortex_expect("timezone replacement requires a timestamp and a UTF-8 ambiguity policy")
}

/// Constructors for expressions whose children have already been bound and type-checked.
///
/// These mirror the constructors in [`crate::expr`] and panic when the supplied children do not
/// form a well-typed expression.
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

// ---- Between ----

/// Creates an expression that checks if values are between two bounds.
///
/// Returns a boolean array indicating which values fall within the specified range.
/// The comparison strictness is controlled by the options parameter.
///
/// ```rust
/// # use vortex_functions::between::BetweenOptions;
/// # use vortex_functions::between::StrictComparison;
/// # use vortex_expr::{lit, root};
/// # use vortex_functions::expr::between;
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

// ---- ListContains ----

/// Creates an expression that checks whether a list contains a value.
///
/// A null element never matches anything: a needle that matches no element yields `false`. For
/// SQL's three-valued `IN`, where a null element makes that `null`, use [`in_list`].
///
/// ```rust
/// # use vortex_expr::{lit, root};
/// # use vortex_functions::expr::list_contains;
/// let expr = list_contains(root(), lit(42));
/// ```
pub fn list_contains(list: Expression, value: Expression) -> Expression {
    list_contains_opts(list, value, ListContainsOptions::default())
}

/// Creates an expression that checks whether a list contains a value, with explicit
/// [`ListContainsOptions`].
pub fn list_contains_opts(
    list: Expression,
    value: Expression,
    options: ListContainsOptions,
) -> Expression {
    ListContains.new_expr(options, [list, value])
}

/// Creates `needle IN (list)` with SQL null semantics: a null element is an unknown value, so a
/// needle that matches no element yields `null` rather than `false` when the list holds one, and
/// `not(in_list(..))` never admits such a row.
///
/// ```rust
/// # use vortex_expr::{lit, root};
/// # use vortex_functions::expr::in_list;
/// let expr = in_list(root(), lit(vec![1, 2, 3]));
/// ```
pub fn in_list(needle: Expression, list: Expression) -> Expression {
    list_contains_opts(
        list,
        needle,
        ListContainsOptions {
            sql_null_semantics: true,
        },
    )
}

/// Creates `needle IN (elements)` with SQL null semantics for a list of constant elements of
/// `element_dtype`, ignoring nullability. An element can also be an untyped null.
///
/// The nullability of `element_dtype` is not used: the list elements are nullable only when a
/// null element is present.
///
/// Non-null integers that hold every value between their smallest and largest become a range
/// check, which is cheaper to evaluate and prunes with tighter bounds than a set lookup, as for
/// the keys of a small dimension table that a join pushes down.
///
/// # Errors
///
/// Returns an error if an element is not a null and not of `element_dtype`.
///
/// ```rust
/// # use vortex_array::dtype::{DType, Nullability, PType};
/// # use vortex_expr::root;
/// # use vortex_functions::expr::in_list_of;
/// let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
/// let expr = in_list_of(root(), dtype, vec![3i32.into(), 1i32.into(), 2i32.into()])?;
/// assert_eq!(expr.to_string(), "(1i32 <= $ <= 3i32)");
/// # Ok::<(), vortex_error::VortexError>(())
/// ```
pub fn in_list_of(
    needle: Expression,
    element_dtype: DType,
    elements: Vec<Scalar>,
) -> VortexResult<Expression> {
    let element_dtype = if elements.iter().any(Scalar::is_null) {
        element_dtype.as_nullable()
    } else {
        element_dtype.as_nonnullable()
    };
    let mut elements = elements
        .iter()
        .map(|element| {
            vortex_ensure!(
                element.dtype() == &DType::Null
                    || element.dtype().eq_ignore_nullability(&element_dtype),
                "IN list element {element} is not of type {element_dtype}"
            );
            element.cast(&element_dtype)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    if let Some((low, high)) = contiguous_integer_bounds(&elements) {
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };
        return Ok(between(
            needle,
            lit(low.clone()),
            lit(high.clone()),
            options,
        ));
    }

    // Ordered elements make equal lists from the same set, so that equal filters deduplicate.
    if elements
        .windows(2)
        .all(|pair| pair[0].partial_cmp(&pair[1]).is_some())
    {
        elements.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    let list = Scalar::list(Arc::new(element_dtype), elements, Nullability::NonNullable);
    Ok(in_list(needle, lit(list)))
}

/// The smallest and largest of non-null integer `scalars` when they hold every integer between
/// those two.
fn contiguous_integer_bounds(scalars: &[Scalar]) -> Option<(&Scalar, &Scalar)> {
    let mut values = scalars
        .iter()
        .map(|scalar| {
            let primitive = scalar.as_primitive_opt()?;
            if !primitive.ptype().is_int() {
                return None;
            }
            let value = if primitive.ptype().is_unsigned_int() {
                primitive.as_opt::<u64>()??.into()
            } else {
                i128::from(primitive.as_opt::<i64>()??)
            };
            Some((value, scalar))
        })
        .collect::<Option<Vec<(i128, &Scalar)>>>()?;
    values.sort_unstable_by_key(|(value, _)| *value);
    values.dedup_by_key(|(value, _)| *value);
    let (low, low_scalar) = values.first()?;
    let (high, high_scalar) = values.last()?;
    (high - low + 1 == i128::try_from(values.len()).ok()?).then_some((*low_scalar, *high_scalar))
}

/// Creates a bound expression that checks whether a list contains a value.
pub fn bound_list_contains(list: BoundExpression, value: BoundExpression) -> BoundExpression {
    ListContains
        .try_new_bound_expr(ListContainsOptions::default(), [list, value])
        .vortex_expect("list-contains expressions require a list child")
}

/// Creates a bound `needle IN (list)` with SQL null semantics. See [`in_list`].
pub fn bound_in_list(needle: BoundExpression, list: BoundExpression) -> BoundExpression {
    ListContains
        .try_new_bound_expr(
            ListContainsOptions {
                sql_null_semantics: true,
            },
            [list, needle],
        )
        .vortex_expect("list-contains expressions require a list child")
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

// ---- IsNan ----

/// Creates an expression that checks for NaN values.
///
/// The expression is strict: null inputs produce null outputs, so the output nullability
/// follows the input. Only primitive float inputs are supported.
///
/// ```rust
/// # use vortex_expr::root;
/// # use vortex_functions::expr::is_nan;
/// let expr = is_nan(root());
/// ```
pub fn is_nan(child: Expression) -> Expression {
    IsNan.new_expr(EmptyOptions, vec![child])
}

/// Creates a bound expression that checks for NaN values.
pub fn bound_is_nan(child: BoundExpression) -> BoundExpression {
    IsNan
        .try_new_bound_expr(EmptyOptions, [child])
        .vortex_expect("is-nan expressions are always well-typed")
}

// ---- Zip ----

/// Creates a zip expression that conditionally selects between two arrays.
///
/// ```rust
/// # use vortex_expr::{lit, root};
/// # use vortex_functions::expr::zip_expr;
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

pub mod bound {
    pub use super::bound_between as between;
    pub use super::bound_byte_length as byte_length;
    pub use super::bound_case_when as case_when;
    pub use super::bound_case_when_no_else as case_when_no_else;
    pub use super::bound_dynamic as dynamic;
    pub use super::bound_dynamic_with_options as dynamic_with_options;
    pub use super::bound_ext_storage as ext_storage;
    pub use super::bound_ilike as ilike;
    pub use super::bound_in_list as in_list;
    pub use super::bound_is_nan as is_nan;
    pub use super::bound_like as like;
    pub use super::bound_list_contains as list_contains;
    pub use super::bound_list_length as list_length;
    pub use super::bound_list_sum as list_sum;
    pub use super::bound_list_sum_opts as list_sum_opts;
    pub use super::bound_merge as merge;
    pub use super::bound_merge_opts as merge_opts;
    pub use super::bound_nested_case_when as nested_case_when;
    pub use super::bound_not_ilike as not_ilike;
    pub use super::bound_not_like as not_like;
    pub use super::bound_replace_time_zone as replace_time_zone;
    pub use super::bound_variant_get as variant_get;
    pub use super::bound_zip_expr as zip_expr;
}
