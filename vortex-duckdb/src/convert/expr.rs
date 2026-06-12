// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use tracing::debug;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::expr::BoundExpr;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::expr::try_get_item;
use vortex::scalar::Scalar;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::Between;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::is_not_null::IsNotNull;
use vortex::scalar_fn::fns::is_null::IsNull;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::list_contains::ListContains;
use vortex::scalar_fn::fns::not::Not;
use vortex::scalar_fn::fns::operators::Operator;

use super::collect_binary;
use super::try_list_scalar;
use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb;
use crate::duckdb::BoundFunction;
use crate::duckdb::BoundOperator;
use crate::duckdb::ExpressionClass;
use crate::duckdb::ExpressionClass::BoundBetween;
use crate::duckdb::ExpressionClass::BoundColumnRef;
use crate::duckdb::ExpressionClass::BoundComparison;
use crate::duckdb::ExpressionClass::BoundConjunction;
use crate::duckdb::ExpressionClass::BoundConstant;
use crate::duckdb::ExpressionClass::BoundRef;

fn from_bound_str(value: &duckdb::ExpressionRef) -> VortexResult<String> {
    // Engine input: an unrecognized expression class must convert to a clean error, never a
    // panic that unwinds across the DuckDB FFI boundary.
    match value.as_class() {
        Some(BoundConstant(constant)) => Ok(constant.value.as_string().as_str().to_owned()),
        Some(_) | None => {
            vortex_bail!("Expected string constant, got {:?}", value.as_class_id())
        }
    }
}

fn try_from_bound_function(
    func: &BoundFunction,
    col_sub: Option<&BoundExpr>,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    let expr = match func.scalar_function.name() {
        "struct_extract" => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(child) = try_from_expression_inner(children[0], col_sub, scope_dtype)? else {
                return Ok(None);
            };
            let field = from_bound_str(children[1])?;
            try_get_item(field, child)?
        }
        like @ ("~~" | "!~~") => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(string) = try_from_expression_inner(children[0], col_sub, scope_dtype)? else {
                return Ok(None);
            };
            let Some(target) = try_from_expression_inner(children[1], col_sub, scope_dtype)? else {
                return Ok(None);
            };
            let opts = LikeOptions {
                negated: like == "!~~",
                case_insensitive: false,
            };
            Like.try_new_expr(opts, [string, target])?
        }
        matchers @ ("contains" | "prefix" | "suffix") => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(value) = try_from_expression_inner(children[0], col_sub, scope_dtype)? else {
                return Ok(None);
            };
            let pattern = from_bound_str(children[1])?;
            let pattern = match matchers {
                "contains" => format!("%{pattern}%"),
                "prefix" => format!("{pattern}%"),
                "suffix" => format!("%{pattern}"),
                _ => unreachable!(),
            };
            Like.try_new_expr(LikeOptions::default(), [value, lit(pattern)])?
        }
        _ => {
            debug!("bound function {}", func.scalar_function.name());
            return Ok(None);
        }
    };

    Ok(Some(expr))
}

pub fn try_from_bound_expression(
    value: &duckdb::ExpressionRef,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    try_from_expression_inner(value, None, scope_dtype)
}

pub(super) fn try_from_bound_expression_with_col_sub(
    value: &duckdb::ExpressionRef,
    col_sub: &BoundExpr,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    try_from_expression_inner(value, Some(col_sub), scope_dtype)
}

/*
 * Called before pushdown_complex_filter or a table filter expression call.
 * As we support complex filter pushdown, Duckdb pushes expressions to Vortex.
 * However, it doesn't know what type of expressions we can handle. Here we list
 * all expressions that are quaranteed to be converted to Vortex expressions.
 *
 * If we return true here, and expression is in the list for
 * pushdown_complex_filter, we must handle it, or query engine will break.
 *
 * Example: we don't support substr() expression so we tell Duckdb we can't
 * push it.
 * Example: optional filters may fail to parse on our side (we return
 * Ok(None)), so we don't allow pushing these.
 */
pub fn can_push_expression(value: &duckdb::ExpressionRef) -> bool {
    let Some(value) = value.as_class() else {
        return false;
    };
    match value {
        BoundColumnRef(_) => true,
        BoundConstant(_) => true,
        BoundRef => true,
        BoundComparison(comp) => can_push_expression(comp.left) && can_push_expression(comp.right),
        BoundBetween(between) => {
            can_push_expression(between.input)
                && can_push_expression(between.lower)
                && can_push_expression(between.upper)
        }
        BoundConjunction(conj) => conj.children().all(can_push_expression),
        ExpressionClass::BoundFunction(func) => {
            match func.scalar_function.name() {
                // These read their second child as a constant string (`from_bound_str`), so
                // only admit them when that child actually is a bound constant — otherwise
                // conversion would fail after we promised DuckDB we could handle the filter.
                "struct_extract" | "contains" | "prefix" | "suffix" => {
                    let children: Vec<_> = func.children().collect();
                    children.len() == 2 && matches!(children[1].as_class(), Some(BoundConstant(_)))
                }
                "~~" | "!~~" => true,
                _ => false,
            }
        }
        ExpressionClass::BoundOperator(op) => {
            if !matches!(
                op.op,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT
                    | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL
                    | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL
                    | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN
                    | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_IN
            ) {
                return false;
            }
            op.children().all(can_push_expression)
        }
    }
}

// If you want to add support for other expressions, also change
// can_push_expression
fn try_from_expression_inner(
    value: &duckdb::ExpressionRef,
    col_sub: Option<&BoundExpr>,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    let Some(value) = value.as_class() else {
        debug!(
            class_id = ?value.as_class_id(),
            "unknown expression class id"
        );
        return Ok(None);
    };
    Ok(Some(match value {
        BoundRef => {
            let Some(col) = col_sub else {
                vortex_bail!("BoundRef requested but no column supplied");
            };
            col.clone()
        }
        BoundColumnRef(col_ref) => try_col(col_ref.name.as_ref(), scope_dtype)?,
        BoundConstant(const_) => lit(Scalar::try_from(const_.value)?),
        BoundComparison(compare) => {
            return try_from_bound_comparison(compare, col_sub, scope_dtype);
        }
        BoundBetween(between) => {
            return try_from_bound_between(between, col_sub, scope_dtype);
        }
        ExpressionClass::BoundOperator(operator) => {
            return try_from_bound_operator(operator, col_sub, scope_dtype);
        }
        ExpressionClass::BoundFunction(func) => {
            return try_from_bound_function(&func, col_sub, scope_dtype);
        }
        BoundConjunction(conj) => {
            return try_from_bound_conjunction(conj, col_sub, scope_dtype);
        }
    }))
}

fn try_from_bound_comparison(
    compare: duckdb::BoundComparison<'_>,
    col_sub: Option<&BoundExpr>,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    let operator: Operator = compare.op.try_into()?;
    let Some(left) = try_from_expression_inner(compare.left, col_sub, scope_dtype)? else {
        return Ok(None);
    };
    let Some(right) = try_from_expression_inner(compare.right, col_sub, scope_dtype)? else {
        return Ok(None);
    };
    Ok(Some(Binary.try_new_expr(operator, [left, right])?))
}

fn try_from_bound_between(
    between: duckdb::BoundBetween<'_>,
    col_sub: Option<&BoundExpr>,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    let Some(array) = try_from_expression_inner(between.input, col_sub, scope_dtype)? else {
        return Ok(None);
    };
    let Some(lower) = try_from_expression_inner(between.lower, col_sub, scope_dtype)? else {
        return Ok(None);
    };
    let Some(upper) = try_from_expression_inner(between.upper, col_sub, scope_dtype)? else {
        return Ok(None);
    };
    Ok(Some(Between.try_new_expr(
        BetweenOptions {
            lower_strict: if between.lower_inclusive {
                StrictComparison::NonStrict
            } else {
                StrictComparison::Strict
            },
            upper_strict: if between.upper_inclusive {
                StrictComparison::NonStrict
            } else {
                StrictComparison::Strict
            },
        },
        [array, lower, upper],
    )?))
}

fn try_from_bound_operator(
    operator: BoundOperator<'_>,
    col_sub: Option<&BoundExpr>,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    match operator.op {
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT
        | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL
        | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL => {
            try_from_unary_operator(operator, col_sub, scope_dtype)
        }
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN => {
            try_from_compare_in(operator, col_sub, scope_dtype, false)
        }
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_IN => {
            try_from_compare_in(operator, col_sub, scope_dtype, true)
        }
        _ => {
            debug!(op=?operator.op, "cannot push down operator");
            Ok(None)
        }
    }
}

fn try_from_unary_operator(
    operator: BoundOperator<'_>,
    col_sub: Option<&BoundExpr>,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    let children: Vec<_> = operator.children().collect();
    vortex_ensure!(children.len() == 1);
    let Some(child) = try_from_expression_inner(children[0], col_sub, scope_dtype)? else {
        return Ok(None);
    };
    Ok(Some(match operator.op {
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT => {
            Not.try_new_expr(EmptyOptions, [child])?
        }
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL => {
            IsNull.try_new_expr(EmptyOptions, [child])?
        }
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL => {
            IsNotNull.try_new_expr(EmptyOptions, [child])?
        }
        _ => unreachable!(),
    }))
}

fn try_from_bound_conjunction(
    conj: duckdb::BoundConjunction<'_>,
    col_sub: Option<&BoundExpr>,
    scope_dtype: &DType,
) -> VortexResult<Option<BoundExpr>> {
    let Some(children) = conj
        .children()
        .map(|c| try_from_expression_inner(c, col_sub, scope_dtype))
        .collect::<VortexResult<Option<Vec<_>>>>()?
    else {
        return Ok(None);
    };
    match conj.op {
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_CONJUNCTION_AND => Ok(Some(
            collect_binary(children, Operator::And)?.vortex_expect("cannot be empty"),
        )),
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_CONJUNCTION_OR => Ok(Some(
            collect_binary(children, Operator::Or)?.vortex_expect("cannot be empty"),
        )),
        _ => vortex_bail!("unexpected operator {:?} in bound conjunction", conj.op),
    }
}

fn try_from_compare_in(
    operator: BoundOperator,
    col_sub: Option<&BoundExpr>,
    scope_dtype: &DType,
    not_in: bool,
) -> VortexResult<Option<BoundExpr>> {
    // First child is element, rest form the list.
    let children: Vec<_> = operator.children().collect();
    vortex_ensure!(
        children.len() >= 2,
        "IN expression must have at least one value"
    );
    let Some(element) = try_from_expression_inner(children[0], col_sub, scope_dtype)? else {
        return Ok(None);
    };

    let Some(list_elements) = children
        .iter()
        .skip(1)
        .map(|c| {
            let Some(value) = try_from_expression_inner(c, col_sub, scope_dtype)? else {
                return Ok(None);
            };
            Ok(Some(
                value
                    .as_literal()
                    .ok_or_else(|| vortex_err!("cannot have a non literal in a in_list"))?
                    .clone(),
            ))
        })
        .collect::<VortexResult<Option<Vec<_>>>>()?
    else {
        return Ok(None);
    };
    let list = try_list_scalar(list_elements)?;

    let expr = ListContains.try_new_expr(EmptyOptions, [lit(list), element])?;
    Ok(Some(if not_in {
        Not.try_new_expr(EmptyOptions, [expr])?
    } else {
        expr
    }))
}

fn try_col(field: impl Into<FieldName>, scope_dtype: &DType) -> VortexResult<BoundExpr> {
    try_get_item(field, root(scope_dtype.clone()))
}

impl TryFrom<DUCKDB_VX_EXPR_TYPE> for Operator {
    type Error = VortexError;

    fn try_from(value: DUCKDB_VX_EXPR_TYPE) -> VortexResult<Self> {
        Ok(match value {
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID => vortex_bail!("invalid expr"),
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL => Operator::Eq,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL => Operator::NotEq,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => Operator::Lt,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => Operator::Gt,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => Operator::Lte,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => Operator::Gte,
            _ => todo!("cannot convert {:?}", value),
        })
    }
}
