// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use tracing::debug;
use vortex::aggregate_fn::Accumulator;
use vortex::aggregate_fn::DynAccumulator;
use vortex::aggregate_fn::EmptyOptions;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::combined::PairOptions;
use vortex::aggregate_fn::fns::first::First;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::mean::Mean;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::sum::Sum;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::byte_length;
use vortex::expr::cast;
use vortex::expr::col;
use vortex::expr::get_item;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::list_length;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::Between;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;

use crate::cpp::DUCKDB_TYPE;
use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::BoundFunction;
use crate::duckdb::BoundOperator;
use crate::duckdb::ExpressionClass;
use crate::duckdb::ExpressionClass::BoundBetween;
use crate::duckdb::ExpressionClass::BoundColumnRef;
use crate::duckdb::ExpressionClass::BoundComparison;
use crate::duckdb::ExpressionClass::BoundConjunction;
use crate::duckdb::ExpressionClass::BoundConstant;
use crate::duckdb::ExpressionClass::BoundRef;
use crate::duckdb::ExpressionRef;
use crate::projection::DuckdbField;

fn from_bound_str(value: &ExpressionRef) -> VortexResult<String> {
    match value.as_class().vortex_expect("unknown class") {
        BoundConstant(constant) => Ok(constant.value.as_string().as_str().to_owned()),
        _ => vortex_bail!("Expected string expression, got {:?}", value.as_class_id()),
    }
}

/// Whether the expression's return type is a `LIST` or fixed-size `ARRAY`.
fn returns_a_list(expr: &ExpressionRef) -> bool {
    matches!(
        expr.return_type().as_type_id(),
        DUCKDB_TYPE::DUCKDB_TYPE_LIST | DUCKDB_TYPE::DUCKDB_TYPE_ARRAY
    )
}

/// Wrap `expr` in `list_length`. Since vortex `list_length` returns u64 but duckdb equivalents
/// return i64, we must cast as well.
fn build_list_length(expr: Expression, nullability: Nullability) -> Expression {
    cast(list_length(expr), DType::Primitive(PType::I64, nullability))
}

fn try_from_bound_function(
    func: &BoundFunction,
    col_sub: Option<&Expression>,
) -> VortexResult<Option<Expression>> {
    let expr = match func.scalar_function.name() {
        "strlen" => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 1);
            let Some(col) = try_from_expression_inner(children[0], col_sub)? else {
                return Ok(None);
            };
            let col = byte_length(col);
            // byte_length returns u64, strlen expects i64.
            // At this point we don't know column's dtype so we ultimately
            // set it to be nullable. For non-nullable column the nullability
            // will be AllValid so it's a marginal cost.
            let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
            cast(col, dtype)
        }
        "struct_extract" => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(child) = try_from_expression_inner(children[0], col_sub)? else {
                return Ok(None);
            };
            let field = from_bound_str(children[1])?;
            get_item(field, child)
        }
        like @ ("~~" | "!~~") => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(string) = try_from_expression_inner(children[0], col_sub)? else {
                return Ok(None);
            };
            let Some(target) = try_from_expression_inner(children[1], col_sub)? else {
                return Ok(None);
            };
            let opts = LikeOptions {
                negated: like == "!~~",
                case_insensitive: false,
            };
            Like.new_expr(opts, [string, target])
        }
        matchers @ ("contains" | "prefix" | "suffix") => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(value) = try_from_expression_inner(children[0], col_sub)? else {
                return Ok(None);
            };
            let pattern = from_bound_str(children[1])?;
            let pattern = match matchers {
                "contains" => format!("%{pattern}%"),
                "prefix" => format!("{pattern}%"),
                "suffix" => format!("%{pattern}"),
                _ => unreachable!(),
            };
            Like.new_expr(LikeOptions::default(), [value, lit(pattern)])
        }
        "array_length" => {
            let children = func.children().collect::<Vec<_>>();
            // Only accept array_length(expr) rather than array_length(expr, dim).
            if children.len() != 1 {
                return Ok(None);
            }
            let Some(col) = try_from_expression_inner(children[0], col_sub)? else {
                return Ok(None);
            };

            // We don't know the column's nullability here, so we set it to nullable.
            build_list_length(col, Nullability::Nullable)
        }
        // len/length semantics depend on the return type of underlying expr.
        "len" | "length" => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 1);
            let child = children[0];

            if returns_a_list(child) {
                let Some(col) = try_from_expression_inner(child, col_sub)? else {
                    return Ok(None);
                };

                // Same nullability rationale as in "array_length" branch.
                let list_len_expr = build_list_length(col, Nullability::Nullable);
                return Ok(Some(list_len_expr));
            } else {
                return Ok(None);
            }
        }
        _ => {
            debug!("bound function {}", func.scalar_function.name());
            return Ok(None);
        }
    };

    Ok(Some(expr))
}

pub fn try_from_bound_expression(value: &ExpressionRef) -> VortexResult<Option<Expression>> {
    try_from_expression_inner(value, None)
}

pub(super) fn try_from_bound_expression_with_col_sub(
    value: &ExpressionRef,
    col_sub: &Expression,
) -> VortexResult<Option<Expression>> {
    try_from_expression_inner(value, Some(col_sub))
}

fn is_supported_length_alias(func: &BoundFunction) -> bool {
    let children: Vec<_> = func.children().collect();
    children.len() == 1 && returns_a_list(children[0])
}

// Called before pushdown_complex_filter or a table filter expression call.
// As we support complex filter pushdown, Duckdb pushes expressions to Vortex.
// However, it doesn't know what type of expressions we can handle. Here we list
// all expressions that are quaranteed to be converted to Vortex expressions.
//
// If we return true here, and expression is in the list for
// pushdown_complex_filter, we must handle it, or query engine will break.
//
// Example: we don't support substr() expression so we tell Duckdb we can't
// push it.
// Example: optional filters may fail to parse on our side (we return
// Ok(None)), so we don't allow pushing these.
pub fn can_push_expression(value: &ExpressionRef) -> bool {
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
            let name = func.scalar_function.name();
            name == "struct_extract"
                || name == "contains"
                || name == "prefix"
                || name == "suffix"
                || name == "~~"
                || name == "!~~"
                || name == "strlen"
                || name == "array_length"
                || (matches!(name, "len" | "length") && is_supported_length_alias(&func))
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
        ExpressionClass::BoundAggregate(_) => false,
    }
}

/// Applies `list_length` expression to a duckdb field
fn list_length_on_field(field: &DuckdbField) -> Expression {
    let col = get_item(field.name.as_str(), root());

    build_list_length(col, field.dtype.nullability())
}

pub fn try_from_projection_expression(
    value: &ExpressionRef,
    field: &DuckdbField,
) -> VortexResult<Option<Expression>> {
    let Some(value) = value.as_class() else {
        return Ok(None);
    };
    let ExpressionClass::BoundFunction(func) = value else {
        return Ok(None);
    };
    Ok(match func.scalar_function.name() {
        "strlen" => {
            let col = byte_length(get_item(field.name.as_str(), root()));
            // byte_length returns u64, strlen expects i64
            let dtype = DType::Primitive(PType::I64, field.dtype.nullability());
            let col = cast(col, dtype);
            Some(col)
        }
        "array_length" => {
            // Only accept array_length(expr) rather than array_length(expr, dim).
            (func.children().count() == 1).then(|| list_length_on_field(field))
        }
        // len/length have different semantics depending on field dtype.
        "len" | "length" => matches!(field.dtype, DType::List(..) | DType::FixedSizeList(..))
            .then(|| list_length_on_field(field)),
        _ => None,
    })
}

/// A pushed-down aggregate kind. Plain data so it can be stored in [`DuckdbField`] and the
/// bind data (which must be `Clone`/`Send`) without carrying a `dyn DynAccumulator`. The
/// actual accumulators are built from it via [`PushedAggregate::build`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushedAggregate {
    Min,
    Max,
    Sum,
    Mean,
    AnyValue,
}

impl PushedAggregate {
    /// Builds a fresh, empty accumulator for this aggregate over `dtype`.
    pub fn build(self, dtype: DType) -> VortexResult<Box<dyn DynAccumulator>> {
        let opts = NumericalAggregateOpts::default();
        Ok(match self {
            Self::Min => Box::new(Accumulator::try_new(Min, opts, dtype)?),
            Self::Max => Box::new(Accumulator::try_new(Max, opts, dtype)?),
            Self::Sum => Box::new(Accumulator::try_new(Sum, opts, dtype)?),
            Self::Mean => Box::new(Accumulator::try_new(
                Mean::combined(),
                PairOptions(opts, opts),
                dtype,
            )?),
            Self::AnyValue => Box::new(Accumulator::try_new(First, EmptyOptions, dtype)?),
        })
    }
}

pub fn try_from_projection_aggregate(
    expr: &ExpressionRef,
) -> VortexResult<Option<PushedAggregate>> {
    let Some(expr) = expr.as_class() else {
        return Ok(None);
    };
    let ExpressionClass::BoundAggregate(agg) = expr else {
        return Ok(None);
    };
    Ok(Some(match agg.aggregate_function.name() {
        "min" => PushedAggregate::Min,
        "max" => PushedAggregate::Max,
        "sum" | "sum_no_overflow" => PushedAggregate::Sum,
        "avg" | "mean" => PushedAggregate::Mean,
        "any_value" => PushedAggregate::AnyValue,
        _ => return Ok(None),
    }))
}

// If you want to add support for other expressions, also change
// can_push_expression
fn try_from_expression_inner(
    value: &ExpressionRef,
    col_sub: Option<&Expression>,
) -> VortexResult<Option<Expression>> {
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
        BoundColumnRef(col_ref) => col(col_ref.name.as_ref()),
        BoundConstant(const_) => lit(Scalar::try_from(const_.value)?),
        BoundComparison(compare) => {
            let operator: Operator = compare.op.try_into()?;

            let Some(left) = try_from_expression_inner(compare.left, col_sub)? else {
                return Ok(None);
            };
            let Some(right) = try_from_expression_inner(compare.right, col_sub)? else {
                return Ok(None);
            };

            Binary.new_expr(operator, [left, right])
        }
        BoundBetween(between) => {
            let Some(array) = try_from_expression_inner(between.input, col_sub)? else {
                return Ok(None);
            };
            let Some(lower) = try_from_expression_inner(between.lower, col_sub)? else {
                return Ok(None);
            };
            let Some(upper) = try_from_expression_inner(between.upper, col_sub)? else {
                return Ok(None);
            };
            Between.new_expr(
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
            )
        }
        ExpressionClass::BoundOperator(operator) => match operator.op {
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT
            | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL
            | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL => {
                let children: Vec<_> = operator.children().collect();
                vortex_ensure!(children.len() == 1);
                let Some(child) = try_from_expression_inner(children[0], col_sub)? else {
                    return Ok(None);
                };
                match operator.op {
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT => not(child),
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL => is_null(child),
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL => {
                        is_not_null(child)
                    }
                    _ => unreachable!(),
                }
            }
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN => {
                return try_from_compare_in(operator, col_sub, false);
            }
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_IN => {
                return try_from_compare_in(operator, col_sub, true);
            }
            _ => {
                debug!(op=?operator.op, "cannot push down operator");
                return Ok(None);
            }
        },
        ExpressionClass::BoundFunction(func) => {
            return try_from_bound_function(&func, col_sub);
        }
        BoundConjunction(conj) => {
            let Some(children) = conj
                .children()
                .map(|c| try_from_expression_inner(c, col_sub))
                .collect::<VortexResult<Option<Vec<_>>>>()?
            else {
                return Ok(None);
            };
            match conj.op {
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_CONJUNCTION_AND => {
                    and_collect(children).vortex_expect("cannot be empty")
                }
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_CONJUNCTION_OR => {
                    or_collect(children).vortex_expect("cannot be empty")
                }
                _ => vortex_bail!("unexpected operator {:?} in bound conjunction", conj.op),
            }
        }
        ExpressionClass::BoundAggregate(_) => return Ok(None),
    }))
}

fn try_from_compare_in(
    operator: BoundOperator,
    col_sub: Option<&Expression>,
    not_in: bool,
) -> VortexResult<Option<Expression>> {
    // First child is element, rest form the list.
    let children: Vec<_> = operator.children().collect();
    assert!(children.len() >= 2);
    let Some(element) = try_from_expression_inner(children[0], col_sub)? else {
        return Ok(None);
    };

    let Some(list_elements) = children
        .iter()
        .skip(1)
        .map(|c| {
            let Some(value) = try_from_expression_inner(c, col_sub)? else {
                return Ok(None);
            };
            Ok(Some(
                value
                    .as_opt::<Literal>()
                    .ok_or_else(|| vortex_err!("cannot have a non literal in a in_list"))?
                    .clone(),
            ))
        })
        .collect::<VortexResult<Option<Vec<_>>>>()?
    else {
        return Ok(None);
    };
    let list = Scalar::list(
        Arc::new(list_elements[0].dtype().clone()),
        list_elements,
        Nullability::Nullable,
    );

    let expr = list_contains(lit(list), element);
    Ok(Some(if not_in { not(expr) } else { expr }))
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
