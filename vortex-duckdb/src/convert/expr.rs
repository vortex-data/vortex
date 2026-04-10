// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use tracing::debug;
use vortex::dtype::Nullability;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::col;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::expr::or_collect;
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

use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb;

const DUCKDB_FUNCTION_NAME_CONTAINS: &str = "contains";

fn like_pattern_str(value: &duckdb::ExpressionRef) -> VortexResult<Option<String>> {
    match value.as_class().vortex_expect("unknown class") {
        duckdb::ExpressionClass::BoundConstant(constant) => {
            Ok(Some(format!("%{}%", constant.value.as_string().as_str())))
        }
        _ => Ok(None),
    }
}

pub fn try_from_bound_expression(
    value: &duckdb::ExpressionRef,
) -> VortexResult<Option<Expression>> {
    let Some(value) = value.as_class() else {
        tracing::debug!("no expression class id {:?}", value.as_class_id());
        return Ok(None);
    };
    Ok(Some(match value {
        duckdb::ExpressionClass::BoundColumnRef(col_ref) => col(col_ref.name.as_ref()),
        duckdb::ExpressionClass::BoundConstant(const_) => lit(Scalar::try_from(const_.value)?),
        duckdb::ExpressionClass::BoundComparison(compare) => {
            let operator: Operator = compare.op.try_into()?;

            let Some(left) = try_from_bound_expression(compare.left)? else {
                return Ok(None);
            };
            let Some(right) = try_from_bound_expression(compare.right)? else {
                return Ok(None);
            };

            Binary.new_expr(operator, [left, right])
        }
        duckdb::ExpressionClass::BoundBetween(between) => {
            let Some(array) = try_from_bound_expression(between.input)? else {
                return Ok(None);
            };
            let Some(lower) = try_from_bound_expression(between.lower)? else {
                return Ok(None);
            };
            let Some(upper) = try_from_bound_expression(between.upper)? else {
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
        duckdb::ExpressionClass::BoundOperator(operator) => match operator.op {
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT
            | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL
            | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL => {
                let children: Vec<_> = operator.children().collect();
                assert_eq!(children.len(), 1);
                let Some(child) = try_from_bound_expression(children[0])? else {
                    return Ok(None);
                };
                match operator.op {
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT => not(child),
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL => is_null(child),
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL => {
                        not(is_null(child))
                    }
                    _ => unreachable!(),
                }
            }
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN => {
                // First child is element, rest form the list.
                let children: Vec<_> = operator.children().collect();
                assert!(children.len() >= 2);
                let Some(element) = try_from_bound_expression(children[0])? else {
                    return Ok(None);
                };

                let Some(list_elements) = children
                    .iter()
                    .skip(1)
                    .map(|c| {
                        let Some(value) = try_from_bound_expression(c)? else {
                            return Ok(None);
                        };
                        Ok(Some(
                            value
                                .as_opt::<Literal>()
                                .ok_or_else(|| {
                                    vortex_err!("cannot have a non literal in a in_list")
                                })?
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
                list_contains(lit(list), element)
            }
            _ => {
                debug!(op=?operator.op, "cannot be pushed down");
                return Ok(None);
            }
        },
        duckdb::ExpressionClass::BoundFunction(func) => match func.scalar_function.name() {
            DUCKDB_FUNCTION_NAME_CONTAINS => {
                let children: Vec<_> = func.children().collect();
                assert_eq!(children.len(), 2);
                let Some(value) = try_from_bound_expression(children[0])? else {
                    return Ok(None);
                };
                let Some(pattern_lit) = like_pattern_str(children[1])? else {
                    vortex_bail!("expected pattern to be bound string")
                };
                let pattern = lit(pattern_lit);
                Like.new_expr(LikeOptions::default(), [value, pattern])
            }
            _ => {
                tracing::debug!("bound function {}", func.scalar_function.name());
                return Ok(None);
            }
        },
        duckdb::ExpressionClass::BoundConjunction(conj) => {
            let Some(children) = conj
                .children()
                .map(try_from_bound_expression)
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
    }))
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
