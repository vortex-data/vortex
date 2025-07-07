// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex::compute::{BetweenOptions, StrictComparison};
use vortex::dtype::Nullability;
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{
    Between, BinaryExpr, ExprRef, Like, Literal, Not, Operator, and_collect, get_item_scope,
    list_contains, lit, or_collect,
};
use vortex::scalar::Scalar;

use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::{Expression, ExpressionClass, TableFilter, TableFilterClass};

const DUCKDB_FUNCTION_NAME_CONTAINS: &str = "contains";

pub fn try_from_table_filter(value: &TableFilter, col: &str) -> VortexResult<Option<ExprRef>> {
    let Some(class) = value.as_class() else {
        return Ok(None);
    };
    Ok(Some(match class {
        TableFilterClass::ConstantComparison(const_) => {
            let scalar: Scalar = const_.value.try_into()?;
            let col = get_item_scope(col);
            BinaryExpr::new_expr(col, const_.operator.try_into()?, lit(scalar))
        }
        TableFilterClass::ConjunctionAnd(conj_and) => {
            let Some(children) = conj_and
                .children()
                .map(|child| try_from_table_filter(&child, col))
                .try_collect::<_, Option<Vec<_>>, _>()?
            else {
                return Ok(None);
            };

            and_collect(children).unwrap_or_else(|| lit(true))
        }
        // This is a disjunction.
        TableFilterClass::ConjunctionOr(disjuction_or) => {
            let Some(children) = disjuction_or
                .children()
                .map(|child| try_from_table_filter(&child, col))
                .try_collect::<_, Option<Vec<_>>, _>()?
            else {
                return Ok(None);
            };

            or_collect(children).unwrap_or_else(|| lit(false))
        }
        _ => todo!("cannot convert table filter {:?}", value),
    }))
}

fn like_pattern_str(value: &Expression) -> VortexResult<Option<String>> {
    match value.as_class().vortex_expect("unknown class") {
        ExpressionClass::BoundConstant(constant) => {
            Ok(Some(format!("%{}%", constant.value.as_string())))
        }
        _ => Ok(None),
    }
}

pub fn try_from_bound_expression(value: &Expression) -> VortexResult<Option<ExprRef>> {
    let Some(value) = value.as_class() else {
        vortex_bail!("no expression class id {:?}", value.as_class_id())
    };
    Ok(Some(match value {
        ExpressionClass::BoundColumnRef(col) => get_item_scope(col.name.to_str()?),
        ExpressionClass::BoundConstant(const_) => lit(Scalar::try_from(const_.value)?),
        ExpressionClass::BoundComparison(compare) => {
            let operator: Operator = compare.op.try_into()?;

            let Some(left) = try_from_bound_expression(&compare.left)? else {
                return Ok(None);
            };
            let Some(right) = try_from_bound_expression(&compare.right)? else {
                return Ok(None);
            };

            BinaryExpr::new_expr(left, operator, right)
        }
        ExpressionClass::BoundBetween(between) => {
            let Some(array) = try_from_bound_expression(&between.input)? else {
                return Ok(None);
            };
            let Some(lower) = try_from_bound_expression(&between.lower)? else {
                return Ok(None);
            };
            let Some(upper) = try_from_bound_expression(&between.upper)? else {
                return Ok(None);
            };
            Between::between(
                array,
                lower,
                upper,
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
            )
        }
        ExpressionClass::BoundOperator(operator) => match operator.op {
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT => {
                let children = operator.children().collect_vec();
                assert_eq!(children.len(), 1);
                let Some(child) = try_from_bound_expression(&children[0])? else {
                    return Ok(None);
                };
                Not::new_expr(child)
            }
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN => {
                // First child is element, rest form the list.
                let children = operator.children().collect_vec();
                assert!(children.len() >= 2);
                let Some(element) = try_from_bound_expression(&children[0])? else {
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
                            Literal::maybe_from(&value)
                                .ok_or_else(|| {
                                    vortex_err!("cannot have a non literal in a in_list")
                                })?
                                .value()
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
            _ => todo!("operator {:?}", operator.op),
        },
        ExpressionClass::BoundFunction(func) => match func.scalar_function.name() {
            DUCKDB_FUNCTION_NAME_CONTAINS => {
                let children = func.children().collect_vec();
                assert_eq!(children.len(), 2);
                let Some(value) = try_from_bound_expression(&children[0])? else {
                    return Ok(None);
                };
                let Some(pattern_lit) = like_pattern_str(&children[1])? else {
                    vortex_bail!("expected pattern to be bound string")
                };
                let pattern = Literal::new_expr(pattern_lit);
                Like::new_expr(value, pattern, false, false)
            }
            _ => {
                log::debug!("bound function {}", func.scalar_function.name());
                return Ok(None);
            }
        },
        ExpressionClass::BoundConjunction(conj) => {
            let Some(children) = conj
                .children()
                .map(|c| try_from_bound_expression(&c))
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
            DUCKDB_VX_EXPR_TYPE::CDUCKDB_VX_EXPR_TYPE_OMPARE_NOTEQUAL => Operator::NotEq,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => Operator::Lt,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => Operator::Gt,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => Operator::Lte,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => Operator::Gte,
            _ => todo!("cannot convert {:?}", value),
        })
    }
}
