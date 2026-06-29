// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use tracing::debug;
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
use vortex::expr::lit;
use vortex::expr::not;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::scalar::Scalar;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::Between;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex_geo::extension::WellKnownBinary;
use vortex_geo::extension::native_geometry_scalar_from_wkb;
use vortex_geo::scalar_fn::distance::GeoDistance;

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
use crate::projection::DuckdbField;

fn from_bound_str(value: &duckdb::ExpressionRef) -> VortexResult<String> {
    match value.as_class().vortex_expect("unknown class") {
        BoundConstant(constant) => Ok(constant.value.as_string().as_str().to_owned()),
        _ => vortex_bail!("Expected string expression, got {:?}", value.as_class_id()),
    }
}

/// Read an `f64` from a constant expression (e.g. the `ST_DWithin` distance literal).
fn from_bound_f64(value: &duckdb::ExpressionRef) -> VortexResult<f64> {
    match value.as_class().vortex_expect("unknown class") {
        BoundConstant(constant) => f64::try_from(&Scalar::try_from(constant.value)?),
        _ => vortex_bail!("Expected f64 constant, got {:?}", value.as_class_id()),
    }
}

/// Lower a geo operand: a `GEOMETRY` literal arrives as WKB, decoded once to its native type so the
/// pushed `GeoDistance` stays native; a column ref recurses. `None` (unsupported type) skips push.
fn geo_operand(
    value: &duckdb::ExpressionRef,
    col_sub: Option<&Expression>,
) -> VortexResult<Option<Expression>> {
    if let Some(BoundConstant(constant)) = value.as_class() {
        let scalar = Scalar::try_from(constant.value)?;
        let DType::Extension(ext_dtype) = scalar.dtype() else {
            return Ok(None);
        };
        if !ext_dtype.is::<WellKnownBinary>() {
            return Ok(None);
        }
        let storage = scalar.as_extension().to_storage_scalar();
        let Some(buf) = storage.as_binary_opt().and_then(|b| b.value()) else {
            return Ok(None);
        };
        return Ok(native_geometry_scalar_from_wkb(buf.as_slice())?.map(lit));
    }
    try_from_expression_inner(value, col_sub)
}

/// Lower geo UDFs to native Vortex geo ops so the work runs in the scan. `None` otherwise.
fn try_from_geo_function(
    name: &str,
    func: &BoundFunction,
    col_sub: Option<&Expression>,
) -> VortexResult<Option<Expression>> {
    // Catch-all for every bound function: reject non-geo names before touching the children.
    if !is_geo_function(name) {
        debug!("bound function {name}");
        return Ok(None);
    }
    let children: Vec<_> = func.children().collect();
    let expr = match name.to_ascii_lowercase().as_str() {
        "vortex_dwithin" => {
            if children.len() != 3 {
                return Ok(None);
            }
            let Some(a) = geo_operand(children[0], col_sub)? else {
                return Ok(None);
            };
            let Some(b) = geo_operand(children[1], col_sub)? else {
                return Ok(None);
            };
            let distance = from_bound_f64(children[2])?;
            let geo_distance = GeoDistance.new_expr(EmptyOptions, [a, b]);
            Binary.new_expr(Operator::Lte, [geo_distance, lit(distance)])
        }
        "st_distance" => {
            if children.len() != 2 {
                return Ok(None);
            }
            let Some(a) = geo_operand(children[0], col_sub)? else {
                return Ok(None);
            };
            let Some(b) = geo_operand(children[1], col_sub)? else {
                return Ok(None);
            };
            GeoDistance.new_expr(EmptyOptions, [a, b])
        }
        _ => return Ok(None),
    };

    Ok(Some(expr))
}

/// Geo UDFs that `try_from_geo_function` lowers — shared with `can_push_expression` so the pushable
/// and lowered sets can't drift.
fn is_geo_function(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "vortex_dwithin" | "st_distance"
    )
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
        // Geo UDFs are handled here.
        name => return try_from_geo_function(name, func, col_sub),
    };

    Ok(Some(expr))
}

pub fn try_from_bound_expression(
    value: &duckdb::ExpressionRef,
) -> VortexResult<Option<Expression>> {
    try_from_expression_inner(value, None)
}

pub(super) fn try_from_bound_expression_with_col_sub(
    value: &duckdb::ExpressionRef,
    col_sub: &Expression,
) -> VortexResult<Option<Expression>> {
    try_from_expression_inner(value, Some(col_sub))
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
            let name = func.scalar_function.name();
            name == "struct_extract"
                || name == "contains"
                || name == "prefix"
                || name == "suffix"
                || name == "~~"
                || name == "!~~"
                || name == "strlen"
                || (is_geo_function(name) && func.children().all(can_push_expression))
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

pub fn try_from_projection_expression(
    value: &duckdb::ExpressionRef,
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
        _ => None,
    })
}

// If you want to add support for other expressions, also change
// can_push_expression
fn try_from_expression_inner(
    value: &duckdb::ExpressionRef,
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
