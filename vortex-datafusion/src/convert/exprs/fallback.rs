// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execute unsupported DF regions with their original input bindings.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Schema;
use datafusion_expr::Operator;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_plan::expressions::BinaryExpr;
use datafusion_physical_plan::expressions::CastExpr;
use datafusion_physical_plan::expressions::Column;
use datafusion_physical_plan::expressions::IsNotNullExpr;
use datafusion_physical_plan::expressions::IsNullExpr;
use datafusion_physical_plan::expressions::Literal;
use datafusion_physical_plan::expressions::NegativeExpr;
use datafusion_physical_plan::expressions::NotExpr;
use datafusion_physical_plan::expressions::TryCastExpr;
use datafusion_physical_plan::joins::HashExpr;
use datafusion_physical_plan::joins::HashTableLookupExpr;
use vortex::expr::Expression;

use super::Conversion;
use super::DefaultExpressionConverter;
use super::Unconverted;
use crate::df_expression::raw_expression;

pub(super) fn eager(expression: &Arc<dyn PhysicalExpr>) -> bool {
    expression.is::<Column>() || expression.is::<Literal>() || expression.is::<CastExpr>()
        || expression.is::<TryCastExpr>() || expression.is::<NegativeExpr>()
        || expression.is::<NotExpr>() || expression.is::<IsNullExpr>() || expression.is::<IsNotNullExpr>()
        // These concrete DF implementations evaluate all children over the input batch.
        || expression.is::<ScalarFunctionExpr>() || expression.is::<HashExpr>() || expression.is::<HashTableLookupExpr>()
        || expression.downcast_ref::<BinaryExpr>().is_some_and(|binary| !matches!(binary.op(), Operator::And | Operator::Or))
}

pub(super) fn compatible(expression: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Conversion<bool> {
    if let Some(binary) = expression.downcast_ref::<BinaryExpr>() {
        let left = binary.left().data_type(schema)?;
        let right = binary.right().data_type(schema)?;
        let floating = |dtype: &DataType| match dtype {
            DataType::Dictionary(_, values) => values.is_floating(),
            other => other.is_floating(),
        };
        if matches!(
            binary.op(),
            Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
        ) && (floating(&left) || floating(&right))
        {
            return Ok(false);
        }
        // DF may wrap integer arithmetic; the Vortex operation is checked. DF does
        // not expose that expression option, so preserve either behavior in DF.
        if matches!(
            binary.op(),
            Operator::Plus | Operator::Minus | Operator::Multiply
        ) && (left.is_integer() || right.is_integer())
        {
            return Ok(false);
        }
    }
    // Native extraction is equivalent only when every parent struct is non-null.
    if let Some(function) = expression.downcast_ref::<ScalarFunctionExpr>()
        && ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(function).is_some()
    {
        let [source, paths @ ..] = function.args() else {
            return Ok(true);
        };
        let mut dtype = source.data_type(schema)?;
        let mut nullable = source.nullable(schema)?;
        for path in paths {
            let Some(name) = path
                .downcast_ref::<Literal>()
                .and_then(|literal| literal.value().try_as_str())
                .flatten()
            else {
                return Ok(true);
            };
            let DataType::Struct(fields) = &dtype else {
                return Ok(true);
            };
            let Some((_, field)) = fields.find(name) else {
                return Ok(true);
            };
            if nullable {
                return Ok(false);
            }
            nullable = field.is_nullable();
            dtype = field.data_type().clone();
        }
    }
    Ok(true)
}

pub(super) fn lower(
    converter: &DefaultExpressionConverter,
    expression: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    advisory: bool,
) -> Conversion<Expression> {
    raw_expression(Arc::clone(expression), schema, &converter.session, advisory)?
        .ok_or(Unconverted::Unsupported)
}
