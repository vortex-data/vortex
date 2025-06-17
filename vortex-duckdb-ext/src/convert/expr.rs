use itertools::Itertools;
use vortex::error::{VortexError, VortexResult, vortex_bail};
use vortex::expr::{BinaryExpr, ExprRef, Operator, and_collect, get_item_scope, lit, or_collect};
use vortex::scalar::Scalar;

use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::{Expression, ExpressionClass, TableFilter, TableFilterClass};

pub fn try_from_table_filter(value: &TableFilter, col: &str) -> VortexResult<ExprRef> {
    let Some(class) = value.as_class() else {
        vortex_bail!("not implemented")
    };
    match class {
        TableFilterClass::ConstantComparison(const_) => {
            let scalar: Scalar = const_.value.try_into()?;
            let col = get_item_scope(col);
            Ok(BinaryExpr::new_expr(
                col,
                const_.operator.try_into()?,
                lit(scalar),
            ))
        }
        TableFilterClass::ConjunctionAnd(conj_and) => {
            let children = conj_and
                .children()
                .map(|child| try_from_table_filter(&child, col))
                .try_collect::<_, Vec<_>, _>()?;

            Ok(and_collect(children).unwrap_or_else(|| lit(true)))
        }
        // This is a disjunction.
        TableFilterClass::ConjunctionOr(disjuction_or) => {
            let children = disjuction_or
                .children()
                .map(|child| try_from_table_filter(&child, col))
                .try_collect::<_, Vec<_>, _>()?;

            Ok(or_collect(children).unwrap_or_else(|| lit(false)))
        }
        _ => todo!("cannot convert table filter {:?}", value),
    }
}

pub fn try_from_bound_expression(value: &Expression) -> VortexResult<ExprRef> {
    let Some(value) = value.as_class() else {
        vortex_bail!("no expression class")
    };
    Ok(match value {
        ExpressionClass::BoundColumnRef(col) => get_item_scope(col.name.to_str()?),
        ExpressionClass::BoundConstant(const_) => lit(Scalar::try_from(const_.value)?),
        ExpressionClass::BoundComparison(compare) => {
            let operator: Operator = compare.op.try_into()?;

            BinaryExpr::new_expr(
                try_from_bound_expression(&compare.left)?,
                operator,
                try_from_bound_expression(&compare.right)?,
            )
        }
        ExpressionClass::BoundBetween(_) => todo!(),
        ExpressionClass::BoundOperator(_) => todo!(),
        ExpressionClass::BoundFunction(_) => todo!(),
    })
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
