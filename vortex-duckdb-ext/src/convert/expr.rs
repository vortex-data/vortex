use vortex::error::{VortexError, VortexResult, vortex_bail};
use vortex::expr::{BinaryExpr, ExprRef, Operator, get_item_scope, lit};
use vortex::scalar::Scalar;

use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::{TableFilter, TableFilterClass};

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
        _ => todo!("cannot convert table filter {:?}", value),
    }
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
