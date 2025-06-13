use Nullability::Nullable;
use PType::{F64, I32};
use vortex::dtype::PType::F32;
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexError, VortexResult, vortex_bail};
use vortex::expr::{BinaryExpr, ExprRef, Operator, get_item_scope, lit};
use vortex::scalar::Scalar;

use crate::cpp;
use crate::cpp::{DUCKDB_TYPE, DUCKDB_VX_EXPR_TYPE};
use crate::duckdb::{TableFilter, TableFilterClass, Value};

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
        _ => todo!(),
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
            _ => todo!(),
        })
    }
}

impl TryFrom<Value> for Scalar {
    type Error = VortexError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value.logical_type().as_type_id() {
            DUCKDB_TYPE::DUCKDB_TYPE_INVALID => vortex_bail!("invalid duckdb type"),
            DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN => {
                if unsafe { cpp::duckdb_is_null_value(value.as_ptr()) } {
                    return Ok(Scalar::null(DType::Bool(Nullable)));
                };
                let bool = unsafe { cpp::duckdb_get_bool(value.as_ptr()) };

                Ok(Scalar::bool(bool, Nullable))
            }
            // DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => {}
            // DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {}
            DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => {
                if unsafe { cpp::duckdb_is_null_value(value.as_ptr()) } {
                    return Ok(Scalar::null(DType::Primitive(I32, Nullable)));
                };
                let value = unsafe { cpp::duckdb_get_int32(value.as_ptr()) };

                Ok(Scalar::primitive(value, Nullable))
            }
            DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => {
                if unsafe { cpp::duckdb_is_null_value(value.as_ptr()) } {
                    return Ok(Scalar::null(DType::Primitive(F32, Nullable)));
                };
                let value = unsafe { cpp::duckdb_get_float(value.as_ptr()) };

                Ok(Scalar::primitive(value, Nullable))
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => {
                if unsafe { cpp::duckdb_is_null_value(value.as_ptr()) } {
                    return Ok(Scalar::null(DType::Primitive(F64, Nullable)));
                };
                let value = unsafe { cpp::duckdb_get_double(value.as_ptr()) };

                Ok(Scalar::primitive(value, Nullable))
            }
            // DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => {}
            // DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => {}
            // DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => {}
            // DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => {}
            // DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => {}
            _ => todo!(),
        }
    }
}
