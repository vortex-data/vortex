//! Connectors to enable DataFusion to read Vortex data.
#![deny(missing_docs)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion_common::Result as DFResult;
use datafusion_expr::{Expr, Operator};
use vortex_array::ArrayData;
use vortex_error::vortex_err;

use crate::memory::VortexMemTable;

pub mod memory;
pub mod persistent;

const SUPPORTED_BINARY_OPS: &[Operator] = &[
    Operator::Eq,
    Operator::NotEq,
    Operator::Gt,
    Operator::GtEq,
    Operator::Lt,
    Operator::LtEq,
];

fn supported_data_types(dt: DataType) -> bool {
    let is_supported = dt.is_integer()
        || dt.is_floating()
        || dt.is_null()
        || dt == DataType::Boolean
        || dt == DataType::Binary
        || dt == DataType::Utf8
        || dt == DataType::Binary
        || dt == DataType::BinaryView
        || dt == DataType::Utf8View
        || dt == DataType::Date32
        || dt == DataType::Date64
        || matches!(
            dt,
            DataType::Timestamp(_, _) | DataType::Time32(_) | DataType::Time64(_)
        );

    if !is_supported {
        log::debug!("DataFusion data type {:?} is not supported", dt);
    }

    is_supported
}

/// Extension function to the DataFusion [`SessionContext`] for registering Vortex tables.
pub trait SessionContextExt {
    /// Register an in-memory Vortex [`ArrayData`] as a DataFusion table.
    fn register_mem_vortex<S: AsRef<str>>(&self, name: S, array: ArrayData) -> DFResult<()>;

    /// Read an in-memory Vortex [`ArrayData`] into a DataFusion [`DataFrame`].
    fn read_mem_vortex(&self, array: ArrayData) -> DFResult<DataFrame>;
}

impl SessionContextExt for SessionContext {
    fn register_mem_vortex<S: AsRef<str>>(&self, name: S, array: ArrayData) -> DFResult<()> {
        if !array.dtype().is_struct() {
            return Err(vortex_err!(
                "Vortex arrays must have struct type, found {}",
                array.dtype()
            )
            .into());
        }

        let vortex_table = VortexMemTable::new(array);
        self.register_table(name.as_ref(), Arc::new(vortex_table))
            .map(|_| ())
    }

    fn read_mem_vortex(&self, array: ArrayData) -> DFResult<DataFrame> {
        if !array.dtype().is_struct() {
            return Err(vortex_err!(
                "Vortex arrays must have struct type, found {}",
                array.dtype()
            )
            .into());
        }

        let vortex_table = VortexMemTable::new(array);

        self.read_table(Arc::new(vortex_table))
    }
}

fn can_be_pushed_down(expr: &Expr, schema: &Schema) -> bool {
    match expr {
        Expr::BinaryExpr(expr)
            if expr.op.is_logic_operator() || SUPPORTED_BINARY_OPS.contains(&expr.op) =>
        {
            can_be_pushed_down(expr.left.as_ref(), schema)
                & can_be_pushed_down(expr.right.as_ref(), schema)
        }
        Expr::Column(col) => match schema.column_with_name(col.name()) {
            Some((_, field)) => supported_data_types(field.data_type().clone()),
            _ => false,
        },
        Expr::Like(like) => {
            can_be_pushed_down(&like.expr, schema) && can_be_pushed_down(&like.pattern, schema)
        }
        Expr::Literal(lit) => supported_data_types(lit.data_type()),
        _ => {
            log::debug!("DataFusion expression can't be pushed down: {:?}", expr);
            false
        }
    }
}
