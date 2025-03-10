//! Connectors to enable DataFusion to read Vortex data.
#![deny(missing_docs)]
#![allow(clippy::cast_possible_truncation)]

use std::fmt::Debug;
use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion_common::Result as DFResult;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_expr::{Expr, Operator};
use vortex_array::ArrayRef;
use vortex_array::stats::Precision;
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
    use DataType::*;
    let is_supported = dt.is_integer()
        || dt.is_floating()
        || dt.is_null()
        || matches!(
            dt,
            Boolean
                | Utf8
                | Utf8View
                | Binary
                | BinaryView
                | Date32
                | Date64
                | Timestamp(_, _)
                | Time32(_)
                | Time64(_)
        );

    if !is_supported {
        log::debug!("DataFusion data type {:?} is not supported", dt);
    }

    is_supported
}

/// Extension function to the DataFusion [`SessionContext`] for registering Vortex tables.
pub trait SessionContextExt {
    /// Register an in-memory Vortex [`ArrayRef`] as a DataFusion table.
    fn register_mem_vortex<S: AsRef<str>>(&self, name: S, array: ArrayRef) -> DFResult<()>;

    /// Read an in-memory Vortex [`ArrayRef`] into a DataFusion [`DataFrame`].
    fn read_mem_vortex(&self, array: ArrayRef) -> DFResult<DataFrame>;
}

impl SessionContextExt for SessionContext {
    fn register_mem_vortex<S: AsRef<str>>(&self, name: S, array: ArrayRef) -> DFResult<()> {
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

    fn read_mem_vortex(&self, array: ArrayRef) -> DFResult<DataFrame> {
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

/// Extension trait to convert our [`Precision`](vortex_array::stats::Precision) to Datafusion's [`Precision`](datafusion_common::stats::Precision)
trait PrecisionExt<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    /// Convert `Precision` to the datafusion equivalent.
    fn to_df(self) -> DFPrecision<T>;
}

impl<T> PrecisionExt<T> for Precision<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Precision::Exact(v) => DFPrecision::Exact(v),
            Precision::Inexact(v) => DFPrecision::Inexact(v),
        }
    }
}

impl<T> PrecisionExt<T> for Option<Precision<T>>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Some(v) => v.to_df(),
            None => DFPrecision::Absent,
        }
    }
}
